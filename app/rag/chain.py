"""RAG 问答链路：embedding → 向量检索 → 上下文组装 → LLM 生成 → 引用解析。

对外只暴露 :func:`answer_question`，返回可直接交给 API 层/前端的结果字典：

.. code-block:: python

    {
        "answer": "...",
        "citations": [
            {"index": 1, "source_file": "第3章.pdf", "page": 12, "excerpt": "..."}
        ],
        "used_chunks": [...],
        "grounded": True,
        "degraded": False,
        "degraded_reason": None,
        "model": "deepseek-chat",
        "latency_ms": 1234,
    }

设计要点：
1. 引用由服务端根据检索元数据构造，不信任模型输出的文件名和页码；
2. 模型答案中的 ``[来源:文件名 第X页]`` 会被解析并映射回真实 chunk；
3. LLM 失败时降级为检索片段原文摘录，并明确标记 ``degraded=True``；
4. 检索/Embedding 失败也返回结构化结果，不向接口层抛裸异常。
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Optional, Sequence

from app.rag import llm, prompts

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 6
DEFAULT_MIN_SCORE = 0.25
MAX_PER_DOCUMENT = 3
MAX_CITATIONS = 8
EXCERPT_LENGTH = 180

# 模型按提示词输出的来源标记：[来源:第3章_动态规划.pdf 第12页]
_SOURCE_TAG_RE = re.compile(
    r"\[\s*来源\s*[:：]\s*(?P<file>[^\[\]]+?)\s*第\s*(?P<page>\d+)\s*页\s*\]"
)
# 兼容仅给文件名的写法：[来源:第3章.pdf]
_SOURCE_FILE_ONLY_RE = re.compile(r"\[\s*来源\s*[:：]\s*(?P<file>[^\[\]第]+?)\s*\]")
# 兼容编号引用写法：[1] [2]
_NUMBER_REF_RE = re.compile(r"\[(\d{1,2})\]")

# 拒答/资料不足的常见话术，用于判断 grounded 与是否自动挂接引用
_REFUSAL_MARKERS = (
    prompts.REFUSAL_PHRASE,
    "资料不足",
    "没有相关",
    "未找到相关",
    "没有足够依据",
    "当前资料未覆盖",
    "资料中没有",
    "无法从资料",
)


# ---------------------------------------------------------------------------
# 懒加载依赖，便于单元测试替换
# ---------------------------------------------------------------------------
def _load_embedder() -> Any:
    """加载 Embedding 服务模块（由知识库模块提供 embed_texts）。"""
    from app.services import embedder

    return embedder


def _load_vectorstore() -> Any:
    """加载向量库模块（由知识库模块提供 query）。"""
    from app.rag import vectorstore

    return vectorstore


def _embed_question(question: str) -> list[float]:
    """把问题向量化；兼容 embed_texts / embed_query / embed_documents 接口。"""
    module = _load_embedder()

    embed_texts = getattr(module, "embed_texts", None)
    if callable(embed_texts):
        vectors = embed_texts([question])
        if isinstance(vectors, Sequence) and len(vectors) > 0:
            first = vectors[0]
            if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
                return [float(value) for value in first]
            if isinstance(first, (int, float)):
                return [float(value) for value in vectors]  # 兼容单条向量的实现

    embed_query = getattr(module, "embed_query", None)
    if callable(embed_query):
        vector = embed_query(question)
        if isinstance(vector, Sequence) and not isinstance(vector, (str, bytes)):
            return [float(value) for value in vector]

    raise RuntimeError("Embedding 模块未提供可用的 embed_texts/embed_query 方法。")


def _build_where(course: str, chapter: Optional[str]) -> dict[str, Any]:
    """构造向量库过滤条件：课程必选，章节可选。"""
    if chapter:
        return {
            "$and": [
                {"course": {"$eq": course}},
                {"chapter": {"$eq": chapter}},
            ]
        }
    return {"course": {"$eq": course}}


# VectorStore 实例缓存：Chroma PersistentClient 初始化成本较高，进程内复用。
_VECTORSTORE_CACHE: dict[str, Any] = {}


def _query_vectorstore(
    question_embedding: Sequence[float],
    where: dict[str, Any],
    top_k: int,
) -> Any:
    """调用向量库查询，兼容 where/filter 两类参数命名。

    知识库模块的接口可能是 ``query(vector, where=..., top_k=...)``，
    也可能是 ``query(question_embedding=..., filter=..., top_k=...)``，
    这里按常见签名依次尝试，避免跨模块集成时反复改代码。
    """
    module = _load_vectorstore()
    query_fn = getattr(module, "query", None)

    if not callable(query_fn):
        getter = getattr(module, "get_vectorstore", None)
        if callable(getter):
            try:
                store = getter()
            except Exception:
                store = None
            if store is not None:
                query_fn = getattr(store, "query", None)

    if not callable(query_fn):
        store_cls = getattr(module, "VectorStore", None)
        if callable(store_cls):
            # VectorStore 内部懒加载 Chroma，缓存实例可避免每次问答重新建客户端
            store = _VECTORSTORE_CACHE.get("default")
            if store is None:
                store = store_cls()
                _VECTORSTORE_CACHE["default"] = store
            query_fn = getattr(store, "query", None)

    if not callable(query_fn):
        raise RuntimeError("向量库模块未提供 query 方法。")

    attempts = (
        lambda: query_fn(question_embedding, where=where, top_k=top_k),
        lambda: query_fn(question_embedding, filter=where, top_k=top_k),
        lambda: query_fn(question_embedding, where, top_k),
        lambda: query_fn(question_embedding, top_k=top_k, where=where),
        lambda: query_fn(
            query_embeddings=[list(question_embedding)],
            n_results=top_k,
            where=where,
        ),
    )

    last_error: Optional[TypeError] = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:  # 参数签名不匹配，换一种写法继续
            last_error = exc
    if last_error is not None:
        raise last_error
    return []  # pragma: no cover


# ---------------------------------------------------------------------------
# 检索结果归一化
# ---------------------------------------------------------------------------
def _first_if_nested(value: Any) -> Any:
    """Chroma 返回的 documents/metadatas 是二维列表，这里取第一组。"""
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        if isinstance(first, (list, tuple)):
            return first
        return value
    return value or []


def _flatten_query_result(raw: Any) -> list[Any]:
    """把 Chroma 原始响应或 list[dict] 统一成条目列表。"""
    if raw is None:
        return []
    if isinstance(raw, dict):
        documents = _first_if_nested(raw.get("documents"))
        metadatas = _first_if_nested(raw.get("metadatas"))
        distances = _first_if_nested(raw.get("distances"))
        ids = _first_if_nested(raw.get("ids"))
        if not documents and not metadatas:
            # 有些封装直接返回 {"chunks": [...]}
            chunks = raw.get("chunks") or raw.get("results")
            return list(chunks) if isinstance(chunks, list) else []
        items: list[dict[str, Any]] = []
        for index, document in enumerate(documents or []):
            metadata = metadatas[index] if index < len(metadatas) else {}
            item: dict[str, Any] = {
                "text": document,
                "metadata": metadata if isinstance(metadata, dict) else {},
            }
            if index < len(ids):
                item["chunk_id"] = ids[index]
            if index < len(distances):
                item["distance"] = distances[index]
            items.append(item)
        return items
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return []


def _merge_metadata(item: Any) -> tuple[str, dict[str, Any], Optional[float]]:
    """从一条检索结果中提取正文、元数据和分数。"""
    if isinstance(item, dict):
        metadata = item.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        # 兼容向量库直接返回扁平 dict 的情况
        flat = {
            key: value
            for key, value in item.items()
            if key
            not in {
                "text",
                "content",
                "document",
                "page_content",
                "metadata",
                "score",
                "distance",
            }
        }
        merged = {**flat, **metadata}

        text = ""
        for key in ("text", "content", "document", "page_content"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
        if not text:
            value = merged.get("text")
            text = value.strip() if isinstance(value, str) else ""

        score = item.get("score")
        if score is None and merged.get("score") is not None:
            score = merged.get("score")
        distance = item.get("distance", merged.get("distance"))
    else:
        metadata = getattr(item, "metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        merged = dict(metadata)
        text = ""
        for key in ("text", "content", "page_content"):
            value = getattr(item, key, None)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
        score = getattr(item, "score", None)
        distance = getattr(item, "distance", None)

    numeric_score: Optional[float] = None
    try:
        if score is not None:
            numeric_score = float(score)
        elif distance is not None:
            numeric_score = 1.0 - float(distance)
    except (TypeError, ValueError):
        numeric_score = None

    return text, merged, numeric_score


def _normalize_chunks(raw: Any, top_k: int, min_score: float) -> list[dict[str, Any]]:
    """归一化、过滤、去重并截断检索结果。"""
    items = _flatten_query_result(raw)
    normalized: list[dict[str, Any]] = []

    for item in items:
        text, metadata, score = _merge_metadata(item)
        if not text:
            continue
        # 只有向量库明确给出分数时才做阈值过滤，避免无分数实现被全部过滤
        if score is not None and score < min_score:
            continue
        normalized.append(
            {
                "text": text,
                "metadata": metadata,
                "score": score,
                "chunk_id": str(
                    metadata.get("chunk_id")
                    or metadata.get("id")
                    or getattr(item, "chunk_id", "")
                    or ""
                ),
            }
        )

    # 有分数的结果按相关度降序；无分数时保持向量库原顺序
    if normalized and all(chunk["score"] is not None for chunk in normalized):
        normalized.sort(key=lambda chunk: chunk["score"], reverse=True)

    deduped: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_texts: set[str] = set()
    per_document: dict[str, int] = {}

    for chunk in normalized:
        metadata = chunk["metadata"]
        dedupe_key = chunk["chunk_id"] or (
            f"{metadata.get('source_file', '')}|{metadata.get('page', '')}|"
            f"{metadata.get('locator', '')}|{chunk['text'][:60]}"
        )
        if dedupe_key in seen_keys:
            continue
        text_key = re.sub(r"\s+", "", chunk["text"])[:120]
        if text_key in seen_texts:
            continue

        doc_key = str(metadata.get("doc_id") or metadata.get("source_file") or "")
        if doc_key and per_document.get(doc_key, 0) >= MAX_PER_DOCUMENT:
            continue

        seen_keys.add(dedupe_key)
        seen_texts.add(text_key)
        if doc_key:
            per_document[doc_key] = per_document.get(doc_key, 0) + 1
        deduped.append(chunk)
        if len(deduped) >= top_k:
            break

    return deduped


# ---------------------------------------------------------------------------
# 引用构造与解析
# ---------------------------------------------------------------------------
def _collapse_text(text: str, limit: int = EXCERPT_LENGTH) -> str:
    """压缩空白并截断为引用摘录。"""
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "……"


def _chunk_source_file(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    for key in ("source_file", "file_name", "filename", "source"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "未知文件"


def _chunk_page(chunk: dict[str, Any]) -> int:
    metadata = chunk.get("metadata") or {}
    try:
        page = int(metadata.get("page", 0) or 0)
    except (TypeError, ValueError):
        page = 0
    return max(page, 0)


def _chunk_chapter(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    return str(metadata.get("chapter") or metadata.get("section") or "").strip()


def _citation_from_chunk(chunk: dict[str, Any], index: int) -> dict[str, Any]:
    """基于检索元数据构造一条引用，文件名/页码不取自模型输出。"""
    metadata = chunk.get("metadata") or {}
    return {
        "index": index,
        "chunk_id": str(metadata.get("chunk_id") or chunk.get("chunk_id") or ""),
        "source_file": _chunk_source_file(chunk),
        "page": _chunk_page(chunk),
        "chapter": _chunk_chapter(chunk),
        "locator": str(metadata.get("locator") or ""),
        "excerpt": _collapse_text(chunk.get("text", "")),
        "score": chunk.get("score"),
    }


def _match_chunk(
    file_name: str,
    page: Optional[int],
    chunks: Sequence[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """把模型答案中的来源标记映射回真实 chunk。"""
    target = (file_name or "").strip().lower()
    if not target:
        return None

    def _name_matches(source: str) -> bool:
        source_lower = source.lower()
        target_base = os.path.basename(target)
        source_base = os.path.basename(source_lower)
        return (
            target == source_lower
            or target_base == source_base
            or target in source_lower
            or source_lower in target
        )

    candidates = [chunk for chunk in chunks if _name_matches(_chunk_source_file(chunk))]
    if not candidates:
        return None
    if page is not None:
        for chunk in candidates:
            if _chunk_page(chunk) == int(page):
                return chunk
    return candidates[0]


def _parse_citations(
    answer: str,
    chunks: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """从答案中解析来源标记并映射为结构化 citations。

    支持两种写法：
    - 首选：``[来源:第3章.pdf 第12页]``
    - 兼容：``[1]`` 编号引用（按上下文顺序对应 chunk）
    """
    citations: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()

    def _append(chunk: dict[str, Any], page_override: Optional[int] = None) -> None:
        if len(citations) >= MAX_CITATIONS:
            return
        actual_page = _chunk_page(chunk) if page_override is None else page_override
        source_file = _chunk_source_file(chunk)
        key = (source_file, actual_page, str(chunk.get("chunk_id") or ""))
        if key in seen:
            return
        seen.add(key)
        citation = _citation_from_chunk(chunk, len(citations) + 1)
        # 页码始终以元数据为准；仅当元数据缺失时才回退到模型标注的页码
        if not citation["page"] and page_override:
            citation["page"] = page_override
        citations.append(citation)

    # 1) 解析 [来源:文件名 第X页]
    for match in _SOURCE_TAG_RE.finditer(answer):
        page_text = match.group("page")
        page_value = int(page_text) if page_text.isdigit() else None
        chunk = _match_chunk(match.group("file"), page_value, chunks)
        if chunk is not None:
            _append(chunk)

    # 2) 解析仅文件名的 [来源:文件名]，页码取该文件首个片段
    if not citations:
        for match in _SOURCE_FILE_ONLY_RE.finditer(answer):
            chunk = _match_chunk(match.group("file"), None, chunks)
            if chunk is not None:
                _append(chunk)

    # 3) 解析 [1] [2] 编号引用
    if not citations:
        for match in _NUMBER_REF_RE.finditer(answer):
            position = int(match.group(1))
            if 1 <= position <= len(chunks):
                _append(chunks[position - 1])

    return citations


def _is_refusal(answer: str) -> bool:
    """判断答案是否为拒答/资料不足话术。"""
    text = (answer or "").strip()
    if not text:
        return True
    if len(text) > 200 and text.count("资料中未提及") > 0:
        # 长答案里顺带提到"资料中未提及"不一定是拒答，结合引用判断
        return False
    return any(marker in text for marker in _REFUSAL_MARKERS)


# ---------------------------------------------------------------------------
# 结果构造
# ---------------------------------------------------------------------------
def _compact_used_chunks(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """返回给接口层的检索片段摘要（不含完整长文本，避免响应过大）。"""
    compact: list[dict[str, Any]] = []
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        compact.append(
            {
                "chunk_id": str(metadata.get("chunk_id") or chunk.get("chunk_id") or ""),
                "source_file": _chunk_source_file(chunk),
                "page": _chunk_page(chunk),
                "chapter": _chunk_chapter(chunk),
                "score": chunk.get("score"),
                "excerpt": _collapse_text(chunk.get("text", ""), 120),
            }
        )
    return compact


def _empty_result(
    answer: str,
    *,
    degraded: bool = False,
    degraded_reason: Optional[str] = None,
    started_at: float,
    used_chunks: Optional[Sequence[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """构造无答案/失败场景的统一返回结构。"""
    return {
        "answer": answer,
        "citations": [],
        "used_chunks": _compact_used_chunks(used_chunks or []),
        "grounded": False,
        "degraded": degraded,
        "degraded_reason": degraded_reason,
        "model": "none",
        "latency_ms": int((time.perf_counter() - started_at) * 1000),
    }


def _degraded_answer(
    chunks: Sequence[dict[str, Any]],
    error: BaseException,
    started_at: float,
) -> dict[str, Any]:
    """LLM 不可用时，用检索片段原文摘录作为降级答案。"""
    citations: list[dict[str, Any]] = []
    excerpt_lines: list[str] = []
    for chunk in chunks[:3]:
        citation = _citation_from_chunk(chunk, len(citations) + 1)
        citations.append(citation)
        excerpt_lines.append(
            f"{prompts.format_source_tag(citation['source_file'], citation['page'])} "
            f"{citation['excerpt']}"
        )

    reason = f"{type(error).__name__}: {str(error)[:200]}"
    answer = (
        "【已降级】大模型暂时不可用，以下为检索到的课程资料原文摘录，未经模型总结，"
        "请结合原文核对：\n\n"
        + ("\n\n".join(excerpt_lines) if excerpt_lines else "（未检索到可用的资料片段）")
        + f"\n\n（降级原因：{reason}）"
    )
    logger.warning("RAG 问答降级为原文摘录：%s", reason)

    return {
        "answer": answer,
        "citations": citations,
        "used_chunks": _compact_used_chunks(chunks),
        "grounded": bool(citations),
        "degraded": True,
        "degraded_reason": "llm_unavailable",
        "model": "none",
        "latency_ms": int((time.perf_counter() - started_at) * 1000),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def answer_question(
    question: str,
    course: str,
    chapter: Optional[str] = None,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    history: Optional[Sequence[dict[str, Any]]] = None,
    question_embedding: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """完成一次课程资料问答，返回答案、引用和检索片段。

    Args:
        question: 用户问题，必填。
        course: 课程名称，作为向量库的硬过滤条件，必填。
        chapter: 可选章节，传入后只在该章节范围内检索。
        top_k: 参与上下文组装的最大片段数，默认 6。
        min_score: 相似度阈值，仅在向量库返回分数时生效。
        history: 可选历史对话，最多取最近若干轮。
        question_embedding: 已计算好的问题向量；主要供单元测试注入，
            为空时才调用 Embedding 服务。

    Returns:
        包含 ``answer``、``citations``、``used_chunks``、``grounded``、
        ``degraded`` 等字段的字典。

    Raises:
        ValueError: question 或 course 为空。
    """
    started_at = time.perf_counter()
    normalized_question = (question or "").strip()
    normalized_course = (course or "").strip()
    if not normalized_question:
        raise ValueError("question 不能为空。")
    if not normalized_course:
        raise ValueError("course 不能为空。")

    top_k = max(1, min(int(top_k or DEFAULT_TOP_K), 20))
    min_score = float(min_score if min_score is not None else DEFAULT_MIN_SCORE)
    normalized_chapter = (chapter or "").strip() or None

    # 1. 问题向量化
    if question_embedding is None:
        try:
            question_embedding = _embed_question(normalized_question)
        except Exception as exc:
            logger.exception("问题向量化失败")
            return _empty_result(
                "问题向量化失败，暂时无法检索课程资料。请检查 Embedding 模型配置后重试。",
                degraded=True,
                degraded_reason="embedding_failed",
                started_at=started_at,
            )

    # 2. 向量检索
    where = _build_where(normalized_course, normalized_chapter)
    try:
        raw_result = _query_vectorstore(question_embedding, where, top_k)
    except Exception as exc:
        logger.exception("向量检索失败")
        return _empty_result(
            f"课程资料检索失败：{type(exc).__name__}。请检查向量库状态后重试。",
            degraded=True,
            degraded_reason="retrieval_failed",
            started_at=started_at,
        )

    chunks = _normalize_chunks(raw_result, top_k=top_k, min_score=min_score)
    if not chunks:
        scope = f"《{normalized_chapter}》" if normalized_chapter else "当前课程"
        return _empty_result(
            f"{scope}资料中未提及该问题的相关内容。建议更换关键词、放宽章节范围，"
            "或先上传对应章节的资料。",
            started_at=started_at,
        )

    # 3. 组装上下文并调用 LLM
    messages = prompts.build_rag_messages(
        normalized_question,
        normalized_course,
        chunks,
        chapter=normalized_chapter,
        history=history,
    )

    try:
        answer = llm.chat(messages)
    except Exception as exc:  # LLMError 及其子类，或未预期的网络异常
        return _degraded_answer(chunks, exc, started_at)

    if not isinstance(answer, str) or not answer.strip():
        return _degraded_answer(chunks, RuntimeError("LLM 返回空答案"), started_at)

    answer = answer.strip()

    # 4. 解析引用并由服务端重建 citation
    citations = _parse_citations(answer, chunks)
    citation_auto_attached = False
    if not citations and not _is_refusal(answer):
        # 模型没有按格式标注来源：自动挂接最相关片段，并保留标记提醒前端
        citations = [_citation_from_chunk(chunks[0], 1)]
        citation_auto_attached = True

    try:
        model_name = llm.get_active_model_name()
    except Exception:  # pragma: no cover - 配置异常不影响答案返回
        model_name = "unknown"

    grounded = bool(citations) and not _is_refusal(answer)
    if citation_auto_attached:
        grounded = False

    return {
        "answer": answer,
        "citations": citations,
        "used_chunks": _compact_used_chunks(chunks),
        "grounded": grounded,
        "degraded": False,
        "degraded_reason": None,
        "model": model_name,
        "latency_ms": int((time.perf_counter() - started_at) * 1000),
        "citation_auto_attached": citation_auto_attached,
    }


__all__ = ["answer_question"]
