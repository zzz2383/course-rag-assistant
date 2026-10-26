"""课程资料问答接口。

对外提供 ``POST /api/chat``：接收问题、课程与可选章节，调用 RAG 链
（``app.rag.chain.answer_question``）完成检索增强生成，并返回答案与引用。

响应结构按接口约定固定为::

    {
        "answer": "中文答案，含 [1] 等引用标记",
        "citations": [
            {"source_file": "第3章.pdf", "page": 12, "excerpt": "原文片段……"}
        ]
    }

异常统一抛出 :class:`app.api.common.APIError`，由全局处理器转换为
``{code, message, detail, request_id}`` 结构。
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from app.api.common import APIError, invoke, truncate_text
from app.models.schemas import ChatRequest

logger = logging.getLogger("course_rag.api.chat")

router = APIRouter(prefix="/chat", tags=["问答"])


def _citation_to_dict(raw: Any) -> dict[str, Any] | None:
    """把 RAG 链返回的引用规范化为前端约定的三个字段。

    兼容 dict、Pydantic 模型与带 ``metadata`` 嵌套的检索片段；缺少文件名
    或完全无内容的引用会被丢弃，避免前端渲染空卡片。
    """
    if raw is None:
        return None
    if hasattr(raw, "model_dump"):
        data = raw.model_dump()
    elif isinstance(raw, dict):
        data = raw
    else:
        data = {key: getattr(raw, key) for key in ("source_file", "page", "excerpt") if hasattr(raw, key)}

    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    source_file = (
        data.get("source_file")
        or data.get("file_name")
        or metadata.get("source_file")
        or ""
    )
    source_file = str(source_file).strip()
    if not source_file:
        return None

    page_value = data.get("page", metadata.get("page", 0))
    try:
        page = max(int(page_value or 0), 0)
    except (TypeError, ValueError):
        page = 0

    excerpt = (
        data.get("excerpt")
        or data.get("snippet")
        or data.get("text")
        or data.get("content")
        or ""
    )
    return {
        "source_file": source_file,
        "page": page,
        "excerpt": truncate_text(excerpt, 200),
    }


def _normalize_citations(raw_citations: Any) -> list[dict[str, Any]]:
    """过滤无效引用并保持原有顺序，返回非空数组（无引用时为 ``[]``）。"""
    if not isinstance(raw_citations, (list, tuple)):
        return []
    citations: list[dict[str, Any]] = []
    for raw in raw_citations:
        citation = _citation_to_dict(raw)
        if citation is not None:
            citations.append(citation)
    return citations


@router.post("")
async def chat(request: Request, payload: ChatRequest) -> dict[str, Any]:
    """课程资料问答。

    Args:
        payload: 包含 ``question``、``course`` 与可选 ``chapter`` 的请求体。

    Returns:
        ``{"answer": str, "citations": [{source_file, page, excerpt}]}``

    Raises:
        APIError: 参数非法（400）或问答链不可用（502）。
    """
    # 延迟导入：避免应用启动时加载 Embedding/LLM 等重量级依赖。
    from app.rag import chain

    answer_question = getattr(chain, "answer_question", None)
    if not callable(answer_question):
        raise APIError(500, "CHAT_UNAVAILABLE", "问答服务未正确初始化")

    history = [
        message.model_dump() if hasattr(message, "model_dump") else dict(message)
        for message in (payload.history or [])
    ]

    try:
        result = await invoke(
            answer_question,
            {
                "question": payload.question,
                "course": payload.course,
                "chapter": payload.chapter,
                "top_k": payload.top_k,
                "history": history or None,
            },
        )
    except APIError:
        raise
    except ValueError as exc:
        # 链条对空问题/空课程等参数错误抛 ValueError
        raise APIError(400, "INVALID_ARGUMENT", str(exc)) from exc
    except Exception as exc:  # 向量库、Embedding、LLM 等未预期异常
        logger.exception("问答处理失败 question=%r course=%r", payload.question, payload.course)
        raise APIError(
            502,
            "CHAT_FAILED",
            "问答服务暂时不可用，请稍后重试",
            detail={"exception": type(exc).__name__},
        ) from exc

    if hasattr(result, "model_dump"):
        result = result.model_dump()
    if not isinstance(result, dict):
        raise APIError(502, "CHAT_FAILED", "问答服务返回了无法识别的结果")

    answer = str(result.get("answer") or "").strip()
    if not answer:
        answer = "根据现有课程资料，无法确定该问题的答案。"

    response = {
        "answer": answer,
        "citations": _normalize_citations(result.get("citations")),
    }

    logger.info(
        "问答完成 request_id=%s course=%s chapter=%s citations=%d",
        getattr(request.state, "request_id", "-"),
        payload.course,
        payload.chapter,
        len(response["citations"]),
    )
    return response
