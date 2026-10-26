"""课程资料管理接口。

提供三个端点：

- ``POST /api/documents/upload``：multipart 上传，校验扩展名与大小后落盘，
  依次执行“解析 → 分块 → 向量化 → 写入向量库”；
- ``GET /api/documents``：列出已入库文档，支持按课程过滤与分页；
- ``DELETE /api/documents/{doc_id}``：删除文档的向量、原始文件与索引元数据。

上传的原始文件保存在 ``data/uploads/{doc_id}/{原文件名}``，文档级元数据
保存在同目录的 ``meta.json``，因此即使向量库暂时不可用，文档列表与删除
操作仍然可用。
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from app.api.common import (
    APIError,
    ensure_data_dirs,
    get_settings_safe,
    invoke,
)

logger = logging.getLogger("course_rag.api.documents")

router = APIRouter(prefix="/documents", tags=["资料管理"])

# 支持的文件类型与大小上限（产品约束为 50MB，优先于配置默认值）。
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".pptx", ".docx", ".txt"})
MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
_UNASSIGNED_CHAPTER = "未分章"

# 向量库单例：VectorStore 内部懒加载 Chroma 客户端，重复实例化会重复打开连接。
_VECTORSTORE: Any | None = None


# ---------------------------------------------------------------------------
# 路径与元数据
# ---------------------------------------------------------------------------


def _validate_doc_id(doc_id: str) -> str:
    """校验 doc_id 只包含安全字符，防止路径穿越。"""
    value = (doc_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise APIError(400, "INVALID_ARGUMENT", "doc_id 格式非法")
    return value


def _upload_root() -> Path:
    """返回原始资料根目录（``data/uploads``）。"""
    return ensure_data_dirs()["uploads"]


def _doc_dir(doc_id: str) -> Path:
    """返回单个文档的存储目录。"""
    return _upload_root() / doc_id


def _meta_path(doc_id: str) -> Path:
    """返回文档元数据文件路径。"""
    return _doc_dir(doc_id) / "meta.json"


def _safe_filename(filename: str | None) -> str:
    """净化上传文件名：仅保留 basename，移除路径分隔符与控制字符。

    保留中文、空格与常见符号，长度限制在 120 字符以内，避免 Windows
    路径过长；净化后为空时回退为 ``upload``。
    """
    raw = str(filename or "")
    # 同时处理 Windows 与 POSIX 路径分隔符，防止路径穿越
    name = raw.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip().strip(".")
    if not name:
        return "upload"
    suffix = Path(name).suffix
    stem = Path(name).stem
    if len(stem) > 100:
        stem = stem[:100]
    if len(suffix) > 16:
        suffix = ""
    # Windows 保留设备名不能直接作为文件名，统一加前缀规避
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {
        f"LPT{i}" for i in range(1, 10)
    }
    if stem.upper() in reserved:
        stem = f"file_{stem}"
    return f"{stem}{suffix}" or "upload"


def _write_meta(doc_id: str, metadata: dict[str, Any]) -> None:
    """写入文档级元数据，供列表、删除与排障使用。"""
    path = _meta_path(doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_meta(doc_id: str) -> dict[str, Any] | None:
    """读取文档元数据；损坏或缺失时返回 ``None``。"""
    path = _meta_path(doc_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("读取文档元数据失败：%s", path, exc_info=True)
        return None
    return data if isinstance(data, dict) else None


def _iter_meta() -> list[dict[str, Any]]:
    """扫描上传目录，返回全部文档元数据（按上传时间倒序）。"""
    root = _upload_root()
    documents: list[dict[str, Any]] = []
    if not root.is_dir():
        return documents
    for child in root.iterdir():
        if not child.is_dir():
            continue
        metadata = _read_meta(child.name)
        if metadata:
            documents.append(metadata)
    documents.sort(key=lambda item: str(item.get("uploaded_at") or ""), reverse=True)
    return documents


# ---------------------------------------------------------------------------
# 向量库与业务服务
# ---------------------------------------------------------------------------


def _get_vectorstore() -> Any:
    """获取进程级 VectorStore 单例。

    使用延迟导入，避免未安装 chromadb 时影响应用启动与其他接口。
    """
    global _VECTORSTORE
    if _VECTORSTORE is None:
        from app.rag.vectorstore import VectorStore

        _VECTORSTORE = VectorStore()
    return _VECTORSTORE


def _chunk_text(chunk: Any) -> str:
    """从 ChunkRecord（或字典）中取出文本。"""
    if isinstance(chunk, dict):
        return str(chunk.get("text") or chunk.get("content") or "")
    return str(getattr(chunk, "text", "") or "")


def _chunk_page(chunk: Any) -> int:
    """从 ChunkRecord（或字典）中取出页码。"""
    metadata = chunk.get("metadata") if isinstance(chunk, dict) else getattr(chunk, "metadata", None)
    if isinstance(metadata, dict):
        value = metadata.get("page")
    else:
        value = getattr(metadata, "page", 0) if metadata is not None else 0
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _chunk_chapter(chunk: Any) -> str:
    """从 ChunkRecord（或字典）中取出章节名。"""
    metadata = chunk.get("metadata") if isinstance(chunk, dict) else getattr(chunk, "metadata", None)
    if isinstance(metadata, dict):
        value = metadata.get("chapter")
    else:
        value = getattr(metadata, "chapter", "") if metadata is not None else ""
    return str(value or "").strip()


# ---------------------------------------------------------------------------
# 上传处理
# ---------------------------------------------------------------------------


async def _save_upload(upload: UploadFile, destination: Path) -> int:
    """以流式方式保存上传文件，并在写入过程中强制校验大小上限。

    Returns:
        实际写入的字节数。

    Raises:
        APIError: 文件为空（422）或超过大小限制（413）。
    """
    total = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("wb") as handle:
            while True:
                block = await upload.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > MAX_UPLOAD_BYTES:
                    raise APIError(
                        413,
                        "FILE_TOO_LARGE",
                        f"文件超过 {MAX_UPLOAD_MB}MB 上限，请压缩或拆分后再上传",
                    )
                handle.write(block)
    except APIError:
        raise
    except OSError as exc:
        raise APIError(500, "UPLOAD_FAILED", "文件保存失败，请稍后重试") from exc
    finally:
        await upload.close()

    if total == 0:
        raise APIError(422, "EMPTY_DOCUMENT", "上传文件为空，请检查后重新上传")
    return total


async def _process_upload(
    upload: UploadFile,
    *,
    course: str,
    chapter: str | None,
) -> dict[str, Any]:
    """处理单个文件：校验 → 落盘 → 解析 → 分块 → 向量化 → 入库。"""
    original_name = str(upload.filename or "").strip()
    if not original_name:
        raise APIError(400, "INVALID_ARGUMENT", "上传文件缺少文件名")

    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = "、".join(sorted(ALLOWED_EXTENSIONS))
        raise APIError(
            415,
            "UNSUPPORTED_FILE_TYPE",
            f"不支持的文件类型 {suffix or '(无扩展名)'}，仅支持：{allowed}",
        )

    # 部分客户端会带上 size，可提前拒绝超大文件，避免无谓落盘。
    declared_size = getattr(upload, "size", None)
    if declared_size is not None and int(declared_size) > MAX_UPLOAD_BYTES:
        raise APIError(
            413,
            "FILE_TOO_LARGE",
            f"文件超过 {MAX_UPLOAD_MB}MB 上限，请压缩或拆分后再上传",
        )

    doc_id = f"doc_{uuid.uuid4().hex[:16]}"
    safe_name = _safe_filename(original_name)
    doc_directory = _doc_dir(doc_id)
    stored_path = doc_directory / safe_name
    uploaded_at = datetime.now().astimezone().isoformat(timespec="seconds")

    try:
        size = await _save_upload(upload, stored_path)
    except Exception:
        shutil.rmtree(doc_directory, ignore_errors=True)
        raise

    # 延迟导入重量级服务，避免未安装解析依赖时影响应用启动。
    store: Any | None = None
    try:
        from app.services import chunker, embedder, parser

        parse_document = getattr(parser, "parse_document", None)
        split_blocks = getattr(chunker, "split_blocks", None)
        embed_texts = getattr(embedder, "embed_texts", None)
        if not callable(parse_document) or not callable(split_blocks):
            raise RuntimeError("解析或分块服务未正确实现")

        try:
            blocks = await invoke(parse_document, {"path": str(stored_path)})
        except Exception as exc:
            raise APIError(
                422,
                "PARSE_FAILED",
                f"文档解析失败：{exc}",
                detail={"file_name": original_name},
            ) from exc

        if not blocks:
            raise APIError(422, "EMPTY_DOCUMENT", "未从文档中提取到有效文本，可能是扫描件或空文件")

        try:
            chunks = await invoke(
                split_blocks,
                {
                    "blocks": blocks,
                    "course": course,
                    "chapter": chapter,
                    "doc_id": doc_id,
                },
            )
        except Exception as exc:
            raise APIError(
                422,
                "CHUNK_FAILED",
                f"文档分块失败：{exc}",
                detail={"file_name": original_name},
            ) from exc

        chunks = list(chunks or [])
        if not chunks:
            raise APIError(422, "EMPTY_DOCUMENT", "文档未切分出有效文本块，请检查内容后重试")

        texts = [text for text in (_chunk_text(chunk) for chunk in chunks) if text.strip()]
        if not callable(embed_texts):
            raise RuntimeError("向量化服务未正确实现")
        try:
            vectors = await invoke(embed_texts, {"texts": texts})
        except Exception as exc:
            raise APIError(
                502,
                "EMBEDDING_FAILED",
                "文本向量化失败，请检查 Embedding 模型配置后重试",
                detail={"exception": type(exc).__name__},
            ) from exc

        store = _get_vectorstore()
        try:
            added = await invoke(
                store.add_chunks,
                {"chunks": chunks, "vectors": vectors},
            )
        except Exception as exc:
            # 向量写入失败时回滚已写入的残块，保持“文档不存在”的一致状态
            try:
                await invoke(store.delete_document, {"doc_id": doc_id})
            except Exception:  # pragma: no cover - 回滚失败只记录日志
                logger.warning("回滚向量失败 doc_id=%s", doc_id, exc_info=True)
            raise APIError(
                500,
                "INDEX_FAILED",
                "文档向量入库失败，请稍后重试",
                detail={"exception": type(exc).__name__},
            ) from exc

        chunk_count = int(added or len(chunks))
        page_count = max((_chunk_page(chunk) for chunk in chunks), default=0)
        char_count = sum(len(str(block.get("text") or "")) for block in blocks if isinstance(block, dict))
        chapter_name = next(
            (name for name in (_chunk_chapter(chunk) for chunk in chunks) if name),
            chapter or _UNASSIGNED_CHAPTER,
        )

        metadata = {
            "doc_id": doc_id,
            "file_name": safe_name,
            "original_name": original_name,
            "file_type": suffix.lstrip("."),
            "size": size,
            "page_count": page_count,
            "char_count": char_count,
            "chunk_count": chunk_count,
            "course": course,
            "chapter": chapter_name,
            "status": "indexed",
            "uploaded_at": uploaded_at,
            "error": None,
        }
        _write_meta(doc_id, metadata)
        logger.info(
            "资料入库成功 doc_id=%s file=%s chunks=%d size=%d",
            doc_id,
            safe_name,
            chunk_count,
            size,
        )
        return metadata
    except APIError:
        # 业务失败：清理落盘文件与可能写入的向量
        shutil.rmtree(doc_directory, ignore_errors=True)
        if store is not None:
            try:
                await invoke(store.delete_document, {"doc_id": doc_id})
            except Exception:  # pragma: no cover - 清理失败不影响原始错误
                pass
        raise
    except Exception as exc:
        shutil.rmtree(doc_directory, ignore_errors=True)
        logger.exception("资料上传处理失败 file=%s", original_name)
        raise APIError(
            500,
            "INTERNAL_ERROR",
            "资料处理失败，请稍后重试",
            detail={"exception": type(exc).__name__},
        ) from exc


@router.post("/upload", status_code=201, summary="上传并索引课程资料")
async def upload_documents(
    request: Request,
    files: list[UploadFile] | None = File(default=None, description="批量上传文件字段"),
    file: UploadFile | None = File(default=None, description="单文件上传字段"),
    course: str = Form(default="未分类课程", description="课程名称"),
    chapter: str | None = Form(default=None, description="章节名称，可选"),
) -> JSONResponse:
    """上传一个或多个课程资料并建立索引。

    - 支持 ``files`` 批量字段与 ``file`` 单文件字段；
    - 扩展名仅允许 ``pdf/pptx/docx/txt``，单文件不超过 50MB；
    - 全部成功返回 201；部分失败返回 207，逐项给出错误信息；
    - 全部失败时抛出首个业务异常（统一错误结构）。
    """
    uploads: list[UploadFile] = []
    if files:
        uploads.extend(files)
    if file is not None:
        uploads.insert(0, file)

    if not uploads:
        raise APIError(400, "INVALID_ARGUMENT", "请至少上传一个文件（字段名 files 或 file）")

    settings = get_settings_safe()
    max_files = int(getattr(settings, "max_files_per_upload", 10) or 10)
    if len(uploads) > max_files:
        raise APIError(
            400,
            "INVALID_ARGUMENT",
            f"单次最多上传 {max_files} 个文件，请分批上传",
        )

    course_name = (course or "").strip() or "未分类课程"
    chapter_name = (chapter or "").strip() or None

    items: list[dict[str, Any]] = []
    failures: list[APIError] = []
    for upload in uploads:
        try:
            items.append(
                await _process_upload(upload, course=course_name, chapter=chapter_name)
            )
        except APIError as exc:
            failures.append(exc)
            items.append(
                {
                    "doc_id": None,
                    "file_name": str(upload.filename or ""),
                    "status": "failed",
                    "chunk_count": 0,
                    "page_count": 0,
                    "char_count": 0,
                    "error": exc.message,
                }
            )

    if not any(item.get("status") == "indexed" for item in items):
        # 全部失败时保持 HTTP 语义（415/413/422/500），不返回 207。
        raise failures[0]

    success_count = sum(1 for item in items if item.get("status") == "indexed")
    payload: dict[str, Any] = {
        "items": items,
        "success_count": success_count,
        "failed_count": len(items) - success_count,
    }
    if len(items) == 1:
        # 单文件上传时同时把文档信息平铺到顶层，方便前端直接取 doc_id。
        payload.update(items[0])

    status_code = 201 if success_count == len(items) else 207
    return JSONResponse(status_code=status_code, content=payload)


# ---------------------------------------------------------------------------
# 文档列表与详情
# ---------------------------------------------------------------------------


def _normalize_document_list_item(raw: dict[str, Any], meta: dict[str, Any] | None) -> dict[str, Any]:
    """把向量库摘要与本地元数据合并为统一的文档对象。"""
    source_file = str(raw.get("source_file") or raw.get("file_name") or "").strip()
    doc_id = str(raw.get("doc_id") or (meta or {}).get("doc_id") or "").strip()
    chapter = str(raw.get("chapter") or (meta or {}).get("chapter") or _UNASSIGNED_CHAPTER)
    chunk_count = raw.get("chunk_count")
    if chunk_count is None:
        chunk_count = (meta or {}).get("chunk_count", 0)
    item = {
        "doc_id": doc_id,
        "file_name": (meta or {}).get("file_name") or source_file,
        "file_type": raw.get("file_type") or (meta or {}).get("file_type", ""),
        "course": str(raw.get("course") or (meta or {}).get("course") or ""),
        "chapter": chapter,
        "chunk_count": int(chunk_count or 0),
        "page_count": int(raw.get("page_count") or (meta or {}).get("page_count") or 0),
        "status": (meta or {}).get("status") or "indexed",
    }
    for key in ("size", "char_count", "uploaded_at", "error"):
        if meta and key in meta:
            item[key] = meta[key]
    return item


@router.get("", summary="查询文档列表")
async def list_documents(
    course: str | None = Query(default=None, description="按课程名称过滤"),
    page: int = Query(default=1, ge=1, description="页码，从 1 开始"),
    size: int = Query(default=50, ge=1, le=200, description="每页数量"),
) -> dict[str, Any]:
    """列出已入库文档。

    以向量库聚合结果为准，并用本地 ``meta.json`` 补充文件大小、上传时间等
    信息；向量库不可用时降级为本地元数据列表，保证管理页面可用。
    """
    documents: dict[str, dict[str, Any]] = {}

    for meta in _iter_meta():
        doc_id = str(meta.get("doc_id") or "")
        if doc_id:
            documents[doc_id] = _normalize_document_list_item({}, meta)

    try:
        store = _get_vectorstore()
        raw_documents = await invoke(store.list_documents, {"course": course})
    except Exception:
        logger.warning("读取向量库文档清单失败，降级为本地元数据", exc_info=True)
        raw_documents = []

    for raw in raw_documents or []:
        if not isinstance(raw, dict):
            continue
        doc_id = str(raw.get("doc_id") or "")
        if not doc_id:
            continue
        merged = _normalize_document_list_item(raw, documents.get(doc_id) or _read_meta(doc_id))
        documents[doc_id] = merged

    items = list(documents.values())
    if course:
        course_name = course.strip()
        items = [item for item in items if item.get("course") == course_name]
    items.sort(key=lambda item: str(item.get("uploaded_at") or ""), reverse=True)

    total = len(items)
    start = (page - 1) * size
    return {
        "items": items[start : start + size],
        "total": total,
        "page": page,
        "size": size,
    }


@router.get("/{doc_id}", summary="查询单个文档详情")
async def get_document(doc_id: str) -> dict[str, Any]:
    """返回单个文档的详细信息；不存在时返回 404。"""
    document_id = _validate_doc_id(doc_id)
    metadata = _read_meta(document_id)
    if metadata is None:
        raise APIError(404, "DOC_NOT_FOUND", f"文档不存在：{document_id}")
    return metadata


# ---------------------------------------------------------------------------
# 删除文档
# ---------------------------------------------------------------------------


@router.delete("/{doc_id}", summary="删除文档及其向量")
async def delete_document(doc_id: str) -> dict[str, Any]:
    """删除指定文档的向量、原始文件、解析产物与元数据。

    删除成功返回实际清理的向量块数；文档完全不存在时返回 404。
    """
    document_id = _validate_doc_id(doc_id)
    metadata = _read_meta(document_id)
    directory = _doc_dir(document_id)
    existed_locally = metadata is not None or directory.exists()

    chunk_count = 0
    store = None
    try:
        store = _get_vectorstore()
        chunk_count = int(
            await invoke(store.delete_document, {"doc_id": document_id}) or 0
        )
    except Exception as exc:
        logger.warning("删除向量失败 doc_id=%s", document_id, exc_info=True)
        if not existed_locally:
            raise APIError(
                503,
                "VECTORSTORE_UNAVAILABLE",
                "向量库暂时不可用，无法确认文档是否存在，请稍后重试",
                detail={"exception": type(exc).__name__},
            ) from exc

    if not existed_locally and chunk_count == 0:
        raise APIError(404, "DOC_NOT_FOUND", f"文档不存在：{document_id}")

    # 清理原始文件、meta.json 与解析中间产物；文件缺失不视为失败。
    shutil.rmtree(directory, ignore_errors=True)
    parsed_file = ensure_data_dirs()["parsed"] / f"{document_id}.json"
    parsed_file.unlink(missing_ok=True)

    if not chunk_count and metadata:
        chunk_count = int(metadata.get("chunk_count") or 0)

    logger.info("文档已删除 doc_id=%s chunks=%d", document_id, chunk_count)
    return {
        "deleted": True,
        "doc_id": document_id,
        "chunk_count": chunk_count,
    }
