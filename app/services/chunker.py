"""文本分块模块。

使用 LangChain 的 ``RecursiveCharacterTextSplitter`` 对解析结果做递归切分，
把同一页（同一 ``page``）的相邻短块先聚合成不超过 ``CHUNK_SIZE`` 的文本，
再按“段落 → 换行 → 中文标点 → 空格”的优先级切分，尽量保持语义完整。

每个 Chunk 会继承 ``page``、``source_file`` 元数据，并生成 UUID 形式的
``chunk_id``、文档内递增的 ``chunk_index`` 与内容哈希。
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.models.schemas import UNASSIGNED_CHAPTER, ChunkMetadata, ChunkRecord
from app.services.parser import parse_document

# 中文课程资料常用的递归分隔符，优先在自然语义边界切分。
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


@dataclass
class _Block:
    """规范化后的文本块，用于分块前的聚合。"""

    text: str
    page: int
    source_file: str


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """构造递归字符分块器。

    Args:
        chunk_size: 块最大字符数，默认取 ``settings.chunk_size``。
        chunk_overlap: 相邻块重叠字符数，默认取 ``settings.chunk_overlap``。

    Raises:
        ValueError: 参数非正或重叠不小于块大小。
    """
    settings = get_settings()
    size = int(chunk_size) if chunk_size is not None else settings.chunk_size
    overlap = int(chunk_overlap) if chunk_overlap is not None else settings.chunk_overlap
    if size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk_overlap 必须满足 0 <= chunk_overlap < chunk_size")

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=DEFAULT_SEPARATORS,
        keep_separator=True,
        add_start_index=False,
    )


def _normalize_blocks(
    blocks: Sequence[dict[str, Any]],
    default_source: str = "",
) -> list[_Block]:
    """校验并规范化解析器输出的文本块。"""
    normalized: list[_Block] = []
    for raw in blocks:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        try:
            page = int(raw.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        source = str(raw.get("source_file") or default_source or "unknown")
        normalized.append(_Block(text=text, page=max(page, 0), source_file=source))
    return normalized


def _group_blocks(blocks: list[_Block], max_chars: int) -> list[_Block]:
    """把同一页/同一来源的相邻短块聚合为不超过 ``max_chars`` 的块。

    聚合只发生在相同 ``page`` 与 ``source_file`` 之间，因此不会把 PDF 的
    不同页或 PPT 的不同幻灯片混在一起。
    """
    groups: list[_Block] = []
    buffer: list[str] = []
    buffer_page = 0
    buffer_source = ""
    buffer_length = 0

    def flush() -> None:
        nonlocal buffer, buffer_length
        if buffer:
            groups.append(
                _Block(
                    text="\n".join(buffer),
                    page=buffer_page,
                    source_file=buffer_source,
                )
            )
        buffer = []
        buffer_length = 0

    for block in blocks:
        if buffer and (
            block.page != buffer_page
            or block.source_file != buffer_source
            or buffer_length + 1 + len(block.text) > max_chars
        ):
            flush()
        if not buffer:
            buffer_page = block.page
            buffer_source = block.source_file
        buffer.append(block.text)
        buffer_length += len(block.text) + (1 if len(buffer) > 1 else 0)
    flush()
    return groups


def _locator(page: int, chunk_index: int, file_type: str) -> str:
    """生成可读定位串：``page:n``、``slide:n`` 或 ``chunk:n``。"""
    if page > 0:
        return f"slide:{page}" if file_type == "pptx" else f"page:{page}"
    return f"chunk:{chunk_index}"


def split_blocks(
    blocks: Sequence[dict[str, Any]],
    *,
    course: str,
    chapter: str | None = None,
    doc_id: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> list[ChunkRecord]:
    """把解析文本块切分为带完整元数据的 Chunk 列表。

    Args:
        blocks: ``parse_document`` 返回的文本块（需含 ``text/page/source_file``）。
        course: 课程名称，写入每条 Chunk 的 metadata。
        chapter: 章节名称；为空时使用 ``"未分章"``。
        doc_id: 文档 ID；为空时自动生成 ``doc_<uuid16>``。
        chunk_size: 块最大字符数，默认取配置。
        chunk_overlap: 相邻块重叠字符数，默认取配置。
        splitter: 可注入自定义分块器（测试用），默认按配置构造。

    Returns:
        ChunkRecord 列表；输入为空时返回空列表。

    Raises:
        ValueError: 课程名为空或分块参数非法。
    """
    course_name = (course or "").strip()
    if not course_name:
        raise ValueError("course 不能为空")

    settings = get_settings()
    size = int(chunk_size) if chunk_size is not None else settings.chunk_size
    text_splitter = splitter or build_splitter(size, chunk_overlap)

    normalized = _normalize_blocks(blocks)
    if not normalized:
        return []

    chapter_name = (chapter or "").strip() or UNASSIGNED_CHAPTER
    document_id = doc_id or f"doc_{uuid4().hex[:16]}"

    records: list[ChunkRecord] = []
    chunk_index = 0
    for group in _group_blocks(normalized, size):
        file_type = Path(group.source_file).suffix.lstrip(".").lower()
        for piece in text_splitter.split_text(group.text):
            text = piece.strip()
            if not text:
                continue
            chunk_id = f"ck_{uuid4().hex}"
            metadata = ChunkMetadata(
                course=course_name,
                chapter=chapter_name,
                source_file=group.source_file,
                page=group.page,
                chunk_id=chunk_id,
                doc_id=document_id,
                file_type=file_type,
                locator=_locator(group.page, chunk_index, file_type),
                chunk_index=chunk_index,
                section="",
                content_hash=sha1(text.encode("utf-8")).hexdigest(),
            )
            records.append(ChunkRecord(chunk_id=chunk_id, text=text, metadata=metadata))
            chunk_index += 1

    return records


def chunk_document(
    path: str | Path,
    *,
    course: str,
    chapter: str | None = None,
    doc_id: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[ChunkRecord]:
    """解析文件并直接分块，等价于 ``parse_document`` + ``split_blocks``。"""
    file_path = Path(path)
    blocks = parse_document(file_path)
    return split_blocks(
        blocks,
        course=course,
        chapter=chapter,
        doc_id=doc_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
