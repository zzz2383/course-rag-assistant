"""资料解析模块。

提供统一的 :func:`parse_document` 入口，根据扩展名分发到 PDF、PPTX、
DOCX、TXT 解析器，输出结构统一的最小文本块列表::

    [
        {"text": "第三章 动态规划……", "page": 12, "source_file": "第3章.pdf"},
        ...
    ]

其中 ``page`` 对 PDF 表示页码、对 PPTX 表示幻灯片号，DOCX/TXT 无页码时
为 ``0``。解析失败统一抛出 :class:`ParseError`，由上层转换为业务错误。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

# 支持的扩展名白名单。
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".pptx", ".docx", ".txt")

ParsedBlock = dict[str, Any]


class ParseError(ValueError):
    """文档解析失败（格式不支持、文件损坏、加密、无文本层等）。"""


def parse_document(path: str | Path) -> list[ParsedBlock]:
    """解析单个课程资料文件。

    Args:
        path: 待解析文件路径，扩展名需在 :data:`SUPPORTED_EXTENSIONS` 内。

    Returns:
        文本块列表，每项包含 ``text``、``page``、``source_file`` 三个键。

    Raises:
        FileNotFoundError: 文件不存在或不是普通文件。
        ParseError: 扩展名不支持、依赖缺失、文件损坏或未提取到有效文本。
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")
    if not file_path.is_file():
        raise ParseError(f"不是普通文件：{file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = "、".join(SUPPORTED_EXTENSIONS)
        raise ParseError(f"不支持的文件类型 {suffix or '(无扩展名)'}，支持：{allowed}")

    parser = _PARSERS[suffix]
    blocks = parser(file_path)
    blocks = [block for block in blocks if block.get("text", "").strip()]
    if not blocks:
        raise ParseError(f"未从 {file_path.name} 中提取到有效文本，可能是空文件或扫描件")
    return blocks


def _clean_text(raw: str) -> str:
    """清洗文本：统一换行、全角空格与连续空白，去除空行。"""
    if not raw:
        return ""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ").replace("\u3000", " ")
    lines: list[str] = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _make_block(text: str, page: int, source_file: str) -> ParsedBlock:
    """构造统一的文本块字典。"""
    return {"text": text, "page": max(int(page), 0), "source_file": source_file}


def _parse_pdf(path: Path) -> list[ParsedBlock]:
    """按页解析 PDF，页码从 1 开始；无文本层的页面会被跳过。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 依赖缺失时的友好提示
        raise ParseError("解析 PDF 需要安装 pypdf：pip install pypdf") from exc

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except Exception as exc:  # noqa: BLE001 - pypdf 内部异常类型不稳定
                raise ParseError(f"PDF 已加密，无法解析：{path.name}") from exc
            if not unlocked:
                raise ParseError(f"PDF 已加密，无法解析：{path.name}")

        blocks: list[ParsedBlock] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = _clean_text(page.extract_text() or "")
            if text:
                blocks.append(_make_block(text, page_number, path.name))
    except ParseError:
        raise
    except Exception as exc:  # noqa: BLE001 - 统一转换为业务异常
        raise ParseError(f"PDF 解析失败（{path.name}）：{exc}") from exc

    if not blocks:
        raise ParseError(f"PDF 无文本层，可能是扫描件：{path.name}")
    return blocks


def _table_rows(table: Any) -> list[str]:
    """把表格逐行拼接为文本，单元格之间用 `` | `` 分隔。"""
    rows: list[str] = []
    for row in table.rows:
        cells = [_clean_text(cell.text) for cell in row.cells]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(" | ".join(cells))
    return rows


def _parse_pptx(path: Path) -> list[ParsedBlock]:
    """按幻灯片解析 PPTX，幻灯片号从 1 开始；提取标题、正文、表格与备注。"""
    try:
        from pptx import Presentation
    except ImportError as exc:  # pragma: no cover
        raise ParseError("解析 PPTX 需要安装 python-pptx：pip install python-pptx") from exc

    try:
        presentation = Presentation(str(path))
        blocks: list[ParsedBlock] = []
        for slide_number, slide in enumerate(presentation.slides, start=1):
            parts: list[str] = []
            title_shape = slide.shapes.title
            title_id = title_shape.shape_id if title_shape is not None else None

            if title_shape is not None and getattr(title_shape, "has_text_frame", False):
                title_text = _clean_text(title_shape.text_frame.text)
                if title_text:
                    parts.append(title_text)

            for shape in slide.shapes:
                if title_id is not None and shape.shape_id == title_id:
                    continue
                if getattr(shape, "has_text_frame", False):
                    shape_text = _clean_text(shape.text_frame.text)
                    if shape_text:
                        parts.append(shape_text)
                if getattr(shape, "has_table", False):
                    parts.extend(_table_rows(shape.table))

            if slide.has_notes_slide:
                notes = _clean_text(slide.notes_slide.notes_text_frame.text)
                if notes:
                    parts.append(f"【备注】{notes}")

            text = "\n".join(parts)
            if text:
                blocks.append(_make_block(text, slide_number, path.name))
    except ParseError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"PPTX 解析失败（{path.name}）：{exc}") from exc

    return blocks


def _parse_docx(path: Path) -> list[ParsedBlock]:
    """按段落解析 DOCX，段落之间保持边界；表格逐行追加，页码固定为 0。"""
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover
        raise ParseError("解析 DOCX 需要安装 python-docx：pip install python-docx") from exc

    try:
        document = Document(str(path))
        blocks: list[ParsedBlock] = []

        for paragraph in document.paragraphs:
            text = _clean_text(paragraph.text)
            if text:
                blocks.append(_make_block(text, 0, path.name))

        for table in document.tables:
            for row_text in _table_rows(table):
                blocks.append(_make_block(row_text, 0, path.name))
    except ParseError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"DOCX 解析失败（{path.name}）：{exc}") from exc

    return blocks


def _parse_txt(path: Path) -> list[ParsedBlock]:
    """按空行分段读取 TXT，自动尝试 UTF-8/GB18030 等常见编码，页码固定为 0。"""
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ParseError(f"TXT 读取失败（{path.name}）：{exc}") from exc

    text: str | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ParseError(f"TXT 编码无法识别，请转换为 UTF-8：{path.name}")

    # 统一换行符后再按空行切分，兼容 Windows CRLF 与旧式 CR 文件。
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n[ \t\u3000]*\n", text)
    blocks: list[ParsedBlock] = []
    for paragraph in paragraphs:
        cleaned = _clean_text(paragraph)
        if cleaned:
            blocks.append(_make_block(cleaned, 0, path.name))
    return blocks


# 扩展名 -> 解析函数；新增格式时在此注册即可。
_PARSERS: dict[str, Callable[[Path], list[ParsedBlock]]] = {
    ".pdf": _parse_pdf,
    ".pptx": _parse_pptx,
    ".docx": _parse_docx,
    ".txt": _parse_txt,
}
