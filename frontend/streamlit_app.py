"""课程资料问答与复习规划助手 - Streamlit 前端。

本文件只负责页面展示与后端 REST 调用，不直接读写课程资料、向量库或模型。
启动方式：
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
import streamlit as st
import streamlit.components.v1 as components

# 允许以「streamlit run frontend/streamlit_app.py」启动时导入同目录的 theme 模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
import theme  # noqa: E402

# ---------------------------------------------------------------------------
# 基础配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000/api"
MAX_UPLOAD_FILES = 10
MAX_UPLOAD_MB = 20

SUPPORTED_FILE_TYPES = ["pdf", "pptx", "docx", "txt"]

# 练习自测页签使用的名称映射，取值与后端 ExerciseRequest 保持一致。
EXERCISE_TYPE_LABELS = {"blank": "填空题", "choice": "选择题"}
EXERCISE_TYPE_ORDER = ("blank", "choice")
EXERCISE_DIFFICULTY_LABELS = {"easy": "简单", "medium": "中等", "hard": "困难"}
EXERCISE_CHOICE_PLACEHOLDER = "（请选择）"

# 课程/章节下拉框使用的固定文案。Streamlit 没有“可选可填”控件，
# 统一用“已有值 + ＋ 新建…”哨兵项来模拟“既能选已有、也能新增”的交互。
ALL_CHAPTERS_LABEL = "全部章节"
NO_CHAPTER_LABEL = "不指定章节"
NEW_COURSE_LABEL = "＋ 新建课程"
NEW_CHAPTER_LABEL = "＋ 新建章节"

STATUS_LABELS = {
    "uploaded": "处理中",
    "parsed": "待索引",
    "indexing": "索引中",
    "indexed": "可用",
    "index_failed": "索引失败",
    "failed": "失败",
    "skipped": "已跳过（重复）",
}


class BackendError(RuntimeError):
    """后端调用异常基类。"""


class BackendUnavailable(BackendError):
    """网络不通、连接被拒绝或请求超时。"""

    def __init__(self, message: str, *, base_url: str) -> None:
        super().__init__(message)
        self.base_url = base_url


class BackendAPIError(BackendError):
    """后端返回了 4xx/5xx 状态码。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        payload: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


def _normalise_api_base_url(raw_value: str) -> str:
    """把配置规范化为以 ``/api`` 结尾的完整前缀。"""
    value = (raw_value or "").strip().rstrip("/")
    if not value:
        return DEFAULT_API_BASE_URL
    if not value.endswith("/api"):
        value = f"{value}/api"
    return value


def _load_api_base_url() -> str:
    """读取 API 地址。

    优先级：环境变量 ``API_BASE_URL`` > 项目根目录 ``.env`` 中的
    ``API_BASE_URL`` > 默认值。这里只读取这一个公开配置项，不读取任何
    DeepSeek/Qwen 密钥，模型密钥只应存在于后端进程。
    """
    value = os.getenv("API_BASE_URL", "").strip()
    if not value:
        env_file = PROJECT_ROOT / ".env"
        try:
            for raw_line in env_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, raw = line.split("=", 1)
                if key.strip() != "API_BASE_URL":
                    continue
                candidate = raw.strip()
                if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "\"'":
                    candidate = candidate[1:-1]
                if candidate:
                    value = candidate
                    break
        except OSError:
            pass
    return _normalise_api_base_url(value)


API_BASE_URL = _load_api_base_url()


# ---------------------------------------------------------------------------
# 后端 REST 调用
# ---------------------------------------------------------------------------
def _format_error_payload(payload: Any) -> str:
    """从 FastAPI/自定义错误响应中提取适合展示的中文错误信息。"""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip()[:500]

    if not isinstance(payload, dict):
        return str(payload)[:500]

    parts: list[str] = []
    code = payload.get("code")
    if code not in (None, "", 0, "0"):
        parts.append(f"[{code}]")

    message = payload.get("message") or payload.get("error")
    if message:
        parts.append(str(message).strip())

    detail = payload.get("detail")
    if isinstance(detail, str) and detail.strip():
        parts.append(detail.strip())
    elif isinstance(detail, list):
        # FastAPI 校验错误通常位于 detail[].loc / detail[].msg
        for item in detail[:4]:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            location = ".".join(str(part) for part in item.get("loc", []) if part != "body")
            text = str(item.get("msg") or item.get("message") or "").strip()
            if text:
                parts.append(f"{location}: {text}" if location else text)

    return "；".join(part for part in parts if part)[:800]


def _request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    data: dict[str, Any] | None = None,
    timeout: tuple[int, int] = (5, 60),
) -> Any:
    """统一调用后端并处理网络异常、HTTP 错误与空响应。"""
    url = f"{API_BASE_URL}{path}"
    try:
        response = requests.request(
            method=method,
            url=url,
            json=json_body,
            files=files,
            data=data,
            timeout=timeout,
        )
    except requests.exceptions.Timeout as exc:
        raise BackendUnavailable(
            f"请求后端超时（{url}），服务可能正在处理或暂时不可用。",
            base_url=API_BASE_URL,
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise BackendUnavailable(
            f"无法连接后端服务（{API_BASE_URL}）。",
            base_url=API_BASE_URL,
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise BackendUnavailable(
            f"调用后端失败（{API_BASE_URL}）：{exc}",
            base_url=API_BASE_URL,
        ) from exc

    payload: Any = None
    if response.content:
        try:
            payload = response.json()
        except ValueError:
            payload = response.text

    if response.status_code >= 400:
        detail = _format_error_payload(payload)
        if not detail:
            detail = (response.reason or response.text or "未知错误").strip()
            detail = detail[:300]
        message = f"HTTP {response.status_code}：{detail}"
        request_id = response.headers.get("X-Request-ID")
        if request_id:
            message = f"{message}（请求 ID：{request_id}）"
        raise BackendAPIError(message, status_code=response.status_code, payload=payload)

    return payload


def _unwrap_data(payload: Any) -> Any:
    """兼容统一响应 ``{code, message, data}`` 与直接返回业务对象两种风格。"""
    if isinstance(payload, dict):
        if "data" in payload and (
            "code" in payload or "message" in payload or "request_id" in payload
        ):
            return payload.get("data")
        if len(payload) == 1 and "data" in payload:
            return payload.get("data")
    return payload


def _extract_list(payload: Any, candidate_keys: tuple[str, ...]) -> list[Any]:
    """从列表本身或包含 items/documents 的对象中取出列表。"""
    data = _unwrap_data(payload)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in candidate_keys:
            value = data.get(key)
            if isinstance(value, list):
                return value
        # 单条记录也包装成列表，便于上传结果展示。
        if any(key in data for key in ("file_name", "source_file", "doc_id")):
            return [data]
    return []


def list_documents() -> list[dict[str, Any]]:
    """GET /api/documents：获取文档列表。"""
    payload = _request("GET", "/documents", timeout=(5, 30))
    documents = _extract_list(payload, ("items", "documents", "files", "results"))
    return [item for item in documents if isinstance(item, dict)]


def upload_documents(files: list[Any], course: str, chapter: str) -> Any:
    """POST /api/documents/upload：批量上传资料。"""
    multipart: list[tuple[str, tuple[str, bytes, str]]] = []
    for uploaded_file in files:
        multipart.append(
            (
                "files",
                (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                ),
            )
        )

    form_data: dict[str, Any] = {"course": course.strip()}
    if chapter.strip():
        form_data["chapter"] = chapter.strip()

    payload = _request(
        "POST",
        "/documents/upload",
        files=multipart,
        data=form_data,
        timeout=(10, 180),
    )
    return _unwrap_data(payload)


def delete_document(doc_id: str) -> Any:
    """DELETE /api/documents/{doc_id}：删除文档及其索引。"""
    payload = _request(
        "DELETE",
        f"/documents/{quote(doc_id, safe='')}",
        timeout=(5, 60),
    )
    return _unwrap_data(payload)


def fetch_document_chunks(doc_id: str) -> dict[str, Any]:
    """GET /api/documents/{doc_id}/chunks：获取该文档的全部原文分块。

    返回结构由后端约定：``{doc_id, file_name, chunk_count, page_count, pages}``，
    ``pages`` 为按页码升序的 ``[{"page": int, "chunks": [...]}]``。
    """
    payload = _request(
        "GET",
        f"/documents/{quote(doc_id, safe='')}/chunks",
        timeout=(5, 30),
    )
    data = _unwrap_data(payload)
    if not isinstance(data, dict):
        raise BackendError("后端返回的原文分块格式不正确。")
    if not isinstance(data.get("pages"), list):
        data["pages"] = []
    return data


def ask_question(question: str, course: str, chapter: str) -> dict[str, Any]:
    """POST /api/chat：基于课程资料提问。"""
    request_body: dict[str, Any] = {
        "question": question.strip(),
        "course": course.strip(),
    }
    if chapter.strip():
        request_body["chapter"] = chapter.strip()

    payload = _request("POST", "/chat", json_body=request_body, timeout=(10, 90))
    data = _unwrap_data(payload)
    if not isinstance(data, dict) or not str(data.get("answer") or "").strip():
        raise BackendError("后端返回的问答结果格式不正确，未找到 answer 字段。")
    if not isinstance(data.get("citations"), list):
        data["citations"] = []
    return data


def generate_plan(
    exam_date: date,
    chapters: list[str],
    weak_points: list[str],
    weak_points_text: str,
) -> Any:
    """POST /api/plan：生成复习计划。

    ``weak_points`` 默认按字符串数组提交；若后端 schema 要求单个字符串，
    会在收到对应 422 校验错误后用原始文本重试一次，兼容两种约定。
    """
    request_body: dict[str, Any] = {
        "exam_date": exam_date.isoformat(),
        "chapters": chapters,
        "weak_points": weak_points,
    }

    try:
        payload = _request("POST", "/plan", json_body=request_body, timeout=(10, 120))
    except BackendAPIError as exc:
        if exc.status_code == 422 and "weak_points" in str(exc):
            retry_body = {**request_body, "weak_points": weak_points_text.strip()}
            payload = _request("POST", "/plan", json_body=retry_body, timeout=(10, 120))
        else:
            raise

    return _unwrap_data(payload)


def generate_exercises(
    course: str,
    chapter: str,
    count: int,
    types: list[str],
    difficulty: str,
    use_llm: bool,
) -> dict[str, Any]:
    """POST /api/exercises：根据课程资料生成练习题。

    ``types`` 只接受 ``blank``（填空）与 ``choice``（选择）；``difficulty``
    取 ``easy`` / ``medium`` / ``hard``。没有资料时后端仍返回 200，
    只是 ``items`` 为空、具体原因写在 ``warnings`` 中。
    """
    request_body: dict[str, Any] = {
        "course": course.strip(),
        "count": int(count),
        "types": list(types),
        "difficulty": difficulty,
        "use_llm": bool(use_llm),
    }
    if chapter.strip():
        request_body["chapter"] = chapter.strip()

    payload = _request("POST", "/exercises", json_body=request_body, timeout=(10, 180))
    data = _unwrap_data(payload)
    if not isinstance(data, dict):
        raise BackendError("后端返回的练习题结果格式不正确，未找到题目数据。")

    if not isinstance(data.get("items"), list):
        data["items"] = []
    if not isinstance(data.get("warnings"), list):
        data["warnings"] = []
    return data


# ---------------------------------------------------------------------------
# 通用页面组件
# ---------------------------------------------------------------------------
def _show_backend_error(exc: Exception) -> None:
    """用友好的方式呈现后端异常，避免直接暴露堆栈。"""
    if isinstance(exc, BackendUnavailable):
        st.error(str(exc))
        st.info(
            "请先确认后端服务已启动，再重试当前操作。\n\n"
            "```powershell\n"
            f"cd {PROJECT_ROOT}\n"
            ".\\run_backend.ps1\n"
            "```\n\n"
            "也可以手动执行 `uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`。"
        )
        st.caption(f"当前 API 地址：{API_BASE_URL}")
        return

    if isinstance(exc, BackendAPIError):
        st.error(f"后端返回错误：{exc}")
        if exc.status_code == 404:
            st.caption("资源不存在，请刷新文档列表或重新生成计划。")
        elif exc.status_code in (401, 403):
            st.caption("鉴权失败，请检查后端 API_KEY 与前端访问配置。")
        elif exc.status_code == 503:
            st.caption("后端依赖（模型或向量库）暂不可用，请稍后重试。")
        return

    st.error(f"操作失败：{exc}")


def _set_flash(message: str, kind: str = "success") -> None:
    """设置跨 rerun 的一次性提示。"""
    st.session_state["_flash_message"] = (kind, message)


def _show_flash() -> None:
    """显示并清除一次性提示。"""
    flash = st.session_state.pop("_flash_message", None)
    if not flash:
        return
    kind, message = flash
    if kind == "warning":
        st.warning(message)
    elif kind == "error":
        st.error(message)
    else:
        st.success(message)


@st.cache_data(ttl=15, show_spinner=False)
def _probe_backend(base_url: str) -> tuple[bool, str]:
    """轻量探测后端是否可达；失败结果只缓存很短时间。"""
    failures: list[str] = []
    for path in ("/health", "/documents"):
        try:
            response = requests.get(f"{base_url}{path}", timeout=(2, 3))
        except requests.exceptions.RequestException:
            failures.append(f"{path} 不可达")
            continue
        if response.status_code < 500:
            return True, f"{path} 返回 HTTP {response.status_code}"
        failures.append(f"{path} 返回 HTTP {response.status_code}")
    return False, "；".join(failures) or "连接失败"


def _render_sidebar() -> None:
    """扉页式侧边栏：书脊铭牌、连接状态、使用顺序。"""
    with st.sidebar:
        st.markdown(
            theme.spine_plate("课程资料问答助手", "COURSE READER", "讲义"),
            unsafe_allow_html=True,
        )

        reachable, detail = _probe_backend(API_BASE_URL)
        st.markdown(
            theme.status_line(
                reachable,
                "后端已连接" if reachable else "后端未连接",
            ),
            unsafe_allow_html=True,
        )
        st.caption(f"API 地址：{API_BASE_URL}")
        st.caption(detail)

        if st.button("重新检测连接", key="probe_backend_btn", use_container_width=True):
            _probe_backend.clear()
            st.rerun()

        if not reachable:
            with st.expander("后端启动方式", expanded=True):
                st.code(
                    f"cd {PROJECT_ROOT}\n.\\run_backend.ps1",
                    language="powershell",
                )
                st.caption("或执行：uvicorn app.main:app --reload")

        st.divider()
        st.markdown(
            theme.side_note(
                "阅 读 顺 序",
                [
                    "上传课程资料，等待索引完成",
                    "在「课程问答」提问并核对引用",
                    "点「在原文中查看」翻阅命中原文",
                    "在「复习计划」生成并导出计划",
                    "在「练习自测」作答并提交判分",
                ],
            ),
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption("前端只通过 REST API 访问后端，不读取任何模型密钥。")


def _display_int(value: Any, default: str = "—") -> int | str:
    """将后端可能返回的数字字符串转换为整数，无法转换时原样展示。"""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _short_time(value: Any) -> str:
    """将 ISO 时间截断为适合表格展示的格式。"""
    if not value:
        return "—"
    text = str(value).replace("T", " ")
    # 表格列宽有限，秒级精度会把单元格挤成两行，保留到分钟即可。
    return text[:16]


def _first_text(mapping: dict[str, Any], *keys: str) -> str:
    """按顺序返回第一个非空字段的文本值。"""
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _status_label(status: Any) -> str:
    """把后端英文状态转换为中文展示。"""
    raw = str(status or "").strip()
    if not raw:
        return "—"
    return STATUS_LABELS.get(raw.lower(), raw)


def _document_rows(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造文档列表表格行。"""
    rows: list[dict[str, Any]] = []
    for document in documents:
        page_count = _display_int(document.get("page_count"))
        if page_count == 0:
            page_count = "—"
        rows.append(
            {
                "文件名": _first_text(document, "source_file", "file_name", "filename", "name")
                or "未命名",
                "课程": _first_text(document, "course", "course_name") or "—",
                "章节": _first_text(document, "chapter", "section") or "未分章",
                "页数": page_count,
                "分块数": _display_int(document.get("chunk_count")),
                "状态": _status_label(
                    document.get("status") or document.get("index_status")
                ),
                "上传时间": _short_time(
                    document.get("created_at")
                    or document.get("uploaded_at")
                    or document.get("updated_at")
                ),
            }
        )
    return rows


def _document_label(document: dict[str, Any]) -> str:
    """文档选择框的可读标签。"""
    filename = _first_text(document, "source_file", "file_name", "filename", "name") or "未命名"
    chapter = _first_text(document, "chapter", "section")
    return f"{filename}｜{chapter}" if chapter else filename


def _render_upload_result(result: Any) -> None:
    """展示上传接口的逐文件结果。"""
    items = _extract_list(result, ("items", "documents", "files", "results"))
    success_count: Any = None
    failed_count: Any = None
    if isinstance(result, dict):
        success_count = result.get("success_count")
        failed_count = result.get("failed_count")

    if not items:
        if success_count is not None or failed_count is not None:
            st.success(
                f"上传完成：成功 {success_count if success_count is not None else '—'} 个，"
                f"失败 {failed_count if failed_count is not None else '—'} 个。"
            )
        else:
            st.success("上传请求已完成，请查看下方文档列表。")
        return

    for item in items:
        if not isinstance(item, dict):
            st.info(str(item))
            continue
        name = _first_text(item, "file_name", "source_file", "filename", "name") or "未命名文件"
        status = str(item.get("status") or "").lower()
        error = _first_text(item, "error", "message", "reason")
        chunk_count = _display_int(item.get("chunk_count"), default="—")
        page_count = _display_int(item.get("page_count"), default="—")

        if error or status in {"failed", "error", "parse_failed"}:
            st.error(f"{name}：上传失败。{error or '请检查文件格式或内容。'}")
        elif status in {"index_failed", "partial_success"}:
            st.warning(
                f"{name}：文件已保存，但索引未完全成功。{error or '请稍后重建索引。'}"
            )
        elif status == "skipped":
            st.info(f"{name}：文件内容已存在，已跳过重复上传。")
        else:
            st.success(
                f"{name}：处理成功（页数 {page_count}，分块数 {chunk_count}，"
                f"状态 {_status_label(item.get('status'))}）。"
            )

    if success_count is not None or failed_count is not None:
        summary = (
            f"汇总：成功 {success_count if success_count is not None else '—'} 个，"
            f"失败 {failed_count if failed_count is not None else '—'} 个。"
        )
        if failed_count not in (None, 0, "0"):
            st.warning(summary)
        else:
            st.caption(summary)


def _sync_documents(*, show_error: bool = True) -> bool:
    """同步文档列表到 session_state。返回是否成功。"""
    try:
        documents = list_documents()
    except Exception as exc:  # 页面层统一兜底，不能因后端未启动而崩溃
        st.session_state["documents_loaded"] = True
        if show_error:
            _show_backend_error(exc)
        return False

    st.session_state["documents"] = documents
    st.session_state["documents_loaded"] = True
    st.session_state["documents_updated_at"] = datetime.now().strftime("%H:%M:%S")
    return True


@st.cache_data(ttl=15, show_spinner=False)
def _course_options() -> list[str]:
    """从已上传文档中提取去重、排序后的课程名称，供下拉框使用。

    课程名的唯一事实来源是后端文档列表：只调用 ``list_documents()``，
    后端不可达时返回空列表，由各页面给出中文提示。缓存 15 秒避免每次
    rerun 都请求后端；上传/删除成功后会显式 clear()，保证新课程立刻可见。
    """
    try:
        documents = list_documents()
    except Exception:  # 后端未启动或返回异常时降级为空列表，不影响页面渲染
        return []

    courses = {_first_text(document, "course", "course_name") for document in documents}
    return sorted(name for name in courses if name)


@st.cache_data(ttl=15, show_spinner=False)
def _chapter_options(course: str) -> list[str]:
    """返回指定课程已有文档的章节列表（去重、排序、排除空值）。

    课程名为空时直接返回空列表，避免把其他课程的章节混进来。
    """
    target = (course or "").strip()
    if not target:
        return []

    try:
        documents = list_documents()
    except Exception:
        return []

    chapters: set[str] = set()
    for document in documents:
        if _first_text(document, "course", "course_name") != target:
            continue
        chapter = _first_text(document, "chapter", "section")
        if chapter:
            chapters.add(chapter)
    return sorted(chapters)


def _clear_document_caches() -> None:
    """失效课程/章节下拉缓存：上传、删除与手动刷新之后调用。"""
    _course_options.clear()
    _chapter_options.clear()


def _render_citations(citations: list[Any], *, entry_number: int = 0) -> None:
    """以脚注形式展示引用来源，并给出「在原文中查看」入口。

    ``entry_number`` 用于生成唯一的控件 key，并支持「在原文中查看」跳转。
    """
    if not citations:
        st.caption("本次回答未返回引用来源。")
        return

    items: list[dict[str, Any]] = []
    for position, citation in enumerate(citations, start=1):
        if not isinstance(citation, dict):
            citation = {"excerpt": str(citation)}

        index = citation.get("index") or position
        filename = _first_text(
            citation, "source_file", "file_name", "filename", "document", "source"
        ) or "未知来源"
        page = citation.get("page")
        page_number = 0
        if page in (None, ""):
            page_text = "页码未知"
        else:
            page_number = _display_int(page, default=0)
            page_text = "无页码" if page_number == 0 else f"第 {page_number} 页"
        excerpt = _first_text(citation, "excerpt", "snippet", "text", "content")
        locator = _first_text(citation, "locator")
        # 「page:4」「chunk-2」这类定位串与页码信息重复，脚注里不再单列
        if re.fullmatch(r"(page|chunk)\s*[:\-_]?\s*\d+", locator, flags=re.IGNORECASE):
            locator = ""

        items.append(
            {
                "index": index,
                "source": filename,
                "page": page_text,
                "locator": locator,
                "excerpt": excerpt,
                "_page_number": page_number,
                "_doc_id": _first_text(citation, "doc_id"),
                "_chunk_id": _first_text(citation, "chunk_id"),
                "_filename": filename,
            }
        )

    st.markdown(theme.footnotes(items), unsafe_allow_html=True)

    jumpable = [item for item in items if item["_doc_id"]]
    if not jumpable:
        return
    columns = st.columns(min(len(jumpable), 4), gap="small")
    for offset, item in enumerate(jumpable):
        with columns[offset % len(columns)]:
            if st.button(
                f"[{item['index']}] 在原文中查看",
                key=f"book_open_{entry_number}_{offset}",
                help="打开原文阅读器，并定位到该引用所在页",
            ):
                st.session_state["book_jump"] = {
                    "doc_id": item["_doc_id"],
                    "page": item["_page_number"],
                    "chunk_id": item["_chunk_id"],
                    "file_name": item["_filename"],
                }
                st.rerun()


def _order_citations(
    citations: list[Any], answer: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按正文出现顺序重排引用，并返回「文件名|页码 → 角标序号」映射。

    模型常在同一句后重复标注同一来源，后端返回的 citations 顺序也未必与正文一致。
    统一以正文首次出现的顺序编号，保证角标与脚注一一对应。
    """
    normalized: list[dict[str, Any]] = []
    for position, citation in enumerate(citations, start=1):
        if not isinstance(citation, dict):
            citation = {"excerpt": str(citation)}
        filename = _first_text(
            citation, "source_file", "file_name", "filename", "document", "source"
        ) or "未知来源"
        page = citation.get("page")
        page_text = "" if page in (None, "") else str(_display_int(page, default=""))
        item = dict(citation)
        item["_name"] = filename
        item["_page_key"] = page_text
        item["_key"] = f"{filename}|{page_text}"
        item["_pos"] = position
        normalized.append(item)

    ordered: list[dict[str, Any]] = []
    used: set[int] = set()

    def take(predicate) -> None:
        for candidate in normalized:
            if id(candidate) in used:
                continue
            if predicate(candidate):
                ordered.append(candidate)
                used.add(id(candidate))
                return

    for key in theme.citation_keys(answer):
        name, _, page = key.partition("|")
        take(lambda c: c["_key"] == key)
        # 页码对不上时退回按文件名匹配，避免角标全部自增
        take(lambda c, n=name: c["_name"] == n)
    for candidate in normalized:
        if id(candidate) not in used:
            ordered.append(candidate)
            used.add(id(candidate))

    mapping: dict[str, Any] = {}
    for index, item in enumerate(ordered, start=1):
        item["index"] = index
        mapping.setdefault(item["_key"], index)
        mapping.setdefault(f'{item["_name"]}|', index)
    return ordered, mapping


def _render_chat_entry(entry: dict[str, Any], number: int) -> None:
    """渲染一条问答历史：问号方印 + 正文段落 + 脚注。"""
    meta_parts = [str(entry.get("created_at") or "")]
    if entry.get("course"):
        meta_parts.append(f"课程：{entry['course']}")
    if entry.get("chapter"):
        meta_parts.append(f"章节：{entry['chapter']}")
    if entry.get("model"):
        meta_parts.append(f"模型：{entry['model']}")
    if entry.get("latency_ms") is not None:
        meta_parts.append(f"耗时：{entry['latency_ms']} ms")

    answer = str(entry.get("answer") or "")
    citations, cite_map = _order_citations(entry.get("citations") or [], answer)
    st.markdown(
        theme.qa_block(
            str(entry.get("question") or ""),
            "",
            " ｜ ".join(part for part in meta_parts if part),
            answer_html=theme.answer_paragraphs(answer, citation_map=cite_map),
            number=number,
        ),
        unsafe_allow_html=True,
    )

    if entry.get("grounded") is False:
        st.warning("后端判断当前资料不足以支撑该答案，建议核对来源或补充资料。")
    if entry.get("degraded"):
        reason = entry.get("degraded_reason")
        st.info(f"本次回答使用了降级模式{f'（{reason}）' if reason else ''}。")

    _render_citations(citations, entry_number=number)


# ---------------------------------------------------------------------------
# 页签一：资料管理
# ---------------------------------------------------------------------------
def _render_documents_tab() -> None:
    _show_flash()

    st.subheader("资料管理")
    st.caption(
        f"支持 PDF / PPTX / DOCX / TXT，单文件不超过 {MAX_UPLOAD_MB} MB，"
        f"单次不超过 {MAX_UPLOAD_FILES} 个文件。上传后后端会自动解析并建立索引。"
    )

    # 课程/章节选择放在 form 外：form 内的控件变更不会触发 rerun，
    # 选完课程后无法立刻刷新该课程下的章节候选；放到 form 前面即可即时联动，
    # form 内只保留文件选择与提交按钮。
    course_options = _course_options()
    col_course, col_chapter = st.columns(2)
    selected_course = col_course.selectbox(
        "课程 *",
        [*course_options, NEW_COURSE_LABEL],
        index=None if course_options else 0,
        placeholder="选择已有课程，或选择「＋ 新建课程」",
        key="upload_course_select",
        help="课程名用于给上传的资料打标签，问答与练习自测从中读取候选。",
    )
    # 只有显式选择「＋ 新建课程」时才显示输入框，避免多余的空输入框干扰。
    if selected_course == NEW_COURSE_LABEL:
        course = col_course.text_input(
            "新课程名称 *",
            placeholder="例如：算法设计与分析",
            key="upload_course_input",
        )
    else:
        course = selected_course or ""

    # 章节候选只来自“已存在课程”下的文档；新建课程名尚无资料，直接手填章节。
    chapter_candidates = _chapter_options(course) if course in course_options else []
    chapter_choice = col_chapter.selectbox(
        "章节（可选）",
        [NO_CHAPTER_LABEL, *chapter_candidates, NEW_CHAPTER_LABEL],
        index=0,
        key="upload_chapter_select",
        help="选择已有章节可保持命名统一；需要新章节时选「＋ 新建章节」再手填。",
    )
    if chapter_choice == NEW_CHAPTER_LABEL:
        chapter = col_chapter.text_input(
            "新章节名称（可选）",
            placeholder="例如：第三章 动态规划",
            key="upload_chapter_input",
        )
    elif chapter_choice == NO_CHAPTER_LABEL:
        chapter = ""
    else:
        chapter = chapter_choice

    with st.form("upload_documents_form", clear_on_submit=False):
        uploaded_files = st.file_uploader(
            "选择课程资料",
            type=SUPPORTED_FILE_TYPES,
            accept_multiple_files=True,
            key="upload_files",
            help="支持批量选择；文件会发送到后端解析并写入向量索引。",
        )
        submitted = st.form_submit_button("上传并建立索引", type="primary")

    if submitted:
        if not course.strip():
            st.warning("请选择已有课程或填写新课程名称后再上传。")
        elif not uploaded_files:
            st.warning("请至少选择一个文件。")
        elif len(uploaded_files) > MAX_UPLOAD_FILES:
            st.error(f"单次最多上传 {MAX_UPLOAD_FILES} 个文件，请分批上传。")
        else:
            oversized = [
                item.name
                for item in uploaded_files
                if (getattr(item, "size", 0) or 0) > MAX_UPLOAD_MB * 1024 * 1024
            ]
            if oversized:
                st.error(f"以下文件超过 {MAX_UPLOAD_MB} MB，请压缩或拆分：{', '.join(oversized)}")
            else:
                with st.spinner("正在上传、解析并建立索引，请稍候..."):
                    try:
                        upload_result = upload_documents(uploaded_files, course, chapter)
                    except Exception as exc:
                        _show_backend_error(exc)
                    else:
                        st.success(f"已提交 {len(uploaded_files)} 个文件。")
                        _render_upload_result(upload_result)
                        # 新课程/新章节已入库，立刻失效下拉缓存，不必等 15 秒 TTL。
                        _clear_document_caches()
                        if not _sync_documents(show_error=False):
                            st.warning("上传请求已返回，但文档列表刷新失败，请稍后点击“刷新列表”。")

    st.divider()
    st.subheader("文档列表")
    col_refresh, col_hint = st.columns([1, 3])
    with col_refresh:
        refresh_clicked = st.button(
            "刷新列表", key="refresh_documents_btn", use_container_width=True
        )
    if refresh_clicked:
        # 手动刷新时同时失效课程/章节下拉缓存，让其他页签的选项也拿到最新数据。
        _clear_document_caches()
        with st.spinner("正在刷新文档列表..."):
            _sync_documents()
    elif not st.session_state.get("documents_loaded"):
        with st.spinner("正在读取文档列表..."):
            _sync_documents()

    # 文案放在同步之后渲染，否则首屏永远显示「正在等待首次加载」。
    with col_hint:
        updated_at = st.session_state.get("documents_updated_at")
        st.caption(f"最近刷新：{updated_at}" if updated_at else "尚未加载文档列表")

    documents = st.session_state.get("documents", [])
    if not documents:
        st.info("暂无可展示的文档。请先上传课程资料；若已上传，请点击“刷新列表”。")
        return

    st.dataframe(
        _document_rows(documents),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("#### 删除文档")
    st.caption("删除后，该文档的解析文件与向量索引也会被后端清理，操作不可恢复。")
    deletable = [doc for doc in documents if doc.get("doc_id")]
    if not deletable:
        st.warning("文档缺少 doc_id，无法执行删除，请刷新列表或检查后端返回。")
        return

    selected_position = st.selectbox(
        "选择要删除的文档",
        options=list(range(len(deletable))),
        format_func=lambda position: _document_label(deletable[position]),
        key="delete_document_select",
    )
    confirmed = st.checkbox(
        "我确认删除该文档及其索引",
        key="delete_document_confirm",
    )
    if st.button(
        "删除选中文档",
        key="delete_document_btn",
        type="secondary",
        disabled=not confirmed,
    ):
        target = deletable[selected_position]
        doc_id = str(target.get("doc_id") or "").strip()
        target_name = _document_label(target)
        if not doc_id:
            st.error("未找到该文档的 doc_id，无法删除。")
            return
        with st.spinner(f"正在删除 {target_name} ..."):
            try:
                delete_document(doc_id)
            except Exception as exc:
                _show_backend_error(exc)
            else:
                _sync_documents(show_error=False)
                # 删除可能让某门课程/章节不再存在，同步清缓存，避免下拉框残留旧值。
                _clear_document_caches()
                _set_flash(f"已删除文档：{target_name}", kind="success")
                st.rerun()


# ---------------------------------------------------------------------------
# 页签二：课程问答
# ---------------------------------------------------------------------------
def _render_chat_tab() -> None:
    st.subheader("课程问答")
    st.caption("答案由后端检索课程资料后生成；请通过下方引用来源核对原文。")

    # 课程/章节下拉框放在 form 外：章节候选依赖已选课程，需要“选课程 → rerun →
    # 刷新章节候选”的即时联动；若放进 st.form 内，form 不会因选择变化而 rerun，
    # 章节列表会停留在上一次提交时的状态。因此 form 内只保留问题与提问按钮。
    course_options = _course_options()
    if not course_options:
        # 这里用 return 而不是 st.stop()：st.stop() 会中断整个脚本，
        # 连其他页签也渲染不出来；return 只跳过本页签的剩余内容。
        st.warning("暂无课程，请先在「资料管理」页上传资料并填写课程名称。")
        return

    col_course, col_chapter = st.columns(2)
    course = col_course.selectbox(
        "课程",
        course_options,
        index=None,
        placeholder="请选择课程",
        key="chat_course_select",
    )
    # 未选课程时章节只有“全部章节”，且控件置灰，避免带着空课程去提问。
    chapter_options = _chapter_options(course) if course else []
    chapter_label = col_chapter.selectbox(
        "章节（可选）",
        [ALL_CHAPTERS_LABEL, *chapter_options],
        index=0,
        key="chat_chapter_select",
        disabled=not course,
    )
    # 选“全部章节”时提交空字符串；后端 chapter 为空即表示不按章节过滤。
    chapter = "" if chapter_label == ALL_CHAPTERS_LABEL else str(chapter_label or "")

    with st.form("chat_form", clear_on_submit=False):
        question = st.text_area(
            "你的问题 *",
            placeholder="例如：动态规划的状态转移方程怎么推导？",
            height=120,
            key="chat_question_input",
        )
        asked = st.form_submit_button("提问", type="primary")

    if asked:
        course_text = (course or "").strip()
        question_text = (question or "").strip()
        if not course_text:
            st.warning("请先选择课程。")
        elif not question_text:
            st.warning("请输入问题。")
        elif len(question_text) > 2000:
            st.warning("问题过长，请控制在 2000 字以内。")
        else:
            with st.spinner("正在检索资料并生成回答..."):
                try:
                    result = ask_question(question_text, course_text, chapter)
                except Exception as exc:
                    # 保留已有会话历史，只提示本次调用失败。
                    _show_backend_error(exc)
                else:
                    citations = result.get("citations")
                    if not isinstance(citations, list):
                        citations = []
                    entry = {
                        "question": question_text,
                        "answer": str(result.get("answer") or "").strip(),
                        "citations": citations,
                        "course": course_text,
                        "chapter": chapter,
                        "grounded": result.get("grounded"),
                        "degraded": bool(result.get("degraded")),
                        "degraded_reason": result.get("degraded_reason"),
                        "model": result.get("model"),
                        "latency_ms": result.get("latency_ms"),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    history = st.session_state.setdefault("chat_history", [])
                    history.append(entry)
                    # 防止长时间会话无限占用内存，保留最近 50 轮。
                    st.session_state["chat_history"] = history[-50:]

    st.divider()
    history = st.session_state.get("chat_history", [])
    col_title, col_clear = st.columns([3, 1])
    with col_title:
        st.subheader(f"会话历史（{len(history)} 条）")
    with col_clear:
        if st.button(
            "清空会话",
            key="clear_chat_btn",
            disabled=not history,
            use_container_width=True,
        ):
            st.session_state["chat_history"] = []
            st.rerun()

    if not history:
        st.info("还没有问答记录。填写上方表单后，答案与引用来源会展示在这里。")
    else:
        st.caption("按最近提问优先展示；刷新页面后会话仍保留在当前浏览器会话中。")
        for number, entry in reversed(list(enumerate(history, start=1))):
            _render_chat_entry(entry, number)

    # 引用卡片上的「在原文中查看」会写入 book_jump，这里把阅读器展开在页面下方
    _render_book_jump_panel()


# ---------------------------------------------------------------------------
# 页签三：复习计划
# ---------------------------------------------------------------------------
def _split_values(raw_text: str) -> list[str]:
    """把多行/逗号分隔的输入拆成去重列表。"""
    parts = re.split(r"[\n\r,，、;；]+", raw_text or "")
    values: list[str] = []
    for part in parts:
        text = part.strip()
        if text and text not in values:
            values.append(text)
    return values


def _extract_chapters(documents: list[dict[str, Any]]) -> list[str]:
    """从文档列表中去重提取章节，供多选框使用。"""
    chapters: list[str] = []
    for document in documents:
        chapter = _first_text(document, "chapter", "section")
        if chapter and chapter not in chapters:
            chapters.append(chapter)
    return chapters


def _iter_task_texts(value: Any) -> list[str]:
    """将后端可能返回的字符串/字典/数组任务统一成文本列表。"""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        title = _first_text(value, "title", "task", "name", "content", "text", "description")
        minutes = value.get("minutes") or value.get("duration") or value.get("duration_minutes")
        if title:
            return [f"{title}（{minutes} 分钟）" if minutes not in (None, "") else title]
        # 没有常见字段时拼接可读的值，避免把整个字典直接展示给用户。
        readable = [
            str(item)
            for key, item in value.items()
            if key not in {"id", "type"} and item not in (None, "", [], {})
        ]
        return ["；".join(readable)] if readable else []
    if isinstance(value, (list, tuple)):
        texts: list[str] = []
        for item in value:
            texts.extend(_iter_task_texts(item))
        return texts
    return [str(value)]


WEEKDAY_LABELS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def _plan_days(plan: list[Any]) -> list[dict[str, Any]]:
    """把后端计划转换为边注式渲染所需的结构（日期大字 + 任务列表）。"""
    days: list[dict[str, Any]] = []
    for position, item in enumerate(plan, start=1):
        if not isinstance(item, dict):
            days.append(
                {
                    "label": f"第{position}天",
                    "weekday": "",
                    "tasks": [str(item)],
                    "focus": "",
                }
            )
            continue

        day_no = item.get("day") or item.get("day_index") or position
        raw_date = _first_text(item, "date", "day_date")
        label = ""
        weekday = ""
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m-%d", "%m/%d"):
            try:
                parsed = datetime.strptime(raw_date, fmt)
            except ValueError:
                continue
            label = parsed.strftime("%m-%d")
            weekday = WEEKDAY_LABELS[parsed.weekday()]
            break
        if not label and raw_date:
            label = raw_date[-5:] if len(raw_date) >= 5 else raw_date
        if not label:
            label = f"第{day_no}天"

        tasks = _iter_task_texts(
            item.get("tasks") or item.get("activities") or item.get("items") or []
        )
        days.append(
            {
                "label": label,
                "weekday": weekday,
                "tasks": tasks,
                "focus": _first_text(item, "focus", "objective", "goal", "chapter"),
            }
        )
    return days


def _extract_plan_payload(raw: Any) -> tuple[list[Any], str, list[str], str]:
    """兼容统一/直接响应，提取计划列表、生成方式、提示与摘要。"""
    plan: Any = []
    generated_by = ""
    warnings_raw: Any = []
    summary = ""

    if isinstance(raw, list):
        plan = raw
    elif isinstance(raw, dict):
        plan = raw.get("plan")
        if isinstance(plan, dict):
            plan = plan.get("days") or plan.get("items") or []
        if not isinstance(plan, list):
            plan = raw.get("days") if isinstance(raw.get("days"), list) else []
        generated_by = str(raw.get("generated_by") or "").strip()
        warnings_raw = raw.get("warnings") or []
        summary = str(raw.get("summary") or "").strip()

    if not isinstance(plan, list):
        plan = []
    if isinstance(warnings_raw, str):
        warnings = [warnings_raw] if warnings_raw.strip() else []
    elif isinstance(warnings_raw, list):
        warnings = [str(item) for item in warnings_raw if str(item).strip()]
    else:
        warnings = []

    return plan, generated_by, warnings, summary


def _generated_by_label(value: str) -> str:
    """把 generated_by 转换为中文说明。"""
    raw = (value or "").strip()
    normalized = raw.lower()
    if normalized in {"llm", "ai", "deepseek", "qwen", "agent"}:
        return f"大模型生成（{raw}）"
    if normalized in {"rule", "rules", "rule_fallback", "fallback", "template"}:
        return f"规则模板生成（{raw}）"
    return raw or "未提供"


def _plan_to_markdown(
    plan: list[Any],
    exam_date: str,
    generated_by: str,
    summary: str,
    warnings: list[str],
) -> str:
    """把计划转换为可下载的 Markdown。"""
    lines: list[str] = [
        "# 课程复习计划",
        "",
        f"- 考试日期：{exam_date or '未提供'}",
        f"- 生成方式：{_generated_by_label(generated_by)}",
    ]
    if summary:
        lines.extend(["", f"> {summary}"])
    lines.append("")

    for position, item in enumerate(plan, start=1):
        if not isinstance(item, dict):
            lines.extend([f"## 第 {position} 天", "", f"- {item}", ""])
            continue

        day = item.get("day") or item.get("day_index") or position
        day_date = _first_text(item, "date", "day_date")
        heading = f"## 第 {day} 天"
        if day_date:
            heading = f"{heading} · {day_date}"
        lines.append(heading)

        focus = _first_text(item, "focus", "objective", "goal", "chapter")
        if focus:
            lines.extend(["", f"**学习重点：** {focus}"])

        tasks = _iter_task_texts(
            item.get("tasks") or item.get("activities") or item.get("items") or []
        )
        if tasks:
            lines.extend(["", "**学习任务：**"])
            lines.extend(f"- {task}" for task in tasks)
        lines.append("")

    if warnings:
        lines.extend(["## 提示", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_saved_plan() -> None:
    """渲染 session_state 中保存的计划与下载按钮。"""
    result = st.session_state.get("plan_result")
    if not isinstance(result, dict):
        return

    plan = result.get("plan") or []
    generated_by = str(result.get("generated_by") or "")
    warnings = result.get("warnings") or []
    summary = str(result.get("summary") or "")
    exam_date = str(result.get("exam_date") or "")

    st.divider()
    st.subheader("计划结果")

    normalized = generated_by.lower()
    if any(keyword in normalized for keyword in ("rule", "fallback", "template")):
        st.warning(
            "当前计划由规则模板降级生成（generated_by="
            f"{generated_by or 'rule'}）。大模型暂不可用或输出校验未通过，"
            "计划仍可执行；模型恢复后可重新生成。"
        )
    elif not generated_by:
        st.caption("后端未返回 generated_by，无法确认生成方式。")

    # 生成方式与日期时间并作一行版记，避免每条元信息各占一条横幅
    plan_meta = [f"考试日期：{exam_date or '—'}"]
    if generated_by:
        plan_meta.append(f"生成方式：{_generated_by_label(generated_by)}")
    if result.get("generated_at"):
        plan_meta.append(f"生成时间：{result['generated_at']}")
    st.caption(" ｜ ".join(plan_meta))

    if summary:
        st.info(summary)
    for warning in warnings:
        st.warning(f"提示：{warning}")

    days = _plan_days(plan)
    if days:
        st.markdown(theme.plan_days(days), unsafe_allow_html=True)
    else:
        st.warning("后端返回的 plan 为空，暂无可展示的每日任务。")

    markdown = _plan_to_markdown(
        plan=plan,
        exam_date=exam_date,
        generated_by=generated_by,
        summary=summary,
        warnings=warnings,
    )
    filename_date = exam_date or date.today().isoformat()
    col_download, col_clear = st.columns(2)
    with col_download:
        st.download_button(
            "导出 Markdown",
            data=markdown.encode("utf-8"),
            file_name=f"复习计划_{filename_date}.md",
            mime="text/markdown",
            key="download_plan_md_btn",
            use_container_width=True,
        )
    with col_clear:
        if st.button(
            "清空当前计划",
            key="clear_plan_btn",
            use_container_width=True,
        ):
            st.session_state.pop("plan_result", None)
            st.rerun()


def _render_plan_tab() -> None:
    st.subheader("复习计划")
    st.caption("根据考试日期、章节和薄弱点生成每日复习安排，并支持导出 Markdown。")

    documents = st.session_state.get("documents", [])
    chapter_options = _extract_chapters(documents)

    col_exam, col_info = st.columns([1, 2])
    with col_exam:
        exam_date = st.date_input(
            "考试日期 *",
            value=date.today() + timedelta(days=14),
            min_value=date.today() + timedelta(days=1),
            max_value=date.today() + timedelta(days=180),
            format="YYYY-MM-DD",
            key="plan_exam_date",
        )
    with col_info:
        st.caption(
            f"已从已上传资料中识别到 {len(chapter_options)} 个章节。"
            "如果章节不完整，可先到「资料管理」刷新列表，或在下方手动补充。"
        )
        if chapter_options:
            st.caption("章节选项：" + "、".join(chapter_options[:10]) + ("…" if len(chapter_options) > 10 else ""))

    selected_chapters = st.multiselect(
        "复习章节 *",
        options=chapter_options,
        default=chapter_options,
        key="plan_selected_chapters",
        placeholder="请选择要复习的章节",
    )
    manual_chapters = st.text_input(
        "补充章节（可选）",
        placeholder="资料未识别章节时可手动填写，多个章节用逗号分隔",
        key="plan_manual_chapters",
    )
    weak_points_text = st.text_area(
        "薄弱点（可选）",
        placeholder="每行一个薄弱点，例如：\n动态规划状态转移\n图的最短路径",
        height=100,
        key="plan_weak_points_input",
    )

    chapters = list(selected_chapters)
    for chapter in _split_values(manual_chapters):
        if chapter not in chapters:
            chapters.append(chapter)
    weak_points = _split_values(weak_points_text)
    if weak_points:
        st.caption(f"已识别 {len(weak_points)} 个薄弱点，将一并提交给后端。")

    if not chapters:
        st.info(
            "未选择章节时，后端会尝试从资料中自动推断章节；"
            "也可以在上方多选或手动补充章节，让计划更聚焦。"
        )

    generated = st.button(
        "生成复习计划",
        key="generate_plan_btn",
        type="primary",
    )

    if generated:
        with st.spinner("正在生成复习计划，请稍候..."):
            try:
                raw = generate_plan(exam_date, chapters, weak_points, weak_points_text)
            except Exception as exc:
                _show_backend_error(exc)
            else:
                plan, generated_by, warnings, summary = _extract_plan_payload(raw)
                if not plan:
                    st.warning("后端未返回可展示的 plan 数据，请检查输入或查看后端日志。")
                else:
                    st.session_state["plan_result"] = {
                        "plan": plan,
                        "generated_by": generated_by,
                        "warnings": warnings,
                        "summary": summary,
                        "exam_date": exam_date.isoformat(),
                        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }

    _render_saved_plan()
    if not st.session_state.get("plan_result"):
        st.caption("生成后会在此处显示每日计划，并提供 Markdown 下载按钮。")


# ---------------------------------------------------------------------------
# 页签四：练习自测
# ---------------------------------------------------------------------------
def _exercise_items_of(payload: Any) -> list[dict[str, Any]]:
    """从接口结果中取出题目列表，丢弃结构异常的条目。"""
    if not isinstance(payload, dict):
        return []
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _exercise_type(item: dict[str, Any]) -> str:
    """判断题目类型；后端字段缺失时按是否带选项推断。"""
    value = str(item.get("type") or "").strip().lower()
    if value in EXERCISE_TYPE_LABELS:
        return value
    options = item.get("options")
    if isinstance(options, list) and len(options) >= 2:
        return "choice"
    return "blank"


def _exercise_options(item: dict[str, Any]) -> list[str]:
    """取出选择题的选项文本，过滤空项。"""
    options = item.get("options")
    if not isinstance(options, list):
        return []
    return [str(option).strip() for option in options if str(option).strip()]


def _choice_option_labels(options: list[str]) -> list[str]:
    """给选项加上 A/B/C/D 前缀，作为单选按钮的展示文本。"""
    letters = "ABCDEFGHIJ"
    return [
        f"{letters[position] if position < len(letters) else position + 1}. {text}"
        for position, text in enumerate(options)
    ]


def _exercise_answer_index(item: dict[str, Any], options: list[str]) -> int:
    """取出选择题的正确选项下标；缺省或越界时返回 -1。"""
    try:
        index = int(item.get("answer_index"))
    except (TypeError, ValueError):
        return -1
    return index if 0 <= index < len(options) else -1


def _normalise_blank_answer(value: Any) -> str:
    """填空题判分归一化：忽略大小写、全角空格与所有空白字符。"""
    text = str(value or "").replace("\u3000", " ")
    return re.sub(r"\s+", "", text).casefold()


def _escape_markdown(text: str) -> str:
    """转义题干/解析中的 Markdown 特殊字符，避免填空线被渲染成斜体。"""
    return re.sub(r"([_*`\[\]\\])", r"\\\1", str(text or ""))


def _exercise_source_text(item: dict[str, Any]) -> str:
    """题目的来源说明：文件名 + 页码。"""
    filename = (
        _first_text(item, "source_file", "file_name", "filename", "source") or "未知来源"
    )
    page = item.get("page")
    if page in (None, ""):
        return f"来源：{filename} ｜ 页码未知"
    page_number = _display_int(page, default=str(page))
    page_text = "无页码" if page_number == 0 else f"第 {page_number} 页"
    return f"来源：{filename} ｜ {page_text}"


def _exercise_answer_key(generation_id: int, position: int) -> str:
    """作答控件的 session_state 键名；重新生成题目后自动换新键。"""
    return f"exercise_answer_{generation_id}_{position}"


def _build_exercise_result(
    items: list[dict[str, Any]],
    answers: dict[int, Any],
) -> dict[str, Any]:
    """按作答内容判分，返回可直接展示的结果字典。

    填空题做大小写与空白归一化比较；选择题先把单选文本还原成下标，
    再与后端返回的 ``answer_index`` 比较。
    """
    details: list[dict[str, Any]] = []
    score = 0
    answered_count = 0

    for position, item in enumerate(items):
        kind = _exercise_type(item)
        raw_answer = answers.get(position)
        explanation = str(item.get("explanation") or "").strip()

        if kind == "choice":
            options = _exercise_options(item)
            labels = _choice_option_labels(options)
            chosen_label = str(raw_answer or "").strip()
            chosen_index = labels.index(chosen_label) if chosen_label in labels else -1
            answer_index = _exercise_answer_index(item, options)
            answered = chosen_index >= 0
            is_correct = answered and answer_index >= 0 and chosen_index == answer_index
            user_answer = labels[chosen_index] if answered else ""
            correct_answer = (
                labels[answer_index]
                if 0 <= answer_index < len(labels)
                else "（后端未返回标准答案）"
            )
        else:
            answer_text = str(item.get("answer") or "").strip()
            user_answer = str(raw_answer or "").strip()
            answered = bool(user_answer)
            is_correct = (
                answered
                and bool(answer_text)
                and _normalise_blank_answer(user_answer)
                == _normalise_blank_answer(answer_text)
            )
            correct_answer = answer_text or "（后端未返回标准答案）"

        if answered:
            answered_count += 1
        if is_correct:
            score += 1

        details.append(
            {
                "number": position + 1,
                "type": kind,
                "stem": str(item.get("stem") or "").strip(),
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "answered": answered,
                "is_correct": is_correct,
                "explanation": explanation,
            }
        )

    total = len(details)
    return {
        "score": score,
        "total": total,
        "answered": answered_count,
        "accuracy": (score / total * 100.0) if total else 0.0,
        "details": details,
        "graded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _render_exercise_meta(meta: dict[str, Any]) -> None:
    """展示本次出题的来源标记与参数摘要。"""
    generated_by = str(meta.get("generated_by") or "")
    normalized = generated_by.lower()
    label = _generated_by_label(generated_by)
    if any(keyword in normalized for keyword in ("rule", "fallback", "template")):
        st.warning(
            f"本题由{label}。大模型暂不可用或输出未通过校验，题目仍可正常作答；"
            "模型恢复后可重新生成。"
        )
    elif not normalized:
        st.caption("后端未返回 generated_by，无法确认题目来源。")

    # 来源与课程/题型等并作一行版记
    meta_parts: list[str] = []
    if normalized:
        meta_parts.append(f"题目来源：{label}")
    if meta.get("course"):
        meta_parts.append(f"课程：{meta['course']}")
    if meta.get("chapter"):
        meta_parts.append(f"章节：{meta['chapter']}")
    types = meta.get("types") or []
    if types:
        meta_parts.append(
            "题型：" + "、".join(EXERCISE_TYPE_LABELS.get(str(item), str(item)) for item in types)
        )
    difficulty = str(meta.get("difficulty") or "")
    if difficulty:
        meta_parts.append(f"难度：{EXERCISE_DIFFICULTY_LABELS.get(difficulty, difficulty)}")
    if meta.get("generated_at"):
        meta_parts.append(f"生成时间：{meta['generated_at']}")
    if meta_parts:
        st.caption(" ｜ ".join(meta_parts))

    for warning in meta.get("warnings") or []:
        st.caption(f"后端提示：{warning}")


def _render_exercise_score(result: dict[str, Any]) -> None:
    """顶部得分条：得分、正确率、已作答与批阅印记。"""
    total = int(result.get("total") or 0)
    score = int(result.get("score") or 0)
    accuracy = float(result.get("accuracy") or 0.0)
    answered = int(result.get("answered") or 0)

    if answered >= total and total:
        stamp = "已批阅"
    elif answered:
        stamp = "未答完"
    else:
        stamp = "待作答"
    st.markdown(
        theme.score_strip(score, total, accuracy, answered, stamp),
        unsafe_allow_html=True,
    )
    st.caption(
        f"判分时间：{result.get('graded_at') or '—'}；修改答案后请重新提交以更新得分。"
    )


def _render_exercise_feedback(detail: dict[str, Any], *, source: str = "") -> None:
    """以朱批形式展示单题判分：对错、标准答案与解析。"""
    answered = bool(detail.get("answered"))
    correct: bool | None = bool(detail.get("is_correct")) if answered else None
    explanation = str(detail.get("explanation") or "").strip()
    if not explanation and not answered:
        explanation = ""
    elif not explanation:
        explanation = "本题为填空题，标准答案直接取自资料原文，可对照下方来源核对。"

    st.markdown(
        theme.mark_block(
            correct=correct,
            correct_answer=str(detail.get("correct_answer") or "—"),
            explanation=explanation,
            source=source,
            user_answer=str(detail.get("user_answer") or ""),
        ),
        unsafe_allow_html=True,
    )


def _render_exercises_tab() -> None:
    """页签四：练习自测，生成题目、作答、提交判分。"""
    _show_flash()

    st.subheader("练习自测")
    st.caption(
        "根据已索引的课程资料生成填空题与选择题；提交后自动判分，"
        "并给出标准答案、解析与原文来源。"
    )

    with st.container():
        # 下拉选项都来自已上传文档；本页不在 form 内，选择课程会立即 rerun 并刷新章节。
        course_options = _course_options()
        col_course, col_chapter = st.columns(2)
        course = col_course.selectbox(
            "课程",
            course_options or ["（暂无课程）"],
            index=None,
            placeholder="请选择课程",
            key="exercise_course_select",
            disabled=not course_options,
        )
        chapter_options = _chapter_options(course) if course and course_options else []
        chapter_label = col_chapter.selectbox(
            "章节（可选）",
            [ALL_CHAPTERS_LABEL, *chapter_options],
            index=0,
            key="exercise_chapter_select",
            disabled=not (course_options and course),
        )
        # 选“全部章节”时提交空字符串；后端 chapter 为空即表示从整门课程中抽题。
        chapter = "" if chapter_label == ALL_CHAPTERS_LABEL else str(chapter_label or "")
        if not course_options:
            st.warning("暂无课程，请先在「资料管理」页上传资料并填写课程名称。")

        col_count, col_difficulty, col_llm = st.columns([2, 1, 1])
        count = col_count.slider(
            "题目数量",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key="exercise_count_slider",
            help="单次最多生成 10 道题目，题型会在所选范围内轮转分配。",
        )
        difficulty = col_difficulty.selectbox(
            "难度",
            options=list(EXERCISE_DIFFICULTY_LABELS),
            format_func=lambda value: EXERCISE_DIFFICULTY_LABELS[value],
            index=1,
            key="exercise_difficulty_select",
        )
        use_llm = col_llm.checkbox(
            "使用大模型出题",
            value=True,
            key="exercise_use_llm_checkbox",
            help="关闭后直接使用规则模板出题，速度更快，但题目样式相对固定。",
        )

        selected_types = st.multiselect(
            "题型 *",
            options=list(EXERCISE_TYPE_ORDER),
            default=list(EXERCISE_TYPE_ORDER),
            format_func=lambda value: EXERCISE_TYPE_LABELS[value],
            key="exercise_types_select",
            placeholder="请选择至少一种题型",
        )
        generate_clicked = st.button(
            "生成题目",
            key="generate_exercises_btn",
            type="primary",
            # 没有任何课程时禁用出题，避免把占位文案当成课程名提交给后端。
            disabled=not course_options or not course,
        )

    if generate_clicked:
        course_text = (course or "").strip()
        types = [value for value in EXERCISE_TYPE_ORDER if value in selected_types]
        if not course_text:
            st.warning("请先选择课程。")
        elif not types:
            st.warning("请至少选择一种题型（填空题或选择题）。")
        else:
            with st.spinner("正在检索资料并生成题目，请稍候..."):
                try:
                    raw = generate_exercises(
                        course_text, chapter, count, types, difficulty, use_llm
                    )
                except Exception as exc:
                    # 只提示本次失败，不清空已有题目，便于稍后重试。
                    _show_backend_error(exc)
                else:
                    items = _exercise_items_of(raw)
                    generation_id = (
                        int(st.session_state.get("exercise_generation_id") or 0) + 1
                    )
                    st.session_state["exercise_generation_id"] = generation_id
                    st.session_state["exercise_items"] = items
                    st.session_state["exercise_meta"] = {
                        "course": str(raw.get("course") or course_text),
                        "chapter": str(raw.get("chapter") or "").strip(),
                        "types": types,
                        "difficulty": difficulty,
                        "use_llm": bool(use_llm),
                        "target_count": int(count),
                        "generated_by": str(raw.get("generated_by") or ""),
                        "warnings": [
                            str(item).strip()
                            for item in (raw.get("warnings") or [])
                            if str(item).strip()
                        ],
                        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "generation_id": generation_id,
                    }
                    # 新题目不继承上一批题目的作答与判分结果
                    st.session_state.pop("exercise_result", None)
                    for stale_key in list(st.session_state.keys()):
                        if str(stale_key).startswith("exercise_answer_"):
                            st.session_state.pop(stale_key, None)
                    if items:
                        _set_flash(f"已生成 {len(items)} 道题目，请作答后提交。")
                    else:
                        _set_flash(
                            "当前课程暂无资料，请先在资料管理页上传并建立索引后再生成题目。",
                            kind="warning",
                        )
                    st.rerun()

    st.divider()
    items = st.session_state.get("exercise_items")
    meta = st.session_state.get("exercise_meta")
    if not isinstance(meta, dict):
        st.info("生成后会在此处显示题目；作答后点击「提交答案」即可查看得分与逐题解析。")
        return

    _render_exercise_meta(meta)

    if not isinstance(items, list) or not items:
        st.warning("当前课程暂无资料，请先在资料管理页上传并建立索引后再生成题目。")
        return

    generation_id = int(meta.get("generation_id") or 0)
    result = st.session_state.get("exercise_result")
    if not isinstance(result, dict) or int(result.get("total") or 0) != len(items):
        result = None
        details: list[Any] = []
    else:
        details = result.get("details") if isinstance(result.get("details"), list) else []
    if result is not None:
        _render_exercise_score(result)

    target = int(meta.get("target_count") or 0)
    heading = f"#### 共 {len(items)} 道题目"
    if target and target != len(items):
        heading = f"{heading}（目标 {target} 道）"
    st.markdown(heading)

    for position, item in enumerate(items):
        kind = _exercise_type(item)
        stem = str(item.get("stem") or "").strip() or "（题干为空）"
        source = _exercise_source_text(item)
        if position:
            st.markdown('<div class="q-sep"></div>', unsafe_allow_html=True)
        with st.container():
            st.markdown(
                theme.question_head(
                    position + 1,
                    EXERCISE_TYPE_LABELS[kind],
                    stem,
                    str(item.get("hint") or "").strip(),
                ),
                unsafe_allow_html=True,
            )

            key = _exercise_answer_key(generation_id, position)
            if kind == "choice":
                labels = _choice_option_labels(_exercise_options(item))
                if labels:
                    # 首项为占位项，默认即“未作答”，不使用 index 预选答案
                    st.radio(
                        f"第 {position + 1} 题选项",
                        options=[EXERCISE_CHOICE_PLACEHOLDER, *labels],
                        key=key,
                        label_visibility="collapsed",
                    )
                else:
                    st.warning("后端未返回该题的选项，暂时无法作答。")
            else:
                st.text_input(
                    f"第 {position + 1} 题作答",
                    key=key,
                    placeholder="填写空缺的内容",
                    label_visibility="collapsed",
                )

            detail = details[position] if position < len(details) else None
            if isinstance(detail, dict):
                _render_exercise_feedback(detail, source=source)
            else:
                st.caption(source)

    col_submit, col_reset, col_tip = st.columns([1, 1, 2])
    with col_submit:
        submitted = st.button(
            "提交答案",
            key="submit_exercises_btn",
            type="primary",
            use_container_width=True,
        )
    with col_reset:
        reset_clicked = st.button(
            "重新作答",
            key="reset_exercises_btn",
            use_container_width=True,
        )
    with col_tip:
        st.caption("填空判分忽略大小写与空白；选择题按所选选项文本比对。")

    if reset_clicked:
        prefix = f"exercise_answer_{generation_id}_"
        for stale_key in list(st.session_state.keys()):
            if str(stale_key).startswith(prefix):
                st.session_state.pop(stale_key, None)
        st.session_state.pop("exercise_result", None)
        st.rerun()

    if submitted:
        answers = {
            position: st.session_state.get(
                _exercise_answer_key(generation_id, position), ""
            )
            for position in range(len(items))
        }
        graded = _build_exercise_result(items, answers)
        if graded["answered"] == 0:
            st.session_state.pop("exercise_result", None)
            _set_flash("还没有作答任何题目，请先填写答案再提交。", kind="warning")
        else:
            st.session_state["exercise_result"] = graded
        st.rerun()


# ---------------------------------------------------------------------------
# 应用入口
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 课本原文：翻书式阅读器
# ---------------------------------------------------------------------------
# Streamlit 的执行模型是“每次交互重跑整个脚本”，原生控件无法承载需要持续
# JS 状态的翻页动画（换页就会被重建）。因此把阅读器放进 st.components 的
# iframe 里：翻页完全由 iframe 内部 JS 处理，Python 只在“用户主动跳转”时
# 重新生成一次 HTML。
#
# 两个必须遵守的约束：
#   1. iframe 高度固定（components.html 不会按内容撑高），所以写死 BOOK_VIEWER_HEIGHT；
#   2. HTML 字符串必须稳定，否则任何无关交互都会让 iframe 重建、阅读进度归零。
#      页面数据只依赖接口返回内容，不含时间戳/随机数。

BOOK_VIEWER_HEIGHT = 620
# 单个段落超过该长度时按句号切分，避免一页只有一大坨文字
BOOK_PARAGRAPH_LIMIT = 260
# 无页码文件（txt/docx）按该字数切成虚拟书页。按 13px 字号、书页可视高度
# 实测：约 350-390 字正好填满一页且不会把底部页码挤出可视区，取 360 兼顾均匀。
BOOK_PAGE_CHAR_CAPACITY = 360


def _book_int(value: Any, default: int = 0) -> int:
    """把后端可能返回的数字或字符串安全地转成 int（供翻书组件内部比较用）。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _paginate_paragraphs(
    paragraphs: list[dict[str, str]],
    capacity: int = BOOK_PAGE_CHAR_CAPACITY,
) -> list[list[dict[str, str]]]:
    """按字数容量把段落切成若干书页。"""
    buckets: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    size = 0
    for paragraph in paragraphs:
        length = len(paragraph["t"])
        if current and size + length > capacity:
            buckets.append(current)
            current = []
            size = 0
        current.append(paragraph)
        size += length
    if current:
        buckets.append(current)
    return buckets or [[]]


_BOOK_HEADING_RE = re.compile(
    r"^(?:\d+(?:\s*[.．]\s*\d+)*[\s.．、]|第[一二三四五六七八九十百]+[章节]|[（(][一二三四五六七八九十]+[）)])"
)
_BOOK_TERMINAL = "。！？；：.!?;:”」"
_BOOK_STOP_CHARS = "。！？；，、：,;:;!?."


def _book_is_heading(line: str, prev_closed: bool) -> bool:
    """判断一行是否为标题（需要单独成段）。"""
    if len(line) <= 40 and _BOOK_HEADING_RE.match(line):
        return True
    # 不含句读的短行（如“本章导读”“（课程讲义·复习用）”）：只有在上文已收尾时才算标题，
    # 否则它更可能是被硬换行截断的正文片段。
    if len(line) <= 24 and not any(ch in line for ch in _BOOK_STOP_CHARS):
        return prev_closed
    return False


def _book_paragraphs(text: str, limit: int = BOOK_PARAGRAPH_LIMIT) -> list[str]:
    """把分块文本切成适合单页排版的段落列表。

    PDF 抽出的文本是“一行一块”，若直接一行一段，书页上会出现大量以缩进
    开头、句子被硬换行截断的碎片。这里先按行合并回自然段：

    - 编号行、第 X 章/节、以及不含句读的短行视为标题，单独成段；
    - 上一行以句末标点结尾则另起一段，否则把下一行接到同一段；
    - 两侧都是 ASCII 字符时补一个空格，避免英文单词粘连。

    合并后再按 ``limit`` 在句末标点处切开，防止单段过长撑破书页。
    """
    blocks: list[str] = []
    buffer = ""
    prev_closed = True
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _book_is_heading(line, prev_closed):
            if buffer:
                blocks.append(buffer)
                buffer = ""
            blocks.append(line)
            prev_closed = True
            continue
        if not buffer:
            buffer = line
        elif buffer[-1] in _BOOK_TERMINAL:
            blocks.append(buffer)
            buffer = line
        elif buffer[-1].isascii() and line[0].isascii() and not buffer[-1].isspace():
            buffer = f"{buffer} {line}"
        else:
            buffer += line
        prev_closed = buffer[-1] in _BOOK_TERMINAL
    if buffer:
        blocks.append(buffer)

    pieces: list[str] = []
    for block in blocks:
        line = block
        while len(line) > limit:
            cut = max(line.rfind(mark, 0, limit) for mark in "。！？；.!?;")
            if cut < limit // 2:
                cut = limit
            pieces.append(line[: cut + 1].strip())
            line = line[cut + 1 :].strip()
        if line:
            pieces.append(line)
    return pieces


def _book_payload(doc_id: str, start_page: int, target_chunk: str) -> dict[str, Any]:
    """把后端分块整理成翻书组件需要的页面数据。

    两种分页策略：
    - 文档带真实页码（PDF/PPT，``page > 0``）时按原页码一页对一页，保持引用可核对；
    - 无页码文档（txt/docx，``page == 0``）按字数切成虚拟书页，否则整份资料只有一页，
      翻书就失去意义。

    起始页优先用 ``target_chunk`` 命中位置（引用跳转最准），否则退回页码比较。
    """
    raw = fetch_document_chunks(doc_id)
    start_page = _book_int(start_page)

    groups: list[tuple[int, list[dict[str, str]]]] = []
    last_text = ""
    for group in raw.get("pages") or []:
        if not isinstance(group, dict):
            continue
        paragraphs: list[dict[str, str]] = []
        for chunk in group.get("chunks") or []:
            if not isinstance(chunk, dict):
                continue
            chunk_id = str(chunk.get("chunk_id") or "")
            for piece in _book_paragraphs(str(chunk.get("text") or "")):
                # 分块之间带 overlap，相邻块的同一段会重复出现，书页上只保留一次
                if piece == last_text:
                    continue
                last_text = piece
                paragraphs.append({"t": piece, "c": chunk_id})
        groups.append((_book_int(group.get("page")), paragraphs))

    pages: list[dict[str, Any]] = []
    if any(page_no > 0 for page_no, _ in groups):
        for page_no, paragraphs in groups:
            pages.append(
                {
                    "page": page_no,
                    "label": "无页码" if page_no <= 0 else f"第 {page_no} 页",
                    "paras": paragraphs,
                }
            )
    else:
        flattened = [para for _, paragraphs in groups for para in paragraphs]
        for position, bucket in enumerate(_paginate_paragraphs(flattened), start=1):
            pages.append({"page": 0, "label": f"第 {position} 页", "paras": bucket})

    start_index = 0
    if target_chunk:
        # 一个分块可能跨多页，取“命中段落最多”的那一页作为落点，比只看第一处更准
        best_index, best_count = 0, 0
        for position, item in enumerate(pages):
            hits = sum(1 for para in item["paras"] if para["c"] == target_chunk)
            if hits > best_count:
                best_index, best_count = position, hits
        start_index = best_index
    elif any(item["page"] > 0 for item in pages):
        # 有真实页码时才按页码定位；虚拟分页的 page 全为 0，不能这样比较
        for position, item in enumerate(pages):
            if item["page"] <= start_page:
                start_index = position

    return {
        "title": str(raw.get("file_name") or ""),
        "pages": pages,
        "start": start_index,
        "target": target_chunk,
    }


def _book_json(payload: dict[str, Any]) -> str:
    """序列化为可安全嵌入 <script> 的 JSON。"""
    # 直接内联 </script> 会提前终止标签，把尖括号转成 JSON 的 \\u 转义即可
    return (
        json.dumps(payload, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


_BOOK_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; height: 100%; }
body {
  background: #EFE9D8;
  color: #3F3D36;
  font-family: "Songti SC", "SimSun", "Noto Serif SC", Georgia, serif;
  overflow: hidden;
}
#app { height: 100%; display: flex; flex-direction: column; }
#bar {
  flex: 0 0 auto; position: relative; z-index: 30;
  display: flex; align-items: center; justify-content: center; gap: 10px;
  padding: 9px 12px;
  background: #E9E2CF;
  border-bottom: 1px solid #D6CCB2;
  font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
}
#bar button {
  font-size: 12px; padding: 4px 13px; cursor: pointer;
  border: 1px solid #C0B59A; border-radius: 2px;
  background: #FAF7EF; color: #5C5648;
  letter-spacing: .08em; transition: border-color .16s ease, color .16s ease;
}
#bar button:hover:not(:disabled) { border-color: #A8442A; color: #A8442A; }
#bar button:disabled { opacity: .38; cursor: default; }
#info { font-size: 12px; color: #6B6553; min-width: 150px; text-align: center; }
#jump {
  width: 54px; font-size: 12px; padding: 3px 6px; text-align: center;
  border: 1px solid #C0B59A; border-radius: 2px;
  background: #FAF7EF; color: #4A4738;
}
#stage {
  flex: 1; position: relative; max-width: 840px;
  /* 两侧各留 28px 余量：纸张翻起时透视会把它放大 5% 左右，留白避免压到书框外 */
  width: calc(100% - 56px);
  margin: 0 auto; perspective: 3000px;
}
/* 书脊中缝：让左右两页看起来是一张摊开的纸，而不是两个并排的框 */
#stage::after {
  content: ""; position: absolute; top: 8px; bottom: 8px; left: 50%;
  width: 22px; margin-left: -11px; pointer-events: none; z-index: 6;
  background: linear-gradient(90deg,
    rgba(120,108,78,0) 0%, rgba(120,108,78,.10) 38%,
    rgba(120,108,78,.20) 50%, rgba(120,108,78,.10) 62%,
    rgba(120,108,78,0) 100%);
}
.pane { position: absolute; top: 8px; bottom: 8px; width: 50%; overflow: hidden; background: #FAF7EF; }
#paneL { left: 0; box-shadow: inset -14px 0 20px -16px rgba(88,78,48,.5); }
#paneR { left: 50%; box-shadow: inset 14px 0 20px -16px rgba(88,78,48,.5); }
.sheet {
  position: absolute; top: 8px; bottom: 8px; left: 50%; width: 50%;
  transform-origin: left center; transform-style: preserve-3d;
  transition: transform .72s cubic-bezier(.44,.05,.28,1);
  opacity: 0; visibility: hidden; z-index: 20;
}
.sheet.run { opacity: 1; visibility: visible; }
.sheet.flipped { transform: rotateY(-180deg); }
.face {
  position: absolute; inset: 0; overflow: hidden; background: #FAF7EF;
  backface-visibility: hidden; -webkit-backface-visibility: hidden;
  /* 外投影让翻起的纸有厚度，底下那页看起来是「被压住」而不是被切碎 */
  box-shadow: 0 4px 22px rgba(96,84,52,.20), 0 1px 3px rgba(96,84,52,.14);
}
.face.front { box-shadow: inset 14px 0 20px -16px rgba(88,78,48,.5), 0 4px 22px rgba(96,84,52,.20); }
.face.back { transform: rotateY(180deg); box-shadow: inset -14px 0 20px -16px rgba(88,78,48,.5), 0 4px 22px rgba(96,84,52,.20); }
.pg {
  position: absolute; inset: 0; overflow: auto;
  padding: 22px 28px 36px;
  font-size: 13px; line-height: 1.82; color: #3F3D36;
}
.pg .pm {
  font-size: 10.5px; color: #A79F8A; letter-spacing: .14em;
  margin: 0 0 14px; text-transform: uppercase;
  font-family: -apple-system, "Segoe UI", sans-serif;
}
.pg p { margin: 0 0 8px; text-align: justify; text-indent: 2em; }
.pg p.hl {
  background: #F7E9A6; text-indent: 0; border-radius: 2px;
  padding: 3px 5px; margin: 0 -5px 8px;
  box-shadow: 0 0 0 2px #F7E9A6;
}
/* 同一个分块可能覆盖整页，只有第一处标成实心黄，其余用淡底表示"同源范围" */
.pg p.hl-soft {
  background: rgba(247,233,166,.48); text-indent: 0; border-radius: 2px;
  padding: 3px 5px; margin: 0 -5px 8px;
}
.pg .pno {
  position: absolute; bottom: 11px; left: 0; right: 0;
  text-align: center; font-size: 11px; color: #9C9480; text-indent: 0;
}
.cover {
  height: 100%; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 11px; padding: 0 26px; text-align: center;
}
.cover-title { font-size: 16px; line-height: 1.65; color: #3F3D36; }
.cover-meta { font-size: 11px; color: #A79F8A; letter-spacing: .12em; }
.cover-hint { font-size: 11px; color: #BDB5A0; margin-top: 4px; }
.pg::-webkit-scrollbar { width: 8px; }
.pg::-webkit-scrollbar-thumb { background: #D8D1BC; border-radius: 4px; }
</style>
</head>
<body>
<div id="app">
  <div id="bar">
    <button id="prev">上一页</button>
    <span id="info"></span>
    <button id="next">下一页</button>
    <input id="jump" type="number" min="1" value="1">
    <button id="go">跳转</button>
  </div>
  <div id="stage">
    <div class="pane" id="paneL"><div class="pg" id="pgL"></div></div>
    <div class="pane" id="paneR"><div class="pg" id="pgR"></div></div>
    <div class="sheet" id="sheet">
      <div class="face front"><div class="pg" id="pgF"></div></div>
      <div class="face back"><div class="pg" id="pgB"></div></div>
    </div>
  </div>
</div>
<script id="book-data" type="application/json">__BOOK_DATA__</script>
<script>
(function () {
  var DATA = JSON.parse(document.getElementById('book-data').textContent);
  var pages = DATA.pages || [];
  var target = DATA.target || '';
  var total = pages.length;
  // 真实书籍里翻动的是一张纸：正面 N、背面 N+1。摊开后左页 = 2s-1、右页 = 2s，
  // s 是跨页序号，所以每次翻页页码前进 2，而不是平移一页。
  var maxSpread = Math.max(0, Math.floor(total / 2));
  var spread = clampSpread(Math.floor(((DATA.start | 0) + 1) / 2));
  var busy = false;
  var byId = function (id) { return document.getElementById(id); };

  function clampSpread(value) {
    if (value < 0) { return 0; }
    if (value > maxSpread) { return maxSpread; }
    return value;
  }

  function buildCover(el) {
    el.innerHTML = '';
    var box = document.createElement('div');
    box.className = 'cover';
    var title = document.createElement('div');
    title.className = 'cover-title';
    title.textContent = DATA.title || '';
    var meta = document.createElement('div');
    meta.className = 'cover-meta';
    meta.textContent = '共 ' + pages.length + ' 页';
    var hint = document.createElement('div');
    hint.className = 'cover-hint';
    hint.textContent = '点击「下一页」开始阅读';
    box.appendChild(title);
    box.appendChild(meta);
    box.appendChild(hint);
    el.appendChild(box);
  }

  function fitPage(el) {
    // 一页 PDF 对应一页书：内容超出时逐步缩小字号，而不是让页面滚动。
    var size = 13;
    el.style.fontSize = size + 'px';
    var guard = 0;
    while (el.scrollHeight > el.clientHeight + 2 && size > 9.5 && guard < 24) {
      size -= 0.5;
      el.style.fontSize = size + 'px';
      guard += 1;
    }
  }

  function buildPage(el, index) {
    el.innerHTML = '';
    if (index < 0) { buildCover(el); return; }
    if (index >= pages.length) { return; }
    var page = pages[index];
    var head = document.createElement('div');
    head.className = 'pm';
    head.textContent = DATA.title || '';
    el.appendChild(head);
    var list = page.paras || [];
    var hit = null;
    for (var i = 0; i < list.length; i++) {
      var p = document.createElement('p');
      p.textContent = list[i].t;
      if (target && list[i].c === target) {
        // 首处实心高亮作为落点，同分块的其余段落用淡底，避免整页糊成一片黄
        p.className = hit ? 'hl-soft' : 'hl';
        if (!hit) { hit = p; }
      }
      el.appendChild(p);
    }
    var no = document.createElement('div');
    no.className = 'pno';
    no.textContent = page.label || '';
    el.appendChild(no);
    fitPage(el);
    el.scrollTop = hit ? Math.max(0, hit.offsetTop - 70) : 0;
  }

  function renderStatic() {
    var left = spread * 2 - 1;
    var right = spread * 2;
    buildPage(byId('pgL'), left);
    buildPage(byId('pgR'), right);
    var text;
    if (!total) {
      text = '暂无内容';
    } else if (left < 0) {
      text = '第 1 页 / 共 ' + total + ' 页';
    } else if (right >= total) {
      text = '第 ' + (left + 1) + ' 页 / 共 ' + total + ' 页';
    } else {
      text = '第 ' + (left + 1) + '–' + (right + 1) + ' 页 / 共 ' + total + ' 页';
    }
    byId('info').textContent = text;
    byId('prev').disabled = spread <= 0;
    byId('next').disabled = spread >= maxSpread;
    byId('jump').max = total;
    byId('jump').value = Math.min(Math.max(spread * 2 + 1, 1), Math.max(total, 1));
  }

  function flip(dir) {
    if (busy || !total) { return; }
    var next = spread + dir;
    if (next < 0 || next > maxSpread) { return; }
    busy = true;
    var forward = dir > 0;
    var sheet = byId('sheet');

    if (forward) {
      // 翻的是当前右页这张纸：正面 2s、背面 2s+1（翻过去正好落到新左页）。
      // 右边底层提前铺上新右页 2s+2，纸一转开露出的就是它。
      buildPage(byId('pgF'), spread * 2);
      buildPage(byId('pgB'), spread * 2 + 1);
      buildPage(byId('pgR'), spread * 2 + 2);
    } else {
      // 往回翻：这张纸现在躺在左边，背面 2s-1 朝上、正面 2s-2 落回右边。
      // 左边底层提前铺上新左页 2s-3，纸卷走后露出来的就是它。
      buildPage(byId('pgB'), spread * 2 - 1);
      buildPage(byId('pgF'), spread * 2 - 2);
      buildPage(byId('pgL'), spread * 2 - 3);
    }

    sheet.style.transition = 'none';
    sheet.classList.add('run');
    sheet.classList.toggle('flipped', !forward);
    void sheet.offsetWidth;
    sheet.style.transition = '';
    void sheet.offsetWidth;
    window.requestAnimationFrame(function () {
      sheet.classList.toggle('flipped', forward);
    });
    window.setTimeout(function () {
      spread = next;
      renderStatic();
      sheet.classList.remove('run');
      busy = false;
    }, 760);
  }

  byId('prev').onclick = function () { flip(-1); };
  byId('next').onclick = function () { flip(1); };
  byId('go').onclick = function () {
    if (busy || !total) { return; }
    var value = parseInt(byId('jump').value, 10);
    if (isNaN(value)) { return; }
    var index = Math.min(Math.max(value - 1, 0), total - 1);
    // 页码换算成跨页：偶数页落在右页、奇数页落在左页，保证目标页一定在视野内
    var next = clampSpread(Math.floor((index + 1) / 2));
    if (next === spread) { return; }
    spread = next;
    renderStatic();
  };
  document.addEventListener('keydown', function (event) {
    if (event.key === 'ArrowLeft') { flip(-1); }
    if (event.key === 'ArrowRight') { flip(1); }
  });
  renderStatic();
})();
</script>
</body>
</html>"""


def _render_book_viewer(
    doc_id: str,
    *,
    start_page: int = 0,
    target_chunk: str = "",
    height: int = BOOK_VIEWER_HEIGHT,
) -> None:
    """渲染翻书式原文阅读器。"""
    try:
        payload = _book_payload(doc_id, start_page, target_chunk)
    except (BackendError, BackendUnavailable) as exc:
        st.warning(f"无法读取原文分块：{exc}")
        return
    except BackendAPIError as exc:
        if exc.status_code == 404:
            st.warning("该资料已不存在，可能已被删除，请刷新文档列表。")
        else:
            st.warning(f"无法读取原文分块：{exc}")
        return

    if not payload["pages"]:
        st.info("该资料目前没有可阅读的原文分块。")
        return

    html = _BOOK_TEMPLATE.replace("__BOOK_DATA__", _book_json(payload))
    components.html(html, height=height, scrolling=False)


def _render_book_jump_panel() -> None:
    """问答页的「原文定位」面板；没有跳转请求时不占用任何空间。"""
    jump = st.session_state.get("book_jump")
    if not jump:
        return

    doc_id = str(jump.get("doc_id") or "")
    if not doc_id:
        st.session_state.pop("book_jump", None)
        return

    st.divider()
    col_title, col_close = st.columns([5, 1])
    with col_title:
        st.subheader("原文定位")
        page_no = _display_int(jump.get("page"), default=0)
        pinpoint = "无页码" if page_no <= 0 else f"第 {page_no} 页"
        st.caption(f"{jump.get('file_name') or doc_id} ｜ {pinpoint}")
    with col_close:
        st.write("")
        if st.button("收起", key="book_jump_close", use_container_width=True):
            st.session_state.pop("book_jump", None)
            st.rerun()

    _render_book_viewer(
        doc_id,
        start_page=_display_int(jump.get("page"), default=0),
        target_chunk=str(jump.get("chunk_id") or ""),
    )


def _render_book_tab() -> None:
    """课本原文页签：选择资料后以翻书方式浏览。"""
    st.subheader("课本原文")
    st.caption(
        "以翻书方式浏览已入库资料的原文分块，页码与问答引用一致；"
        "在「课程问答」的引用卡片上可以直接跳到对应页。"
    )

    try:
        documents = list_documents()
    except (BackendError, BackendUnavailable) as exc:
        st.error(f"读取文档列表失败：{exc}")
        return

    if not documents:
        st.info("还没有已入库的资料，请先在「资料管理」上传。")
        return

    options: dict[str, str] = {}
    for document in documents:
        doc_id = str(document.get("doc_id") or "")
        if not doc_id:
            continue
        name = str(document.get("file_name") or doc_id)
        chunk_count = _display_int(document.get("chunk_count"), default=0)
        options[doc_id] = f"{name}（{chunk_count} 块）"

    if not options:
        st.info("暂无可阅读的资料。")
        return

    doc_ids = list(options)
    selected = st.selectbox(
        "选择资料",
        options=doc_ids,
        format_func=lambda value: options.get(value, value),
        key="book_doc_select",
    )

    # 引用跳转过来时，起始页由会话状态决定；否则从第一页开始
    jump = st.session_state.get("book_jump") or {}
    start_page = 0
    target_chunk = ""
    if jump.get("doc_id") == selected:
        start_page = _display_int(jump.get("page"), default=0)
        target_chunk = str(jump.get("chunk_id") or "")

    _render_book_viewer(selected, start_page=start_page, target_chunk=target_chunk)


def main() -> None:
    st.set_page_config(
        page_title="课程资料问答与复习规划助手",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(theme.CSS, unsafe_allow_html=True)
    _render_sidebar()

    st.markdown(
        theme.masthead(
            "课程资料问答与复习规划助手",
            "单门课程 · RAG 可追溯问答 · 规则降级复习计划",
        ),
        unsafe_allow_html=True,
    )

    tab_documents, tab_chat, tab_plan, tab_exercises, tab_book = st.tabs(
        ["资料管理", "课程问答", "复习计划", "练习自测", "课本原文"]
    )
    with tab_documents:
        _render_documents_tab()
    with tab_chat:
        _render_chat_tab()
    with tab_plan:
        _render_plan_tab()
    with tab_exercises:
        _render_exercises_tab()
    with tab_book:
        _render_book_tab()


if __name__ == "__main__":
    main()
