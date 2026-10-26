"""API 层公共组件。

本模块集中提供 HTTP 适配所需的通用能力，避免各个路由文件重复实现：

- :class:`APIError`：业务异常，携带 HTTP 状态码与统一错误码；
- 全局异常处理器：把业务异常、HTTP 异常、参数校验失败与未知异常统一
  转换为 ``{code, message, detail, request_id}`` 结构；
- :class:`RequestIDMiddleware`：为每个请求生成或透传 ``X-Request-ID``，
  便于日志排查与前端报障；
- 数据目录、配置读取与跨模块函数调用的小工具。由于项目各模块由不同
  开发者并行实现，这里对函数签名做了有限的兼容处理（按参数名匹配、
  支持同步/异步函数），但不会吞掉业务异常。

成功响应遵循各接口文档定义的字段结构；错误响应统一由本模块生成。
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger("course_rag.api")

# 项目根目录：app/api/common.py -> app/api -> app -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# 异常定义与统一错误响应
# ---------------------------------------------------------------------------


class APIError(Exception):
    """业务异常。

    路由层只抛出该异常，由全局处理器转换为统一的 JSON 错误结构，
    从而保证所有接口的错误格式一致。
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail


def get_request_id(request: Request) -> str:
    """读取当前请求的 request_id；中间件未生效时回退为短横线。"""
    return str(getattr(request.state, "request_id", "") or "-")


def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    detail: Any = None,
) -> JSONResponse:
    """构造统一错误响应。

    结构固定为 ``code / message / detail / request_id``，其中 ``code`` 为
    供前端判断的稳定错误码，``message`` 为可直接展示的中文提示。
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "code": code,
            "message": message,
            "detail": detail,
            "request_id": get_request_id(request),
        },
        headers={"X-Request-ID": get_request_id(request)},
    )


async def handle_api_error(request: Request, exc: APIError) -> JSONResponse:
    """处理路由主动抛出的业务异常。"""
    logger.warning(
        "业务异常 request_id=%s path=%s code=%s message=%s",
        get_request_id(request),
        request.url.path,
        exc.code,
        exc.message,
    )
    return error_response(request, exc.status_code, exc.code, exc.message, exc.detail)


async def handle_http_exception(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """处理 FastAPI/Starlette 抛出的 HTTP 异常（404、405 等）。"""
    # 常见状态码映射为更直观的中文提示与错误码
    code_map = {
        400: ("INVALID_ARGUMENT", "请求参数不合法"),
        401: ("UNAUTHORIZED", "未授权"),
        403: ("FORBIDDEN", "无权访问"),
        404: ("NOT_FOUND", "请求的资源不存在"),
        405: ("METHOD_NOT_ALLOWED", "请求方法不被允许"),
        413: ("FILE_TOO_LARGE", "上传文件过大"),
        415: ("UNSUPPORTED_FILE_TYPE", "不支持的文件类型"),
        422: ("INVALID_ARGUMENT", "请求参数校验失败"),
        429: ("RATE_LIMITED", "请求过于频繁，请稍后重试"),
    }
    code, message = code_map.get(exc.status_code, ("HTTP_ERROR", "请求处理失败"))
    return error_response(request, exc.status_code, code, message, detail=None)


async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """处理请求体/查询参数校验失败，返回字段级错误明细。"""
    details: list[dict[str, Any]] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        details.append(
            {
                "field": location,
                "message": error.get("msg", "参数不合法"),
                "type": error.get("type", "value_error"),
            }
        )
    logger.info(
        "参数校验失败 request_id=%s path=%s errors=%s",
        get_request_id(request),
        request.url.path,
        details,
    )
    return error_response(
        request,
        422,
        "INVALID_ARGUMENT",
        "请求参数校验失败",
        detail=details,
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """兜底异常处理：记录堆栈，对外只返回通用的内部错误提示。"""
    logger.exception(
        "未处理异常 request_id=%s path=%s", get_request_id(request), request.url.path
    )
    return error_response(
        request,
        500,
        "INTERNAL_ERROR",
        "服务器内部错误，请稍后重试或联系管理员",
        detail={"exception": type(exc).__name__},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """把统一异常处理器注册到 FastAPI 应用。"""
    app.add_exception_handler(APIError, handle_api_error)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, handle_validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, handle_unexpected_error)


# ---------------------------------------------------------------------------
# 请求 ID 中间件
# ---------------------------------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    """为每个请求生成 ``req_xxx`` 标识，并写入响应头与日志上下文。

    如果客户端已传入 ``X-Request-ID``，则原样透传，方便前后端串联链路。
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# 配置与目录
# ---------------------------------------------------------------------------


def get_settings_safe() -> Any | None:
    """尽最大努力读取全局配置；配置模块不可用时返回 ``None``。

    这样即使配置依赖尚未安装完整，健康检查等只读接口仍可启动。
    """
    try:
        from app.core.config import get_settings

        return get_settings()
    except Exception as exc:  # pragma: no cover - 仅在配置缺失时触发
        logger.warning("读取应用配置失败，将使用默认值：%s", exc)
        return None


def get_data_dir() -> Path:
    """返回数据根目录，优先使用配置，其次环境变量，最后是 ``项目根/data``。"""
    settings = get_settings_safe()
    if settings is not None:
        configured = getattr(settings, "data_dir", None)
        if configured:
            return Path(configured).expanduser().resolve()

    env_dir = os.getenv("COURSE_RAG_DATA_DIR") or os.getenv("DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return PROJECT_ROOT / "data"


def ensure_data_dirs() -> dict[str, Path]:
    """创建运行所需的数据目录（幂等），返回目录映射。

    目录清单：``data``、``uploads``（原始资料）、``parsed``（解析中间产物）、
    ``plans``（复习计划 JSON）、``chroma``（向量库持久化目录）。
    """
    settings = get_settings_safe()
    base = get_data_dir()

    upload_dir = base / "uploads"
    if settings is not None:
        upload_value = getattr(settings, "upload_dir", None)
        if upload_value:
            upload_dir = Path(upload_value).expanduser().resolve()

    directories: dict[str, Path] = {
        "data": base,
        "uploads": upload_dir,
        "parsed": base / "parsed",
        "plans": base / "plans",
        "chroma": base / "chroma",
    }
    for directory in directories.values():
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # 目录不可写时保留原始异常信息，由调用方决定是否降级
            logger.error("创建数据目录失败 path=%s error=%s", directory, exc)
    return directories


# ---------------------------------------------------------------------------
# 跨模块调用适配
# ---------------------------------------------------------------------------

_MISSING = object()


def resolve_callable(
    module: Any,
    names: Sequence[str],
    fuzzy_keywords: Sequence[str] = (),
) -> Callable[..., Any] | None:
    """在模块中按候选名称查找可调用对象。

    项目由多人并行开发，部分函数的命名可能略有差异；优先精确匹配
    ``names``，找不到时再按 ``fuzzy_keywords`` 做一次保守的模糊匹配。
    """
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate) and not inspect.isclass(candidate):
            return candidate

    for keyword in fuzzy_keywords:
        lowered = keyword.lower()
        for name in dir(module):
            if name.startswith("_"):
                continue
            candidate = getattr(module, name, None)
            if (
                name.lower().endswith(lowered) or lowered in name.lower()
            ) and callable(candidate) and not inspect.isclass(candidate):
                return candidate
    return None


def build_call_arguments(
    func: Callable[..., Any],
    payload: dict[str, Any],
    aliases: dict[str, Sequence[str]] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """根据函数签名从 ``payload`` 中挑选可用的参数。

    - 同名参数直接传入；
    - 通过 ``aliases`` 支持 ``weak_chapters`` -> ``weak_points`` 这类别名；
    - 缺省值参数不强行传入，避免覆盖实现方默认逻辑；
    - 带 ``**kwargs`` 的函数只接收已识别的命名参数，避免把整包数据重复写入。
    """
    aliases = aliases or {}
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        # 内建函数等无法读取签名时，退化为全部按关键字传入
        return [], dict(payload)

    args: list[Any] = []
    kwargs: dict[str, Any] = {}

    for name, param in signature.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue

        value: Any = _MISSING
        if name in payload:
            value = payload[name]
        else:
            for alternate in aliases.get(name, ()):  # 参数名 -> payload 中的别名
                if alternate in payload:
                    value = payload[alternate]
                    break

        if value is _MISSING:
            # 未提供的可选参数交给实现方默认值；必填参数缺失时让调用处报错，
            # 错误信息会包含函数名，便于定位集成问题。
            continue

        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[name] = value

    return args, kwargs


async def invoke(
    func: Callable[..., Any],
    payload: dict[str, Any],
    *,
    aliases: dict[str, Sequence[str]] | None = None,
) -> Any:
    """按签名调用函数，兼容同步函数与协程函数。

    同步的解析、向量化、向量库操作会放入线程池执行，避免阻塞事件循环。
    """
    args, kwargs = build_call_arguments(func, payload, aliases)
    if inspect.iscoroutinefunction(func):
        result = func(*args, **kwargs)
    else:
        result = await run_in_threadpool(func, *args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


def truncate_text(text: Any, limit: int = 200) -> str:
    """压缩空白并截断长文本，用于生成引用摘要。"""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "……"
