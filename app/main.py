"""FastAPI 应用入口。

职责：
- 创建应用实例并注册 CORS、请求 ID 中间件；
- 挂载 ``/api`` 下的资料、问答、计划三组路由；
- 启动时初始化 ``data`` 下的上传、解析、向量库等目录；
- 提供 ``GET /api/health`` 健康检查，供前端与部署脚本探测。

运行方式::

    uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

接口文档：http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import importlib.util
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, documents, plan
from app.api.common import (
    RequestIDMiddleware,
    ensure_data_dirs,
    register_exception_handlers,
)
from app.core.config import get_settings

APP_VERSION = "1.0.0"
# 记录进程启动时间，健康检查中可据此判断是否为新实例。
_STARTED_AT = time.time()

settings = get_settings()
logger = logging.getLogger("course_rag")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：启动时初始化数据目录，关闭时留出扩展点。"""
    try:
        settings.ensure_directories()
        directories = ensure_data_dirs()
        logger.info(
            "数据目录就绪：uploads=%s chroma=%s",
            directories.get("uploads"),
            directories.get("chroma"),
        )
    except Exception:  # 目录创建失败不阻断启动，健康检查会给出明确状态
        logger.exception("初始化数据目录失败")

    logger.info(
        "%s 已启动（env=%s, version=%s）",
        settings.app_name,
        settings.app_env,
        APP_VERSION,
    )
    yield
    logger.info("%s 已关闭", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    description="单门课程资料 RAG 问答与复习计划助手后端接口",
    version=APP_VERSION,
    lifespan=lifespan,
)

# 请求 ID 中间件先生效（内层），再由 CORS 包裹，确保错误响应同样带有跨域头。
app.add_middleware(RequestIDMiddleware)

_origins = settings.cors_origin_list or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials="*" not in _origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# 统一异常处理：业务异常、HTTP 异常、参数校验、未知异常。
register_exception_handlers(app)

# 业务路由统一挂载到 /api 前缀下。
app.include_router(documents.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(plan.router, prefix="/api")


def _module_available(module_name: str) -> bool:
    """判断模块是否可以导入，不触发模块内的重量级初始化。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


@app.get("/api/health", tags=["系统"], summary="健康检查")
async def health_check() -> dict[str, Any]:
    """返回服务与关键依赖的可用状态。

    ``status`` 仅在数据目录不可写时降级为 ``degraded``；LLM 未配置 API Key
    属于常见开发态，不影响服务整体可用性，因此单独标记为 ``missing_key``。
    """
    directories = ensure_data_dirs()
    upload_dir = directories["uploads"]

    data_dir_ok = False
    try:
        probe = upload_dir / ".health_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        data_dir_ok = True
    except OSError:
        logger.exception("数据目录不可写：%s", upload_dir)

    llm_profile = settings.llm_profile
    llm_configured = bool(str(llm_profile.get("api_key") or "").strip())

    # 健康检查只做“依赖是否可用”的轻量判断，不触发模型加载或网络请求。
    chroma_ready = (
        _module_available("chromadb")
        and _module_available("app.rag.vectorstore")
        and directories["chroma"].exists()
    )
    if settings.embedding_mode == "api":
        embedding_ready = bool(str(settings.embedding_api_key or "").strip()) and _module_available(
            "app.services.embedder"
        )
    else:
        embedding_ready = (
            _module_available("sentence_transformers")
            and _module_available("torch")
            and _module_available("app.services.embedder")
        )

    return {
        "status": "ok" if data_dir_ok else "degraded",
        "app": settings.app_name,
        "version": APP_VERSION,
        "env": settings.app_env,
        "time": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": int(time.time() - _STARTED_AT),
        # 以下三个字段与设计文档的健康检查口径保持一致。
        "chroma": "ready" if chroma_ready else "unavailable",
        "embedding": "ready" if embedding_ready else "unavailable",
        "embedding_mode": settings.embedding_mode,
        "embedding_model": settings.active_embedding_model,
        "llm": "configured" if llm_configured else "missing_key",
        "checks": {
            "data_dir": str(directories["data"]),
            "upload_dir_writable": data_dir_ok,
            "chroma_dir": str(directories["chroma"]),
        },
    }


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """根路径返回简要导航信息，便于快速确认服务已启动。"""
    return {
        "app": settings.app_name,
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":  # pragma: no cover - 本地开发入口
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )
