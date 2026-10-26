"""应用配置模块。

基于 pydantic-settings，从环境变量与项目根目录的 ``.env`` 加载配置，
集中管理 LLM 供应商、Embedding 模式、Chroma/上传目录、分块与检索参数。

典型用法::

    from app.core.config import get_settings

    settings = get_settings()
    settings.ensure_directories()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

# 各 LLM 供应商的默认接入信息；显式配置（LLM_BASE_URL/LLM_MODEL）优先。
LLM_PROFILES: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


class Settings(BaseSettings):
    """全局配置对象。

    字段名与 ``.env`` 中的大写环境变量一一对应（大小写不敏感），例如
    ``CHUNK_SIZE`` 映射到 ``chunk_size``。未在 ``.env`` 中出现的字段使用
    下方默认值。
    """

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------------- 应用
    app_name: str = "课程资料问答与复习规划助手"
    app_env: str = "dev"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    api_key: str = ""
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    # ---------------------------------------------------------------- 目录
    data_dir: Path = PROJECT_ROOT / "data"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    chroma_collection: str = "course_chunks_v1"

    # ----------------------------------------------------------------- LLM
    llm_provider: Literal["deepseek", "qwen"] = "deepseek"
    # 通用覆盖项：留空时按 llm_provider 回退到对应的供应商配置。
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_timeout: int = Field(default=60, ge=1)

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"

    # ----------------------------------------------------------- Embedding
    embedding_mode: Literal["local", "api"] = "local"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = Field(default=384, ge=1)
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_api_key: str = ""
    embedding_api_base_url: str = ""
    embedding_api_model: str = "text-embedding-3-small"

    # --------------------------------------------------------------- RAG
    chunk_size: int = Field(default=500, ge=50)
    chunk_overlap: int = Field(default=50, ge=0)
    top_k: int = Field(default=5, ge=1, le=50)

    # --------------------------------------------------------------- 上传
    max_file_size_mb: int = Field(default=20, ge=1)
    max_files_per_upload: int = Field(default=10, ge=1)

    # ------------------------------------------------------------- 校验器
    @field_validator("llm_provider", "embedding_mode", mode="before")
    @classmethod
    def _normalize_choice(cls, value: object) -> str:
        """把供应商/模式值统一为小写并去除首尾空白。"""
        return str(value).strip().lower()

    @field_validator("data_dir", "upload_dir", "chroma_dir", mode="after")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        """展开 ``~``，并把相对路径解析为相对项目根目录的绝对路径。"""
        value = value.expanduser()
        if not value.is_absolute():
            value = (PROJECT_ROOT / value).resolve()
        return value

    @model_validator(mode="after")
    def _check_rag_params(self) -> "Settings":
        """校验分块与检索参数的组合是否合法。"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
        return self

    # ------------------------------------------------------------- 辅助属性
    @property
    def cors_origin_list(self) -> list[str]:
        """把逗号分隔的 CORS 配置解析为列表。"""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def llm_profile(self) -> dict[str, str]:
        """返回当前生效的 LLM 接入信息（通用配置优先于供应商配置）。"""
        defaults = LLM_PROFILES[self.llm_provider]
        if self.llm_provider == "deepseek":
            provider_key = self.deepseek_api_key
        else:
            provider_key = self.qwen_api_key
        return {
            "provider": self.llm_provider,
            "api_key": self.llm_api_key or provider_key,
            "base_url": self.llm_base_url or defaults["base_url"],
            "model": self.llm_model or defaults["model"],
        }

    @property
    def active_embedding_model(self) -> str:
        """返回当前 Embedding 模式下实际使用的模型名。"""
        if self.embedding_mode == "api":
            return self.embedding_api_model
        return self.embedding_model

    def ensure_directories(self) -> None:
        """创建运行所需的数据目录（幂等，可在应用启动时调用）。"""
        for directory in (self.data_dir, self.upload_dir, self.chroma_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取进程级单例配置，避免重复读取 ``.env``。"""
    return Settings()


def reload_settings() -> Settings:
    """清空缓存并重新加载配置，主要供测试与热更新场景使用。"""
    get_settings.cache_clear()
    return get_settings()
