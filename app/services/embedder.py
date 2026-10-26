"""向量化模块。

提供统一的 :func:`embed_texts` 入口，支持两种模式：

* ``local``：懒加载 ``sentence-transformers`` 的
  ``paraphrase-multilingual-MiniLM-L12-v2`` 模型，CPU 推理，首次调用时才下载/加载。
* ``api``：调用 OpenAI 兼容的 ``/embeddings`` 接口（如通义千问、DeepSeek 或
  自建网关），按批提交并保持返回顺序。

模型对象按 ``(模型名, 设备)`` 缓存在进程内，避免重复加载。
"""

from __future__ import annotations

import threading
from typing import Any, Sequence

from app.core.config import Settings, get_settings

# API 模式单次请求超时时间（秒）。
_EMBEDDING_API_TIMEOUT = 60

# 本地模型缓存：{(模型名, 设备): SentenceTransformer}
_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_LOCK = threading.Lock()


def embed_texts(
    texts: Sequence[str],
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    settings: Settings | None = None,
) -> list[list[float]]:
    """把一批文本转换为向量。

    Args:
        texts: 待向量化文本列表，不允许包含空字符串。
        mode: ``"local"`` 或 ``"api"``；默认取 ``settings.embedding_mode``。
        batch_size: 批大小；默认取 ``settings.embedding_batch_size``。
        settings: 可注入配置对象（测试用），默认读取全局配置。

    Returns:
        与输入等长、顺序一致的向量列表，每个向量为 ``list[float]``。

    Raises:
        ValueError: 文本为空、模式不支持或批大小非法。
        RuntimeError: 缺少依赖、模型加载失败或 Embedding API 调用失败。
    """
    config = settings or get_settings()
    selected_mode = (mode or config.embedding_mode).strip().lower()

    cleaned = [str(text).strip() for text in texts]
    if not cleaned:
        return []
    if any(not text for text in cleaned):
        raise ValueError("待向量化文本不能为空")

    size = int(batch_size) if batch_size is not None else config.embedding_batch_size
    if size <= 0:
        raise ValueError("batch_size 必须大于 0")

    if selected_mode == "local":
        return _embed_local(cleaned, config, size)
    if selected_mode == "api":
        return _embed_api(cleaned, config, size)
    raise ValueError(f"不支持的 EMBEDDING_MODE：{selected_mode}（可选 local/api）")


def embed_query(
    text: str,
    *,
    mode: str | None = None,
    settings: Settings | None = None,
) -> list[float]:
    """把单条查询文本转换为向量（检索时使用）。"""
    vectors = embed_texts([text], mode=mode, batch_size=1, settings=settings)
    return vectors[0] if vectors else []


def clear_model_cache() -> None:
    """清空本地模型缓存并释放内存，主要供测试或切换模型时使用。"""
    with _MODEL_LOCK:
        _MODEL_CACHE.clear()


def _get_local_model(settings: Settings) -> Any:
    """按需加载并缓存本地 SentenceTransformer 模型。"""
    key = (settings.embedding_model, settings.embedding_device or "cpu")
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - 依赖缺失时的友好提示
                raise RuntimeError(
                    "本地 Embedding 模式需要安装 sentence-transformers，"
                    "或在 .env 中设置 EMBEDDING_MODE=api"
                ) from exc
            try:
                model = SentenceTransformer(
                    settings.embedding_model,
                    device=settings.embedding_device or None,
                )
            except Exception as exc:  # noqa: BLE001 - 模型下载/加载异常类型不稳定
                raise RuntimeError(
                    f"加载本地 Embedding 模型失败：{settings.embedding_model}（{exc}）"
                ) from exc
            _MODEL_CACHE[key] = model
    return model


def _embed_local(texts: list[str], settings: Settings, batch_size: int) -> list[list[float]]:
    """使用本地 sentence-transformers 模型批量推理。"""
    model = _get_local_model(settings)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    # 单条输入在部分版本会返回一维向量，统一补成二维再逐行转换。
    if len(texts) == 1 and getattr(vectors, "ndim", 2) == 1:
        vectors = [vectors]
    return [[float(value) for value in vector] for vector in vectors]


def _embed_api(texts: list[str], settings: Settings, batch_size: int) -> list[list[float]]:
    """调用 OpenAI 兼容的 ``/embeddings`` 接口批量向量化。"""
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("API Embedding 模式需要安装 requests") from exc

    base_url = settings.embedding_api_base_url.strip().rstrip("/")
    api_key = settings.embedding_api_key.strip()
    if not base_url:
        raise RuntimeError("API Embedding 模式需配置 EMBEDDING_API_BASE_URL")
    if not api_key:
        raise RuntimeError("API Embedding 模式需配置 EMBEDDING_API_KEY")

    endpoint = f"{base_url}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        payload = {"model": settings.embedding_api_model, "input": batch}
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=_EMBEDDING_API_TIMEOUT,
            )
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Embedding API 调用失败：{exc}") from exc
        except ValueError as exc:
            raise RuntimeError("Embedding API 返回内容不是合法 JSON") from exc

        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != len(batch):
            raise RuntimeError("Embedding API 返回格式异常：data 数量与输入不一致")

        ordered = sorted(
            data,
            key=lambda item: item.get("index", 0) if isinstance(item, dict) else 0,
        )
        for item in ordered:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list) or not embedding:
                raise RuntimeError("Embedding API 返回格式异常：缺少 embedding 字段")
            vectors.append([float(value) for value in embedding])

    return vectors
