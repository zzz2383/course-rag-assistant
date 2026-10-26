"""统一的大模型调用封装。

本模块提供唯一的 LLM 出口 ``chat(messages) -> str``，通过 OpenAI 兼容的
``/chat/completions`` 接口调用 DeepSeek 或通义千问（DashScope 兼容模式）。

设计约束：
1. 不依赖 LangChain 的模型对象，只用 ``requests``，便于离线测试和替换；
2. 超时 60 秒，网络错误 / 5xx / 429 自动重试 2 次（指数退避）；
3. 所有失败都转换为 :class:`LLMError` 体系内的可控异常，调用方可以据此降级；
4. 不在日志中输出 API Key、完整提示词或完整响应，避免泄露课程资料。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional, Sequence

try:  # requests 属于项目运行依赖；导入失败时延迟到调用阶段报错，便于纯单元测试
    import requests
except ImportError:  # pragma: no cover - 仅在极端环境触发
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT = 60  # 秒，需求规定的单次调用超时
DEFAULT_MAX_RETRIES = 2  # 失败后最多重试 2 次，即最多尝试 3 次
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1500
CONNECT_TIMEOUT = 10  # 秒

# 两种供应商的默认配置；Settings 中对应字段存在时会被覆盖
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "deepseek": {
        "api_key_field": "DEEPSEEK_API_KEY",
        "base_url_field": "DEEPSEEK_BASE_URL",
        "model_field": "DEEPSEEK_MODEL",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "qwen": {
        "api_key_field": "QWEN_API_KEY",
        "base_url_field": "QWEN_BASE_URL",
        "model_field": "QWEN_MODEL",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}

# 供应商别名，兼容配置里常见的写法
PROVIDER_ALIASES = {
    "deepseek": "deepseek",
    "ds": "deepseek",
    "qwen": "qwen",
    "dashscope": "qwen",
    "tongyi": "qwen",
    "通义千问": "qwen",
}


# ---------------------------------------------------------------------------
# 异常体系
# ---------------------------------------------------------------------------
class LLMError(RuntimeError):
    """LLM 调用失败的基类，消息可以直接展示给用户或写入降级原因。"""

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        retryable: bool = False,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.retryable = retryable
        self.cause = cause


class LLMConfigError(LLMError):
    """配置缺失或非法（例如未提供 API Key、供应商名称不支持）。"""


class LLMTimeoutError(LLMError):
    """请求超时。"""


class LLMUpstreamError(LLMError):
    """供应商返回错误状态或无法解析的响应。"""


# ---------------------------------------------------------------------------
# 配置读取
# ---------------------------------------------------------------------------
def _load_settings() -> Any:
    """延迟加载 ``app.core.config`` 中的 Settings 实例。

    兼容三种常见写法：``get_settings()``、模块级 ``settings``、``Settings()``。
    延迟导入可以避免本模块在单元测试时强制依赖配置模块。
    """
    try:
        from app.core import config as config_module  # 延迟导入，避免循环依赖
    except Exception as exc:  # pragma: no cover - 配置模块缺失时才会触发
        raise LLMConfigError(
            "无法加载应用配置 app.core.config，请检查配置文件是否存在。",
            cause=exc,
        ) from exc

    getter = getattr(config_module, "get_settings", None)
    if callable(getter):
        try:
            return getter()
        except Exception as exc:  # pragma: no cover
            raise LLMConfigError("读取应用配置失败，请检查 .env 配置。", cause=exc) from exc

    settings_obj = getattr(config_module, "settings", None)
    if settings_obj is not None:
        return settings_obj

    settings_cls = getattr(config_module, "Settings", None)
    if settings_cls is not None:
        try:
            return settings_cls()
        except Exception as exc:
            raise LLMConfigError("初始化 Settings 失败，请检查环境变量。", cause=exc) from exc

    raise LLMConfigError("app.core.config 中未找到 Settings 或 settings，无法读取模型配置。")


def _read_setting(settings_obj: Any, name: str, default: Any = None) -> Any:
    """从 Settings 对象读取字段，兼容大写/小写字段名以及同名环境变量。"""
    for candidate in (name, name.lower()):
        if hasattr(settings_obj, candidate):
            value = getattr(settings_obj, candidate)
            if value is not None and value != "":
                return value
    env_value = os.environ.get(name)
    if env_value not in (None, ""):
        return env_value
    return default


def normalize_provider(provider: Optional[str]) -> str:
    """把供应商名称归一化为 ``deepseek`` 或 ``qwen``。"""
    name = (provider or "deepseek").strip().lower()
    normalized = PROVIDER_ALIASES.get(name)
    if normalized is None:
        raise LLMConfigError(
            f"不支持的 LLM_PROVIDER：{provider!r}，可选值为 deepseek 或 qwen。"
        )
    return normalized


def _resolve_provider_config(provider: Optional[str] = None) -> dict[str, Any]:
    """解析实际使用的供应商、密钥、地址与模型名。"""
    settings_obj = _load_settings()
    raw_provider = provider or _read_setting(settings_obj, "LLM_PROVIDER", "deepseek")
    provider_name = normalize_provider(str(raw_provider))
    defaults = PROVIDER_DEFAULTS[provider_name]

    api_key = _read_setting(settings_obj, defaults["api_key_field"], "")
    base_url = _read_setting(settings_obj, defaults["base_url_field"], defaults["base_url"])
    model = _read_setting(settings_obj, defaults["model_field"], defaults["model"])

    # 通用 LLM_* 配置优先，但只在它与所选供应商一致时生效，
    # 避免临时切换到 qwen 时误用 deepseek 的通用密钥。
    configured_provider = normalize_provider(
        str(_read_setting(settings_obj, "LLM_PROVIDER", "deepseek"))
    )
    if provider_name == configured_provider:
        api_key = _read_setting(settings_obj, "LLM_API_KEY", "") or api_key
        base_url = _read_setting(settings_obj, "LLM_BASE_URL", "") or base_url
        model = _read_setting(settings_obj, "LLM_MODEL", "") or model

    # 兼容部分项目把通义千问写成 DASHSCOPE_* 的命名
    if provider_name == "qwen" and not api_key:
        api_key = _read_setting(settings_obj, "DASHSCOPE_API_KEY", "")
    if provider_name == "qwen" and not base_url:
        base_url = _read_setting(settings_obj, "DASHSCOPE_BASE_URL", defaults["base_url"])

    if not api_key:
        raise LLMConfigError(
            f"未配置 {provider_name} 的 API Key（{defaults['api_key_field']}），"
            "请先在 .env 中设置。",
            provider=provider_name,
            model=str(model),
        )

    return {
        "provider": provider_name,
        "api_key": str(api_key).strip(),
        "base_url": str(base_url or defaults["base_url"]).strip(),
        "model": str(model or defaults["model"]).strip(),
    }


def _build_endpoint(base_url: str) -> str:
    """把 base_url 拼接为 ``/chat/completions`` 完整地址，避免重复路径。"""
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def _resolve_option(settings_obj: Any, name: str, default: Any) -> Any:
    """读取可选的调用参数（温度、最大 token、重试次数等）。"""
    value = _read_setting(settings_obj, name, default)
    try:
        return type(default)(value)
    except (TypeError, ValueError):  # pragma: no cover - 配置值非法时使用默认值
        return default


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------
def get_active_provider() -> str:
    """返回当前配置的供应商名称（deepseek/qwen），用于日志与响应元数据。"""
    settings_obj = _load_settings()
    raw_provider = _read_setting(settings_obj, "LLM_PROVIDER", "deepseek")
    return normalize_provider(str(raw_provider))


def get_active_model_name(provider: Optional[str] = None) -> str:
    """返回当前配置的模型名；未配置密钥时也能返回模型名，便于降级信息展示。"""
    try:
        return str(_resolve_provider_config(provider)["model"])
    except LLMConfigError:
        settings_obj = _load_settings()
        provider_name = normalize_provider(
            str(provider or _read_setting(settings_obj, "LLM_PROVIDER", "deepseek"))
        )
        defaults = PROVIDER_DEFAULTS[provider_name]
        generic_model = _read_setting(settings_obj, "LLM_MODEL", "")
        return str(generic_model or _read_setting(settings_obj, defaults["model_field"], defaults["model"]))


def _validate_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """校验并规范化消息列表，防止空消息或非法角色进入请求体。"""
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise ValueError("messages 必须是 list[dict]，每个元素包含 role 和 content。")
    if not messages:
        raise ValueError("messages 不能为空。")

    cleaned: list[dict[str, str]] = []
    allowed_roles = {"system", "user", "assistant", "tool"}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] 必须是 dict。")
        role = str(message.get("role", "")).strip()
        content = message.get("content")
        if role not in allowed_roles:
            raise ValueError(f"messages[{index}] 的 role 非法：{role!r}。")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}] 的 content 不能为空。")
        cleaned.append({"role": role, "content": content})
    return cleaned


def _extract_content(payload: Any) -> str:
    """从 OpenAI 兼容响应中取出首条消息文本。"""
    if not isinstance(payload, dict):
        raise LLMUpstreamError("LLM 响应不是合法 JSON 对象。")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMUpstreamError("LLM 响应缺少 choices 字段。")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        # 少数推理模型只返回 reasoning_content，作为兜底
        reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
        if isinstance(reasoning, str) and reasoning.strip():
            content = reasoning
        else:
            raise LLMUpstreamError("LLM 返回了空内容。")
    return content.strip()


def _should_retry_status(status_code: int) -> bool:
    """判断 HTTP 状态码是否值得重试。"""
    return status_code in (408, 409, 425, 429) or status_code >= 500


def chat(
    messages: Sequence[dict[str, Any]],
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    provider: Optional[str] = None,
    response_format: Optional[dict[str, Any]] = None,
) -> str:
    """调用大模型并返回纯文本答案。

    Args:
        messages: OpenAI 风格消息列表，例如
            ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``。
        temperature: 采样温度，默认取配置 ``LLM_TEMPERATURE``（0.2）。
        max_tokens: 最大输出 token 数，默认取配置 ``LLM_MAX_TOKENS``（1500）。
        timeout: 单次请求超时秒数，默认 60 秒。
        provider: 临时指定供应商（deepseek/qwen），默认取配置 ``LLM_PROVIDER``。
        response_format: 可选的 OpenAI ``response_format`` 参数，例如
            ``{"type": "json_object"}``；默认不发送。

    Returns:
        模型返回的文本内容（已去除首尾空白）。

    Raises:
        ValueError: messages 结构非法。
        LLMConfigError: 配置缺失、密钥为空或供应商不支持。
        LLMTimeoutError: 多次尝试后仍超时。
        LLMUpstreamError: 供应商返回错误状态或响应无法解析。
        LLMError: 其他调用失败（所有重试耗尽）。
    """
    if requests is None:  # pragma: no cover
        raise LLMConfigError("缺少 requests 依赖，无法调用大模型接口。")

    normalized_messages = _validate_messages(messages)
    settings_obj = _load_settings()
    config = _resolve_provider_config(provider)

    request_timeout = float(
        timeout
        if timeout is not None
        else _read_setting(settings_obj, "LLM_TIMEOUT", DEFAULT_TIMEOUT)
    )
    if request_timeout <= 0:
        request_timeout = DEFAULT_TIMEOUT

    if temperature is None:
        temperature = float(
            _read_setting(settings_obj, "LLM_TEMPERATURE", DEFAULT_TEMPERATURE)
        )
    if max_tokens is None:
        max_tokens = int(_read_setting(settings_obj, "LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    if max_tokens <= 0:
        max_tokens = DEFAULT_MAX_TOKENS

    max_retries = int(
        _read_setting(settings_obj, "LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES)
    )
    max_retries = max(0, min(max_retries, 5))

    payload: dict[str, Any] = {
        "model": config["model"],
        "messages": normalized_messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    url = _build_endpoint(config["base_url"])
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    attempts = max_retries + 1
    last_error: Optional[LLMError] = None

    for attempt in range(attempts):
        started = time.perf_counter()
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=(CONNECT_TIMEOUT, request_timeout),
            )
        except requests.Timeout as exc:
            last_error = LLMTimeoutError(
                f"调用 {config['provider']} 超时（{request_timeout:.0f} 秒）。",
                provider=config["provider"],
                model=config["model"],
                retryable=True,
                cause=exc,
            )
        except requests.RequestException as exc:
            last_error = LLMError(
                f"调用 {config['provider']} 网络失败：{type(exc).__name__}。",
                provider=config["provider"],
                model=config["model"],
                retryable=True,
                cause=exc,
            )
        else:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            status_code = int(response.status_code)

            if status_code == 200:
                try:
                    content = _extract_content(response.json())
                except ValueError as exc:  # JSON 解析失败
                    last_error = LLMUpstreamError(
                        "LLM 响应不是合法 JSON。",
                        provider=config["provider"],
                        model=config["model"],
                        status_code=status_code,
                        retryable=True,
                        cause=exc,
                    )
                except LLMError as exc:
                    exc.provider = exc.provider or config["provider"]
                    exc.model = exc.model or config["model"]
                    exc.status_code = status_code
                    exc.retryable = True
                    last_error = exc
                else:
                    logger.info(
                        "LLM 调用成功 provider=%s model=%s elapsed_ms=%d attempts=%d",
                        config["provider"],
                        config["model"],
                        elapsed_ms,
                        attempt + 1,
                    )
                    return content

            elif status_code in (401, 403):
                # 密钥错误不重试，直接抛出配置类异常
                raise LLMConfigError(
                    f"{config['provider']} 鉴权失败（HTTP {status_code}），请检查 API Key。",
                    provider=config["provider"],
                    model=config["model"],
                    status_code=status_code,
                )

            elif status_code == 400:
                detail = response.text[:300] if response.text else ""
                raise LLMUpstreamError(
                    f"{config['provider']} 拒绝了请求（HTTP 400）：{detail}",
                    provider=config["provider"],
                    model=config["model"],
                    status_code=status_code,
                )

            else:
                detail = response.text[:300] if response.text else ""
                last_error = LLMUpstreamError(
                    f"{config['provider']} 返回 HTTP {status_code}：{detail}",
                    provider=config["provider"],
                    model=config["model"],
                    status_code=status_code,
                    retryable=_should_retry_status(status_code),
                )
                if not _should_retry_status(status_code):
                    raise last_error

        # 重试退避：1 秒、2 秒；最后一次失败后不再等待
        if attempt < attempts - 1:
            backoff = min(2 ** attempt, 4)
            logger.warning(
                "LLM 调用失败，%s 秒后重试（%d/%d）：%s",
                backoff,
                attempt + 2,
                attempts,
                last_error,
            )
            time.sleep(backoff)

    if last_error is None:  # pragma: no cover - 逻辑兜底
        last_error = LLMError(
            "LLM 调用失败，原因未知。",
            provider=config["provider"],
            model=config["model"],
        )
    raise last_error


__all__ = [
    "LLMError",
    "LLMConfigError",
    "LLMTimeoutError",
    "LLMUpstreamError",
    "chat",
    "get_active_provider",
    "get_active_model_name",
    "normalize_provider",
]
