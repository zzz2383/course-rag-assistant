"""练习题目生成接口。

对外提供 ``POST /api/exercises``：根据课程（可选章节）从已索引资料中生成
填空题与选择题，响应体与 :class:`app.agents.exerciser.ExerciseSet` 一致::

    {
        "course": "计算机网络",
        "chapter": "第三章 传输层",
        "items": [
            {"type": "blank", "stem": "TCP 通过____建立连接。", "answer": "三次握手",
             "hint": null, "source_file": "第3章.pdf", "page": 12},
            {"type": "choice", "stem": "……____……", "options": ["A", "B", "C", "D"],
             "answer_index": 0, "explanation": "……", "source_file": "第3章.pdf", "page": 12}
        ],
        "generated_by": "llm" | "rule",
        "warnings": []
    }

``generated_by=rule`` 表示大模型不可用或输出不合法，已自动降级为规则出题；
没有资料时返回空 ``items``，具体原因写在 ``warnings`` 中，HTTP 状态仍为 200。
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.common import APIError, invoke

logger = logging.getLogger("course_rag.api.exercises")

router = APIRouter(prefix="/exercises", tags=["练习题目"])

# 与 exerciser.QUESTION_TYPES 保持一致，避免导入 agent 层影响应用启动。
_ALLOWED_TYPES = ("blank", "choice")


class ExerciseRequest(BaseModel):
    """出题请求体。"""

    model_config = ConfigDict(extra="forbid")

    course: str = Field(min_length=1, max_length=120, description="课程名称")
    chapter: str | None = Field(
        default=None, max_length=200, description="可选章节；为空时跨章节抽取"
    )
    count: int = Field(default=5, ge=1, le=20, description="目标题目数量")
    types: list[str] = Field(
        default_factory=lambda: ["blank", "choice"],
        description="题型列表，可选 blank / choice",
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium", description="难度"
    )
    use_llm: bool = Field(default=True, description="是否优先使用大模型出题")

    @field_validator("course")
    @classmethod
    def _strip_course(cls, value: str) -> str:
        """课程名称去空白并拒绝纯空白。"""
        value = value.strip()
        if not value:
            raise ValueError("课程名称不能为空")
        return value

    @field_validator("chapter")
    @classmethod
    def _strip_chapter(cls, value: str | None) -> str | None:
        """可选章节去空白，空串转为 None。"""
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("types")
    @classmethod
    def _clean_types(cls, values: list[str]) -> list[str]:
        """题型小写归一化、去重，并拒绝未知题型。"""
        cleaned: list[str] = []
        for value in values or []:
            text = str(value or "").strip().lower()
            if text and text not in _ALLOWED_TYPES:
                raise ValueError(f"不支持的题型：{value!r}，可选 blank 或 choice")
            if text and text not in cleaned:
                cleaned.append(text)
        if not cleaned:
            raise ValueError("types 至少包含 blank 或 choice 中的一种")
        return cleaned


def _normalize_result(raw: Any) -> dict[str, Any]:
    """把出题结果转换为接口约定的响应字典。"""
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="json")
    if not isinstance(raw, dict):
        raise APIError(500, "EXERCISE_GENERATION_FAILED", "出题器返回了无法识别的结果")

    items = raw.get("items")
    if not isinstance(items, list):
        raise APIError(500, "EXERCISE_GENERATION_FAILED", "出题结果缺少题目列表")

    generated_by = "llm" if str(raw.get("generated_by") or "").lower() == "llm" else "rule"
    warnings = [str(item) for item in raw.get("warnings") or [] if str(item).strip()]
    return {
        "course": str(raw.get("course") or ""),
        "chapter": raw.get("chapter"),
        "items": items,
        "generated_by": generated_by,
        "warnings": warnings,
    }


@router.post("", summary="生成练习题")
async def create_exercises(request: Request, payload: ExerciseRequest) -> dict[str, Any]:
    """根据课程资料生成填空题/选择题。

    Args:
        payload: 包含 ``course``、可选 ``chapter`` / ``count`` / ``types`` /
            ``difficulty`` / ``use_llm`` 的请求体。

    Returns:
        ``ExerciseSet`` 结构的字典，每道题都带 ``source_file`` 与 ``page``。

    Raises:
        APIError: 参数不可行（422）或生成流程异常（500）。
    """
    # 延迟导入出题器，避免未配置 LLM 时影响应用启动。
    try:
        from app.agents import exerciser
    except Exception as exc:  # noqa: BLE001 - 依赖缺失时给出统一错误
        logger.exception("练习题模块加载失败")
        raise APIError(
            500,
            "EXERCISE_GENERATION_FAILED",
            "练习题服务未正确初始化",
            detail={"exception": type(exc).__name__},
        ) from exc

    generate = getattr(exerciser, "generate_exercises", None)
    if not callable(generate):
        raise APIError(500, "EXERCISE_GENERATION_FAILED", "练习题服务未正确初始化")

    try:
        result = await invoke(
            generate,
            {
                "course": payload.course,
                "chapter": payload.chapter,
                "count": payload.count,
                "types": payload.types,
                "difficulty": payload.difficulty,
                "use_llm": payload.use_llm,
            },
        )
    except APIError:
        raise
    except ValueError as exc:
        # 课程为空、数量/题型非法等参数问题
        raise APIError(422, "EXERCISE_VALIDATION_FAILED", str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - 对外只暴露通用错误提示
        logger.exception("练习题生成失败 course=%s", payload.course)
        raise APIError(
            500,
            "EXERCISE_GENERATION_FAILED",
            "练习题生成失败，请稍后重试",
            detail={"exception": type(exc).__name__},
        ) from exc

    normalized = _normalize_result(result)
    logger.info(
        "练习题生成完成 request_id=%s course=%s items=%d generated_by=%s",
        getattr(request.state, "request_id", "-"),
        normalized["course"],
        len(normalized["items"]),
        normalized["generated_by"],
    )
    return normalized
