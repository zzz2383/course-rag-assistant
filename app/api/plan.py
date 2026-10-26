"""复习计划生成接口。

对外提供 ``POST /api/plan``：把考试日期、章节列表与薄弱点交给规划器
（``app.agents.planner.generate_plan``），生成逐日复习计划。

响应结构按接口约定固定为::

    {
        "plan": [
            {"day": 1, "date": "2026-10-27", "tasks": ["……", "……"], "focus": "第三章 动态规划"}
        ],
        "generated_by": "llm" | "rule"
    }

``generated_by=rule`` 表示大模型不可用或输出不合法，已自动降级为规则模板。
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.api.common import APIError, invoke

logger = logging.getLogger("course_rag.api.plan")

router = APIRouter(prefix="/plan", tags=["复习计划"])

# 没有可用章节时的兜底标签，保证规划器始终有可分配的“章节”。
_FALLBACK_CHAPTER = "课程整体复习"


class PlanRequest(BaseModel):
    """复习计划请求。

    在接口约定的 ``exam_date / chapters / weak_points`` 之外，兼容设计文档
    中的 ``course / start_date / daily_minutes / weak_chapters`` 字段，
    方便前端渐进式接入。
    """

    model_config = ConfigDict(extra="allow")

    exam_date: date = Field(description="考试日期，格式 YYYY-MM-DD")
    chapters: list[str] = Field(default_factory=list, description="需要覆盖的章节列表，可为空")
    weak_points: list[str] = Field(default_factory=list, description="薄弱章节或知识点")
    course: str = Field(default="当前课程", max_length=120, description="课程名称")
    start_date: date | None = Field(default=None, description="计划开始日期，默认今天")
    daily_minutes: int = Field(default=90, ge=15, le=480, description="每日学习分钟数")
    days: int | None = Field(default=None, ge=1, le=180, description="计划天数，默认按日期计算")
    use_llm: bool = Field(default=True, description="是否优先使用大模型优化计划")

    @model_validator(mode="before")
    @classmethod
    def _compat_weak_chapters(cls, data: Any) -> Any:
        """兼容 ``weak_chapters`` 字段名（设计文档口径）。"""
        if isinstance(data, dict) and not data.get("weak_points") and data.get("weak_chapters"):
            data = dict(data)
            data["weak_points"] = data["weak_chapters"]
        return data

    @field_validator("chapters", "weak_points")
    @classmethod
    def _clean_list(cls, values: list[str]) -> list[str]:
        """去除空项与首尾空白，并保持原顺序去重。"""
        cleaned: list[str] = []
        for value in values or []:
            text = str(value or "").strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned


def _auto_chapters(course: str) -> list[str]:
    """从向量库文档清单中尽力推断课程章节（失败时返回空列表）。"""
    try:
        from app.rag.vectorstore import VectorStore

        documents = VectorStore().list_documents(course)
    except Exception:
        logger.warning("自动读取课程章节失败 course=%s", course, exc_info=True)
        return []

    chapters: list[str] = []
    for document in documents or []:
        if not isinstance(document, dict):
            continue
        chapter = str(document.get("chapter") or "").strip()
        if chapter and chapter != "未分章" and chapter not in chapters:
            chapters.append(chapter)
    return chapters


def _resolve_chapters(request: PlanRequest) -> list[str]:
    """确定用于排期的章节列表。

    优先使用请求显式传入的章节；为空时从向量库自动推断；仍为空时用薄弱点
    兜底，最后退化为“课程整体复习”，避免规划器因章节为空而失败。
    """
    if request.chapters:
        return list(request.chapters)

    detected = _auto_chapters(request.course)
    if detected:
        logger.info("请求未指定章节，自动推断出 %d 个章节：%s", len(detected), detected)
        return detected

    if request.weak_points:
        return list(request.weak_points)
    return [_FALLBACK_CHAPTER]


def _iso_date(value: Any) -> str:
    """把 date/datetime/字符串统一转换为 ``YYYY-MM-DD``。"""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value or "").strip()
    return text[:10]


def _task_titles(raw_task: dict[str, Any]) -> list[str]:
    """从一天的规划结果中提取 2～4 条任务文案。

    优先使用 ``tasks`` 字段；规划器内部也可能输出 ``activities``，
    两者都没有时退回 ``objective``，保证前端每天都有可展示的任务。
    """
    titles: list[str] = []

    raw_tasks = raw_task.get("tasks")
    if isinstance(raw_tasks, list):
        for task in raw_tasks:
            if isinstance(task, str) and task.strip():
                titles.append(task.strip())
            elif isinstance(task, dict):
                value = task.get("title") or task.get("task") or task.get("content")
                if isinstance(value, str) and value.strip():
                    titles.append(value.strip())
    elif isinstance(raw_tasks, str) and raw_tasks.strip():
        titles.append(raw_tasks.strip())

    if not titles and isinstance(raw_task.get("activities"), list):
        for activity in raw_task["activities"]:
            if isinstance(activity, dict):
                value = activity.get("title") or activity.get("task")
                if isinstance(value, str) and value.strip():
                    titles.append(value.strip())

    if not titles:
        objective = str(raw_task.get("objective") or "").strip()
        if objective:
            titles.append(objective)

    # 去重后限制最多 4 条，避免前端卡片过长
    unique: list[str] = []
    for title in titles:
        if title not in unique:
            unique.append(title)
    return unique[:4]


def _normalize_plan(raw: Any) -> dict[str, Any]:
    """把规划器返回的 ``PlanResponse`` 转换为接口约定的精简结构。"""
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="json")
    if not isinstance(raw, dict):
        raise APIError(500, "PLAN_GENERATION_FAILED", "规划器返回了无法识别的结果")

    raw_tasks = raw.get("plan")
    if not isinstance(raw_tasks, list):
        raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, list):
        raw_tasks = raw.get("days")
    if not isinstance(raw_tasks, list):
        raise APIError(500, "PLAN_GENERATION_FAILED", "规划结果缺少逐日任务列表")

    plan: list[dict[str, Any]] = []
    for index, item in enumerate(raw_tasks, start=1):
        if not isinstance(item, dict):
            continue
        try:
            day = int(item.get("day") or item.get("day_index") or index)
        except (TypeError, ValueError):
            day = index
        focus = (
            item.get("focus")
            or item.get("chapter")
            or item.get("objective")
            or item.get("phase")
            or ""
        )
        plan.append(
            {
                "day": day,
                "date": _iso_date(item.get("date")),
                "tasks": _task_titles(item),
                "focus": str(focus).strip(),
            }
        )

    if not plan:
        raise APIError(500, "PLAN_GENERATION_FAILED", "规划结果为空，请稍后重试")

    generated_by_raw = str(raw.get("generated_by") or "").strip().lower()
    generated_by = "llm" if generated_by_raw in {"llm", "agent", "model"} else "rule"
    return {"plan": plan, "generated_by": generated_by}


@router.post("", summary="生成复习计划")
async def generate_plan(request: Request, payload: PlanRequest) -> dict[str, Any]:
    """根据考试日期与章节生成逐日复习计划。

    Args:
        payload: 包含 ``exam_date``、``chapters``、``weak_points`` 的请求体。

    Returns:
        ``{"plan": [{day, date, tasks, focus}], "generated_by": "llm"|"rule"}``

    Raises:
        APIError: 日期等参数不可行（422）或生成过程失败（500）。
    """
    # 延迟导入规划器，避免未配置 LLM 时影响应用启动。
    from app.agents import planner

    generate = getattr(planner, "generate_plan", None)
    if not callable(generate):
        raise APIError(500, "PLAN_GENERATION_FAILED", "复习计划服务未正确初始化")

    chapters = _resolve_chapters(payload)

    try:
        result = await invoke(
            generate,
            {
                "exam_date": payload.exam_date,
                "chapters": chapters,
                "weak_points": payload.weak_points,
                "days": payload.days,
                "start_date": payload.start_date,
                "daily_minutes": payload.daily_minutes,
                "course": payload.course,
                "use_llm": payload.use_llm,
            },
        )
    except APIError:
        raise
    except ValueError as exc:
        # 考试日期早于今天、每日时长越界等参数问题
        raise APIError(422, "PLAN_VALIDATION_FAILED", str(exc)) from exc
    except Exception as exc:
        logger.exception("复习计划生成失败 exam_date=%s", payload.exam_date)
        raise APIError(
            500,
            "PLAN_GENERATION_FAILED",
            "复习计划生成失败，请稍后重试",
            detail={"exception": type(exc).__name__},
        ) from exc

    normalized = _normalize_plan(result)
    logger.info(
        "计划生成完成 request_id=%s days=%d generated_by=%s",
        getattr(request.state, "request_id", "-"),
        len(normalized["plan"]),
        normalized["generated_by"],
    )
    return normalized
