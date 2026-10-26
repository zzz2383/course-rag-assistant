"""复习计划生成模块。

对外入口 :func:`generate_plan`，按"先规则、后 LLM"的两条路径生成逐日复习计划：

1. **规则模板（保底）**：不依赖任何外部服务，按剩余天数均分章节，
   薄弱章节加权靠前并适当重复，最后 1～2 天安排模拟测试与错题回顾；
2. **LLM 优化（优先）**：把章节、薄弱点、日期和每日时长交给大模型，
   要求返回严格 JSON；解析失败、字段缺失或超时后自动回退规则模板，
   并在 ``warnings`` 中说明降级原因。

返回值与 :class:`app.models.schemas.PlanResponse` 完全对齐，
可直接执行 ``PlanResponse(**plan)`` 或作为 FastAPI 响应体返回。
"""

from __future__ import annotations

import json
import logging
import math
import re
import uuid
from datetime import date as Date
from datetime import datetime, timedelta
from typing import Any, Optional, Sequence

from app.models.schemas import PlanResponse
from app.rag import llm, prompts

logger = logging.getLogger(__name__)

DEFAULT_DAILY_MINUTES = 90
MIN_DAILY_MINUTES = 15
MAX_DAILY_MINUTES = 480
MAX_PLAN_DAYS = 180

# PlanActivity.type 允许的取值
_ACTIVITY_TYPES = {"read", "practice", "review", "summary", "test"}


# ---------------------------------------------------------------------------
# 日期与文本工具
# ---------------------------------------------------------------------------
def _to_date(value: Any, field: str = "date") -> Date:
    """把 ``date``/``datetime``/ISO 字符串转换为 ``date``。"""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, Date):
        return value
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} 不能为空。")
    try:
        return Date.fromisoformat(text[:10])
    except ValueError as exc:
        raise ValueError(f"{field} 格式非法：{value!r}，应为 YYYY-MM-DD。") from exc


def _compact_name(value: Any) -> str:
    """去掉空白和常见标点，用于章节名模糊匹配。"""
    return re.sub(r"[\s\-_、,，.。:：;；()（）\[\]【】]+", "", str(value or "")).lower()


def _match_chapter(candidate: Any, chapter_names: Sequence[str]) -> Optional[str]:
    """把薄弱点/LLM 返回的章节名匹配到真实章节。

    先精确匹配（忽略空白标点），再尝试包含关系，最后取最长匹配项，
    避免把"动态规划"误判成其他章节。
    """
    target = _compact_name(candidate)
    if not target:
        return None

    for name in chapter_names:
        if _compact_name(name) == target:
            return name

    fuzzy = [
        name
        for name in chapter_names
        if target in _compact_name(name) or _compact_name(name) in target
    ]
    if fuzzy:
        return max(fuzzy, key=lambda name: len(_compact_name(name)))
    return None


def _normalize_chapters(chapters: Sequence[Any]) -> tuple[list[str], dict[str, int]]:
    """规范化章节列表，兼容 ``list[str]`` 与带统计信息的 ``list[dict]``。

    Returns:
        ``(chapter_names, chapter_stats)``；``chapter_stats`` 记录每章片段/页数。
    """
    if isinstance(chapters, (str, bytes)):
        chapters = [chapters]

    names: list[str] = []
    stats: dict[str, int] = {}
    seen: set[str] = set()

    for entry in chapters or []:
        name = ""
        count = 0
        if isinstance(entry, str):
            name = entry.strip()
        elif isinstance(entry, dict):
            for key in ("chapter", "name", "title"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    name = value.strip()
                    break
            for key in ("chunk_count", "count", "size", "page_count"):
                value = entry.get(key)
                if isinstance(value, (int, float)):
                    count = max(int(value), 0)
                    break
        else:
            name = str(entry or "").strip()

        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
        if count:
            stats[name] = count

    if not names:
        raise ValueError("chapters 不能为空，请先上传课程资料或确认章节列表。")
    return names, stats


def _normalize_weak_points(weak_points: Optional[Sequence[Any]]) -> list[str]:
    """规范化薄弱点列表，兼容字符串与 ``{"chapter": ...}`` 形式。"""
    if isinstance(weak_points, (str, bytes)):
        weak_points = [weak_points]
    result: list[str] = []
    seen: set[str] = set()
    for entry in weak_points or []:
        if isinstance(entry, dict):
            value = entry.get("chapter") or entry.get("name") or entry.get("title") or ""
        else:
            value = entry
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _is_weak_chapter(chapter: str, weak_points: Sequence[str]) -> bool:
    """判断章节是否命中任一薄弱点（忽略空白标点、支持包含关系）。"""
    chapter_key = _compact_name(chapter)
    for point in weak_points:
        point_key = _compact_name(point)
        if point_key and (point_key in chapter_key or chapter_key in point_key):
            return True
    return False


# ---------------------------------------------------------------------------
# 日期与章节分配
# ---------------------------------------------------------------------------
def _resolve_dates(
    exam_date: Any,
    start_date: Any,
    days: Optional[int],
    warnings: list[str],
) -> tuple[Date, Date, int]:
    """计算开始日期、考试日期和计划总天数。"""
    exam = _to_date(exam_date, "exam_date")
    start = _to_date(start_date, "start_date") if start_date else Date.today()
    if exam <= start:
        raise ValueError("exam_date 必须晚于 start_date（或今天）。")

    available = (exam - start).days
    total = int(days) if days is not None else available
    if total < 1:
        raise ValueError("计划天数必须至少为 1 天。")
    if total > available:
        warnings.append(f"计划天数 {total} 超过考试前可用天数 {available}，已按 {available} 天生成。")
        total = available
    if total > MAX_PLAN_DAYS:
        warnings.append(f"计划天数超过 {MAX_PLAN_DAYS} 天，已截断为 {MAX_PLAN_DAYS} 天。")
        total = MAX_PLAN_DAYS
    return start, exam, total


def _order_chapters(
    chapter_names: Sequence[str],
    weak_points: Sequence[str],
    chapter_stats: dict[str, int],
) -> list[str]:
    """按薄弱程度和资料量排序，薄弱章节优先。"""
    max_count = max(chapter_stats.values(), default=0) or 1
    scored: list[tuple[float, int, str]] = []
    for index, name in enumerate(chapter_names):
        weight = 2.0 if _is_weak_chapter(name, weak_points) else 1.0
        if chapter_stats:
            weight += 0.3 * (chapter_stats.get(name, 0) / max_count)
        scored.append((weight, index, name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, _, name in scored]


def _group_chapters(total_days: int, ordered: Sequence[str]) -> list[list[str]]:
    """天数少于章节数时，把章节尽量均匀地合并到每天。"""
    groups: list[list[str]] = []
    base = len(ordered) // total_days
    extra = len(ordered) % total_days
    cursor = 0
    for index in range(total_days):
        size = base + (1 if index < extra else 0)
        groups.append(list(ordered[cursor : cursor + size]))
        cursor += size
    return groups


def _chapter_sequence(total_days: int, ordered: Sequence[str]) -> list[str]:
    """天数不少于章节数时，按轮次重复章节，薄弱章节在靠前轮次优先出现。"""
    quotas = {name: 1 for name in ordered}
    remaining = total_days - len(ordered)
    cursor = 0
    while remaining > 0:
        name = ordered[cursor % len(ordered)]
        quotas[name] += 1
        remaining -= 1
        cursor += 1

    sequence: list[str] = []
    while len(sequence) < total_days:
        progressed = False
        for name in ordered:
            if len(sequence) >= total_days:
                break
            if quotas.get(name, 0) > 0:
                sequence.append(name)
                quotas[name] -= 1
                progressed = True
        if not progressed:  # pragma: no cover - 理论上不会发生
            break
    return sequence


def _phase_for(index: int, total_days: int) -> str:
    """按进度划分基础巩固/强化练习/冲刺复盘三阶段。"""
    if total_days <= 4:
        split = max(1, math.ceil(total_days * 0.6))
        return "基础巩固" if index < split else "冲刺复盘"
    ratio = (index + 1) / total_days
    if ratio <= 0.5:
        return "基础巩固"
    if ratio <= 0.8:
        return "强化练习"
    return "冲刺复盘"


def _split_minutes(total_minutes: int, count: int) -> list[int]:
    """把一天的总时长拆成 count 份，保证总和恰好等于总时长。"""
    total_minutes = max(int(total_minutes), 0)
    if count <= 0:
        return []
    if count == 1:
        return [total_minutes]

    ratio_map = {
        2: (0.6, 0.4),
        3: (0.5, 0.3, 0.2),
        4: (0.4, 0.3, 0.2, 0.1),
    }
    ratios = ratio_map.get(count)
    if ratios is None:
        base = total_minutes // count
        parts = [base] * (count - 1)
        return parts + [total_minutes - sum(parts)]

    parts = [int(round(total_minutes * ratio)) for ratio in ratios[:-1]]
    parts.append(total_minutes - sum(parts))
    if parts[-1] < 0:  # 极端取整情况下回退为均分
        parts = [total_minutes // count] * (count - 1)
        parts.append(total_minutes - sum(parts))
    return parts


def _activity_types(count: int, is_final: bool) -> list[str]:
    """按活动数量与阶段选择活动类型。"""
    if is_final:
        base = ["test", "review", "summary"]
    else:
        base = ["read", "practice", "review", "summary"]
    return [base[index % len(base)] for index in range(count)]


def _rule_titles(chapter_label: str, is_weak: bool, is_final: bool) -> list[str]:
    """规则模板的任务文案（过长的合并章节名做截断，满足 schema 长度限制）。"""
    label = chapter_label if len(chapter_label) <= 80 else chapter_label[:80] + "……"
    if is_final:
        return [
            f"完成一套覆盖《{label}》的限时模拟测试",
            "对照答案订正错题，记录错误原因与正确解法",
            "回顾薄弱章节与易错公式，整理考前速查清单",
        ]
    prefix = "重点补强：" if is_weak else ""
    return [
        f"{prefix}阅读并梳理《{label}》的核心概念、定义与公式",
        f"完成《{label}》的典型例题或课后练习 2-3 道",
        f"合上资料复述《{label}》要点，整理 3 条笔记",
    ]


def _build_rule_day(
    *,
    index: int,
    total_days: int,
    start_date: Date,
    chapters: Sequence[str],
    weak_points: Sequence[str],
    daily_minutes: int,
) -> dict[str, Any]:
    """构造规则模板中的一天。"""
    chapter_label = "、".join(chapters)
    is_weak = any(_is_weak_chapter(name, weak_points) for name in chapters)
    is_final = index >= total_days - 2
    titles = _rule_titles(chapter_label, is_weak, is_final)
    minutes_list = _split_minutes(daily_minutes, len(titles))
    types = _activity_types(len(titles), is_final)

    if is_final:
        objective = "完成综合模拟，定位并订正错题，形成考前速查清单"
        checkpoint = "模拟测试完成并订正全部错题，能说出 3 条考前易错点"
    elif is_weak:
        objective = f"补强薄弱章节《{chapter_label}》，能独立解释核心概念并完成典型题"
        checkpoint = f"不看资料写出《{chapter_label}》的 3 个核心结论"
    else:
        objective = f"掌握《{chapter_label}》的核心概念与典型题型"
        checkpoint = f"能用自己的话复述《{chapter_label}》的 3 个要点"

    return {
        "date": start_date + timedelta(days=index),
        "day_index": index + 1,
        "phase": _phase_for(index, total_days),
        "chapter": chapter_label,
        "objective": objective,
        "minutes": daily_minutes,
        "activities": [
            {
                "type": types[position],
                "title": title[:300],
                "minutes": minutes_list[position],
            }
            for position, title in enumerate(titles)
        ],
        "resources": [],
        "checkpoint": checkpoint,
    }


def _build_rule_days(
    *,
    total_days: int,
    start_date: Date,
    ordered_chapters: Sequence[str],
    weak_points: Sequence[str],
    daily_minutes: int,
) -> list[dict[str, Any]]:
    """规则模板：按剩余天数均分章节，薄弱章节靠前。"""
    if total_days < len(ordered_chapters):
        groups = _group_chapters(total_days, ordered_chapters)
    else:
        sequence = _chapter_sequence(total_days, ordered_chapters)
        groups = [[name] for name in sequence]

    return [
        _build_rule_day(
            index=index,
            total_days=total_days,
            start_date=start_date,
            chapters=group,
            weak_points=weak_points,
            daily_minutes=daily_minutes,
        )
        for index, group in enumerate(groups)
    ]


# ---------------------------------------------------------------------------
# 结果组装与校验
# ---------------------------------------------------------------------------
def _new_plan_id() -> str:
    """生成计划 ID，例如 ``plan_20261026_201833_ab12``。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"plan_{stamp}_{uuid.uuid4().hex[:4]}"


def _compose_plan(
    *,
    course: str,
    start_date: Date,
    exam_date: Date,
    total_days: int,
    daily_minutes: int,
    generated_by: str,
    summary: str,
    tasks: Sequence[dict[str, Any]],
    warnings: Sequence[str],
) -> dict[str, Any]:
    """组装并校验 PlanResponse，返回可直接 JSON 序列化的字典。"""
    payload = {
        "plan_id": _new_plan_id(),
        "course": course,
        "start_date": start_date,
        "exam_date": exam_date,
        "total_days": total_days,
        "daily_minutes": daily_minutes,
        "generated_by": generated_by,
        "summary": summary,
        "tasks": list(tasks),
        "warnings": list(dict.fromkeys(warnings)),
    }
    response = PlanResponse.model_validate(payload)
    return response.model_dump(mode="json")


def _default_summary(
    total_days: int,
    weak_chapters: Sequence[str],
    daily_minutes: int,
) -> str:
    """规则计划的摘要文案。"""
    weak_text = "、".join(weak_chapters) if weak_chapters else "无"
    if total_days <= 4:
        return (
            f"共 {total_days} 天、每天约 {daily_minutes} 分钟，按基础巩固与冲刺复盘两阶段安排；"
            f"薄弱章节（{weak_text}）优先，最后一天安排模拟测试与错题回顾。"
        )
    return (
        f"共 {total_days} 天、每天约 {daily_minutes} 分钟，按基础巩固、强化练习、冲刺复盘"
        f"三阶段安排；薄弱章节（{weak_text}）优先并适当重复，最后两天安排模拟测试与错题回顾。"
    )


# ---------------------------------------------------------------------------
# LLM 计划路径
# ---------------------------------------------------------------------------
def _strip_code_fence(text: str) -> str:
    """去掉模型偶尔输出的 Markdown 代码围栏。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _parse_json_payload(raw: str) -> Any:
    """尽可能从 LLM 文本中解析 JSON 对象或数组。"""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("LLM 返回了空文本。")
    text = _strip_code_fence(raw)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("LLM 返回的内容不是合法 JSON。")


def _extract_days_payload(data: Any) -> list[Any]:
    """从 LLM JSON 中取出逐日计划数组。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("days", "tasks", "plan", "schedule", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("LLM JSON 缺少 days/tasks 数组。")


def _coerce_task_titles(item: dict[str, Any]) -> list[str]:
    """从 LLM 返回的一天中提取 2～4 条任务文案。"""
    titles: list[str] = []

    raw_tasks = item.get("tasks")
    if isinstance(raw_tasks, list):
        for task in raw_tasks:
            if isinstance(task, str) and task.strip():
                titles.append(task.strip())
            elif isinstance(task, dict):
                value = task.get("title") or task.get("task") or task.get("content")
                if isinstance(value, str) and value.strip():
                    titles.append(value.strip())

    if not titles and isinstance(item.get("activities"), list):
        for activity in item["activities"]:
            if isinstance(activity, dict):
                value = activity.get("title") or activity.get("task")
                if isinstance(value, str) and value.strip():
                    titles.append(value.strip())

    # 去重并限制数量
    unique: list[str] = []
    for title in titles:
        if title not in unique:
            unique.append(title)
    if len(unique) == 1:
        unique.append("复盘本节内容，完成 3 道自测题并记录易错点")
    return unique[:4]


def _coerce_resources(item: dict[str, Any]) -> list[dict[str, Any]]:
    """校验 LLM 返回的资料关联，丢弃缺少文件名的项。"""
    resources: list[dict[str, Any]] = []
    raw_resources = item.get("resources")
    if not isinstance(raw_resources, list):
        return resources
    for resource in raw_resources:
        if not isinstance(resource, dict):
            continue
        source_file = resource.get("source_file") or resource.get("file_name")
        if not isinstance(source_file, str) or not source_file.strip():
            continue
        try:
            page = int(resource.get("page", 0) or 0)
        except (TypeError, ValueError):
            page = 0
        resources.append(
            {
                "source_file": source_file.strip()[:255],
                "page": max(page, 0),
                "chunk_id": str(resource.get("chunk_id") or "")[:64],
            }
        )
    return resources


def _normalize_llm_days(
    data: Any,
    *,
    total_days: int,
    start_date: Date,
    ordered_chapters: Sequence[str],
    weak_points: Sequence[str],
    daily_minutes: int,
    rule_days: Sequence[dict[str, Any]],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], str]:
    """把 LLM 返回的逐日计划修复为可校验的 PlanTask 列表。"""
    payload = _extract_days_payload(data)
    if not payload:
        raise ValueError("LLM 返回的逐日计划为空。")
    if len(payload) > total_days:
        warnings.append(f"LLM 返回 {len(payload)} 天计划，超过预期 {total_days} 天，已截断。")
    elif len(payload) < total_days:
        warnings.append(f"LLM 返回 {len(payload)} 天计划，不足 {total_days} 天，已用规则模板补齐。")

    date_mismatch = False
    chapter_mismatch = False
    over_budget = False
    days: list[dict[str, Any]] = []

    for index in range(total_days):
        expected_date = start_date + timedelta(days=index)
        fallback = rule_days[index]
        raw_item = payload[index] if index < len(payload) else None
        item = raw_item if isinstance(raw_item, dict) else {}
        if not item:
            days.append(fallback)
            continue

        # 日期以服务端计算的连续日期为准，模型日期仅作参考
        raw_date = item.get("date")
        if raw_date:
            try:
                if _to_date(raw_date) != expected_date:
                    date_mismatch = True
            except ValueError:
                date_mismatch = True

        candidate_chapter = item.get("chapter") or item.get("focus") or ""
        chapter = _match_chapter(candidate_chapter, ordered_chapters)
        if chapter is None:
            if str(candidate_chapter).strip():
                chapter_mismatch = True
            chapter = str(fallback["chapter"])

        titles = _coerce_task_titles(item)
        if not titles:
            chapter_mismatch = True
            titles = [activity["title"] for activity in fallback["activities"]]

        try:
            minutes = int(item.get("minutes") or daily_minutes)
        except (TypeError, ValueError):
            minutes = daily_minutes
        if minutes <= 0:
            minutes = daily_minutes
        if minutes > daily_minutes:
            over_budget = True
            minutes = daily_minutes

        is_final = index >= total_days - 2
        minutes_list = _split_minutes(minutes, len(titles))
        types = _activity_types(len(titles), is_final)
        activities = [
            {
                "type": types[position],
                "title": title[:300],
                "minutes": minutes_list[position],
            }
            for position, title in enumerate(titles)
        ]

        phase = str(item.get("phase") or _phase_for(index, total_days))[:50]
        objective = str(item.get("objective") or fallback["objective"])[:500]
        checkpoint = str(item.get("checkpoint") or fallback["checkpoint"])[:500]
        resources = _coerce_resources(item)

        days.append(
            {
                "date": expected_date,
                "day_index": index + 1,
                "phase": phase,
                "chapter": chapter[:200],
                "objective": objective,
                "minutes": minutes,
                "activities": activities,
                "resources": resources,
                "checkpoint": checkpoint,
            }
        )

    if date_mismatch:
        warnings.append("LLM 返回的日期与连续日期不一致，已按开始日期重新排布。")
    if chapter_mismatch:
        warnings.append("LLM 返回了无法匹配的章节或空任务，已用规则模板修正。")
    if over_budget:
        warnings.append(f"LLM 有单日任务超过 {daily_minutes} 分钟，已压缩到每日上限。")

    summary = ""
    if isinstance(data, dict):
        summary = str(data.get("summary") or "").strip()[:500]
    if not summary:
        summary = _default_summary(total_days, weak_points, daily_minutes)
    return days, summary


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------
def _merge_chapter_stats(
    stats: dict[str, int],
    chapter_stats: Optional[Any],
) -> dict[str, int]:
    """把外部传入的章节统计合并进 ``stats``，兼容 dict 与列表两种形式。"""
    if not chapter_stats:
        return stats
    if isinstance(chapter_stats, dict):
        items = list(chapter_stats.items())
        for name, value in items:
            try:
                stats.setdefault(str(name), max(int(value), 0))
            except (TypeError, ValueError):
                continue
        return stats

    if isinstance(chapter_stats, Sequence) and not isinstance(chapter_stats, (str, bytes)):
        try:
            _, extra_stats = _normalize_chapters(chapter_stats)
        except ValueError:
            return stats
        for name, count in extra_stats.items():
            stats.setdefault(name, count)
    return stats


def generate_plan(
    exam_date: Any,
    chapters: Sequence[Any],
    weak_points: Optional[Sequence[Any]] = None,
    days: Optional[int] = None,
    *,
    start_date: Any = None,
    daily_minutes: int = DEFAULT_DAILY_MINUTES,
    course: str = "",
    chapter_stats: Optional[dict[str, Any]] = None,
    use_llm: bool = True,
    weak_chapters: Optional[Sequence[Any]] = None,
) -> dict[str, Any]:
    """生成符合 :class:`PlanResponse` 的复习计划。

    Args:
        exam_date: 考试日期（``date`` 或 ``YYYY-MM-DD``）。
        chapters: 章节列表，元素可以是字符串，也可以是
            ``{"chapter": ..., "chunk_count": ...}`` 形式的统计条目。
        weak_points: 薄弱章节/知识点列表，支持与章节名模糊匹配。
        days: 计划天数；省略时按"开始日期到考试前一日"的自然天数计算。
        start_date: 计划开始日期，默认今天。
        daily_minutes: 每日可用学习分钟数，默认 90。
        course: 课程名称，写入响应。
        chapter_stats: 章节片段数映射，用于给资料量大的章节略微加权。
        use_llm: 是否优先尝试 LLM；``False`` 时直接走规则模板。
        weak_chapters: ``weak_points`` 的别名，兼容 PlanRequest.weak_chapters 命名。

    Returns:
        与 PlanResponse 字段一致的字典，``generated_by`` 为 ``llm`` 或 ``rule``。

    Raises:
        ValueError: 日期非法、章节为空、每日时长越界等参数问题。
    """
    warnings: list[str] = []
    course_text = (course or "当前课程").strip()[:120] or "当前课程"

    # 兼容 PlanRequest 中的 weak_chapters 命名
    if weak_chapters:
        merged_points = list(weak_points or [])
        merged_points.extend(item for item in weak_chapters if item not in merged_points)
        weak_points = merged_points

    if not isinstance(daily_minutes, int):
        try:
            daily_minutes = int(daily_minutes)
        except (TypeError, ValueError) as exc:
            raise ValueError("daily_minutes 必须是整数。") from exc
    if not MIN_DAILY_MINUTES <= daily_minutes <= MAX_DAILY_MINUTES:
        raise ValueError(
            f"daily_minutes 必须在 {MIN_DAILY_MINUTES}～{MAX_DAILY_MINUTES} 分钟之间。"
        )

    start, exam, total_days = _resolve_dates(exam_date, start_date, days, warnings)
    chapter_names, stats = _normalize_chapters(chapters)
    stats = _merge_chapter_stats(stats, chapter_stats)

    weak_list = _normalize_weak_points(weak_points)
    matched_weak: list[str] = []
    for point in weak_list:
        matched = _match_chapter(point, chapter_names)
        if matched is None:
            warnings.append(f"薄弱点「{point}」未匹配到课程章节，已在计划中忽略。")
        elif matched not in matched_weak:
            matched_weak.append(matched)

    ordered_chapters = _order_chapters(chapter_names, weak_list, stats)
    rule_days = _build_rule_days(
        total_days=total_days,
        start_date=start,
        ordered_chapters=ordered_chapters,
        weak_points=weak_list,
        daily_minutes=daily_minutes,
    )
    summary = _default_summary(total_days, matched_weak, daily_minutes)

    # 1) 优先尝试 LLM
    if use_llm:
        try:
            messages = prompts.build_plan_messages(
                exam_date=exam.isoformat(),
                start_date=start.isoformat(),
                chapters=chapter_names,
                weak_points=weak_list,
                days=total_days,
                daily_minutes=daily_minutes,
                course=course,
                chapter_stats=stats,
            )
            raw = llm.chat(messages, temperature=0.3)
            data = _parse_json_payload(raw)
            llm_days, llm_summary = _normalize_llm_days(
                data,
                total_days=total_days,
                start_date=start,
                ordered_chapters=ordered_chapters,
                weak_points=matched_weak,
                daily_minutes=daily_minutes,
                rule_days=rule_days,
                warnings=warnings,
            )
            return _compose_plan(
                course=course_text,
                start_date=start,
                exam_date=exam,
                total_days=total_days,
                daily_minutes=daily_minutes,
                generated_by="llm",
                summary=llm_summary,
                tasks=llm_days,
                warnings=warnings,
            )
        except Exception as exc:  # LLM 网络、配额、JSON 或校验失败均降级
            reason = f"{type(exc).__name__}: {str(exc)}"[:180]
            logger.warning("LLM 计划生成失败，使用规则模板：%s", reason)
            warnings.append(f"LLM 计划生成失败，已使用规则模板（{reason}）。")

    # 2) 规则模板保底
    return _compose_plan(
        course=course_text,
        start_date=start,
        exam_date=exam,
        total_days=total_days,
        daily_minutes=daily_minutes,
        generated_by="rule",
        summary=summary,
        tasks=rule_days,
        warnings=warnings,
    )


__all__ = ["generate_plan"]
