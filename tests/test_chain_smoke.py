"""RAG 问答与复习计划的离线冒烟测试。

测试不访问任何外部 API：Embedding 由 ``question_embedding`` 直接注入，
向量库用内存假对象替换，LLM 用 monkeypatch 替换掉 ``llm.chat``。

覆盖点：
1. ``answer_question`` 能返回 answer / citations / used_chunks 结构，
   且引用来自检索元数据而不是模型自由文本；
2. ``answer_question`` 在 LLM 失败时走原文摘录降级路径；
3. ``generate_plan`` 在 LLM 不可用时走规则模板，天数、日期与任务结构正确；
4. 规则模板优先安排薄弱章节，且每日活动时长不超过每日上限。
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

# 允许直接执行 `python -m pytest tests/test_chain_smoke.py`（项目根目录未安装为包）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents import planner  # noqa: E402
from app.models.schemas import ChatResponse, PlanResponse  # noqa: E402
from app.rag import chain  # noqa: E402


# ---------------------------------------------------------------------------
# 测试数据与假对象
# ---------------------------------------------------------------------------
def _chunk(
    text: str,
    *,
    source_file: str,
    page: int,
    chapter: str,
    chunk_id: str,
    score: float = 0.9,
) -> dict:
    """构造一条与 vectorstore.query 返回结构一致的检索结果。"""
    return {
        "text": text,
        "metadata": {
            "source_file": source_file,
            "page": page,
            "chapter": chapter,
            "chunk_id": chunk_id,
            "doc_id": f"doc_{chunk_id}",
        },
        "score": score,
    }


def _fake_vectorstore(rows: list[dict]) -> SimpleNamespace:
    """返回带 query 方法的假向量库模块。"""

    def query(*args, **kwargs):  # noqa: ANN002, ANN003 - 兼容多种调用签名
        return rows

    return SimpleNamespace(query=query)


def _today_iso(offset: int = 0) -> str:
    return (date.today() + timedelta(days=offset)).isoformat()


# ---------------------------------------------------------------------------
# RAG 问答
# ---------------------------------------------------------------------------
def test_answer_question_returns_citations_and_used_chunks(monkeypatch) -> None:
    """正常路径：返回答案、两条引用和检索片段摘要。"""
    rows = [
        _chunk(
            "动态规划的状态转移方程由最优子结构与重叠子问题推导得到。",
            source_file="第3章_动态规划.pdf",
            page=12,
            chapter="第三章 动态规划",
            chunk_id="ck_dp_01",
            score=0.93,
        ),
        _chunk(
            "0-1 背包问题的状态定义为前 i 个物品、容量为 j 时的最大价值。",
            source_file="算法教材.pdf",
            page=96,
            chapter="第三章 动态规划",
            chunk_id="ck_dp_02",
            score=0.81,
        ),
    ]
    monkeypatch.setattr(chain, "_load_vectorstore", lambda: _fake_vectorstore(rows))

    captured: dict = {}

    def fake_chat(messages, **kwargs):
        captured["messages"] = messages
        return (
            "动态规划通过最优子结构推导状态转移方程"
            "[来源:第3章_动态规划.pdf 第12页]。\n"
            "0-1 背包用二维状态表示前 i 个物品与容量 j 的最大价值"
            "[来源:算法教材.pdf 第96页]。"
        )

    monkeypatch.setattr(chain.llm, "chat", fake_chat)

    result = chain.answer_question(
        "动态规划的状态转移方程怎么推导？",
        "算法设计与分析",
        chapter="第三章 动态规划",
        question_embedding=[0.1, 0.2, 0.3],
    )

    assert isinstance(result["answer"], str) and result["answer"]
    assert len(result["used_chunks"]) == 2

    citations = result["citations"]
    assert isinstance(citations, list) and len(citations) == 2
    for citation in citations:
        assert {"source_file", "page", "excerpt"} <= set(citation)
        assert citation["source_file"] in {"第3章_动态规划.pdf", "算法教材.pdf"}
        assert citation["page"] in {12, 96}
        assert citation["excerpt"]

    assert citations[0]["source_file"] == "第3章_动态规划.pdf"
    assert citations[0]["page"] == 12
    assert result["degraded"] is False
    assert result["grounded"] is True

    # 提示词里必须包含检索到的原文和来源标记
    assert captured["messages"]
    prompt_text = "\n".join(message["content"] for message in captured["messages"])
    assert "动态规划的状态转移方程" in prompt_text
    assert "[来源:第3章_动态规划.pdf 第12页]" in prompt_text

    # 返回值去除内部字段后应能通过 ChatResponse 校验
    chat_payload = {
        key: result[key]
        for key in (
            "answer",
            "citations",
            "grounded",
            "degraded",
            "degraded_reason",
            "model",
            "latency_ms",
        )
    }
    filtered = []
    for citation in chat_payload["citations"]:
        filtered.append(
            {
                "index": citation["index"],
                "chunk_id": citation["chunk_id"],
                "source_file": citation["source_file"],
                "page": citation["page"],
                "chapter": citation.get("chapter", ""),
                "locator": citation.get("locator", ""),
                "snippet": citation.get("excerpt", ""),
                "score": citation.get("score") or 0.0,
            }
        )
    chat_payload["citations"] = filtered
    assert ChatResponse.model_validate(chat_payload).answer == result["answer"]


def test_answer_question_degrades_when_llm_unavailable(monkeypatch) -> None:
    """LLM 抛错时降级为原文摘录，仍返回 citations 且标记 degraded。"""
    rows = [
        _chunk(
            "快速排序的平均时间复杂度为 O(n log n)，最坏为 O(n^2)。",
            source_file="第2章_排序.pdf",
            page=7,
            chapter="第二章 排序",
            chunk_id="ck_sort_01",
        )
    ]
    monkeypatch.setattr(chain, "_load_vectorstore", lambda: _fake_vectorstore(rows))

    def broken_chat(messages, **kwargs):  # noqa: ANN002, ANN003
        raise chain.llm.LLMTimeoutError("模拟 LLM 超时")

    monkeypatch.setattr(chain.llm, "chat", broken_chat)

    result = chain.answer_question(
        "快速排序的时间复杂度是多少？",
        "算法设计与分析",
        question_embedding=[0.4, 0.5],
    )

    assert result["degraded"] is True
    assert result["degraded_reason"] == "llm_unavailable"
    assert result["model"] == "none"
    assert "已降级" in result["answer"]
    assert "O(n log n)" in result["answer"]
    assert len(result["citations"]) >= 1
    assert result["citations"][0]["source_file"] == "第2章_排序.pdf"
    assert result["citations"][0]["page"] == 7


def test_answer_question_without_chunks_returns_refusal(monkeypatch) -> None:
    """无检索结果时直接拒答，不调用 LLM。"""
    monkeypatch.setattr(chain, "_load_vectorstore", lambda: _fake_vectorstore([]))

    called = {"count": 0}

    def should_not_call(messages, **kwargs):  # noqa: ANN002, ANN003
        called["count"] += 1
        return "不应被调用"

    monkeypatch.setattr(chain.llm, "chat", should_not_call)

    result = chain.answer_question(
        "什么是最优二叉搜索树？",
        "算法设计与分析",
        question_embedding=[0.9],
    )

    assert called["count"] == 0
    assert result["citations"] == []
    assert result["grounded"] is False
    assert "资料中未提及" in result["answer"]


# ---------------------------------------------------------------------------
# 复习计划
# ---------------------------------------------------------------------------
def test_generate_plan_rule_fallback_has_correct_days(monkeypatch) -> None:
    """LLM 不可用时，规则模板必须生成天数、日期、任务结构正确的计划。"""
    exam = date.today() + timedelta(days=5)
    chapters = ["第一章 绪论", "第二章 排序", "第三章 动态规划", "第四章 图算法"]

    def broken_chat(messages, **kwargs):  # noqa: ANN002, ANN003
        raise planner.llm.LLMError("模拟断网")

    monkeypatch.setattr(planner.llm, "chat", broken_chat)

    plan = planner.generate_plan(
        exam,
        chapters,
        ["第二章"],
        days=5,
        start_date=date.today(),
        daily_minutes=90,
        course="算法设计与分析",
    )

    assert plan["generated_by"] == "rule"
    assert plan["total_days"] == 5
    assert len(plan["tasks"]) == 5
    assert plan["course"] == "算法设计与分析"

    expected_dates = [_today_iso(index) for index in range(5)]
    assert [task["date"] for task in plan["tasks"]] == expected_dates
    assert [task["day_index"] for task in plan["tasks"]] == [1, 2, 3, 4, 5]

    # 薄弱章节加权靠前
    assert plan["tasks"][0]["chapter"] == "第二章 排序"
    assert any("模拟测试" in activity["title"] for activity in plan["tasks"][-1]["activities"])
    assert plan["tasks"][-1]["phase"] == "冲刺复盘"

    for task in plan["tasks"]:
        assert task["activities"], "每天至少有一项活动"
        assert sum(item["minutes"] for item in task["activities"]) == task["minutes"]
        assert task["minutes"] <= 90
        assert task["chapter"] in "、".join(chapters) or any(
            chapter in task["chapter"] for chapter in chapters
        )

    assert any("LLM 计划生成失败" in warning for warning in plan["warnings"])

    # 输出必须能被 PlanResponse 反序列化（与 API 层契约一致）
    parsed = PlanResponse.model_validate(plan)
    assert parsed.total_days == 5
    assert parsed.generated_by == "rule"


def test_generate_plan_defaults_days_to_exam_window() -> None:
    """未显式传 days 时，规则计划天数等于开始日期到考试前一日的天数。"""
    exam = date.today() + timedelta(days=4)
    plan = planner.generate_plan(
        exam,
        ["第一章 绪论", "第二章 排序"],
        ["第二章"],
        start_date=date.today(),
        daily_minutes=60,
        course="算法设计与分析",
        use_llm=False,
    )

    assert plan["generated_by"] == "rule"
    assert plan["total_days"] == 4
    assert len(plan["tasks"]) == 4
    assert plan["tasks"][0]["chapter"] == "第二章 排序"
    assert plan["tasks"][-1]["date"] == (exam - timedelta(days=1)).isoformat()


def test_generate_plan_accepts_llm_json(monkeypatch) -> None:
    """LLM 返回合法 JSON 时使用 llm 路径，并修正连续日期。"""
    exam = date.today() + timedelta(days=3)
    payload = {
        "summary": "三天冲刺计划",
        "days": [
            {
                "date": "2000-01-01",  # 故意错误，验证服务端会修正
                "focus": "第二章 排序",
                "tasks": ["梳理排序算法复杂度", "完成 2 道堆排序练习"],
                "minutes": 80,
                "phase": "基础巩固",
                "objective": "掌握常见排序",
                "checkpoint": "默写快排流程",
            },
            {
                "date": _today_iso(1),
                "focus": "第三章 动态规划",
                "tasks": ["整理状态转移方程", "完成背包例题"],
                "minutes": 80,
            },
            {
                "date": _today_iso(2),
                "focus": "第四章 图算法",
                "tasks": ["复习最短路算法", "完成限时模拟测试"],
                "minutes": 80,
            },
        ],
    }
    monkeypatch.setattr(planner.llm, "chat", lambda messages, **kwargs: json.dumps(payload, ensure_ascii=False))

    plan = planner.generate_plan(
        exam,
        ["第二章 排序", "第三章 动态规划", "第四章 图算法"],
        ["第三章 动态规划"],
        days=3,
        start_date=date.today(),
        daily_minutes=90,
        course="算法设计与分析",
    )

    assert plan["generated_by"] == "llm"
    assert plan["total_days"] == 3
    assert len(plan["tasks"]) == 3
    assert [task["date"] for task in plan["tasks"]] == [
        _today_iso(0),
        _today_iso(1),
        _today_iso(2),
    ]
    assert any("日期" in warning for warning in plan["warnings"])
    assert plan["summary"] == "三天冲刺计划"

    for task in plan["tasks"]:
        assert sum(item["minutes"] for item in task["activities"]) == task["minutes"]
        assert task["minutes"] <= 90
    PlanResponse.model_validate(plan)
