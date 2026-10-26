"""练习题目生成模块的离线单元测试。

测试不访问网络、不依赖真实大模型与向量库：

* ``VectorStore`` 用内存假对象替换；
* ``llm.chat`` 用 monkeypatch 替换为固定文本或抛错函数。

覆盖点：
1. LLM 返回合法 JSON 数组时能解析为填空题/选择题，且保留来源；
2. LLM 返回非法 JSON 或超时时自动降级到规则出题；
3. 规则填空题正确挖空、答案取自原文、来源可回查；
4. 规则选择题干扰项不含正确答案，且 4 个选项互不相同；
5. 没有课程资料时返回空集并给出 warning。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 允许直接执行 `python -m pytest tests/test_exerciser.py`（项目根目录未安装为包）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from app.agents import exerciser  # noqa: E402
from app.agents.exerciser import (  # noqa: E402
    BlankQuestion,
    ChoiceQuestion,
    ExerciseSet,
)
from app.rag.llm import LLMTimeoutError  # noqa: E402
from app.rag.vectorstore import RetrievedChunk  # noqa: E402

SOURCE_FILE = "计算机网络-第3章-传输层.txt"
CHAPTER = "第三章 传输层"

CHUNK_ONE_TEXT = (
    "传输层为两台主机的应用进程提供端到端的逻辑通信服务。"
    "TCP 协议通过三次握手建立可靠连接，并使用滑动窗口进行流量控制。"
    "端口号用于标识主机上的不同应用进程。"
)
CHUNK_TWO_TEXT = (
    "UDP 协议不保证可靠传输，适用于实时音视频等对时延敏感的场景。"
    "拥塞控制通过慢启动、拥塞避免、快重传与快恢复四个阶段调整发送速率。"
)


# ---------------------------------------------------------------------------
# 测试数据与假对象
# ---------------------------------------------------------------------------
def _chunk(
    text: str,
    *,
    source_file: str = SOURCE_FILE,
    page: int = 3,
    chapter: str = CHAPTER,
    chunk_id: str = "ck_1",
) -> RetrievedChunk:
    """构造一条与 vectorstore.query 返回结构一致的检索结果。"""
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={
            "source_file": source_file,
            "page": page,
            "chapter": chapter,
            "chunk_id": chunk_id,
            "doc_id": f"doc_{chunk_id}",
        },
        score=0.9,
    )


def _rows() -> list[RetrievedChunk]:
    """两页、两个分块的测试资料。"""
    return [
        _chunk(CHUNK_ONE_TEXT, page=3, chunk_id="ck_1"),
        _chunk(CHUNK_TWO_TEXT, page=4, chunk_id="ck_2"),
    ]


def _patch_store(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[RetrievedChunk],
    documents: list[dict] | None = None,
) -> None:
    """用假 VectorStore 替换出题器里的实现，按章节与 top_k 模拟过滤。"""

    class _FakeStore:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003 - 兼容真实构造签名
            pass

        def query(
            self,
            query_text=None,
            *,
            course=None,
            chapter=None,
            top_k=None,
            **kwargs,
        ):  # noqa: ANN001, ANN003
            selected = list(rows)
            if chapter:
                selected = [
                    row
                    for row in selected
                    if str(row.metadata.get("chapter") or "") == str(chapter)
                ]
            if top_k is not None:
                selected = selected[: int(top_k)]
            return selected

        def list_documents(self, course=None):  # noqa: ANN001
            return list(documents or [])

    monkeypatch.setattr(exerciser, "VectorStore", _FakeStore)


def _source_texts(rows: list[RetrievedChunk]) -> dict[tuple[str, int], str]:
    """构造 (来源文件, 页码) -> 原文 的映射，用于核对题目来源。"""
    return {(row.source_file, row.page): row.text for row in rows}


# ---------------------------------------------------------------------------
# LLM 路径
# ---------------------------------------------------------------------------
def test_llm_json_is_parsed_into_exercises(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM 返回合法 JSON 数组时，解析出填空题与选择题并保留来源。"""
    rows = _rows()
    _patch_store(monkeypatch, rows)

    payload = [
        {
            "type": "blank",
            "stem": "TCP 协议通过____建立可靠连接。",
            "answer": "三次握手",
            "hint": "共 4 个字",
            "source_file": SOURCE_FILE,
            "page": 3,
        },
        {
            "type": "choice",
            "stem": "传输层为应用进程提供____的逻辑通信服务。",
            "options": ["端到端", "点到点", "广播", "组播"],
            "answer_index": 0,
            "explanation": "原文：传输层为两台主机的应用进程提供端到端的逻辑通信服务。",
            "source_file": SOURCE_FILE,
            "page": 4,
        },
    ]

    captured: dict = {}

    def fake_chat(messages, **kwargs):  # noqa: ANN001, ANN003
        captured["messages"] = messages
        return json.dumps(payload, ensure_ascii=False)

    monkeypatch.setattr(exerciser.llm, "chat", fake_chat)

    result = exerciser.generate_exercises(
        "计算机网络",
        chapter=CHAPTER,
        count=2,
        types=["blank", "choice"],
        difficulty="medium",
        use_llm=True,
    )

    assert isinstance(result, ExerciseSet)
    assert result.generated_by == "llm"
    assert result.course == "计算机网络"
    assert result.chapter == CHAPTER
    assert len(result.items) == 2

    blank, choice = result.items
    assert isinstance(blank, BlankQuestion)
    assert blank.type == "blank"
    assert blank.stem == "TCP 协议通过____建立可靠连接。"
    assert blank.answer == "三次握手"
    assert blank.hint == "共 4 个字"
    assert blank.source_file == SOURCE_FILE
    assert blank.page == 3

    assert isinstance(choice, ChoiceQuestion)
    assert choice.type == "choice"
    assert len(choice.options) == 4
    assert len(set(choice.options)) == 4
    assert choice.options[choice.answer_index] == "端到端"
    assert choice.explanation and "端到端" in choice.explanation
    assert choice.source_file == SOURCE_FILE
    assert choice.page == 4

    # 提示词中必须带上原文与来源，要求按 JSON 输出
    prompt_text = "\n".join(message["content"] for message in captured["messages"])
    assert "TCP 协议通过三次握手" in prompt_text
    assert SOURCE_FILE in prompt_text
    assert "JSON" in prompt_text


def test_llm_invalid_json_falls_back_to_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM 返回非法 JSON 时降级到规则出题，并给出降级提示。"""
    rows = _rows()
    _patch_store(monkeypatch, rows)
    monkeypatch.setattr(
        exerciser.llm,
        "chat",
        lambda messages, **kwargs: "抱歉，我无法按要求输出 JSON。",  # noqa: ARG005
    )

    result = exerciser.generate_exercises(
        "计算机网络",
        chapter=CHAPTER,
        count=3,
        types=["blank", "choice"],
        difficulty="medium",
        use_llm=True,
    )

    assert result.generated_by == "rule"
    assert len(result.items) == 3
    assert any("LLM" in warning for warning in result.warnings)
    for item in result.items:
        assert item.source_file
        assert item.page >= 0


def test_llm_timeout_falls_back_to_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM 超时同样降级到规则出题，保证无 Key/断网时仍能出题。"""
    _patch_store(monkeypatch, _rows())

    def broken_chat(messages, **kwargs):  # noqa: ANN001, ANN003
        raise LLMTimeoutError("模拟 LLM 超时")

    monkeypatch.setattr(exerciser.llm, "chat", broken_chat)

    result = exerciser.generate_exercises(
        "计算机网络",
        chapter=CHAPTER,
        count=2,
        types=["blank"],
        use_llm=True,
    )

    assert result.generated_by == "rule"
    assert len(result.items) == 2
    assert all(isinstance(item, BlankQuestion) for item in result.items)


# ---------------------------------------------------------------------------
# 规则路径
# ---------------------------------------------------------------------------
def test_rule_blank_question_has_correct_answer_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """规则填空题必须把原文术语挖成 ____，答案能在来源分块中找到。"""
    rows = _rows()
    _patch_store(monkeypatch, rows)
    monkeypatch.setattr(
        exerciser.llm,
        "chat",
        lambda *args, **kwargs: pytest.fail("use_llm=False 时不应调用大模型"),
    )

    result = exerciser.generate_exercises(
        "计算机网络",
        chapter=CHAPTER,
        count=4,
        types=["blank"],
        difficulty="medium",
        use_llm=False,
    )

    assert result.generated_by == "rule"
    assert len(result.items) == 4
    assert not result.warnings

    source_texts = _source_texts(rows)
    for item in result.items:
        assert isinstance(item, BlankQuestion)
        assert item.type == "blank"
        assert "____" in item.stem
        assert item.answer
        assert item.answer not in item.stem
        assert item.source_file == SOURCE_FILE
        assert (item.source_file, item.page) in source_texts
        # 答案必须来自原文，便于回原文核对
        assert item.answer in source_texts[(item.source_file, item.page)]


def test_rule_choice_options_are_unique_and_exclude_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """规则选择题：4 个选项互不相同，干扰项不包含正确答案。"""
    rows = _rows()
    _patch_store(monkeypatch, rows)

    result = exerciser.generate_exercises(
        "计算机网络",
        chapter=CHAPTER,
        count=4,
        types=["choice"],
        difficulty="medium",
        use_llm=False,
    )

    assert result.generated_by == "rule"
    assert len(result.items) == 4
    source_texts = _source_texts(rows)

    for item in result.items:
        assert isinstance(item, ChoiceQuestion)
        assert item.type == "choice"
        assert len(item.options) == 4
        assert len(set(item.options)) == 4
        assert 0 <= item.answer_index < 4

        answer = item.options[item.answer_index]
        assert answer
        assert answer in source_texts[(item.source_file, item.page)]
        assert answer in item.explanation

        for index, option in enumerate(item.options):
            if index == item.answer_index:
                continue
            # 干扰项不能等于正确答案，也不能包含正确答案
            assert option != answer
            assert answer not in option
            assert option not in answer


# ---------------------------------------------------------------------------
# 无资料
# ---------------------------------------------------------------------------
def test_without_material_returns_empty_set_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """向量库没有资料时返回空集并说明原因，不抛异常。"""
    _patch_store(monkeypatch, [], [])

    result = exerciser.generate_exercises(
        "计算机网络",
        count=3,
        types=["blank", "choice"],
        use_llm=False,
    )

    assert isinstance(result, ExerciseSet)
    assert result.items == []
    assert result.generated_by == "rule"
    assert result.warnings
    assert any("资料" in warning for warning in result.warnings)

    # 序列化后 items 仍为空数组而不是 null，符合接口契约
    dumped = result.model_dump(mode="json")
    assert dumped["items"] == []
    assert dumped["warnings"]
