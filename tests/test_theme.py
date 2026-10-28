"""书卷风主题模块的渲染契约测试。

这些函数只拼 HTML 字符串，出问题时页面会静默错位而不是抛异常，
所以用测试固定几条关键规则：文本必须转义、角标必须与脚注同号、
模型输出的轻量 markdown 必须被正确识别。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

import theme  # noqa: E402


def test_esc_blocks_html_injection() -> None:
    """外部文本不能当作标签解析。"""
    assert theme.esc("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"
    assert theme.esc(None) == ""
    assert theme.esc(123) == "123"


def test_answer_paragraphs_replaces_citation_with_footnote_number() -> None:
    """正文角标编号取自 citation_map，与脚注序号一致。"""
    answer = "三次握手确认双方能力。[来源:计算机网络-第3章-传输层.pdf 第4页]"
    html = theme.answer_paragraphs(
        answer, citation_map={"计算机网络-第3章-传输层.pdf|4": 2}
    )
    assert '<sup class="fnref">[2]</sup>' in html
    assert "来源:" not in html


def test_answer_paragraphs_collapses_repeated_marks() -> None:
    """同一句后重复标注同一来源只保留一个角标。"""
    answer = "结论。[来源:a.pdf 第1页][来源:a.pdf 第1页]"
    html = theme.answer_paragraphs(answer, citation_map={"a.pdf|1": 1})
    assert html.count('class="fnref"') == 1


def test_answer_paragraphs_renders_bold_and_bullets() -> None:
    """模型输出的 **加粗** 与 - 列表要变成 HTML，而不是原样露出星号。"""
    answer = "**三次握手的目的**\n\n- 第一次握手：SYN\n- 第二次握手：SYN+ACK"
    html = theme.answer_paragraphs(answer)
    assert "<strong>三次握手的目的</strong>" in html
    assert "**" not in html
    assert html.count('class="a-li"') == 2


def test_answer_paragraphs_escapes_model_text() -> None:
    """模型若输出尖括号，必须被转义。"""
    html = theme.answer_paragraphs("对比 a<b 与 c>d")
    assert "&lt;b" in html and "&gt;d" in html


def test_citation_keys_follow_appearance_order() -> None:
    """来源键按首次出现顺序返回，已去重。"""
    answer = (
        "甲。[来源:b.pdf 第2页] 乙。[来源:a.pdf 第1页] 丙。[来源:b.pdf 第2页]"
    )
    assert theme.citation_keys(answer) == ["b.pdf|2", "a.pdf|1"]


def test_footnotes_escapes_source_and_excerpt() -> None:
    """脚注里的文件名与摘录同样要转义。"""
    html = theme.footnotes(
        [{"index": 1, "source": "<bad>.pdf", "page": "第 1 页", "excerpt": "a & b"}]
    )
    assert "&lt;bad&gt;.pdf" in html
    assert "a &amp; b" in html


def test_mark_block_handles_unanswered() -> None:
    """未作答时用中性判定，不显示「回答不正确」。"""
    html = theme.mark_block(correct=None, correct_answer="16")
    assert "未作答" in html
    assert "回答不正确" not in html
    assert "mark bad" not in html


def test_mark_block_wrong_answer_shows_user_choice() -> None:
    """答错时要同时给出用户答案与正确答案。"""
    html = theme.mark_block(
        correct=False, correct_answer="B．网络层交付给主机", user_answer="A．网络层交付给进程"
    )
    assert "回答不正确" in html
    assert "A．网络层交付给进程" in html
    assert "B．网络层交付给主机" in html


@pytest.mark.parametrize(
    ("raw", "label", "weekday"),
    [
        ("2026-11-01", "11-01", "周日"),
        ("2026/11/05", "11-05", "周四"),
    ],
)
def test_plan_days_parses_date(raw: str, label: str, weekday: str) -> None:
    """日期要拆成「月-日 + 星期」两行大字。"""
    import streamlit_app

    days = streamlit_app._plan_days([{"day": 1, "date": raw, "tasks": ["复习"], "focus": "第四章"}])
    assert days[0]["label"] == label
    assert days[0]["weekday"] == weekday


def test_plan_days_falls_back_to_day_index() -> None:
    """没有日期时退回「第 N 天」，不能渲染成空白。"""
    import streamlit_app

    days = streamlit_app._plan_days([{"day": 2, "tasks": ["复习"]}])
    assert days[0]["label"] == "第2天"


def test_css_covers_mobile_breakpoint() -> None:
    """手机端必须有独立的覆盖规则，否则纸页 40px 内边距会把内容挤成一条。"""
    assert "@media (max-width: 640px)" in theme.CSS
    mobile = theme.CSS.split("@media (max-width: 640px)", 1)[1]
    # 页签横滑、列堆叠、纸页内边距是手机端三个关键点
    assert '[data-testid="stTabs"]' in mobile
    assert '[data-testid="stHorizontalBlock"]' in mobile
    assert "stAppViewBlockContainer" in mobile


def test_book_viewer_degrades_to_single_page_on_narrow() -> None:
    """窄屏下书页要退化成单页，而不是把双页压成两条缝。"""
    import streamlit_app

    html = streamlit_app._BOOK_TEMPLATE
    assert "#stage.single" in html          # 单页模式的样式
    assert "function renderNarrow" in html  # 单页渲染分支
    assert "function stepPage" in html      # 单页逐页翻
    assert "max-width: " in html and "640px" in html  # 断点判定
