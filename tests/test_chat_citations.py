"""``app.api.chat._citation_to_dict`` 的离线单元测试。

引用规范化是“答案可追溯”的最后一道关口：链上已经带上了 doc_id 与
chunk_id，接口层若把它们丢掉，前端就无法把引用跳转到原文阅读器。
本测试固定住这一契约：

1. 保留 source_file / page / excerpt 三个展示字段；
2. 同时保留 doc_id / chunk_id / locator，供「在原文中查看」定位；
3. 缺少文件名的引用被丢弃，避免前端渲染空卡片。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.chat import _citation_to_dict  # noqa: E402


def test_citation_keeps_locator_fields() -> None:
    citation = _citation_to_dict(
        {
            "index": 2,
            "chunk_id": "ck_0001",
            "doc_id": "doc_123e143c09b24f46",
            "source_file": "计算机网络-第3章-传输层.pdf",
            "page": 4,
            "locator": "page:4",
            "excerpt": "TCP 建立连接需要三次握手",
        }
    )

    assert citation is not None
    assert citation["source_file"] == "计算机网络-第3章-传输层.pdf"
    assert citation["page"] == 4
    assert citation["excerpt"]
    assert citation["doc_id"] == "doc_123e143c09b24f46"
    assert citation["chunk_id"] == "ck_0001"
    assert citation["locator"] == "page:4"
    assert citation["index"] == 2


def test_citation_reads_nested_metadata() -> None:
    citation = _citation_to_dict(
        {
            "text": "慢启动阶段拥塞窗口按指数增长",
            "metadata": {
                "source_file": "计算机网络-第3章-传输层.pdf",
                "page": 7,
                "doc_id": "doc_abc",
                "chunk_id": "ck_0009",
            },
        }
    )

    assert citation is not None
    assert citation["page"] == 7
    assert citation["doc_id"] == "doc_abc"
    assert citation["chunk_id"] == "ck_0009"


def test_citation_without_file_is_dropped() -> None:
    assert _citation_to_dict({"page": 3, "excerpt": "无来源片段"}) is None
    assert _citation_to_dict(None) is None
