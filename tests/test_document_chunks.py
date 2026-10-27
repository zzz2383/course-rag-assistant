"""``GET /api/documents/{doc_id}/chunks`` 的离线接口测试。

测试不访问真实 Chroma、不读取真实磁盘文件：
``app.api.documents._get_vectorstore`` 与 ``_read_meta`` 都用 monkeypatch 替换，
因此可以在任意机器上稳定运行。

覆盖点：
1. 正常多页文档按页码升序分组；
2. ``page=0`` 的 txt 场景（无页码不是错误，必须保留）；
3. 文档存在但向量库里没有分块（200 + 空 pages）；
4. 文档完全不存在（404 + DOC_NOT_FOUND）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# 允许直接执行 `python -m pytest tests/test_document_chunks.py`（项目根目录未安装为包）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.api import documents  # noqa: E402
from app.main import app  # noqa: E402

CLIENT = TestClient(app)


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------
def _chunk(
    chunk_id: str,
    text: str,
    *,
    page: int,
    chapter: str = "第三章 传输层",
    source_file: str = "计算机网络-第3章-传输层.pdf",
) -> dict[str, Any]:
    """构造与 VectorStore.get_document_chunks 返回值一致的分块字典。"""
    metadata = {
        "chunk_id": chunk_id,
        "page": page,
        "chapter": chapter,
        "source_file": source_file,
        "doc_id": "abc123",
    }
    return {
        "chunk_id": chunk_id,
        "text": text,
        "page": page,
        "chapter": chapter,
        "metadata": metadata,
    }


class _FakeStore:
    """只实现被测路由用到的 get_document_chunks。"""

    def __init__(self, chunks: list[dict[str, Any]] | None = None, *, broken: bool = False) -> None:
        self._chunks = chunks or []
        self._broken = broken
        self.calls = 0

    def get_document_chunks(self, doc_id: str) -> list[dict[str, Any]]:
        self.calls += 1
        if self._broken:
            raise RuntimeError("模拟向量库不可用")
        return list(self._chunks)


def _install(monkeypatch: pytest.MonkeyPatch, store: _FakeStore, meta: dict[str, Any] | None) -> None:
    """打桩向量库与元数据读取。"""
    monkeypatch.setattr(documents, "_get_vectorstore", lambda: store)
    monkeypatch.setattr(documents, "_read_meta", lambda doc_id: meta)


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------
def test_chunks_grouped_by_page_in_ascending_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """正常路径：12 个块分布在 3 页，pages 按页码升序且组内保持原顺序。"""
    store = _FakeStore(
        [
            _chunk("c1", "第 0 页 第一段", page=1),
            _chunk("c2", "第 0 页 第二段", page=1),
            _chunk("c3", "第 1 页 第一段", page=2),
            _chunk("c4", "第 2 页 第一段", page=3),
        ]
    )
    _install(
        monkeypatch,
        store,
        {"doc_id": "abc123", "file_name": "计算机网络-第3章-传输层.pdf", "chunk_count": 4},
    )

    response = CLIENT.get("/api/documents/abc123/chunks")

    assert response.status_code == 200
    payload = response.json()
    assert payload["doc_id"] == "abc123"
    assert payload["file_name"] == "计算机网络-第3章-传输层.pdf"
    assert payload["chunk_count"] == 4
    assert payload["page_count"] == 3
    assert [page["page"] for page in payload["pages"]] == [1, 2, 3]

    first_page = payload["pages"][0]
    assert [item["chunk_id"] for item in first_page["chunks"]] == ["c1", "c2"]
    assert set(first_page["chunks"][0]) == {"chunk_id", "text", "chapter"}
    assert first_page["chunks"][0]["text"] == "第 0 页 第一段"
    assert first_page["chunks"][0]["chapter"] == "第三章 传输层"
    # 页码升序：即使 Chroma 返回乱序，也要按 page 重排
    assert [page["page"] for page in payload["pages"]] == sorted(
        page["page"] for page in payload["pages"]
    )


def test_txt_without_page_keeps_page_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """txt 无页码：page=0 必须保留，且 page_count 为 1 而不是 0。"""
    store = _FakeStore(
        [
            _chunk(
                "c1",
                "传输层提供端到端的通信服务。",
                page=0,
                source_file="计算机网络-第3章-传输层.txt",
            ),
            _chunk(
                "c2",
                "TCP 通过三次握手建立连接。",
                page=0,
                source_file="计算机网络-第3章-传输层.txt",
            ),
        ]
    )
    _install(
        monkeypatch,
        store,
        {"doc_id": "abc123", "file_name": "计算机网络-第3章-传输层.txt"},
    )

    response = CLIENT.get("/api/documents/abc123/chunks")

    assert response.status_code == 200
    payload = response.json()
    assert payload["file_name"] == "计算机网络-第3章-传输层.txt"
    assert payload["chunk_count"] == 2
    assert payload["page_count"] == 1
    assert [page["page"] for page in payload["pages"]] == [0]
    assert [item["chunk_id"] for item in payload["pages"][0]["chunks"]] == ["c1", "c2"]


def test_existing_document_without_chunks_returns_empty_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """meta 存在但向量库里查不到块：200，pages 为空。"""
    store = _FakeStore([])
    _install(
        monkeypatch,
        store,
        {"doc_id": "abc123", "file_name": "计算机网络-第3章-传输层.txt", "chunk_count": 0},
    )

    response = CLIENT.get("/api/documents/abc123/chunks")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "doc_id": "abc123",
        "file_name": "计算机网络-第3章-传输层.txt",
        "chunk_count": 0,
        "page_count": 0,
        "pages": [],
    }


def test_unknown_document_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """meta 与向量库都没有该文档：404 + DOC_NOT_FOUND。"""
    store = _FakeStore([])
    _install(monkeypatch, store, None)

    response = CLIENT.get("/api/documents/abc123/chunks")

    assert response.status_code == 404
    payload = response.json()
    assert payload["code"] == "DOC_NOT_FOUND"
    assert payload["message"]
    assert payload["request_id"]


def test_vectorstore_failure_degrades_to_empty_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """向量库抛出异常时不返回 500：meta 存在即为 200 + 空分块。"""
    store = _FakeStore([], broken=True)
    _install(
        monkeypatch,
        store,
        {"doc_id": "abc123", "file_name": "计算机网络-第3章-传输层.txt"},
    )

    response = CLIENT.get("/api/documents/abc123/chunks")

    assert response.status_code == 200
    payload = response.json()
    assert store.calls == 1
    assert payload["chunk_count"] == 0
    assert payload["page_count"] == 0
    assert payload["pages"] == []
