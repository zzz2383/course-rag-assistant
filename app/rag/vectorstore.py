"""Chroma 向量库封装模块。

对 ``chromadb.PersistentClient`` 做轻量封装，向上提供：

* :meth:`VectorStore.add_chunks`：向量化并幂等写入 Chunk（``upsert``）。
* :meth:`VectorStore.query`：按 ``course`` / ``chapter`` metadata 过滤后检索。
* :meth:`VectorStore.delete_document`：按 ``doc_id`` 删除某文档的全部向量。
* :meth:`VectorStore.list_documents`：聚合向量库中的文档清单。

chromadb 采用懒加载导入，未安装依赖时仅在实际访问向量库时抛出明确错误。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from app.core.config import Settings, get_settings
from app.models.schemas import ChunkRecord

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


@dataclass
class RetrievedChunk:
    """一条召回结果。"""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None = None
    score: float = 0.0

    @property
    def source_file(self) -> str:
        """原始文件名。"""
        return str(self.metadata.get("source_file") or "")

    @property
    def page(self) -> int:
        """页码/幻灯片号，无页码为 0。"""
        try:
            return int(self.metadata.get("page") or 0)
        except (TypeError, ValueError):
            return 0


def _default_embedder(texts: Sequence[str]) -> list[list[float]]:
    """默认向量化实现：延迟导入 embedder，避免模块级循环依赖。"""
    from app.services.embedder import embed_texts

    return embed_texts(list(texts))


class VectorStore:
    """Chroma collection 封装。

    Args:
        settings: 配置对象，默认使用全局配置。
        embedder: 向量化函数，接收文本列表并返回等长向量列表；默认使用
            :func:`app.services.embedder.embed_texts`。
        collection_name: 覆盖 collection 名称，默认取 ``settings.chroma_collection``。
        persist_directory: 覆盖持久化目录，默认取 ``settings.chroma_dir``。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: EmbedFn | None = None,
        collection_name: str | None = None,
        persist_directory: str | Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.collection_name = collection_name or self.settings.chroma_collection
        self.persist_directory = Path(persist_directory or self.settings.chroma_dir)
        self._embedder: EmbedFn = embedder or _default_embedder
        self._client: Any | None = None
        self._collection: Any | None = None

    # ------------------------------------------------------------- 连接管理
    def _get_client(self) -> Any:
        """懒加载 Chroma 持久化客户端。"""
        if self._client is None:
            try:
                import chromadb
            except ImportError as exc:  # pragma: no cover - 依赖缺失时的友好提示
                raise RuntimeError("向量库功能需要安装 chromadb：pip install chromadb") from exc

            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        return self._client

    def _get_collection(self) -> Any:
        """懒加载并缓存 collection 句柄。"""
        if self._collection is None:
            client = self._get_client()
            metadata = {
                "hnsw:space": "cosine",
                "embedding_model": self.settings.active_embedding_model,
                "embedding_mode": self.settings.embedding_mode,
                "embedding_dim": int(self.settings.embedding_dim),
                "schema_version": "1.0",
            }
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata=metadata,
            )
        return self._collection

    def ensure_collection(self) -> None:
        """确保 collection 已创建（应用启动或健康检查时调用）。"""
        self._get_collection()

    def count(self) -> int:
        """返回 collection 中的 Chunk 总数。"""
        return int(self._get_collection().count())

    # ------------------------------------------------------------- 写入
    def add_chunks(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[Sequence[float]] | None = None,
        *,
        rebuild: bool = False,
    ) -> int:
        """批量写入 Chunk（按 ``chunk_id`` 幂等 upsert）。

        Args:
            chunks: 待写入的 Chunk 列表。
            vectors: 预计算向量；为空时调用 embedder 生成。
            rebuild: 为 ``True`` 时先删除这些 Chunk 所属 ``doc_id`` 的旧向量，
                避免重建索引残留旧块。

        Returns:
            实际写入的 Chunk 数量。

        Raises:
            ValueError: 输入为空或向量数量与 Chunk 数量不一致。
            RuntimeError: Chroma 未安装或写入失败。
        """
        records = list(chunks)
        if not records:
            return 0

        if vectors is None:
            computed = self._embedder([record.text for record in records])
        else:
            computed = vectors
        if len(computed) != len(records):
            raise ValueError(
                f"向量数量（{len(computed)}）与 Chunk 数量（{len(records)}）不一致"
            )

        if rebuild:
            doc_ids = {record.metadata.doc_id for record in records if record.metadata.doc_id}
            for doc_id in doc_ids:
                self.delete_document(doc_id)

        created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        model_name = self.settings.active_embedding_model
        metadatas = [
            record.to_chroma_metadata(
                embedding_model=model_name,
                created_at=created_at,
            )
            for record in records
        ]
        embeddings = [[float(value) for value in vector] for vector in computed]

        try:
            self._get_collection().upsert(
                ids=[record.chunk_id for record in records],
                documents=[record.text for record in records],
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:  # noqa: BLE001 - Chroma 异常类型不统一
            raise RuntimeError(f"写入 Chroma 失败：{exc}") from exc
        return len(records)

    # ------------------------------------------------------------- 查询
    @staticmethod
    def _build_where(
        course: str | None = None,
        chapter: str | None = None,
        extra_where: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """组合课程/章节过滤条件与额外 where 子句。"""
        conditions: list[dict[str, Any]] = []
        if course and course.strip():
            conditions.append({"course": {"$eq": course.strip()}})
        if chapter and chapter.strip():
            conditions.append({"chapter": {"$eq": chapter.strip()}})
        if extra_where:
            if set(extra_where.keys()) == {"$and"} and isinstance(extra_where["$and"], list):
                conditions.extend(extra_where["$and"])
            else:
                conditions.append(extra_where)

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def query(
        self,
        query_text: str | Sequence[float] | None = None,
        *,
        query_embedding: Sequence[float] | None = None,
        course: str | None = None,
        chapter: str | None = None,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """按课程/章节过滤后做向量检索。

        Args:
            query_text: 查询文本；未提供 ``query_embedding`` 时用于现场向量化。
                为兼容其他封装，也接受直接传入查询向量（此时按向量检索）。
            query_embedding: 已计算好的查询向量，优先级高于 ``query_text``。
            course: 课程过滤条件（推荐必传）。
            chapter: 章节过滤条件（可选）。
            top_k: 召回数量，默认取 ``settings.top_k``。
            where: 额外的 Chroma where 子句，会与课程/章节条件合并。

        Returns:
            按相似度从高到低排列的召回结果，``score = 1 - cosine_distance``。

        Raises:
            ValueError: 缺少查询向量/文本，或 top_k 非法。
            RuntimeError: Chroma 查询失败。
        """
        if query_embedding is None and query_text is not None and not isinstance(query_text, str):
            query_embedding = query_text
            query_text = None

        if query_embedding is None:
            if not query_text or not str(query_text).strip():
                raise ValueError("query_text 与 query_embedding 至少提供一个")
            vectors = self._embedder([str(query_text).strip()])
            if not vectors:
                raise ValueError("查询文本向量化失败")
            query_embedding = vectors[0]
        if not query_embedding:
            raise ValueError("查询向量不能为空")

        k = int(top_k) if top_k is not None else self.settings.top_k
        if k <= 0:
            raise ValueError("top_k 必须大于 0")

        where_clause = self._build_where(course, chapter, where)
        kwargs: dict[str, Any] = {
            "query_embeddings": [[float(value) for value in query_embedding]],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_clause:
            kwargs["where"] = where_clause

        try:
            result = self._get_collection().query(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Chroma 查询失败：{exc}") from exc

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for position, chunk_id in enumerate(ids):
            text = documents[position] if position < len(documents) else ""
            metadata = dict(metadatas[position] or {}) if position < len(metadatas) else {}
            distance: float | None = None
            if position < len(distances) and distances[position] is not None:
                distance = float(distances[position])
            score = 1.0 - distance if distance is not None else 0.0
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    text=str(text or ""),
                    metadata=metadata,
                    distance=distance,
                    score=score,
                )
            )
        return retrieved

    # ------------------------------------------------------------- 删除与清单
    def delete_document(self, doc_id: str) -> int:
        """删除指定文档的全部向量。

        Args:
            doc_id: 文档 ID。

        Returns:
            删除的 Chunk 数量。

        Raises:
            ValueError: ``doc_id`` 为空。
            RuntimeError: Chroma 读写失败。
        """
        document_id = (doc_id or "").strip()
        if not document_id:
            raise ValueError("doc_id 不能为空")

        collection = self._get_collection()
        where = {"doc_id": {"$eq": document_id}}
        try:
            existing = collection.get(where=where, include=["metadatas"])
            ids = list(existing.get("ids") or [])
            if ids:
                collection.delete(ids=ids)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"删除文档向量失败（doc_id={document_id}）：{exc}") from exc
        return len(ids)

    def list_documents(self, course: str | None = None) -> list[dict[str, Any]]:
        """按 ``doc_id`` 聚合列出向量库中的文档。

        Args:
            course: 可选课程过滤条件。

        Returns:
            文档摘要列表，每项含 ``doc_id``、``course``、``chapter``、
            ``source_file``、``file_type``、``page_count``、``chunk_count``。
        """
        get_kwargs: dict[str, Any] = {"include": ["metadatas"]}
        if course and course.strip():
            get_kwargs["where"] = {"course": {"$eq": course.strip()}}

        try:
            data = self._get_collection().get(**get_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"读取向量库文档清单失败：{exc}") from exc

        summaries: dict[str, dict[str, Any]] = {}
        for metadata in data.get("metadatas") or []:
            if not metadata:
                continue
            doc_id = str(metadata.get("doc_id") or metadata.get("source_file") or "unknown")
            summary = summaries.get(doc_id)
            if summary is None:
                summary = {
                    "doc_id": doc_id,
                    "course": str(metadata.get("course") or ""),
                    "chapter": str(metadata.get("chapter") or ""),
                    "source_file": str(metadata.get("source_file") or ""),
                    "file_type": str(metadata.get("file_type") or ""),
                    "page_count": 0,
                    "chunk_count": 0,
                }
                summaries[doc_id] = summary

            summary["chunk_count"] += 1
            try:
                page = int(metadata.get("page") or 0)
            except (TypeError, ValueError):
                page = 0
            if page > summary["page_count"]:
                summary["page_count"] = page
            if not summary["chapter"] and metadata.get("chapter"):
                summary["chapter"] = str(metadata["chapter"])

        return sorted(summaries.values(), key=lambda item: item["source_file"])
