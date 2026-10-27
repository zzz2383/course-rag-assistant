"""Pydantic 数据契约模块。

集中定义跨层复用的请求/响应模型：

* 文档与分块：``DocumentInfo``、``ChunkMetadata``、``ChunkRecord``
* 问答：``ChatRequest``、``ChatResponse``、``Citation``
* 复习计划：``PlanRequest``、``PlanResponse``、``PlanTask``

所有模型均基于 Pydantic v2，可直接用于 FastAPI 的请求校验与响应序列化。
"""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# 章节识别失败时的兜底值，同时作为 Chroma metadata 的必填字符串。
UNASSIGNED_CHAPTER = "未分章"

DocumentStatus = Literal["uploaded", "parsed", "indexed", "index_failed", "failed", "skipped"]


class ChunkMetadata(BaseModel):
    """单个 Chunk 的元数据。

    字段与 Chroma metadata 保持扁平对应，取值仅限 ``str/int/float/bool``，
    不写 ``None``、dict 或 list，便于直接作为 ``metadatas`` 写入向量库。
    """

    model_config = ConfigDict(extra="forbid")

    course: str = Field(min_length=1, max_length=120, description="课程名称")
    chapter: str = Field(default=UNASSIGNED_CHAPTER, max_length=200, description="章节名称")
    source_file: str = Field(min_length=1, max_length=255, description="原始文件名（不含路径）")
    page: int = Field(default=0, ge=0, description="PDF 页码/PPT 幻灯片号，无页码为 0")
    chunk_id: str = Field(min_length=1, max_length=64, description="Chunk 唯一 ID")
    doc_id: str = Field(default="", max_length=64, description="所属文档 ID")
    file_type: str = Field(default="", max_length=16, description="pdf/pptx/docx/txt")
    locator: str = Field(default="", max_length=64, description="page:12 / slide:5 / chunk:3")
    chunk_index: int = Field(default=0, ge=0, description="块在文档内的顺序，从 0 开始")
    section: str = Field(default="", max_length=200, description="小节标题，未知为空串")
    content_hash: str = Field(default="", max_length=64, description="文本 SHA-1")
    embedding_model: str = Field(default="", max_length=200, description="生成该向量的模型名")

    def to_chroma_metadata(
        self,
        *,
        embedding_model: str | None = None,
        created_at: str | None = None,
        index_version: str = "v1",
    ) -> dict[str, str | int | float | bool]:
        """转换为可写入 Chroma 的扁平字典。

        Args:
            embedding_model: 覆盖模型名；传入时写入 ``embedding_model`` 字段。
            created_at: ISO 8601 时间；传入时附加 ``created_at`` 字段。
            index_version: 索引结构版本，默认 ``v1``。
        """
        metadata: dict[str, str | int | float | bool] = dict(self.model_dump())
        if embedding_model:
            metadata["embedding_model"] = embedding_model
        if created_at:
            metadata["created_at"] = created_at
        metadata["index_version"] = index_version
        return {key: value for key, value in metadata.items() if value is not None}


class ChunkRecord(BaseModel):
    """带完整元数据的文本分块，供向量库写入与引用回查使用。"""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1)
    metadata: ChunkMetadata

    @model_validator(mode="after")
    def _sync_chunk_id(self) -> "ChunkRecord":
        """确保内层 metadata 的 chunk_id 与外层主键一致。"""
        if self.metadata.chunk_id != self.chunk_id:
            self.metadata.chunk_id = self.chunk_id
        return self

    def to_chroma_metadata(
        self,
        *,
        embedding_model: str | None = None,
        created_at: str | None = None,
        index_version: str = "v1",
    ) -> dict[str, str | int | float | bool]:
        """返回写入 Chroma 的扁平 metadata（含主键复制的 ``chunk_id``）。"""
        return self.metadata.to_chroma_metadata(
            embedding_model=embedding_model,
            created_at=created_at,
            index_version=index_version,
        )


class DocumentInfo(BaseModel):
    """文档注册表/文档列表中的单个文档信息。"""

    model_config = ConfigDict(extra="ignore")

    doc_id: str = Field(min_length=1, max_length=64)
    file_name: str = Field(min_length=1, max_length=255)
    course: str = Field(min_length=1, max_length=120)
    chapter: str = Field(default=UNASSIGNED_CHAPTER, max_length=200)
    file_type: str = Field(default="", max_length=16)
    page_count: int = Field(default=0, ge=0)
    char_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    status: DocumentStatus = "uploaded"
    error: str | None = None
    created_at: datetime | None = None


class Citation(BaseModel):
    """答案中的一条引用，编号与正文中的 ``[n]`` 对应。"""

    index: int = Field(ge=1, description="引用编号，从 1 开始")
    chunk_id: str = Field(min_length=1, max_length=64)
    doc_id: str = Field(default="", max_length=64)
    source_file: str = Field(min_length=1, max_length=255)
    course: str = Field(default="", max_length=120)
    chapter: str = Field(default="", max_length=200)
    page: int = Field(default=0, ge=0)
    locator: str = Field(default="", max_length=64)
    snippet: str = Field(default="", description="原文片段，供前端展示")
    score: float = Field(default=0.0, description="相似度分数，越大越相关")


class ChatMessage(BaseModel):
    """会话历史中的一条消息。"""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    """RAG 问答请求。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000, description="用户问题")
    course: str = Field(min_length=1, max_length=120, description="课程名称，检索必选")
    chapter: str | None = Field(default=None, max_length=200, description="可选章节过滤")
    session_id: str | None = Field(default=None, max_length=64, description="会话标识")
    top_k: int = Field(default=5, ge=3, le=10, description="召回片段数")
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=20,
        description="最近若干轮对话（最多 20 条消息）",
    )

    @field_validator("question", "course")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        """去除首尾空白并拒绝纯空白字符串。"""
        value = value.strip()
        if not value:
            raise ValueError("不能为空")
        return value

    @field_validator("chapter", "session_id")
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        """可选字符串统一去除首尾空白，空串转为 None。"""
        if value is None:
            return None
        value = value.strip()
        return value or None


class ChatResponse(BaseModel):
    """RAG 问答响应。

    无有效召回时 ``citations`` 仍为空数组（不得为 null），``grounded=False``。
    """

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = False
    degraded: bool = False
    degraded_reason: str | None = Field(
        default=None,
        description="llm_timeout / llm_switch / extractive_fallback 等",
    )
    model: str = "none"
    latency_ms: int = Field(default=0, ge=0)
    session_id: str | None = None


class PlanActivity(BaseModel):
    """计划任务中的一项学习活动。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["read", "practice", "review", "summary", "test"] = "read"
    title: str = Field(min_length=1, max_length=300)
    minutes: int = Field(ge=0)


class PlanResource(BaseModel):
    """计划任务关联的课程资料片段。"""

    model_config = ConfigDict(extra="forbid")

    source_file: str = Field(min_length=1, max_length=255)
    page: int = Field(default=0, ge=0)
    chunk_id: str = Field(default="", max_length=64)


class PlanTask(BaseModel):
    """一天的学习任务。"""

    model_config = ConfigDict(extra="forbid")

    date: Date
    day_index: int = Field(ge=1)
    phase: str = Field(default="", max_length=50)
    chapter: str = Field(default=UNASSIGNED_CHAPTER, max_length=200)
    objective: str = Field(default="", max_length=500)
    minutes: int = Field(ge=0)
    activities: list[PlanActivity] = Field(default_factory=list)
    resources: list[PlanResource] = Field(default_factory=list)
    checkpoint: str = Field(default="", max_length=500)


class PlanRequest(BaseModel):
    """复习计划生成请求。"""

    model_config = ConfigDict(extra="forbid")

    course: str = Field(min_length=1, max_length=120)
    exam_date: Date = Field(description="考试日期")
    start_date: Date = Field(default_factory=Date.today, description="计划开始日期，默认今天")
    daily_minutes: int = Field(default=90, ge=15, le=480, description="每日可用学习分钟数")
    weak_chapters: list[str] = Field(default_factory=list, max_length=50)
    preferred_slots: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("course")
    @classmethod
    def _strip_course(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("课程名称不能为空")
        return value

    @field_validator("weak_chapters", "preferred_slots")
    @classmethod
    def _clean_text_list(cls, values: list[str]) -> list[str]:
        """去除空项与首尾空白。"""
        return [item.strip() for item in values if item and item.strip()]

    @model_validator(mode="after")
    def _check_date_range(self) -> "PlanRequest":
        """考试日期必须晚于开始日期，且计划跨度不超过 180 天。"""
        if self.exam_date <= self.start_date:
            raise ValueError("EXAM_DATE 必须晚于 START_DATE")
        if (self.exam_date - self.start_date).days > 180:
            raise ValueError("计划跨度不能超过 180 天")
        return self

    @property
    def total_days(self) -> int:
        """计划覆盖的自然天数（含开始日与考试前一日）。"""
        return (self.exam_date - self.start_date).days


class PlanResponse(BaseModel):
    """复习计划响应，``generated_by=rule`` 表示已降级为规则模板。"""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(default="", max_length=64)
    course: str = Field(min_length=1, max_length=120)
    start_date: Date
    exam_date: Date
    total_days: int = Field(ge=1)
    daily_minutes: int = Field(ge=15, le=480)
    generated_by: Literal["llm", "rule"] = "rule"
    summary: str = ""
    tasks: list[PlanTask] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# 与总体设计文档中的 StudyPlan 命名保持兼容。
StudyPlan = PlanResponse
