"""练习题目生成模块。

对外入口 :func:`generate_exercises`，按"LLM 优先 + 规则降级"两条路径，从
已索引的课程资料中生成填空题与选择题：

1. **LLM 出题（优先）**：把检索到的原文分块交给大模型，要求严格输出 JSON
   数组，每题必须给出 ``source_file`` / ``page`` 等来源信息；
2. **规则出题（保底）**：不依赖任何外部服务，从句子中挑选关键词挖空生成
   填空题，或用同一批分块中的其他术语构造 4 选 1 选择题。

无论走哪条路径，每道题都带 ``source_file`` 与 ``page``，便于回原文核对；
无资料或全部生成失败时返回空 ``items``，并通过 ``warnings`` 说明原因。
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.models.schemas import UNASSIGNED_CHAPTER
from app.rag import llm
from app.rag.vectorstore import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

# 支持的题型与难度，与接口层 Literal 校验保持一致。
QUESTION_TYPES: tuple[str, ...] = ("blank", "choice")
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
MIN_COUNT = 1
MAX_COUNT = 20
# 单次出题最多参考的分块数，避免提示词过长与规则题重复。
MAX_CHUNKS = 10
# 未指定章节时随机抽取的章节数。
MAX_CHAPTERS_PER_REQUEST = 3

_CHINESE_RANGE = "\u4e00-\u9fff"
_CONTENT_RE = re.compile(f"[{_CHINESE_RANGE}A-Za-z]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])")
_ENGLISH_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")
_FORMULA_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\s*\([^()]{1,30}\)")
_NUMBER_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|ms|s|MB|GB|KB|bit|byte|Hz|次|个|种|层|位|秒|分钟|小时|倍)"
)
_PURE_NUMBER_RE = re.compile(r"\b\d{2,}(?:\.\d+)?\b")

# 英文停用词：避免把 the/and 之类的词挖成答案。
_ENGLISH_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
        "not", "you", "can", "will", "has", "have", "had", "but", "all", "any",
        "our", "their", "its", "his", "her", "they", "them", "then", "than",
        "into", "over", "under", "use", "used", "using", "may", "also", "such",
    }
)

# 中文停用词/功能字：作为切分边界，帮助从连续汉字串中切出术语。
_CN_STOPWORDS = (
    "其中", "因此", "所以", "但是", "如果", "那么", "例如", "包括", "属于",
    "具有", "使得", "得到", "即可", "应该", "能够", "一种", "这个", "这些",
    "我们", "他们", "可以", "需要", "进行", "通过", "以及", "并且", "或者",
    "而且", "由于", "为了", "对于", "关于", "以下", "如下", "如图", "所示",
    "注意", "定义", "称为", "表示", "一种", "不同", "相关", "主要", "什么",
    "怎么", "如何", "一个", "两个", "这种", "那种", "这时", "那时", "任何",
    "所有", "是指", "分别", "同时", "适用于", "用于", "基于", "位于", "处于",
    "在于", "用来",
)
_CN_STOP_CHARS = (
    "的了是在和与或由为把被从将就也都而且其这那有无不很更最"
)
_CN_SPLIT_RE = re.compile(
    "|".join([re.escape(word) for word in _CN_STOPWORDS] + list(_CN_STOP_CHARS))
)
_CN_RUN_RE = re.compile(f"[{_CHINESE_RANGE}]+")

# 过长汉字串只取句段首尾的候选词；以下字符通常不是术语的开头/结尾，
# 用于过滤"个阶段调""快恢复四"这类机械切分产生的噪声。
_CN_LEADING_GLUE = set("个位次种项条组步批届期第整段阶送供使被把对从向于在是的有和与或及等")
_CN_TRAILING_GLUE = set(
    "个的了吗呢吧啊是在和与或由为中可及等被把对从到将就也都而且并其这那有无很更最"
    "为上下内外之于时会能要该每各某些此即则若但却因故又再还只已未非着过进"
    "一二三四五六七八九十百千万亿两"
)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------
class BlankQuestion(BaseModel):
    """填空题：题干中的 ``____`` 为待填内容。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["blank"] = "blank"
    stem: str = Field(min_length=1, max_length=1000, description="题干，空位用 ____ 表示")
    answer: str = Field(min_length=1, max_length=200, description="标准答案")
    hint: str | None = Field(default=None, max_length=200, description="可选提示")
    source_file: str = Field(min_length=1, max_length=255, description="来源文件名")
    page: int = Field(default=0, ge=0, description="来源页码/幻灯片号")


class ChoiceQuestion(BaseModel):
    """选择题：固定 4 个互不相同的选项，``answer_index`` 从 0 开始。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    stem: str = Field(min_length=1, max_length=1000, description="题干，空位用 ____ 表示")
    options: list[str] = Field(min_length=4, max_length=4, description="4 个互不相同的选项")
    answer_index: int = Field(ge=0, le=3, description="正确选项下标，从 0 开始")
    explanation: str = Field(default="", max_length=1000, description="解析，含原文依据")
    source_file: str = Field(min_length=1, max_length=255, description="来源文件名")
    page: int = Field(default=0, ge=0, description="来源页码/幻灯片号")

    @field_validator("options")
    @classmethod
    def _check_options(cls, values: list[str]) -> list[str]:
        """选项必须非空、4 项且互不重复。"""
        cleaned = [str(value).strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("选项不能为空")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("选项不能重复")
        if len(cleaned) != 4:
            raise ValueError("选项必须为 4 项")
        return cleaned


class ExerciseSet(BaseModel):
    """一次出题的结果；``items`` 为空时看 ``warnings`` 了解原因。"""

    model_config = ConfigDict(extra="forbid")

    course: str = Field(min_length=1, max_length=120, description="课程名称")
    chapter: str | None = Field(default=None, max_length=200, description="请求限定的章节")
    items: list[BlankQuestion | ChoiceQuestion] = Field(
        default_factory=list, description="题目列表，填空题与选择题可混合"
    )
    generated_by: Literal["llm", "rule"] = Field(
        default="rule", description="llm 表示大模型出题，rule 表示规则降级"
    )
    warnings: list[str] = Field(default_factory=list, description="降级/补足等提示")


# ---------------------------------------------------------------------------
# 参数与文本工具
# ---------------------------------------------------------------------------
def _truncate(text: str, limit: int) -> str:
    """按字符数截断文本，超出部分用省略号代替。"""
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "……"


def _normalize_inputs(
    course: Any,
    chapter: Any,
    count: Any,
    types: Sequence[str] | None,
    difficulty: Any,
) -> tuple[str, str | None, int, list[str], str]:
    """校验并规范化入参，非法时抛出 ``ValueError`` 供接口层转 422。"""
    course_text = str(course or "").strip()[:120]
    if not course_text:
        raise ValueError("course 不能为空。")

    try:
        count_value = int(count)
    except (TypeError, ValueError) as exc:
        raise ValueError("count 必须是整数。") from exc
    if not MIN_COUNT <= count_value <= MAX_COUNT:
        raise ValueError(f"count 必须在 {MIN_COUNT}～{MAX_COUNT} 之间。")

    normalized_types: list[str] = []
    for raw_type in types or []:
        value = str(raw_type or "").strip().lower()
        if value in QUESTION_TYPES and value not in normalized_types:
            normalized_types.append(value)
    if not normalized_types:
        raise ValueError("types 必须包含 blank 或 choice。")

    difficulty_text = str(difficulty or "medium").strip().lower()
    if difficulty_text not in DIFFICULTIES:
        raise ValueError("difficulty 必须是 easy / medium / hard。")

    chapter_text = str(chapter or "").strip()[:200] or None
    return course_text, chapter_text, count_value, normalized_types, difficulty_text


def _dedupe_warnings(warnings: Sequence[str]) -> list[str]:
    """按出现顺序去重提示信息。"""
    return list(dict.fromkeys(str(item) for item in warnings if str(item).strip()))


def _basename(value: str) -> str:
    """取出文件名并统一小写，便于与 LLM 返回的来源做宽松匹配。"""
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1].strip().lower()


def _safe_page(value: Any) -> int:
    """把页码转换为非负整数，非法值按 0 处理。"""
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# 资料检索
# ---------------------------------------------------------------------------
def _list_chapters(store: Any, course: str, warnings: list[str]) -> list[str]:
    """从向量库文档清单中尽力推断课程章节，失败时返回空列表。"""
    try:
        documents = store.list_documents(course=course)
    except Exception:  # noqa: BLE001 - 章节推断失败不影响课程整体检索
        logger.warning("读取课程章节失败 course=%s", course, exc_info=True)
        warnings.append("读取课程章节失败，已按课程整体检索资料。")
        return []

    chapters: list[str] = []
    for document in documents or []:
        if not isinstance(document, dict):
            continue
        name = str(document.get("chapter") or "").strip()
        if name and name != UNASSIGNED_CHAPTER and name not in chapters:
            chapters.append(name)
    return chapters


def _coerce_chunk(raw: Any) -> RetrievedChunk | None:
    """把检索结果统一转换为 ``RetrievedChunk``。

    正常路径返回的就是 ``RetrievedChunk``；这里同时兼容单元测试常用的
    ``{"text": ..., "metadata": {...}}`` 字典形式，避免调用方因假对象类型
    不同而无法出题。
    """
    if isinstance(raw, RetrievedChunk):
        chunk = raw
    else:
        metadata: dict[str, Any] = {}
        if isinstance(raw, dict):
            raw_metadata = raw.get("metadata")
            if isinstance(raw_metadata, dict):
                metadata.update(raw_metadata)
            # 兼容来源信息直接写在顶层的字典
            for key in ("source_file", "page", "chapter", "chunk_id", "doc_id"):
                if raw.get(key) is not None and key not in metadata:
                    metadata[key] = raw[key]
            text = str(raw.get("text") or raw.get("document") or "").strip()
            chunk_id = str(raw.get("chunk_id") or raw.get("id") or "").strip()
            score = raw.get("score")
        else:
            raw_metadata = getattr(raw, "metadata", {})
            metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            text = str(getattr(raw, "text", "") or "").strip()
            chunk_id = str(getattr(raw, "chunk_id", "") or "").strip()
            score = getattr(raw, "score", 0.0)
        if not text:
            return None
        try:
            score_value = float(score or 0.0)
        except (TypeError, ValueError):
            score_value = 0.0
        chunk = RetrievedChunk(
            chunk_id=chunk_id or f"chunk_{abs(hash(text))}",
            text=text,
            metadata=metadata,
            score=score_value,
        )

    if not chunk.text.strip() or not chunk.source_file.strip():
        return None
    return chunk


def _dedupe_chunks(chunks: Sequence[Any]) -> list[RetrievedChunk]:
    """去重并丢弃缺少正文或来源文件的分块，保证每题都能标注来源。"""
    result: list[RetrievedChunk] = []
    seen: set[str] = set()
    for raw in chunks or []:
        chunk = _coerce_chunk(raw)
        if chunk is None:
            continue
        key = (
            chunk.chunk_id
            or f"{chunk.source_file}|{chunk.page}|{chunk.text[:80]}"
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(chunk)
    return result


def _retrieve_chunks(
    course: str,
    chapter: str | None,
    count: int,
    rng: random.Random,
    warnings: list[str],
) -> list[RetrievedChunk]:
    """按课程/章节取出若干原文分块。

    指定 ``chapter`` 时只检索该章节；未指定时先读取章节清单，随机抽取最多
    3 个章节分别检索，读取不到章节则退化为按课程整体检索。
    """
    top_k = max(6, min(count * 3, 20))
    query_text = (
        f"{chapter} 核心概念 定义 公式 典型例题"
        if chapter
        else "课程核心概念 定义 公式 典型例题"
    )

    try:
        store = VectorStore()
        if chapter:
            chunks = store.query(
                query_text, course=course, chapter=chapter, top_k=top_k
            )
        else:
            chapters = _list_chapters(store, course, warnings)
            if chapters:
                picked = rng.sample(
                    chapters, k=min(MAX_CHAPTERS_PER_REQUEST, len(chapters))
                )
                logger.info("未指定章节，随机抽取章节出题：%s", picked)
                chunks = []
                for name in picked:
                    chunks.extend(
                        store.query(
                            f"{name} 核心概念 定义 公式",
                            course=course,
                            chapter=name,
                            top_k=top_k,
                        )
                    )
                if not _dedupe_chunks(chunks):
                    # 章节过滤可能因索引元数据不完整而落空，退化为课程整体检索
                    warnings.append("指定章节未检索到内容，已改为按课程整体检索资料。")
                    chunks = store.query(query_text, course=course, top_k=top_k)
            else:
                chunks = store.query(query_text, course=course, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 - 向量库异常统一降级为无资料
        logger.warning("检索课程资料失败 course=%s error=%s", course, exc, exc_info=True)
        warnings.append(f"检索课程资料失败，暂时无法出题（{type(exc).__name__}）。")
        return []

    deduped = _dedupe_chunks(chunks or [])
    if len(deduped) > MAX_CHUNKS:
        deduped = rng.sample(deduped, MAX_CHUNKS)
    if not deduped:
        warnings.append("未在课程资料中检索到可用于出题的内容，请先上传并索引资料。")
    return deduped


# ---------------------------------------------------------------------------
# 规则出题：句子切分与术语抽取
# ---------------------------------------------------------------------------
def _split_sentences(text: str) -> list[str]:
    """按中英文标点与换行切分句子，并过滤过短的碎片。"""
    sentences: list[str] = []
    for line in str(text or "").splitlines():
        for piece in _SENTENCE_SPLIT_RE.split(line):
            sentence = re.sub(r"\s+", " ", piece).strip(" \t　")
            if len(sentence) < 10:
                continue
            if not _CONTENT_RE.search(sentence):
                continue
            sentences.append(sentence)
    return sentences


def _is_clean_cn_term(term: str) -> bool:
    """判断候选中文词是否像完整术语（过滤机械切分产生的碎片）。"""
    if len(term) < 2:
        return False
    if term[0] in _CN_LEADING_GLUE or term[-1] in _CN_TRAILING_GLUE:
        return False
    return True


def _chinese_terms(text: str) -> list[str]:
    """从连续汉字串中切出候选术语。

    先用停用词/功能字做边界切分；不超过 6 字的片段整体保留，过长的片段
    只取首尾的 4 字或 3 字作为候选，避免滑动窗口产生"个阶段调"这类噪声。
    """
    terms: list[str] = []
    for run in _CN_RUN_RE.findall(text):
        for fragment in _CN_SPLIT_RE.split(run):
            fragment = fragment.strip()
            if len(fragment) < 2:
                continue
            if len(fragment) <= 6:
                if _is_clean_cn_term(fragment):
                    terms.append(fragment)
                continue

            # 长片段：最多取一个前缀和一个后缀，长度 4 优先、其次 3
            prefix = next(
                (
                    candidate
                    for candidate in (fragment[:4], fragment[:3])
                    if _is_clean_cn_term(candidate)
                ),
                "",
            )
            suffix = next(
                (
                    candidate
                    for candidate in (fragment[-4:], fragment[-3:])
                    if _is_clean_cn_term(candidate)
                ),
                "",
            )
            for candidate in (prefix, suffix):
                if candidate and candidate not in terms:
                    terms.append(candidate)
    return terms


def _rank_terms(sentence: str) -> list[str]:
    """抽取并排序句子中的候选术语。

    排序优先级：4 字以上中文词 > 英文术语 > 公式/数字 > 2～3 字中文词，
    同一优先级内长词优先。
    """
    scored: dict[str, tuple[str, int, int]] = {}

    def add(term: str, priority: int) -> None:
        value = str(term or "").strip()
        if len(value) < 2:
            return
        key = value.lower()
        rank = (priority, len(value))
        current = scored.get(key)
        if current is None or rank > (current[1], current[2]):
            scored[key] = (value, priority, len(value))

    for match in _FORMULA_RE.finditer(sentence):
        add(match.group(0), 2)
    for match in _ENGLISH_TERM_RE.finditer(sentence):
        if match.group(0).lower() in _ENGLISH_STOPWORDS:
            continue
        add(match.group(0), 3)
    for match in _NUMBER_RE.finditer(sentence):
        add(match.group(0), 2)
    for match in _PURE_NUMBER_RE.finditer(sentence):
        add(match.group(0), 1)
    for term in _chinese_terms(sentence):
        add(term, 4 if len(term) >= 4 else 0)

    ordered = sorted(
        scored.values(), key=lambda item: (-item[1], -item[2], item[0].lower())
    )
    return [item[0] for item in ordered]


def _term_matches_difficulty(term: str, difficulty: str) -> bool:
    """按难度筛选术语：简单偏短、困难偏长或公式/英文。"""
    length = len(term)
    if difficulty == "easy":
        return 2 <= length <= 6
    if difficulty == "hard":
        return length >= 5 or bool(_FORMULA_RE.fullmatch(term))
    # medium：4 字以上中文词，或英文术语/公式
    return length >= 4 or bool(re.search(r"[A-Za-z]", term))


def _make_stem(sentence: str, term: str, limit: int = 300) -> str:
    """把句子中的术语替换为 ``____``；题目过长时围绕空位截取窗口。"""
    blank = "____"
    stem = sentence.replace(term, blank)
    if blank not in stem:
        return ""

    if len(stem) <= limit:
        return stem

    index = stem.find(blank)
    start = max(0, index - 100)
    end = min(len(stem), start + limit)
    window = stem[start:end]
    if start > 0:
        window = "……" + window
    if end < len(stem):
        window = window + "……"
    return window if blank in window else ""


def _valid_distractor(term: str, answer: str) -> bool:
    """干扰项必须非空、不等于正确答案，且与答案不存在包含关系。"""
    value = str(term or "").strip()
    if len(value) < 2:
        return False
    if value.lower() == answer.lower():
        return False
    lowered = value.lower()
    answer_lowered = answer.lower()
    if answer_lowered in lowered or lowered in answer_lowered:
        return False
    return True


def _blank_question(
    chunk: RetrievedChunk,
    sentence: str,
    term: str,
    difficulty: str,
) -> BlankQuestion | None:
    """由句子与术语构造一道填空题。"""
    stem = _make_stem(sentence, term)
    if not stem:
        return None
    hint: str | None = None
    if difficulty == "easy":
        hint = f"提示：共 {len(term)} 个字，首字是「{term[0]}」"
    return BlankQuestion(
        stem=stem,
        answer=_truncate(term, 200),
        hint=_truncate(hint, 200) if hint else None,
        source_file=chunk.source_file,
        page=chunk.page,
    )


def _choice_question(
    chunk: RetrievedChunk,
    sentence: str,
    answer: str,
    term_pool: Sequence[str],
    rng: random.Random,
) -> ChoiceQuestion | None:
    """由句子与正确术语构造一道 4 选 1 选择题。"""
    stem = _make_stem(sentence, answer)
    if not stem:
        return None

    candidates = [term for term in term_pool if _valid_distractor(term, answer)]
    if len(candidates) < 3:
        return None

    distractors = rng.sample(candidates, 3)
    options = [answer, *distractors]
    rng.shuffle(options)
    try:
        answer_index = options.index(answer)
    except ValueError:  # pragma: no cover - 选项由 answer 构造，理论不会缺失
        return None

    explanation = (
        f"正确答案是「{answer}」。原文：{_truncate(sentence, 200)}"
        f"（来源：《{chunk.source_file}》第 {chunk.page} 页）"
    )
    return ChoiceQuestion(
        stem=stem,
        options=[_truncate(option, 100) for option in options],
        answer_index=answer_index,
        explanation=_truncate(explanation, 1000),
        source_file=chunk.source_file,
        page=chunk.page,
    )


def _collect_sentence_records(
    chunks: Sequence[RetrievedChunk], rng: random.Random
) -> list[tuple[RetrievedChunk, str, list[str]]]:
    """把分块拆成 ``(分块, 句子, 候选术语)`` 记录并打乱顺序。"""
    records: list[tuple[RetrievedChunk, str, list[str]]] = []
    for chunk in chunks:
        for sentence in _split_sentences(chunk.text):
            terms = _rank_terms(sentence)
            if terms:
                records.append((chunk, sentence, terms))
    rng.shuffle(records)
    return records


def _build_term_pool(
    records: Sequence[tuple[RetrievedChunk, str, list[str]]]
) -> list[str]:
    """汇总同一批分块中的全部术语（忽略大小写去重），作为干扰项来源。"""
    pool: list[str] = []
    seen: set[str] = set()
    for _, _, terms in records:
        for term in terms:
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            pool.append(term)
    return pool


def _candidate_items(
    *,
    question_type: str,
    records: Sequence[tuple[RetrievedChunk, str, list[str]]],
    term_pool: Sequence[str],
    count: int,
    difficulty: str,
    rng: random.Random,
) -> tuple[list[BlankQuestion | ChoiceQuestion], set[str]]:
    """为单一题型生成候选题目。

    Returns:
        ``(候选题目, 放宽难度后才生成的题干集合)``；第二轮放宽筛选只为
        避免资料术语偏短时一道题也生成不出来。
    """
    candidates: list[BlankQuestion | ChoiceQuestion] = []
    relaxed_stems: set[str] = set()
    used_stems: set[str] = set()
    target = max(count * 3, count)

    for relax in (False, True):
        for chunk, sentence, terms in records:
            if len(candidates) >= target:
                break
            for term in terms:
                if not relax and not _term_matches_difficulty(term, difficulty):
                    continue
                if not term.strip():
                    continue
                try:
                    if question_type == "blank":
                        item = _blank_question(chunk, sentence, term, difficulty)
                    else:
                        item = _choice_question(chunk, sentence, term, term_pool, rng)
                except ValidationError:
                    logger.debug("规则题目校验失败，已跳过 term=%s", term)
                    continue
                if item is None or item.stem in used_stems:
                    continue
                used_stems.add(item.stem)
                candidates.append(item)
                if relax:
                    relaxed_stems.add(item.stem)
        if len(candidates) >= count:
            break
    return candidates, relaxed_stems


def _question_type_order(types: Sequence[str], count: int) -> list[str]:
    """按题型轮流排列 count 个出题位置，保证混合出题时题型均衡。"""
    return [types[index % len(types)] for index in range(count)]


def _rule_questions(
    chunks: Sequence[RetrievedChunk],
    *,
    types: Sequence[str],
    count: int,
    difficulty: str,
    rng: random.Random,
    warnings: list[str],
    exclude_stems: set[str] | None = None,
) -> list[BlankQuestion | ChoiceQuestion]:
    """规则降级路径：按题型轮流从分块中出题，不足时轮转补足。"""
    records = _collect_sentence_records(chunks, rng)
    if not records:
        return []

    term_pool = _build_term_pool(records)
    candidates: dict[str, list[BlankQuestion | ChoiceQuestion]] = {}
    relaxed_stems: set[str] = set()
    for question_type in types:
        items, relaxed = _candidate_items(
            question_type=question_type,
            records=records,
            term_pool=term_pool,
            count=count,
            difficulty=difficulty,
            rng=rng,
        )
        candidates[question_type] = items
        relaxed_stems |= relaxed

    selected: list[BlankQuestion | ChoiceQuestion] = []
    used = set(exclude_stems or ())
    pointers = {question_type: 0 for question_type in types}

    # 第一轮：按题型轮流取题，避免靠前的题型把所有句子先用完
    for question_type in _question_type_order(types, count):
        items = candidates.get(question_type, [])
        pointer = pointers.get(question_type, 0)
        while pointer < len(items):
            item = items[pointer]
            pointer += 1
            if item.stem in used:
                continue
            selected.append(item)
            used.add(item.stem)
            break
        pointers[question_type] = pointer

    # 第二轮：某题型候选不足时，用其他题型的剩余候选轮转补足
    while len(selected) < count:
        progressed = False
        for question_type in types:
            items = candidates.get(question_type, [])
            pointer = pointers.get(question_type, 0)
            while pointer < len(items):
                item = items[pointer]
                pointer += 1
                if item.stem in used:
                    continue
                selected.append(item)
                used.add(item.stem)
                progressed = True
                break
            pointers[question_type] = pointer
            if len(selected) >= count:
                break
        if not progressed:
            break

    if len(selected) < count:
        warnings.append(
            f"资料中的可用术语有限，实际生成 {len(selected)} 道题目（目标 {count} 道）。"
        )
    if relaxed_stems and any(item.stem in relaxed_stems for item in selected):
        warnings.append("部分资料术语偏短，已放宽难度筛选以保证出题数量。")
    return selected[:count]


# ---------------------------------------------------------------------------
# LLM 出题：提示词、JSON 解析与结果校验
# ---------------------------------------------------------------------------
def _difficulty_label(difficulty: str) -> str:
    """把难度枚举转换为提示词中的中文描述。"""
    return {"easy": "简单（基础概念记忆）", "medium": "中等（概念理解与辨析）", "hard": "困难（综合运用与易混点辨析）"}[
        difficulty
    ]


def _build_llm_messages(
    *,
    course: str,
    chapter: str | None,
    chunks: Sequence[RetrievedChunk],
    count: int,
    types: Sequence[str],
    difficulty: str,
) -> list[dict[str, str]]:
    """组装要求严格输出 JSON 数组的出题提示词。"""
    type_labels = {"blank": "填空题", "choice": "选择题"}
    type_text = "、".join(type_labels[item] for item in types)

    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        chapter_name = str(chunk.metadata.get("chapter") or UNASSIGNED_CHAPTER)
        blocks.append(
            f"【资料 {index}】文件：{chunk.source_file}；页码：{chunk.page}；"
            f"章节：{chapter_name}\n{_truncate(chunk.text, 800)}"
        )

    scope = f"章节《{chapter}》" if chapter else "整门课程（可跨章节）"
    system = "你是一名严谨的高校课程助教，只依据给定资料命制练习题，不编造事实。"
    user = f"""请基于下面的课程资料，为课程《{course}》的{scope}生成 {count} 道练习题。

题型：{type_text}
难度：{_difficulty_label(difficulty)}

要求：
1. 只能使用资料中明确出现的事实与术语，不得引入资料之外的知识；
2. 每道题必须标注 source_file 与 page，且必须与资料清单中的真实取值完全一致；
3. 填空题用 ____ 表示空位，answer 为原文中的完整术语或数值，hint 可选；
4. 选择题必须提供 4 个互不相同的选项，answer_index 为正确选项下标（从 0 开始），
   explanation 说明理由并引用原文；
5. 题目不得重复，题干尽量完整、可独立作答；
6. 只输出 JSON 数组，不要 Markdown 代码块、不要任何解释性文字。

JSON 结构示例：
[
  {{"type": "blank", "stem": "TCP 通过____建立可靠连接。", "answer": "三次握手",
    "hint": "共 4 个字", "source_file": "示例.pdf", "page": 12}},
  {{"type": "choice", "stem": "传输层提供的是____通信服务。",
    "options": ["端到端", "点到点", "广播", "组播"], "answer_index": 0,
    "explanation": "传输层为应用进程提供端到端的逻辑通信。",
    "source_file": "示例.pdf", "page": 12}}
]

资料清单：
{chr(10).join(blocks)}"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


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


def _parse_llm_json(raw: str) -> list[Any]:
    """从 LLM 文本中解析题目数组，兼容代码围栏与包了一层的对象。"""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("LLM 返回了空文本。")

    text = _strip_code_fence(raw)
    candidates = [text]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("items", "questions", "exercises", "data"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
    raise ValueError("LLM 返回的内容不是合法 JSON 数组。")


def _detect_question_type(raw: dict[str, Any]) -> str:
    """识别 LLM 返回的题型，兼容中文写法；无法判断时返回空串。"""
    value = str(raw.get("type") or raw.get("question_type") or "").strip().lower()
    if value in QUESTION_TYPES:
        return value
    if value in {"填空", "填空题", "blank_question"}:
        return "blank"
    if value in {"选择", "选择题", "choice_question"}:
        return "choice"
    if isinstance(raw.get("options"), list):
        return "choice"
    if raw.get("answer") is not None and not raw.get("options"):
        return "blank"
    return ""


def _resolve_source(
    raw: dict[str, Any], chunks: Sequence[RetrievedChunk]
) -> RetrievedChunk | None:
    """把 LLM 标注的来源匹配回真实分块，匹配不到时返回 ``None``。

    优先按 ``source_file`` + ``page`` 精确匹配；只给文件名时取该文件首个
    分块；文件与页码都不可信时，再用答案/题干在原文中回查。
    """
    raw_file = str(
        raw.get("source_file") or raw.get("file") or raw.get("source") or ""
    ).strip()
    raw_page = _safe_page(raw.get("page"))
    answer = str(raw.get("answer") or raw.get("correct_answer") or "").strip()
    stem = str(raw.get("stem") or raw.get("question") or "").strip()

    if raw_file:
        matched = [
            chunk
            for chunk in chunks
            if _basename(chunk.source_file) == _basename(raw_file)
        ]
        if matched:
            if raw_page:
                for chunk in matched:
                    if chunk.page == raw_page:
                        return chunk
            return matched[0]

    # 回查原文：答案或题干片段能在正文中找到，就认为来源可信
    probes = [probe for probe in (answer, stem[:30]) if probe]
    for probe in probes:
        for chunk in chunks:
            if probe in chunk.text:
                return chunk
    return None


def _normalize_llm_items(
    data: Sequence[Any],
    *,
    chunks: Sequence[RetrievedChunk],
    types: Sequence[str],
    count: int,
    warnings: list[str],
) -> list[BlankQuestion | ChoiceQuestion]:
    """把 LLM 返回的原始条目校验为 Pydantic 题目，丢弃不合法项。"""
    items: list[BlankQuestion | ChoiceQuestion] = []
    used_stems: set[str] = set()
    dropped_source = 0
    dropped_invalid = 0

    for raw in data:
        if len(items) >= count:
            break
        if not isinstance(raw, dict):
            dropped_invalid += 1
            continue

        question_type = _detect_question_type(raw)
        if question_type not in types:
            dropped_invalid += 1
            continue

        source = _resolve_source(raw, chunks)
        if source is None:
            dropped_source += 1
            continue

        try:
            if question_type == "blank":
                question = _normalize_llm_blank(raw, source)
            else:
                question = _normalize_llm_choice(raw, source)
        except (ValueError, ValidationError):
            dropped_invalid += 1
            continue

        if question is None or question.stem in used_stems:
            dropped_invalid += 1
            continue
        used_stems.add(question.stem)
        items.append(question)

    if dropped_source:
        warnings.append(f"LLM 有 {dropped_source} 道题缺少可核对的来源，已丢弃。")
    if dropped_invalid:
        warnings.append(f"LLM 有 {dropped_invalid} 道题字段不完整或重复，已丢弃。")
    return items


def _normalize_llm_blank(
    raw: dict[str, Any], source: RetrievedChunk
) -> BlankQuestion | None:
    """校验 LLM 返回的填空题；题干缺少空位时尝试用答案替换。"""
    stem = str(raw.get("stem") or raw.get("question") or "").strip()
    answer = str(raw.get("answer") or raw.get("correct_answer") or "").strip()
    if not stem or not answer:
        return None
    if "____" not in stem:
        if answer not in stem:
            return None
        stem = stem.replace(answer, "____")

    hint_value = raw.get("hint")
    hint = str(hint_value).strip() if isinstance(hint_value, str) else None
    return BlankQuestion(
        stem=_truncate(stem, 1000),
        answer=_truncate(answer, 200),
        hint=_truncate(hint, 200) if hint else None,
        source_file=source.source_file,
        page=source.page,
    )


def _normalize_llm_choice(
    raw: dict[str, Any], source: RetrievedChunk
) -> ChoiceQuestion | None:
    """校验 LLM 返回的选择题；选项不足 4 项或下标非法时丢弃。"""
    stem = str(raw.get("stem") or raw.get("question") or "").strip()
    raw_options = raw.get("options")
    if not stem or not isinstance(raw_options, list):
        return None

    options = [_truncate(str(option).strip(), 100) for option in raw_options]
    if len(options) != 4 or any(not option for option in options):
        return None
    if len(set(options)) != 4:
        return None

    answer_text = str(raw.get("answer") or raw.get("correct_answer") or "").strip()
    if raw.get("answer_index") is not None:
        try:
            answer_index = int(raw.get("answer_index"))
        except (TypeError, ValueError):
            return None
    elif answer_text in options:
        answer_index = options.index(answer_text)
    else:
        return None
    if not 0 <= answer_index < 4:
        return None

    explanation = str(raw.get("explanation") or raw.get("analysis") or "").strip()
    if not explanation:
        explanation = f"正确答案是「{options[answer_index]}」。"
    return ChoiceQuestion(
        stem=_truncate(stem, 1000),
        options=options,
        answer_index=answer_index,
        explanation=_truncate(explanation, 1000),
        source_file=source.source_file,
        page=source.page,
    )


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------
def generate_exercises(
    course: str,
    chapter: str | None = None,
    count: int = 5,
    types: Sequence[str] | None = None,
    difficulty: str = "medium",
    use_llm: bool = True,
) -> ExerciseSet:
    """从课程资料生成填空题/选择题。

    Args:
        course: 课程名称，决定检索范围，必填。
        chapter: 可选章节过滤；为空时从多个章节中随机抽取资料。
        count: 目标题目数量，取值 1～20，默认 5。
        types: 题型列表，可选 ``blank`` / ``choice``，默认两者都出。
        difficulty: 难度，``easy`` / ``medium`` / ``hard``，默认 ``medium``。
        use_llm: 是否优先调用大模型；``False`` 时直接走规则出题。

    Returns:
        :class:`ExerciseSet`；没有资料或生成失败时 ``items`` 为空，
        具体原因写在 ``warnings`` 中。每道题都带 ``source_file`` 与 ``page``。

    Raises:
        ValueError: 课程为空、数量/题型/难度非法等参数问题。
    """
    warnings: list[str] = []
    course_text, chapter_text, count_value, types_list, difficulty_text = (
        _normalize_inputs(course, chapter, count, types, difficulty)
    )
    rng = random.Random()  # 不固定种子，多次出题可得到不同题目

    chunks = _retrieve_chunks(
        course_text, chapter_text, count_value, rng, warnings
    )
    if not chunks:
        # 无资料：返回空集 + warning，不抛出异常，接口保持 200
        return ExerciseSet(
            course=course_text,
            chapter=chapter_text,
            items=[],
            generated_by="rule",
            warnings=_dedupe_warnings(warnings),
        )

    # 1) 优先尝试 LLM 出题
    if use_llm:
        try:
            messages = _build_llm_messages(
                course=course_text,
                chapter=chapter_text,
                chunks=chunks,
                count=count_value,
                types=types_list,
                difficulty=difficulty_text,
            )
            raw = llm.chat(messages, temperature=0.4)
            data = _parse_llm_json(raw)
            items = _normalize_llm_items(
                data,
                chunks=chunks,
                types=types_list,
                count=count_value,
                warnings=warnings,
            )
            if not items:
                raise ValueError("LLM 未返回符合要求的题目")

            if len(items) < count_value:
                supplement = _rule_questions(
                    chunks,
                    types=types_list,
                    count=count_value - len(items),
                    difficulty=difficulty_text,
                    rng=rng,
                    warnings=warnings,
                    exclude_stems={item.stem for item in items},
                )
                if supplement:
                    warnings.append(
                        f"LLM 仅返回 {len(items)} 道可用题目，已用规则补足 "
                        f"{len(supplement)} 道。"
                    )
                    items.extend(supplement)

            if len(items) > count_value:
                items = items[:count_value]
            if len(items) < count_value:
                warnings.append(
                    f"可用资料有限，实际生成 {len(items)} 道题目"
                    f"（目标 {count_value} 道）。"
                )
            return ExerciseSet(
                course=course_text,
                chapter=chapter_text,
                items=items,
                generated_by="llm",
                warnings=_dedupe_warnings(warnings),
            )
        except Exception as exc:  # noqa: BLE001 - 网络/JSON/校验失败均降级
            reason = f"{type(exc).__name__}: {str(exc)}"[:180]
            logger.warning("LLM 出题失败，改用规则出题：%s", reason)
            warnings.append(f"LLM 出题失败，已使用规则生成（{reason}）。")

    # 2) 规则出题保底
    items = _rule_questions(
        chunks,
        types=types_list,
        count=count_value,
        difficulty=difficulty_text,
        rng=rng,
        warnings=warnings,
    )
    if not items:
        warnings.append("资料中的句子不足以生成题目，请补充或重新索引课程资料。")
    elif len(items) < count_value:
        warnings.append(
            f"可用术语有限，实际生成 {len(items)} 道题目"
            f"（目标 {count_value} 道）。"
        )
    return ExerciseSet(
        course=course_text,
        chapter=chapter_text,
        items=items,
        generated_by="rule",
        warnings=_dedupe_warnings(warnings),
    )


__all__ = [
    "BlankQuestion",
    "ChoiceQuestion",
    "ExerciseSet",
    "generate_exercises",
]
