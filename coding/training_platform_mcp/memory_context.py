"""Agent 记忆与上下文工程的最小教学实现。

本模块不连接向量数据库或真实 LLM，重点展示四条边界：

1. ``TaskState`` 是 Runtime 的权威状态，不能由长期记忆覆盖；
2. ``MemoryStore`` 负责租户/用户隔离、版本、过期和软删除；
3. 检索结果只是候选记忆，进入 Context 前还要做来源与注入检查；
4. ``ContextBuilder`` 在 token 预算内组装模型视图，但这个视图本身不权威。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Mapping, Sequence


class MemoryType(str, Enum):
    """三类常见长期记忆。"""

    EPISODIC = "EPISODIC"      # 某次具体经历，例如“上次部署因显存不足失败”
    SEMANTIC = "SEMANTIC"      # 稳定事实或偏好，例如“用户偏好中文回答”
    PROCEDURAL = "PROCEDURAL"  # 可复用流程，例如“发布前必须跑回归测试”


class Sensitivity(str, Enum):
    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"
    RESTRICTED = "RESTRICTED"


class SourceTrust(str, Enum):
    SYSTEM = "SYSTEM"
    USER_CONFIRMED = "USER_CONFIRMED"
    MODEL_INFERRED = "MODEL_INFERRED"
    EXTERNAL_UNTRUSTED = "EXTERNAL_UNTRUSTED"


class MemoryScopeHint(str, Enum):
    TASK_ONLY = "TASK_ONLY"
    CROSS_TASK = "CROSS_TASK"


class WriteDecision(str, Enum):
    DO_NOT_STORE = "DO_NOT_STORE"
    TASK_STATE_ONLY = "TASK_STATE_ONLY"
    WRITE_LONG_TERM = "WRITE_LONG_TERM"
    REQUIRES_CONFIRMATION = "REQUIRES_CONFIRMATION"


@dataclass(frozen=True)
class MemoryCandidate:
    """从对话或 Trace 提取出的候选；候选不等于已获准写入。"""

    subject_key: str
    content: str
    memory_type: MemoryType
    source_event_id: str
    source_trust: SourceTrust
    scope_hint: MemoryScopeHint
    sensitivity: Sensitivity = Sensitivity.PRIVATE
    high_impact: bool = False


class MemoryWriteGate:
    """用确定性规则决定候选信息的保存层级。"""

    def decide(self, candidate: MemoryCandidate) -> WriteDecision:
        if not candidate.subject_key or not candidate.content.strip():
            return WriteDecision.DO_NOT_STORE
        if _looks_like_prompt_injection(candidate.content):
            return WriteDecision.DO_NOT_STORE
        if candidate.source_trust is SourceTrust.EXTERNAL_UNTRUSTED:
            return WriteDecision.DO_NOT_STORE
        if candidate.scope_hint is MemoryScopeHint.TASK_ONLY:
            return WriteDecision.TASK_STATE_ONLY
        if candidate.high_impact or candidate.sensitivity is Sensitivity.RESTRICTED:
            return WriteDecision.REQUIRES_CONFIRMATION
        if candidate.source_trust is SourceTrust.MODEL_INFERRED:
            return WriteDecision.REQUIRES_CONFIRMATION
        return WriteDecision.WRITE_LONG_TERM


@dataclass(frozen=True)
class MemoryRecord:
    """一条带作用域、来源、时效和版本的长期记忆记录。

    同一个 ``memory_id`` 的更高 ``version`` 表示新版本。生产系统通常还会保存
    embedding、审计人、法律保留策略和加密信息；这些不是本教学示例的重点。
    """

    memory_id: str
    tenant_id: str
    user_id: str
    subject_key: str
    content: str
    memory_type: MemoryType
    source_event_id: str
    created_at: datetime
    expires_at: datetime | None
    version: int
    confidence: float
    sensitivity: Sensitivity
    source_trust: SourceTrust
    tags: frozenset[str] = field(default_factory=frozenset)
    deleted: bool = False

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("version 必须从 1 开始")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence 必须位于 [0, 1]")
        if not self.memory_id or not self.tenant_id or not self.user_id or not self.subject_key:
            raise ValueError("memory_id、tenant_id、user_id 和 subject_key 不能为空")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at 必须包含时区")
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            raise ValueError("expires_at 必须包含时区")


@dataclass(frozen=True)
class RetrievalQuery:
    tenant_id: str
    user_id: str
    text: str
    now: datetime
    allowed_sensitivity: frozenset[Sensitivity] = field(
        default_factory=lambda: frozenset({Sensitivity.PUBLIC, Sensitivity.PRIVATE})
    )
    min_confidence: float = 0.5
    limit: int = 5


@dataclass(frozen=True)
class RetrievedMemory:
    record: MemoryRecord
    score: float


def _tokens(text: str) -> set[str]:
    """无依赖的教学检索分词；真实系统可替换为 embedding/BM25。"""

    return set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower()))


class InMemoryMemoryStore:
    """保留所有版本、查询时只返回最新有效版本的内存存储。"""

    def __init__(self) -> None:
        self._versions: dict[tuple[str, str, str], list[MemoryRecord]] = {}

    def write(self, record: MemoryRecord) -> None:
        family_key = (record.tenant_id, record.user_id, record.subject_key)
        versions = self._versions.setdefault(family_key, [])
        if versions:
            latest = versions[-1]
            if record.version != latest.version + 1:
                raise ValueError("新版本必须严格递增 1")
        elif record.version != 1:
            raise ValueError("新记忆的首个版本必须为 1")
        versions.append(record)

    def retrieve(self, query: RetrievalQuery) -> list[RetrievedMemory]:
        if query.now.tzinfo is None:
            raise ValueError("query.now 必须包含时区")
        if query.limit < 1:
            return []

        query_tokens = _tokens(query.text)
        candidates: list[RetrievedMemory] = []
        for versions in self._versions.values():
            record = versions[-1]
            if record.tenant_id != query.tenant_id or record.user_id != query.user_id:
                continue
            if record.deleted or record.sensitivity not in query.allowed_sensitivity:
                continue
            if record.expires_at is not None and record.expires_at <= query.now:
                continue
            if record.confidence < query.min_confidence:
                continue

            record_tokens = _tokens(record.content + " " + " ".join(record.tags))
            overlap = len(query_tokens & record_tokens)
            # confidence 只是排序因素，不能把低可信记忆变成事实。
            score = float(overlap) + record.confidence
            if overlap > 0 or not query_tokens:
                candidates.append(RetrievedMemory(record=record, score=score))

        candidates.sort(
            key=lambda item: (item.score, item.record.created_at, item.record.version),
            reverse=True,
        )
        return candidates[: query.limit]


@dataclass(frozen=True)
class ContextItem:
    kind: str
    content: str
    priority: int
    estimated_tokens: int
    evidence_ref: str | None = None


@dataclass(frozen=True)
class ContextBundle:
    items: tuple[ContextItem, ...]
    used_tokens: int
    dropped_refs: tuple[str, ...]
    rejected_memories: Mapping[str, str]


@dataclass(frozen=True)
class HistoryWindow:
    """按时间连续保留的最近消息窗口。"""

    messages: tuple[str, ...]
    used_tokens: int


class SlidingWindowHistoryBuilder:
    """从最新消息向前填充预算；用于展示仅靠近期历史的召回盲区。"""

    def build(self, messages: Sequence[str], token_budget: int) -> HistoryWindow:
        if token_budget < 1:
            raise ValueError("token_budget 必须大于 0")

        selected_reversed: list[str] = []
        used = 0
        for message in reversed(messages):
            message_tokens = _estimate_tokens(message)
            if used + message_tokens > token_budget:
                # 滑动窗口保持时间连续，不跳过中间大消息去捞更早内容。
                break
            selected_reversed.append(message)
            used += message_tokens

        return HistoryWindow(
            messages=tuple(reversed(selected_reversed)),
            used_tokens=used,
        )


class ContextBudgetError(RuntimeError):
    """目标或权威 TaskState 无法放入预算时阻止调用模型。"""


MANDATORY_CONTEXT_KINDS = frozenset({"GOAL", "AUTHORITATIVE_TASK_STATE"})


_INJECTION_PATTERNS = (
    re.compile(r"ignore (all |the )?(previous|system) instructions", re.I),
    re.compile(r"忽略.{0,8}(之前|系统).{0,8}(指令|提示)"),
    re.compile(r"泄露.{0,8}(密钥|密码|token)", re.I),
)


def _estimate_tokens(text: str) -> int:
    """保守的教学估算，不代替具体模型 tokenizer。"""

    return max(1, (len(text) + 3) // 4)


def _looks_like_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


class ContextBuilder:
    """把权威状态、近期 Trace 和候选记忆组装为受预算约束的模型 Context。"""

    def build(
        self,
        *,
        goal: str,
        task_state: Mapping[str, object],
        recent_trace: Sequence[Mapping[str, object]],
        retrieved_memories: Iterable[RetrievedMemory],
        token_budget: int,
    ) -> ContextBundle:
        if token_budget < 1:
            raise ValueError("token_budget 必须大于 0")

        candidates: list[ContextItem] = [
            self._item("GOAL", goal, priority=100),
            self._item(
                "AUTHORITATIVE_TASK_STATE",
                json.dumps(task_state, ensure_ascii=False, sort_keys=True, default=str),
                priority=95,
            ),
        ]
        for index, event in enumerate(reversed(recent_trace)):
            candidates.append(
                self._item(
                    "RECENT_TRACE",
                    json.dumps(event, ensure_ascii=False, sort_keys=True, default=str),
                    priority=max(70, 85 - index),
                    evidence_ref=str(event.get("event_id", f"trace-{index}")),
                )
            )

        rejected: dict[str, str] = {}
        for item in retrieved_memories:
            record = item.record
            if _looks_like_prompt_injection(record.content):
                rejected[record.memory_id] = "疑似 Prompt Injection，隔离而不进入 Context"
                continue
            if (
                record.memory_type is MemoryType.PROCEDURAL
                and record.source_trust not in {SourceTrust.SYSTEM, SourceTrust.USER_CONFIRMED}
            ):
                rejected[record.memory_id] = "流程记忆来源不可信，不能作为操作指令"
                continue

            # 明确标记为非权威候选证据，避免模型把召回内容误当 Runtime 状态。
            memory_text = (
                f"[候选记忆，不得覆盖 TaskState] {record.content} "
                f"(version={record.version}, confidence={record.confidence:.2f})"
            )
            candidates.append(
                self._item(
                    "RETRIEVED_MEMORY",
                    memory_text,
                    priority=50 + min(20, int(item.score * 2)),
                    evidence_ref=f"memory:{record.memory_id}:v{record.version}",
                )
            )

        mandatory = [
            item for item in candidates if item.kind in MANDATORY_CONTEXT_KINDS
        ]
        optional = [
            item for item in candidates if item.kind not in MANDATORY_CONTEXT_KINDS
        ]
        mandatory_tokens = sum(item.estimated_tokens for item in mandatory)
        if mandatory_tokens > token_budget:
            # 缺少目标或权威状态时继续请求模型会产生安全歧义。生产实现可先按白名单
            # 对 TaskState 做确定性裁剪；裁剪后仍超预算则应失败关闭。
            raise ContextBudgetError(
                "目标和权威 TaskState 超出 token 预算，不能静默丢弃后继续调用模型"
            )

        selected: list[ContextItem] = list(mandatory)
        dropped: list[str] = []
        used = mandatory_tokens
        optional.sort(key=lambda item: item.priority, reverse=True)
        for item in optional:
            if used + item.estimated_tokens <= token_budget:
                selected.append(item)
                used += item.estimated_tokens
            else:
                dropped.append(item.evidence_ref or item.kind)

        return ContextBundle(
            items=tuple(selected),
            used_tokens=used,
            dropped_refs=tuple(dropped),
            rejected_memories=rejected,
        )

    @staticmethod
    def _item(
        kind: str,
        content: str,
        *,
        priority: int,
        evidence_ref: str | None = None,
    ) -> ContextItem:
        return ContextItem(
            kind=kind,
            content=content,
            priority=priority,
            estimated_tokens=_estimate_tokens(content),
            evidence_ref=evidence_ref,
        )


def utc_datetime(year: int, month: int, day: int) -> datetime:
    """让教学测试更易读的 UTC 时间构造器。"""

    return datetime(year, month, day, tzinfo=timezone.utc)
