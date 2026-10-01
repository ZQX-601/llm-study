"""训练平台 MCP 的输入、输出与领域 Schema。

Pydantic 类型可被 MCP SDK 用于生成 JSON Schema。Schema 只负责结构合同；任务是否
存在、用户是否有权限、状态是否允许取消等语义规则仍由 Server/后端校验。
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class JobState(str, Enum):
    """训练任务的有限状态集合。"""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"


class OperationStatus(str, Enum):
    """写操作的权威执行状态。"""

    NOT_FOUND = "NOT_FOUND"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    FAILED_FINAL = "FAILED_FINAL"
    UNKNOWN = "UNKNOWN"


class ErrorInfo(BaseModel):
    """提供给 Host 的稳定错误合同，而不是泄露后端异常栈。"""

    code: str
    message: str
    retryable: bool = False
    result_unknown: bool = False


class JobView(BaseModel):
    """允许返回给 Agent 的任务视图，敏感字段应在后端或适配层脱敏。"""

    job_id: str = Field(pattern=r"^train-[0-9]+$")
    state: JobState
    progress: float = Field(ge=0.0, le=1.0)
    version: int = Field(ge=1)
    owner_id: str
    latest_error: str | None = None
    updated_at: datetime


class GetJobOutput(BaseModel):
    ok: bool
    job: JobView | None = None
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def validate_branch(self) -> "GetJobOutput":
        """成功必须有 job，失败必须有 error，避免含糊的半结构化结果。"""
        if self.ok == (self.job is None):
            raise ValueError("ok=true 时必须有 job，ok=false 时 job 必须为空")
        if self.ok == (self.error is not None):
            raise ValueError("ok=true 时不能有 error，ok=false 时必须有 error")
        return self


class PrepareCancelInput(BaseModel):
    job_id: str = Field(pattern=r"^train-[0-9]+$", description="训练任务 ID，例如 train-2048")
    expected_version: int = Field(ge=1, description="用户刚刚看到的任务版本，用于检测并发状态变化")
    reason: str = Field(min_length=5, max_length=500, description="用户提供的取消原因")


class CancelProposal(BaseModel):
    proposal_id: str
    job_id: str
    expected_version: int
    current_state: JobState
    summary: str
    expires_at: datetime


class PrepareCancelOutput(BaseModel):
    ok: bool
    proposal: CancelProposal | None = None
    error: ErrorInfo | None = None


class CommitCancelInput(BaseModel):
    proposal_id: str = Field(description="prepare 工具返回的不可执行 Proposal ID")
    confirmation_id: str = Field(min_length=16, description="Host 在用户确认具体 Proposal 后签发的确认凭据")
    idempotency_key: str = Field(
        min_length=16,
        max_length=128,
        description="同一逻辑操作的重试必须复用同一个幂等键",
    )


class CommitCancelOutput(BaseModel):
    ok: bool
    job_id: str | None = None
    final_state: JobState | None = None
    operation_id: str | None = None
    error: ErrorInfo | None = None


class CancelOperationResult(BaseModel):
    """训练平台 POST /cancel 的规范化内部结果。"""

    ok: bool
    operation_id: str | None = None
    error_code: str | None = None
    retryable: bool = False
    result_unknown: bool = False


class CancelOperationView(BaseModel):
    """通过幂等键查询到的权威取消操作记录。"""

    idempotency_key: str
    status: OperationStatus
    operation_id: str | None = None
    job_id: str | None = None
    updated_at: datetime | None = None
    error_code: str | None = None


class GetCancelOperationOutput(BaseModel):
    ok: bool
    operation: CancelOperationView | None = None
    error: ErrorInfo | None = None
