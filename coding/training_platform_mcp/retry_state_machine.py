"""Host 侧 RESULT_UNKNOWN 有界核对状态机。

该模块不依赖具体模型或 MCP SDK。真实 Host 只需把 query_operation 适配为调用
`get_cancel_operation` MCP Tool 的异步函数即可。
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import Enum


class ActionStatus(str, Enum):
    PENDING = "PENDING"
    IN_FLIGHT = "IN_FLIGHT"
    RESULT_UNKNOWN = "RESULT_UNKNOWN"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    ESCALATED = "ESCALATED"


class RemoteOperationStatus(str, Enum):
    NOT_FOUND = "NOT_FOUND"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    FAILED_FINAL = "FAILED_FINAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class PendingAction:
    """Host 必须持久化的最小写操作状态，而不是只留在模型上下文中。"""

    tool_name: str
    arguments: dict[str, object]
    request_hash: str
    idempotency_key: str
    status: ActionStatus = ActionStatus.PENDING
    submit_attempts: int = 0
    status_checks: int = 0


QueryOperation = Callable[[str], Awaitable[RemoteOperationStatus]]
Sleep = Callable[[float], Awaitable[None]]


async def reconcile_result_unknown(
    action: PendingAction,
    *,
    query_operation: QueryOperation,
    backoff_seconds: Sequence[float] = (1, 2, 4, 8, 16),
    sleep: Sleep = asyncio.sleep,
) -> ActionStatus:
    """只查询原操作，不在状态未知时自动重新提交写操作。

    输入：处于 RESULT_UNKNOWN 或 PROCESSING 的 PendingAction。
    输出：SUCCEEDED、FAILED 或 ESCALATED；每次查询都复用原 idempotency_key。
    """
    if action.status not in {ActionStatus.RESULT_UNKNOWN, ActionStatus.PROCESSING}:
        raise ValueError("只有结果未知或仍在处理的动作可以进入核对状态机")

    for delay in backoff_seconds:
        await sleep(delay)
        remote_status = await query_operation(action.idempotency_key)
        action.status_checks += 1

        if remote_status == RemoteOperationStatus.SUCCEEDED:
            action.status = ActionStatus.SUCCEEDED
            return action.status

        if remote_status == RemoteOperationStatus.FAILED_FINAL:
            action.status = ActionStatus.FAILED
            return action.status

        if remote_status == RemoteOperationStatus.PROCESSING:
            action.status = ActionStatus.PROCESSING
            continue

        # UNKNOWN 继续有界查询。NOT_FOUND 默认也不自动重提；只有后端提供
        # “未接收且未来不会异步执行”的强合同后，才能在更高层复用原 key 重提。
        action.status = ActionStatus.RESULT_UNKNOWN

    action.status = ActionStatus.ESCALATED
    return action.status
