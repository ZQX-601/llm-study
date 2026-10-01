"""有界核对状态机的标准库测试，不需要连接 MCP Server。"""

import unittest

from retry_state_machine import (
    ActionStatus,
    PendingAction,
    RemoteOperationStatus,
    reconcile_result_unknown,
)


async def no_wait(_: float) -> None:
    """测试中跳过真实退避等待。"""


def make_action() -> PendingAction:
    return PendingAction(
        tool_name="commit_cancel_training_job",
        arguments={"proposal_id": "proposal-1"},
        request_hash="hash-of-stable-business-arguments",
        idempotency_key="cancel-train-2048-v7",
        status=ActionStatus.RESULT_UNKNOWN,
        submit_attempts=1,
    )


class RetryStateMachineTest(unittest.IsolatedAsyncioTestCase):
    async def test_processing_then_succeeded(self) -> None:
        statuses = iter(
            [RemoteOperationStatus.PROCESSING, RemoteOperationStatus.SUCCEEDED]
        )
        seen_keys: list[str] = []

        async def query(key: str) -> RemoteOperationStatus:
            seen_keys.append(key)
            return next(statuses)

        action = make_action()
        final_status = await reconcile_result_unknown(
            action,
            query_operation=query,
            backoff_seconds=(0, 0, 0),
            sleep=no_wait,
        )

        self.assertEqual(final_status, ActionStatus.SUCCEEDED)
        self.assertEqual(action.status_checks, 2)
        self.assertEqual(action.submit_attempts, 1)
        self.assertEqual(seen_keys, [action.idempotency_key, action.idempotency_key])

    async def test_unknown_until_limit_escalates(self) -> None:
        async def query(_: str) -> RemoteOperationStatus:
            return RemoteOperationStatus.UNKNOWN

        action = make_action()
        final_status = await reconcile_result_unknown(
            action,
            query_operation=query,
            backoff_seconds=(0, 0, 0, 0, 0),
            sleep=no_wait,
        )

        self.assertEqual(final_status, ActionStatus.ESCALATED)
        self.assertEqual(action.status_checks, 5)
        self.assertEqual(action.submit_attempts, 1)

    async def test_final_failure_stops_immediately(self) -> None:
        async def query(_: str) -> RemoteOperationStatus:
            return RemoteOperationStatus.FAILED_FINAL

        action = make_action()
        final_status = await reconcile_result_unknown(
            action,
            query_operation=query,
            backoff_seconds=(0, 0, 0),
            sleep=no_wait,
        )

        self.assertEqual(final_status, ActionStatus.FAILED)
        self.assertEqual(action.status_checks, 1)


if __name__ == "__main__":
    unittest.main()
