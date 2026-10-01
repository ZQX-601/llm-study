"""已有训练平台 REST API 的异步适配层。"""

import json
from typing import Any

import httpx

from schemas import CancelOperationResult, CancelOperationView, JobView, OperationStatus


class TrainingPlatformClient:
    """把 HTTP 状态、超时和响应 JSON 归一化为领域对象。"""

    def __init__(self, *, base_url: str, service_token: str) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {service_token}"},
            timeout=httpx.Timeout(10.0),
        )

    async def get_job(self, *, job_id: str, tenant_id: str) -> JobView | None:
        response = await self._http.get(
            f"/jobs/{job_id}",
            headers={"X-Tenant-ID": tenant_id},
        )
        if response.status_code in {403, 404}:
            # 对 Agent 统一返回不可访问，避免枚举其他租户的任务 ID。
            return None
        response.raise_for_status()
        return JobView.model_validate(response.json())

    async def cancel_job(
        self,
        *,
        job_id: str,
        expected_version: int,
        idempotency_key: str,
        actor_id: str,
        tenant_id: str,
        reason: str,
    ) -> CancelOperationResult:
        try:
            response = await self._http.post(
                f"/jobs/{job_id}/cancel",
                json={"expected_version": expected_version, "reason": reason},
                headers={
                    "Idempotency-Key": idempotency_key,
                    "X-Actor-ID": actor_id,
                    "X-Tenant-ID": tenant_id,
                },
            )
        except httpx.TimeoutException:
            # POST 超时不能断言取消失败；服务端可能已经完成但响应丢失。
            return CancelOperationResult(
                ok=False,
                error_code="RESULT_UNKNOWN",
                retryable=False,
                result_unknown=True,
            )

        if response.status_code == 409:
            return CancelOperationResult(ok=False, error_code="JOB_VERSION_CONFLICT")
        if response.status_code in {403, 404}:
            return CancelOperationResult(ok=False, error_code="JOB_NOT_ACCESSIBLE")

        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return CancelOperationResult(ok=True, operation_id=str(payload["operation_id"]))

    async def get_cancel_operation(
        self,
        *,
        idempotency_key: str,
        actor_id: str,
        tenant_id: str,
    ) -> CancelOperationView | None:
        """按用户、租户和幂等键查询权威执行记录，不产生业务副作用。"""
        response = await self._http.get(
            f"/operations/by-idempotency-key/{idempotency_key}",
            headers={
                "X-Actor-ID": actor_id,
                "X-Tenant-ID": tenant_id,
            },
        )

        if response.status_code == 403:
            return None
        if response.status_code == 404:
            # 只有后端明确承诺 404 代表当前身份作用域内从未接收该 key，
            # Host 才能把 NOT_FOUND 用作复用原 key 重试的依据。
            return CancelOperationView(
                idempotency_key=idempotency_key,
                status=OperationStatus.NOT_FOUND,
            )

        response.raise_for_status()
        return CancelOperationView.model_validate(response.json())

    async def get_job_logs(self, *, job_id: str, tenant_id: str) -> str | None:
        """读取任务日志 Resource；服务端限制大小，避免一次响应无限增长。"""
        response = await self._http.get(
            f"/jobs/{job_id}/logs",
            headers={"X-Tenant-ID": tenant_id},
        )
        if response.status_code in {403, 404}:
            return None
        response.raise_for_status()

        max_characters = 256_000
        text = response.text
        if len(text) <= max_characters:
            return text
        return text[:max_characters] + "\n\n[内容已由 MCP Server 截断，请使用日志搜索 Tool 定向查询]"

    async def get_job_config(self, *, job_id: str, tenant_id: str) -> str | None:
        """以 JSON 文本返回配置 Resource，供模型阅读且保留稳定字段结构。"""
        response = await self._http.get(
            f"/jobs/{job_id}/config",
            headers={"X-Tenant-ID": tenant_id},
        )
        if response.status_code in {403, 404}:
            return None
        response.raise_for_status()
        return json.dumps(response.json(), ensure_ascii=False, indent=2)

    async def close(self) -> None:
        await self._http.aclose()
