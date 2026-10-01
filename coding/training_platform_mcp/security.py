"""身份、权限、确认和审计的教学接口。

身份来自经过验证的连接上下文，不允许模型通过 Tool arguments 自报 user_id 或
tenant_id。这里使用 ContextVar 表示认证中间件向当前请求注入身份的边界。
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant_id: str
    permissions: frozenset[str]


_current_principal: ContextVar[Principal | None] = ContextVar("current_principal", default=None)


@contextmanager
def bind_principal(principal: Principal) -> Iterator[None]:
    """供认证中间件或测试绑定身份；不应由模型直接调用。"""
    token = _current_principal.set(principal)
    try:
        yield
    finally:
        _current_principal.reset(token)


def current_principal() -> Principal:
    principal = _current_principal.get()
    if principal is None:
        raise PermissionError("请求没有经过身份认证")
    return principal


def require_permission(principal: Principal, permission: str) -> None:
    if permission not in principal.permissions:
        raise PermissionError(f"缺少权限：{permission}")


class ConfirmationVerifier:
    """生产实现应验证签名、用户、Proposal 摘要、有效期和一次性使用状态。"""

    def verify(self, *, confirmation_id: str, proposal_id: str, user_id: str) -> bool:
        # 教学占位：故意拒绝，避免示例在未接入真实确认服务时执行写操作。
        del confirmation_id, proposal_id, user_id
        return False


class AuditLog:
    """审计接口占位；生产实现应写入不可随意修改的持久化存储。"""

    def write(self, **event: object) -> None:
        # 不向 stdout 打印，避免 stdio Transport 的协议流被日志污染。
        del event


confirmation_verifier = ConfirmationVerifier()
audit_log = AuditLog()
