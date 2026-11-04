"""Actor/Learner 权重同步的确定性教学状态机。

真实系统会接对象存储、分布式 checkpoint 和推理引擎热加载。这里不搬运模型参数，只验证
三个不能混写的版本：Coordinator 期望版本、Actor 已加载版本、当前轨迹锁定版本。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SnapshotStatus(str, Enum):
    WRITING = "writing"
    READY = "ready"


class RefreshStatus(str, Enum):
    READY = "ready"
    PENDING = "pending"
    FAILED = "failed"


@dataclass(frozen=True)
class PolicySnapshotManifest:
    """Actor 原子加载前必须整体校验的不可变 Snapshot 身份。"""

    policy_version: int
    status: SnapshotStatus
    weights_hash: str
    base_model_hash: str
    adapter_hash: str | None
    tokenizer_version: str
    chat_template_version: str


@dataclass
class ActorPolicyState:
    """区分期望、实际加载和当前 trajectory 使用的 Policy 版本。"""

    desired_policy_version: int
    loaded_policy_version: int
    base_model_hash: str
    tokenizer_version: str
    chat_template_version: str
    active_trajectory_policy_version: int | None = None
    refresh_status: RefreshStatus = RefreshStatus.READY

    def request_version(self, policy_version: int) -> None:
        """Learner/Coordinator 发布新目标；只标记期望，不等于已经加载。"""

        if policy_version < self.desired_policy_version:
            raise ValueError("不能把 desired policy 回退到更旧版本")
        self.desired_policy_version = policy_version
        if policy_version != self.loaded_policy_version:
            self.refresh_status = RefreshStatus.PENDING

    def install_snapshot(self, manifest: PolicySnapshotManifest) -> None:
        """只在空闲边界原子切换；任一合同不匹配都保留旧 loaded version。"""

        if self.active_trajectory_policy_version is not None:
            raise RuntimeError("trajectory 执行中不能热切换 Policy")
        if manifest.status is not SnapshotStatus.READY:
            self.refresh_status = RefreshStatus.FAILED
            raise ValueError("Snapshot 尚未 READY")
        if manifest.policy_version != self.desired_policy_version:
            self.refresh_status = RefreshStatus.FAILED
            raise ValueError("Snapshot 版本不是 Actor 当前 desired version")
        if manifest.base_model_hash != self.base_model_hash:
            self.refresh_status = RefreshStatus.FAILED
            raise ValueError("LoRA Adapter 与 Actor Base Model 不兼容")
        if manifest.tokenizer_version != self.tokenizer_version:
            self.refresh_status = RefreshStatus.FAILED
            raise ValueError("Tokenizer 版本不兼容")
        if manifest.chat_template_version != self.chat_template_version:
            self.refresh_status = RefreshStatus.FAILED
            raise ValueError("Chat Template 版本不兼容")
        if not manifest.weights_hash:
            self.refresh_status = RefreshStatus.FAILED
            raise ValueError("Snapshot 缺少 weights hash")

        # 生产环境应先在候选模型实例完成加载/hash/健康检查，再原子替换 serving handle。
        self.loaded_policy_version = manifest.policy_version
        self.refresh_status = RefreshStatus.READY

    def begin_trajectory(self, *, required_policy_version: int) -> int:
        """在 trajectory 边界锁定实际 Policy；不允许把 desired 冒充 loaded。"""

        if self.active_trajectory_policy_version is not None:
            raise RuntimeError("Actor 已有正在执行的 trajectory")
        if self.loaded_policy_version != required_policy_version:
            raise RuntimeError(
                "Actor 实际 loaded version 不满足 Coordinator 的任务要求"
            )
        self.active_trajectory_policy_version = self.loaded_policy_version
        return self.active_trajectory_policy_version

    def finish_trajectory(self) -> int:
        """返回轨迹必须记录的真实 behavior version，并释放版本锁。"""

        if self.active_trajectory_policy_version is None:
            raise RuntimeError("Actor 当前没有正在执行的 trajectory")
        completed_version = self.active_trajectory_policy_version
        self.active_trajectory_policy_version = None
        return completed_version
