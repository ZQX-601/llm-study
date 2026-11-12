"""检查当前命令实际运行环境，不把另一台机器的 GPU 误记为本次证据。"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass


PACKAGES = ("torch", "transformers", "peft", "trl", "vllm", "ray")


@dataclass(frozen=True)
class EnvironmentReport:
    os: str
    machine: str
    python: str
    executable: str
    packages: dict[str, bool]
    nvidia_smi_available: bool
    gpu_summary: str | None


def _read_gpu_summary() -> tuple[bool, str | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return False, None

    completed = subprocess.run(
        [
            executable,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    summary = completed.stdout.strip() if completed.returncode == 0 else None
    return True, summary


def inspect_environment() -> EnvironmentReport:
    """返回可归档的环境事实；不安装依赖，也不访问网络。"""

    nvidia_smi_available, gpu_summary = _read_gpu_summary()
    return EnvironmentReport(
        os=platform.platform(),
        machine=platform.machine(),
        python=platform.python_version(),
        executable=sys.executable,
        packages={name: importlib.util.find_spec(name) is not None for name in PACKAGES},
        nvidia_smi_available=nvidia_smi_available,
        gpu_summary=gpu_summary,
    )


def main() -> None:
    report = inspect_environment()
    print("当前执行环境（不是用户其他机器的环境）：")
    print(f"- OS: {report.os}")
    print(f"- machine: {report.machine}")
    print(f"- Python: {report.python} ({report.executable})")
    print(f"- packages: {report.packages}")
    print(f"- nvidia-smi: {report.nvidia_smi_available}")
    print(f"- GPU: {report.gpu_summary or '未检测到可记录的 NVIDIA GPU'}")


if __name__ == "__main__":
    main()

