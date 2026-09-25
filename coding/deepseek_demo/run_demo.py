"""打印两层 SWA 的张量形状和局部索引。

安装 PyTorch 后，在本目录运行 `python run_demo.py`。
默认配置：B=2、T=4、dim=16、H=4、Dh=8、W=2。
"""

import torch

from swa_attention import LocalSWA, SWAConfig


def show_trace(name: str, x: torch.Tensor, trace: dict[str, torch.Tensor],
               output: torch.Tensor) -> None:
    """按 attention 前向计算的顺序显示各个中间张量的形状。"""
    print(f"\n{name}：输入 {tuple(x.shape)}")
    for key in ("qr", "q", "kv", "indices", "weights", "head_output",
                "grouped", "compressed", "cache"):
        print(f"  {key:12s} {tuple(trace[key].shape)}")
    print("  最终输出     ", tuple(output.shape))
    # prefill 的索引指向 prompt KV；decode 的索引指向环形缓存槽位。
    print("  第 0 个样本的局部索引：\n", trace["indices"][0])
    # weights:[B,T,H,K]；打印第 0 个样本、第 0 个头的全部 T 个 query。
    # 因为 sink 参与归一化，即使 K 个槽位均有效，其权重之和也可能小于 1。
    print("  第 0 个样本、第 0 个头的局部权重：\n", trace["weights"][0, :, 0])


def main() -> None:
    torch.manual_seed(7)
    config = SWAConfig()
    # 每个样本有 4 个 prompt token：[B=2,T=4,dim=16]。
    x = torch.randn(2, 4, config.dim)
    # 两个实例有不同的投影参数，也各自维护一份 KV 环形缓存。
    first, second = LocalSWA(config), LocalSWA(config)
    first.eval()
    second.eval()

    with torch.no_grad():
        # prefill：第一层内部同时处理 4 个位置；第二层必须先收到
        # 第一层的完整输出 [B,T,dim]，再用自己的参数继续计算。
        h1, trace1 = first(x, return_trace=True)
        h2, trace2 = second(h1, return_trace=True)
        show_trace("第 1 层 prefill", x, trace1, h1)
        show_trace("第 2 层 prefill", h1, trace2, h2)

        # decode：只处理绝对位置 p=4 的新 token，输入 [B,1,dim]。
        # 每层先写入各自的 p%W 缓存槽位，再计算本层的注意力。
        next_x = torch.randn(2, 1, config.dim)
        next_h1, next_trace1 = first(next_x, start_pos=4, return_trace=True)
        next_h2, next_trace2 = second(next_h1, start_pos=4, return_trace=True)
        show_trace("第 1 层 decode", next_x, next_trace1, next_h1)
        show_trace("第 2 层 decode", next_h1, next_trace2, next_h2)


if __name__ == "__main__":
    main()
