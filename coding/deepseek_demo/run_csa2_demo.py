"""运行 CSA2 的小尺寸 prefill 演示：python run_csa2_demo.py"""

import torch

from csa2_attention import CSA2Config, SharedCSA2State, build_demo_stack


def shape(tensor: torch.Tensor | None) -> str:
    return "None" if tensor is None else str(tuple(tensor.shape))


def main() -> None:
    torch.manual_seed(7)
    config = CSA2Config()
    x = torch.randn(1, 8, 16)  # B=1,T=8,d=16；m=2，所以 Tc=4。
    shared = SharedCSA2State()

    print("输入 X:", shape(x))
    print("层序列: Full -> Reuse -> Reindex -> Reuse\n")
    for layer_id, layer in enumerate(build_demo_stack(config)):
        x, shared, trace = layer.forward_prefill(x, shared)
        print(f"第 {layer_id} 层，模式={layer.mode}")
        print("  Q:                ", shape(trace["q"]))
        print("  本层局部 KV:      ", shape(trace["local_kv"]))
        print("  跨层共享主 KV:    ", shape(trace["main_kv"]))
        print("  跨层共享 Index K: ", shape(trace["index_k"]))
        print("  局部窗口索引:      ", shape(trace["local_indices"]))
        print("  全局 Top-K 索引:   ", shape(trace["global_indices"]))
        print("  合并稀疏索引:      ", shape(trace["all_indices"]))
        print("  稀疏注意力权重:     ", shape(trace["attention_weights"]), "(B,T,H,W+K)")
        print("  层输出:            ", shape(x))
        print("  token 7 全局索引:  ", trace["global_indices"][0, 7].tolist())
        if "compressor_weights" in trace:
            print("  压缩门控权重:      ", shape(trace["compressor_weights"]), "(B,Tc,m,Dh)")
        print("  本层运行 Indexer:  ", "是" if "index_scores" in trace else "否，复用 Top-K")
        print()


if __name__ == "__main__":
    main()
