# DeepSeek V4.1 Flash：SWA 逐步拆解

- 日期：2026-09-25
- 类型：论文答疑，接续 V4.1 输入到输出导读
- 用户问题：从 SWA 开始，按流程逐步拆解。
- 当前进度：已系统讲解标准因果 SWA 的动机、掩码、计算、训练/解码、复杂度、跨层传播及与 V4.1 扩展的边界。QA 不设置检查题，也不记录掌握结论。
- 重要性：SWA 的掩码和 KV 缓存属于岗位必备；V4.1 的 Bounded Replay 属于需要掌握的系统设计案例。

## 学习路径

1. 一个 token 的可见窗口与掩码。
2. 窗口内 Q/K/V、分数、softmax 与输出。
3. 增量解码的 KV 环形缓存与计算成本。
4. 多层 SWA 的信息传播和状态依赖。
5. 精确回放与 Bounded Replay 的近似取舍。
6. 回到 V4.1：SWA 与 CSA2 全局分支如何配合，读代码并做最小复现。

## 第 1 步：直接可见范围

对位置 `i`，窗口宽度 `W` 包含当前位置，单层因果 SWA 只允许注意到 `[max(0, i-W+1), i]`。序列 `A B C D E`、`W=3` 时，各位置的直接可见集合为：

| 当前 token | 可直接看见 |
| --- | --- |
| A | A |
| B | A B |
| C | A B C |
| D | B C D |
| E | C D E |

因此 E 在本层不会直接与 A、B 的 key 算注意力分数。这不等于跨层后完全没有早期信息；间接传播留到第 4 步。Full attention 对长度 `T` 的一层有约 `T²` 个注意力位置对，固定宽度 SWA 约为 `T×W`。DeepSeek-V4.1-Flash 报告使用 `W=128`；这里的简单 SWA 不包含 CSA2 的全局稀疏分支，也不是 Bounded Replay。

## 第 2 步：窗口内的注意力计算

每个位置的输入表示分别投影为 Q、K、V。以 E 为例，先取当前层 E 的 query，仅与本层窗口 `C,D,E` 的 key 计算缩放点积；被窗口排除的 A、B 不参与本层 E 的 softmax。三个分数经过 softmax 成为权重，再加权求和三个 value，得到 E 的 attention 输出。多头注意力会在各头分别做这一过程，再合并输出。窗口改变的是允许参与注意力的 K/V 位置，而不是取消 Q/K/V 投影。

在增量解码中，新 token 的 query 只需要最近 `W` 个位置的 K/V；连续生成时，最旧的局部 K/V 可以从固定长度的缓存中移出。局部缓存界约随 `W` 而非全部历史长度增长，但 V4.1 还有独立的全局 CSA2 路径，不能把整个模型的上下文能力或总 KV 成本等同于纯 SWA。

## 系统解释与工程边界

- **掩码与公式**：一层内第 `i` 个位置只让 `j` 落在 `max(0,i-W+1) <= j <= i` 的 K/V 参与 `softmax(q_i k_j^T/sqrt(d_head))`，随后对这些 `v_j` 加权求和。窗口外等价于分数为负无穷。训练时整个序列可并行计算；增量解码时一次计算新位置的 Q/K/V，并把新 K/V 写入各层长度为 `W` 的局部环形缓存。
- **复杂度**：仅就注意力分数和加权求和而言，完整因果注意力的时间/权重矩阵为 `O(T^2)`，固定窗口为 `O(TW)`；解码每层局部 KV 缓存为 `O(W*d_kv)`，并不会使投影和 FFN/MoE 成本消失。朴素地构造完整 `T×T` 掩码不自动节省显存或计算，需要局部/稀疏 kernel 或等价索引实现。
- **跨层有效感受野**：若每层都只用宽 `W` 的因果 SWA，且没有其他全局混合路径，则 `L` 层后当前位置在理论上可间接受至多 `1+L(W-1)` 个最近原始位置影响（边界处截断）。这只是可能的依赖路径，不保证远处信息能被无损保留或有效检索。
- **V4.1 的扩展**：报告中局部 SWA 窗口为 128，每层有自己的 SWA KV；多数层另有 CSA2 全局稀疏分支，所以整个模型不受纯 SWA 理论感受野限制。SWA Bounded Replay 是部署时丢失局部 KV 后只回放最近一个窗口的近似状态重建，不是 SWA 公式本身；精确重建因跨层依赖要回放约 `L×W` 个 token。论文报告的质量影响是作者测得结果，仓库未复现。
- **代码入口**：官方 `get_window_topk_idxs` 在 prefill 时为每个 query 建立自己的因果窗口索引，在 decode 时从环形缓存取窗口位置；`-1` 标记未填充槽位。读代码时注意窗口索引是位置选择，不是生成 Q/K/V 的全部逻辑。
- **工作用途**：选择窗口大小时要同时评估长距离引用任务、prefill 吞吐、decode 延迟、局部 KV 容量和缓存恢复边界；有全局分支时还要单独核算全局 KV 和检索成本。

## 4×4 prefill 手算与层的含义

用户追问：prefill 怎样计算，能否演算 `4×4` 矩阵；同层并行如何实现；第一层、第二层是什么。

为便于手算，仅设一个注意力头、`d_head=1`、`W=2`，输入位置为 `A,B,C,D`。假设本层投影得到 `Q=[1,1,1,1]^T`、`K=[0,ln2,0,ln2]^T`、`V=[10,20,30,40]^T`。这是构造的教学数值，不是 DeepSeek 的真实权重；缩放因子 `1/sqrt(d_head)=1`。未掩码的 `QK^T` 四行相同，均为 `[0,ln2,0,ln2]`。

添加因果窗口掩码后，分数矩阵的四行分别为 `[0,-inf,-inf,-inf]`、`[0,ln2,-inf,-inf]`、`[-inf,ln2,0,-inf]`、`[-inf,-inf,0,ln2]`。逐行 softmax 得到权重：`[1,0,0,0]`、`[1/3,2/3,0,0]`、`[0,2/3,1/3,0]`、`[0,0,1/3,2/3]`。矩阵乘 `V` 后输出 `[10,50/3,70/3,110/3]^T`，约 `[10,16.67,23.33,36.67]^T`。

同层并行：整个输入先同时投影到所有位置的 Q/K/V；矩阵乘法一次形成所有位置的得分（实际高效实现可只形成局部块），各行掩码、softmax、加权求和可并行。因果性由掩码保证，不需要按 A→B→C→D 顺序运行同一层。层间不能这样并行：`H^0` 是 embedding，第一层由 `H^0` 生成自己的 Q/K/V 并输出 `H^1`，第二层再由 `H^1` 用**另一组参数**生成 Q/K/V 并输出 `H^2`。残差、归一化、FFN/MoE 等细节在手算例中省略。以 `W=2` 为例，第二层 D 可读取第一层 C，而第一层 C 已读 B，所以原始 B 可间接影响第二层 D；第二层不能沿用第一层的 Q/K/V。

完整 `4×4` 密集矩阵是教学演示；实际长序列的局部/稀疏实现必须避免构造大量窗口外分数，否则掩码本身不产生计算节约。标准 prefill 是同层并行、逐层推进；最终 logits 通常从最后位置取，服务时再保留所需的 KV 缓存。官方 `get_window_topk_idxs` 为 prefill 的每个 query 选择自己的因果窗口，印证这种逐行可见性。

## 多头张量形状与教学代码

用户追问：若 `X:[B,T,d]`，多头 SWA 的每一步形状如何变化，要求结合代码。

- 最初用 `B=2,T=4,d=8,H=2,Dh=4,W=2` 的密集矩阵脚本说明基础形状；用户指出它过于简略，因此已将代码升级并集中到 [coding/deepseek_demo](../coding/deepseek_demo/README.md)。新版取 `B=2,T=4,d=16,H=4,Dh=8,W=2`，增加低秩 Q、RoPE、共享局部 KV、索引式局部聚合、attention sink、分组输出投影、两层及环形缓存单步 decode。源码映射与每步形状见 [SOURCE_MAP.md](../coding/deepseek_demo/SOURCE_MAP.md)。
- 此教学示例不是常规每头独立 K/V 的 MHA。常规 MHA 的 K/V 一般为 `[B,H,T,Dh]`；DeepSeek V4.1 参考代码则是多 Q 头而共享局部 latent KV，局部缓存 `[B,W,Dh]`。报告配置 `d=5120,H=64,Dh=512,W=128`；真实 Q 先低秩投影，KV 经 norm/RoPE/量化，局部窗口与压缩全局 KV 合并送入 `sparse_attn`，输出有分组低秩投影及其他机制。教学代码省略这些细节，也采用密集 `T×T` 分数矩阵，仅说明形状，不代表高效 SWA kernel。
- 官方入口：`Attention.forward`、`Attention._window_kv`、`get_window_topk_idxs`，见 [model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)；`sparse_attn` 见 [kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py)。旧脚本尝试运行时默认环境因缺少 PyTorch 在 `import torch` 处失败。新目录中的 Python 文件已通过 AST 语法检查，但单元测试因同一依赖缺失未执行；不要写成运行验证通过。
- 用户后续反馈：源码 demo 内的注释还需更详细，直接说明每一步的计算目的与 shape 变化。已在 `coding/deepseek_demo/swa_attention.py` 中按六个前向阶段补充注释，并为 RoPE、窗口索引、稀疏 gather/softmax、prefill/decode 缓存分别加输入输出形状说明；`run_demo.py` 和测试也标注了运行路径与验证意图。今后源码分析默认采用此粒度；本轮仍只做语法检查，未因补注释而声称执行通过。
- 用户明确要求今后的学习文档与 Python demo 中的说明性文字使用中文。已将 `coding/deepseek_demo/` 的两份文档、三个 Python 文件中的注释、文档字符串及运行/报错提示改成中文，保留代码标识符与必要的技术术语；已更新 `AGENTS.md` 为后续规则。翻译后三个 Python 文件再次通过 AST 语法检查，运行测试仍因缺少 PyTorch 而未验证。

## 一手资料

- [DeepSeek-V4.1-Flash 技术报告](https://arxiv.org/html/2609.19969v1)：SWA 窗口、SWA Bounded Replay。
- [官方参考实现 model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)：`get_window_topk_idxs` 的 prefill 窗口索引。
