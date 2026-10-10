# ReAct vs Plan-and-Execute 成对基准结果

## 实验约束

- 环境：macOS、Python 3.13.15、纯标准库确定性 mock；
- 两种模式使用同一任务输入、Tool 语义、权限和整次任务预算；
- 默认总预算：8 个 Action、4 次 Tool、1 次 Replan，同一 Action 最多执行 2 次；
- Plan Runtime 同时保留 Step 内上限，但总上限不会随 Step 数增加；
- 不接真实 LLM/MCP，因此 Token、输出 Token 和费用遥测为 `None`，不伪造为 0；
- 时延只是单次本地教学脚本运行值，用于验证采集链路，不用于性能显著性结论。

## 一次实际运行结果

| Case | Mode | 成功 | Evidence | 安全 | Action | Tool | Replan | 时延 ms |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 单工具查询 | ReAct | 是 | 完整 | 通过 | 2 | 1 | 0 | 0.116 |
| 单工具查询 | Plan-and-Execute | 是 | 完整 | 通过 | 2 | 1 | 0 | 0.838 |
| 多步延迟诊断 | ReAct | 是 | 完整 | 通过 | 3 | 2 | 0 | 0.088 |
| 多步延迟诊断 | Plan-and-Execute | 是 | 完整 | 通过 | 4 | 2 | 0 | 0.597 |
| 可恢复超时 | ReAct | 是 | 完整 | 通过 | 3 | 2 | 0 | 0.072 |
| 可恢复超时 | Plan-and-Execute | 是 | 完整 | 通过 | 3 | 2 | 0 | 0.468 |
| 目标订单变化 | ReAct | 是 | 完整 | 通过 | 3 | 2 | 0 | 0.068 |
| 目标订单变化 | Plan-and-Execute | 是 | 完整 | 通过 | 3 | 2 | 1 | 0.581 |

## 结论

硬门槛方面，四组任务的两种模式都成功，Evidence 完整且没有安全违规。效率方面：

- 单工具查询中，两者 Action/Tool 数相同，但 Plan 的建计划、校验和 Trace 管理有固定时延；
- 多步诊断中，Plan 每个 Step 都要显式完成验证，因此比 ReAct 多 1 个 Action；
- 可恢复超时中，两边都真实调用 2 次 Tool，证明瞬时失败没有被误记为成功；
- 目标变化中，两边都能到达新订单，但只有 Plan Trace 显式记录旧 Step 失效、Evidence
  重验证和 `plan_revised`，更适合恢复和失败归因。

不能据此宣布某一种模式全面胜出：mock 时延远小于真实模型推理，而且本实验没有观测
Token 与费用。当前可执行路由规则是：

1. 单步、低风险、目标稳定的查询优先 ReAct；
2. 多步骤、存在依赖或恢复点、目标可能变化的任务优先 Plan-and-Execute；
3. 高风险副作用不由模式选择替代安全门，仍须 Runtime 确认、幂等和权限校验；
4. 任何模式超过全局 Action/Tool/Deadline 预算都应停止，不能靠增加 Step 获得额外预算。

## 复现

```bash
UV_CACHE_DIR=/tmp/llm-agent-uv-cache \
UV_PYTHON_INSTALL_DIR=/tmp/llm-agent-uv-python \
uv run --offline --python 3.13 python coding/plan_execute_agent/paired_benchmark.py
```

实际时延会随机器负载变化；硬门槛、Action、Tool 和 Replan 计数应保持一致。
