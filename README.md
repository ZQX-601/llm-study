# 大模型算法与 Agent 工程学习项目归档

## 在仓库中使用学习教练

在此仓库中与 Codex 对话，说“开始今天的学习计划吧”会从 [当前状态](study_agent/STATE.md)、[未来 12 周路线](study_agent/ROADMAP.md) 和既有归档接续主线，原理与代码默认各约一半；说“将今天的学习内容存档”后才生成下一个 `archives/dayNN.md`。直接提出论文、架构或代码问题则进入 QA：先讲原理、架构/数据流、作用和源码，允许继续追问；话题收尾时确认是否掌握，用户确认后再按具体主题正式存入 `qa/`。两类已确认存档都可在后续计划学习中穿插复习。详细约定见 [AGENTS.md](AGENTS.md) 和 [归档模板](study_agent/SESSION_TEMPLATE.md)。首次主线会话从 Day19 的未答检查和权重同步继续。

当前版本需要在对话中主动开始学习或复习；尚无定时推送、自动论文监控或已配置的云训练环境。以下内容保留原始导出说明和历史归档结构。

导出时间：2026-09-10（Asia/Shanghai）

## 导出范围

本目录是当前“大模型算法与 Agent 工程学习”主线的完整可访问副本。原始导出包含 11 个独立 Day 议题；此后已在本地续写 Day14–Day18 学习归档，并按主题新增独立的答疑归档（见下）：

| 学习日 | Multica 议题 | 说明 |
|---|---|---|
| Day02 | WS-108 | Transformer / Decoder-only 预训练基础 |
| Day03 | WS-114 | Transformer、BERT/GPT 与训练机制衔接 |
| Day04 | WS-129 | Decoder-only SFT、mask、LoRA |
| Day05 | WS-136 | QLoRA、显存诊断、SFT 配置审查 |
| Day06 | WS-137 | QLoRA 实验设计，并在同一议题内承接 Day07 |
| Day08 | WS-147 | SFT / DPO 与偏好学习 |
| Day09 | WS-150 | 高风险结构化任务的数据、评测与治理 |
| Day10 | WS-155 | 高风险评测、版本化与受控迭代 |
| Day11 | WS-157 | 风险驱动阈值、硬门槛与上线判断 |
| Day12 | WS-159 | PPO：目标、GAE、value、训练稳定性 |
| Day13 | WS-160 | PPO 代码级闭环，GRPO 入口 |
| Day14 | 本地续学 | GRPO 原理、Reward Model 与最小训练闭环 |
| Day15 | 本地续学 | GRPO 低方差组、有效组 Mask 与 Loss 拼接 |
| Day16 | 本地续学 | GRPO 端到端训练源码、Old/New Log-prob 与 Mini-batch |
| Day17 | 本地续学 | GRPO 梯度累积、严格 Token 平均与测试设计 |
| Day18 | 本地续学 | 分布式 GRPO、全局 Group 与变长 Batching |

没有发现独立的 Day01 议题；后续归档多次明确说明，Day02–03 已覆盖 Transformer / 预训练前置。没有独立的 Day07 议题；Day07 内容在 Day06 的完整评论记录中。目录保留这一真实结构，没有虚构缺失文件。Day14–Day18 是后续对话形成的本地学习归档，不对应原始 Multica 议题，因此没有配套的 `raw/day14/` 至 `raw/day18/`。

## 答疑归档

答疑内容不占用 `archives/dayNN.md` 逐日序列，单独存放于 `qa/`，按 `日期-主题.md` 命名：

| 日期 | 文件 | 说明 |
|---|---|---|
| 2026-09-21 | `qa/2026-09-21-rollout-trainer-staleness.md` | Rollout/Trainer 分离与 Policy Staleness 答疑 |
| 2026-09-21 | `qa/2026-09-21-kimi-k3-kda-mla.md` | Kimi K3 的 KDA 原理、KDA–MLA 配合、完整前向与手算演练答疑（含 Gated MLA、LatentMoE、Attention Residuals 三节扩展） |
| 2026-09-26 | `qa/2026-09-26-deepseek-v4-1-swa-and-architecture.md` | DeepSeek V4.1 的 SWA、prefill/decode、Sparse Attention、逆 RoPE、整体架构与源码学习汇总 |

该文件完整保留学习教练的原始讲解结构（Rollout 与 Training 的形态差异、生产三段架构、policy version 与 staleness、同步/异步权重同步、过期数据准入），并保留尚未作答的 Policy Staleness 五问。

## 目录结构

```text
study/
├── README.md                 # 本说明与阅读入口
├── LEARNING_PLAN.md          # 从全部归档重建的总学习计划
├── CURRENT_PROGRESS.md       # 当前掌握情况、易错点、下一步
├── MANIFEST.tsv              # 文件大小与 SHA-256 清单
├── archives/                 # 便于阅读的逐日完整 Markdown
│   ├── day02.md
│   └── ...
├── qa/                       # 答疑归档，按 日期-主题.md 命名
│   ├── 2026-09-21-rollout-trainer-staleness.md
│   └── 2026-09-21-kimi-k3-kda-mla.md
└── raw/                      # 可审计的 Multica 原始 JSON
    ├── day02/
    │   ├── issue.json
    │   ├── comments.json
    │   └── metadata.json
    └── ...
```

原始的 `archives/day02.md` 至 `day13.md` 含议题详情、所有评论正文、评论关系和元数据；`raw/` 保留机器可读的原始导出。原始 11 个议题共 941 条评论，附件总数为 0。`archives/day14.md` 至 `archives/day18.md` 是后续本地学习摘要，`grpo_end_to_end.py` 是 Day16 配套教学源码。答疑类内容不进入 `archives/dayNN.md` 序列，统一放在 `qa/` 目录。

## 推荐阅读顺序

1. 先读 `CURRENT_PROGRESS.md`，快速恢复当前学习位置和未完成题。
2. 再读 `LEARNING_PLAN.md`，确认总路线和阶段验收。
3. 需要最近学习的完整摘要时，打开 `archives/day18.md`；需要最近答疑全文时，打开 `qa/2026-09-21-rollout-trainer-staleness.md`；源码上下文见 `grpo_end_to_end.py`。
4. 需要核对 Day02–13 的原始上下文时，使用 `raw/<day>/comments.json`。

## 边界说明

- 当前 Day13 议题没有绑定 Multica `project_id`；因此范围按当前学习教练 Agent 名下、由同一发起人创建的 Day 学习主线确定。
- 未导出工作区中听视频、推荐系统、数据分析等无关项目，避免混入他人或无关私密内容。
- 未复制运行时内部指令、凭据或系统配置；它们不是学习项目内容。
- 本目录是截至导出时刻的快照，后续 Multica 新评论不会自动同步。
