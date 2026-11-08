# GPT 6.7B：A800 / H100 新模型配置

更新：2026-10-10。按用户要求，把后续实验主模型从约 0.64B 扩为 **约 6.7B**。本次只修改配置、准备/校验逻辑和文档，**不编译、不运行 CPU/GPU 测试、不提交作业**。下文命令供之后获准运行时使用。

## 模型定义

| 项目 | 新主模型 |
| --- | --- |
| 层数 | 32 个 Transformer 层 |
| hidden / FFN | 4096 / 16384，FFN=4H |
| attention heads | 32，每头维度 128 |
| 序列长度 | 默认 2048 |
| micro / global batch，DP1 | 1 / 4，4 个微批，8192 token/optimizer step |
| 合成 token 范围 | 0–32767，`--vocab-size 32768`，NullTokenizer 另加 EOD |
| 实际 padded vocabulary | TP4=33280；TP8=33792，按 Megatron 默认 128×TP 补齐 |
| 嵌入与输出头 | 保留 `--untie-embeddings-and-output-weights`，不共享权重 |
| 位置编码 / 计算 | learned absolute / BF16，sequence parallel |
| 优化器与统计 | 沿用 Adam 和完整 optimizer-step 口径；DP=PP=CP=1 为单节点默认 |

按当前带 bias、LayerNorm、非门控 FFN、独立输入/输出嵌入的结构计算，参数量为：

`L × (4H² + 2HF + 9H + F) + 2 × padded_vocab × H + S × H + 2H`。

S2048 时，TP4 约 **6,725,181,440（6.725B）**，TP8 约 **6,729,375,744（6.729B）**，统一简称 GPT 6.7B。这是配置推算，不是本次加载模型的实测计数；不能将每 rank 参数量直接乘 TP，因为位置嵌入和部分参数在 TP ranks 上重复。该模型是本项目沿用结构的随机初始化合成数据基准，不代表加载了某个预训练 GPT-3 checkpoint。

## 形状变化与计划失效

| 项目，默认 S2048/mb1 | 旧 0.64B | 新 6.7B |
| --- | --- | --- |
| MLP `[M,N,K_global]` | `[2048,2048,8192]` | `[2048,4096,16384]` |
| MLP TP4 / TP8 的 `K_local` | 2048 / 1024 | 4096 / 2048 |
| attention `[M,N,K_global]` | `[2048,2048,2048]` | `[2048,4096,4096]` |
| attention TP4 / TP8 的 `K_local` | 512 / 256 | 1024 / 512 |
| 每来源物理输出 tile 数，128×128 | 256 | 512 |
| MLP 目标模块数 | 12 | 32 |

新形状要求重新 BF16 校准、拟合到达表和物理量化 mask、构建候选。不能把旧表扩一倍，不能仅改模型层数后复用 H2048 的静态表。每个基础顺序独立校准；window64、远端量化预算上限1/64及 BF16 STE 规则保持原定义。TP4 旧 M8192/N2048 compact-v1 特化不适用于新模型，使用通用解码路径。

没有进行显存/吞吐/正确性测试。保留原来未启用 activation checkpointing、CUDA Graph、PP/CP 的口径；新模型的可用显存和数值行为必须在之后的 preflight 确认，不能沿用 0.64B 的峰值显存或耗时。

## A800 入口

新公共模型文件为 [model-6.7b.json](../../tools/a800/phase6/model-6.7b.json)。历史 Phase4/5 模板和冻结回放维持原配置，新准备器会将完整模型信息写入本次快照。相邻 Megatron-LM 保持固定提交和干净工作区，不修改其历史 `train_gpt3_small_tp4_synthetic.sh`。

### 单节点 TP4 / TP8 完整六组量化对比

A800 路线依赖该机器已有的冻结 Flux baseline 和 BF16 sampler 构建。普通 baseline 保持原库并在新模型上重新检查/计时；候选根据新表另建。克隆到没有这些产物的 A800 机器时，须先准备相应 A800 构建，不能将 H100 二进制作为替代。

下面以同一全新会话名说明文件归属；作业 ID 由后续实际分配提供。准备命令不执行 GPU；生成的 `driver.py` 会执行 GPU 校准，须在对应节点的有效分配中运行。本次没有执行这些命令。

```bash
export MODEL="$PWD/tools/a800/phase6/model-6.7b.json"
export A800_JOB_ID="替换为新分配作业ID"
export A800_NODE="替换为实际节点名"
export SESSION=gpt67-tp4-NEW

python tools/a800/phase6/prepare_calibration.py "logs/a800/phase6/$SESSION-calibration" \
  --model-config "$MODEL" --world-size 4 --job-id "$A800_JOB_ID"

# 之后在该分配中使用训练 Python 执行 driver.py，完成真实校准后再继续。
python tools/a800/phase6/plan.py "logs/a800/phase6/$SESSION-calibration" \
  "outputs/a800/phase6/$SESSION/selection-plan.json"
python tools/a800/phase6/build.py \
  "outputs/a800/phase6/$SESSION/selection-plan.json" \
  "outputs/a800/phase6/$SESSION/candidates" --bases remote_first interleaved
python tools/a800/phase6/prepare_scenario.py "logs/a800/phase6/$SESSION-compare" \
  --model-config "$MODEL" \
  --build "outputs/a800/phase6/$SESSION/candidates" \
  --plan "outputs/a800/phase6/$SESSION/selection-plan.json" \
  --bases remote_first interleaved --decoder legacy --job-id "$A800_JOB_ID"
```

`prepare_scenario.py` 不传 `--hidden` 时默认采用 6.7B；显式 `--hidden 512/1024/2048` 且不传模型文件保留历史场景默认值，供旧脚本复现。新模型建议始终显式指定 `--model-config`。可用 `--layers`、`--heads`、`--vocab`、`--global-batch` 覆盖相应值，但修改 GEMM 形状后必须匹配新的校准计划。

TP8 使用同一模型文件、单独的八卡校准及 TP8 构建/准备器：

```bash
export SESSION=gpt67-tp8-NEW
python tools/a800/phase6/prepare_calibration.py "logs/a800/phase6/$SESSION-calibration" \
  --model-config "$MODEL" --world-size 8 --job-id "$A800_JOB_ID"

# 等真实八卡校准完成后。
python tools/a800/phase6/tp8_quant/plan.py "logs/a800/phase6/$SESSION-calibration" \
  "outputs/a800/phase6/$SESSION/selection-plan.json" --model-config "$MODEL"
python tools/a800/phase6/tp8_quant/build.py \
  "outputs/a800/phase6/$SESSION/selection-plan.json" \
  "outputs/a800/phase6/$SESSION/candidates"
python tools/a800/phase6/tp8_quant/prepare.py "logs/a800/phase6/$SESSION-compare" \
  --model-config "$MODEL" \
  --build "outputs/a800/phase6/$SESSION/candidates" \
  --plan "outputs/a800/phase6/$SESSION/selection-plan.json" \
  --job-id "$A800_JOB_ID" --node "$A800_NODE"
```

TP8 计划、映射检查、模型预检形状、padded vocabulary 和报告均从新模型派生；不再用固定 `[2048,2048,8192]` 或词表9216校验新模型。TP4/TP8 默认 sampler 根分别为 `outputs/a800/phase3/mechanism-build`、`outputs/a800/arrival8/mechanism-build`，可通过 `--sampler-root` 指定同架构的独立 sampler 根；每个根包含 `remote_first/` 与 `interleaved/`。运行阶段仍使用准备目录内的 campaign，按对应 [TP4](../../tools/a800/phase6/README.md) / [TP8](../../tools/a800/phase6/tp8_quant/README.md) 协议先预检，再两轮计时。

### 多节点与 BF16 对照

- **TP4/DP2 六组**：[dp2/prepare.py](../../tools/a800/phase6/dp2/prepare.py) 新增 `--source <新6.7B的TP4准备目录>` 和 `--model-config`；必须提供同模型、同形状的新 TP4 计划和构建。保持每副本工作量后 global batch 从4变8，共16384 token/step。该步骤不表示已经在 DP2 负载下重新校准。
- **通用 BF16 / 原生对照**：[distributed/config.py](../../tools/a800/e2e/distributed/config.py) 默认已改为32层/H4096/FFN16384/词表32768。单节点完整模型配置见 [TP4](../../tools/a800/e2e/distributed/examples/gpt-6.7b-tp4.json)、[TP8](../../tools/a800/e2e/distributed/examples/gpt-6.7b-tp8.json)。不带 `--submit` 仅打印计划。该通用历史 BF16 适配器作用于 attention 和 MLP，不能将它当成六组 MLP-only 量化验收入口。
- **跨节点 TP8**：通用 BF16 入口可配置6.7B与分层传输；旧 `phase6/tp8` 的固定形状回放继续归属0.64B历史实验。跨节点完整选择性量化尚未实现，本次模型扩大不改变这一边界。

## H100 入口

| 文件 | 新默认模型 / 场景 |
| --- | --- |
| [config.json](../../tools/h100/phase6/config.json) | 6.7B，TP8，S2048/mb1/global4 |
| [config-tp4-s1024.json](../../tools/h100/phase6/config-tp4-s1024.json) | 6.7B，TP4，S1024/mb1/global8，8192 token/step |
| [config-0.64b-tp8.json](../../tools/h100/phase6/config-0.64b-tp8.json) | 保留原12层TP8配置 |
| [config-0.64b-tp4-s1024.json](../../tools/h100/phase6/config-0.64b-tp4-s1024.json) | 保留原12层TP4、S1024/mb8/global8配置 |

TP4 辅助场景将 micro batch 从8降至1，通过累积维持8192 token/step，减少新大模型一次前向的激活规模；其 MLP shape 为 `[1024,4096,16384]`。这是一套新配置，不与旧 mb8 的性能直接相除。

按 [H100 部署指南](design-h100/deploy-tp8.md) 执行相同的“baseline/sampler构建 → 本机校准 → 新计划 → 候选构建 → 预检 → 计时”流程，但务必使用全新会话名，例如 `gpt67-tp8-YYYYMMDD-HHMMSS`。生成文件分类与目录规范见 [H100 instruction](design-h100/instruction.md#生成数据与文件放置规范)。

## 历史证据与本次状态

所有 A800 原0.64B性能表、四层小模型记录，以及 H100 `gxn74` 的2026-10-10六组实测原样保留并标注历史模型。新的6.7B配置没有测试结果；不将旧正确性检查、15项CPU检查、GPU峰值显存、到达计划或已编译库记录表述成新模型验证结果。本次仅做代码/文档编辑与静态阅读。
