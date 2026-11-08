# A800 项目快速接续说明

实验与资源快照更新：2026-10-06（北京时间）；文档目录与链接更新：2026-10-09。本页只保留 A800 接续必需的信息；详细实验、统计和历史过程见下方文档。资源表是历史核验快照，使用前须查询Slurm。

A800 文档统一位于 `docs/design/design-a800/`；上级导航见 [设计文档索引](../README.md)。本次目录整理不新增实验结果。

## 2026-10-10：后续模型升级到6.7B，未测试

新主模型为 **32层/H4096/FFN16384/32 heads/词表32768**，默认S2048/mb1/global4、BF16；保留输入/输出嵌入不共享。参数量约6.725B（TP4）/6.729B（TP8），完整定义与新入口见[模型说明](../model-gpt-6.7b.md)。本次只修改A800/H100模型配置、准备器和文档，不编译、不运行测试、不申请资源。

A800单节点六组使用 `phase6/prepare_scenario.py --model-config ...`（TP4）或 `phase6/tp8_quant/prepare.py --model-config ...`（TP8），新模型文件为 [model-6.7b.json](../../../tools/a800/phase6/model-6.7b.json)。新MLP shape为 `[2048,4096,16384]`，须重新校准并构建到达/选择性量化候选；不得复用0.64B静态表。TP4/DP2需传入同模型的新TP4准备目录。跨节点TP8完整量化仍未实现。

下面的性能结论与冻结库清单均为**历史0.64B实验**，不构成新6.7B的运行证据。历史专用入口继续用于旧结果回放；新的通用BF16入口默认已改成6.7B，旧复跑须显式使用历史模型配置。

## 当前进度与结果入口

| 场景 | 当前结论 | 完整记录 |
| --- | --- | --- |
| 单节点TP4 | S1024/mb8通过4%验收，S256/mb32远端组合约5%；原S2048/mb4仍未达标但已相差不远。三形状已冻结存档 | [实验](base-phase6-tp4.md) · [存档与复跑](tp4-archive.md) |
| 双节点TP4、DP2 | 两场景已完成六组、两轮测量，均未稳定通过4%门槛；尚未优化 | [base-phase6-tp4dp2.md](base-phase6-tp4dp2.md) |
| 单节点TP8、DP1 | 限时优化已停止；window16六组两轮对融合v2快1.881%–2.498%，未达本次3%及历史4%；保留旧window64推荐配置 | [base-phase6-tp8.md](base-phase6-tp8.md) |
| 双节点TP8、DP1 | 仅测现有BF16分层路径，第1–4轮已按实际样本记录；尚未确认稳定排序收益 | [base-phase6-tp8双节点.md](base-phase6-tp8双节点.md) |

S2048的TP8第4轮只有1个配对块：Megatron、原始Flux、远端排序各1窗口，交替无采样，95%区间不估计。该记录按用户要求以已有窗口统计，不能按完整四块复测解释。

单节点4卡优化暂不继续；单节点TP8限时优化及六组两轮验证已完成，13:50按用户要求停止，不自动续跑补充对照；详见[TP8优化记录](optimize/tp8/optimize1.md)。S1024/mb8、S2048/mb4、S256/mb32的TP4配置、库、到达表、mask及复跑入口见[冻结存档](tp4-archive.md)；此前“S512”已确认指S256/mb32。其他历史准备目录不自动续跑，单节点8卡结论不能替代双节点TP8。

总体思路见[idea.md](idea.md)，两轮优化细节见[optimize1](optimize/tp4/optimize1.md)、[optimize2](optimize/tp4/optimize2.md)；早期阶段见[base.md](base.md)、[only-tile.md](only-tile.md)、[base-phase4.md](base-phase4.md)、[base-phase5.md](base-phase5.md)、[base-e2e.md](base-e2e.md)。

## 目标、基线与验收

只优化 **MLP `linear_fc2` 前向通信**。两条候选为“远端优先＋到达优先＋选择性量化”和“交替排序＋到达优先＋选择性量化”；attention保持原后端，GEMM及本地贡献保持BF16。选中的远端贡献经TACO H128编码为FP8 E4M3，接收端恢复为BF16；反向保持BF16 STE，正式候选不启用CUDA Graph。

| 类别 | 基线 | 定义 |
| --- | --- | --- |
| 必须baseline | Megatron＋量化（`native_taco`） | 原生Megatron，MLP远端贡献使用朴素TACO，attention原生 |
| 必须baseline | Flux＋融合量化（`taco_fused`） | 原始Flux顺序、全量远端TACO融合v2，不含自定义重排 |
| 参考baseline | 原生Megatron、原生Flux | 不加量化或自定义重排，分别报告与候选的耗时差距 |

本次TP8限时优化另按用户约3%目标评估（`user-goal.json`），未达到；下述历史4%标准及旧结果不改写。

**目标：同一候选在两轮反序测试中，对两个必须baseline的配对耗时下降均≥4%，且各95%区间下界>0，同时尽量接近两个参考。** “接近”尚无硬阈值；不以四舍五入、区间上界或只优于一个baseline宣称通过。

- 点估计：`100 × (1 − exp(mean(log(T_candidate / T_baseline))))`；按配对块bootstrap 10000次计算95%区间，不能用汇总中位数直接相除。
- 正式量化验收同轮包含两候选、两必须baseline和两原生参考；同模型、TP/DP布局、batch、精度配对，每窗口通常10步预热＋20步完整optimizer step，profile另跑。算子收益不能替代完整step收益。
- 必须Flux基线冻结于`outputs/a800/phase4/taco-fused-warp-20261003`，核验动态库SHA256，不重建、替换或削弱；`remote_all`等带排序的辅助组不能冒充该基线。
- 新配置明确冻结4%门槛；缺少`acceptance`的旧配置仍按历史5%解释，不改写历史验收。
- TP4/DP2首测为16384全局token/step，原DP1为8192。不同工作量、节点或场景的耗时不能直接相除当作实现收益；DP同步不等于跨节点TP通信。
- 固定合成token与有限loss/codec误差通过不证明真实数据收敛。保留初始化、RNG、输入、GPU身份、数值预算、映射、库哈希和无跳步/回退核验。

## 当前实现与冻结版本

- TP4最新候选：`outputs/a800/phase6/tp4-opt2-final-build-20261004`，准备时显式传 **`--decoder compact-v1`**；复现历史三形状优先直接使用[归档](tp4-archive.md)，其中S256保留旧库及八组配置。
- 单节点TP8候选及计划：`outputs/a800/phase6/tp8-quant-adapt-20261006`；已验证S2048/mb1/global4、TP8/DP1，实际词表9216。完整结果见[TP8记录](base-phase6-tp8.md)，不能直接与TP4的mb4/词表8704作强扩展比。
- TP8本轮window16实验构建：`outputs/a800/phase6/tp8-opt1-20261006/build-window16`；正式结果`logs/a800/phase6/tp8-opt1-final-window16-20261006`。通用解码、默认GEMM、原3 tile/来源mask，未证明比旧window64稳定更快，保留原推荐。补充对照已停止。
- v13短列表解码仅默认用于TP4、M8192/N2048、每rank≤32个混合tile且元数据有效的路径；其他形状或稠密mask沿用旧分派。改变ABI时使用[rebuild_candidate.py](../../../tools/a800/phase6/rebuild_candidate.py)在新目录重建，不混用旧封装与新runtime。
- 两套完整BF16/FP8工作区交替，保留每次GEMM的全rank发布barrier和输出生命周期保护；不能仅双缓冲FP8包就删除同步。
- 原推荐到达计划使用同目标分区64个合法槽位，本轮TP8试验候选为16；每来源最多1/64远端tile量化是**预算上限**。TP4计划来自`logs/a800/phase6/communication-confirm-h2048-tp4-20261003/scripts/selection-plan.json`；单节点TP8使用独立八卡校准表，不套用TP4表。均未在重排/量化后联合重校准，复用时须注明来源。
- TP4四种基础映射是rank+1分区偏移、`L,R1,R2,R3`、`R1,L,R2,R3`、`R1,R2,R3,L`；名称不保证全局严格远端先行。各基础的顺序和mask可能同时不同，不能把整个组合收益归因于单一改动。

**跨节点TP8完整量化尚未实现。** 现有`GemmRSTaco`依赖同节点GPU IPC；跨节点需要另行适配传输、同步、到达表、mask及必须baseline。现有双节点TP8入口只是BF16分层对照，不适用六组量化4%验收。原单节点worker仍限制DP=1，双节点使用独立入口。

## GPU资源与运行规则

最近核验：2026-10-06 14:11；表中四个作业均已按用户明确指令取消，目前这些分配均不可复用。原分配均为`gpu_a800`分区、`normal` QoS，时限从实际启动起算。

| 作业 | 资源 / 节点 | 最近状态与时限（北京时间） | 控制目录 |
| --- | --- | --- | --- |
| **187459** | 4×A800、8 CPU；`d1n41a23g03` | CANCELLED；10-06 13:48:45，用户要求取消 | `logs/a800/allocations/a800-tp4-hold-20261005/` |
| **189341** | 4×A800、8 CPU；`d1n41a12g02` | CANCELLED；10-06 13:48:45，用户要求取消 | `logs/a800/allocations/a800-tp4-hold-20261006/` |
| **187567** | 单节点8×A800、16 CPU；取消前未分配节点 | CANCELLED；10-06 13:48:45，用户要求取消 | `logs/a800/allocations/a800-tp8-hold-20261005/` |
| **179139** | 单节点8×A800、16 CPU；`d1n41a15g02` | CANCELLED；10-06 14:11:04，用户要求取消 | `outputs/a800/arrival-v2-tp8-20261003/` |

179147、183972、182708均已到期，不再用于启动；182708于2026-10-06 00:00:54结束，由新申请189341接替。四卡作业187459/189341现均已取消，不再用于启动；187567也已取消。新作业[申请回执](../../../logs/a800/allocations/a800-tp4-hold-20261006/submission.json)与[状态核验](../../../logs/a800/allocations/a800-tp4-hold-20261006/validation.json)保存在控制目录。

**保护外层驻留：不在实验清理中取消有效作业、终止batch或创建`RELEASE`。** 只清理本次内部step；同节点性能计时串行，`--overlap`仅用于与驻留共存。新节点须重新核验GPU身份、拓扑和实际NCCL/IB通路，不能沿用强制单节点通信的配置。

```bash
cd /data/run01/scyb672/hjr/TempName
squeue -u scyb672
squeue --steps -u scyb672
```

每次准备全新结果目录，显式指定有效`--job-id`且与`srun --jobid`一致；不能沿用脚本中可能残留的旧作业默认值。原双节点作业187459/189341及单节点TP8作业179139/187567均已取消，不能再用于启动。后续实验须先获得新的有效分配并核验实时资源；单节点TP8仍需8卡/16 CPU。不得覆盖冻结目录、补写旧节点测量或自动续跑已停止实验。

## 代码、环境与资料位置

项目根目录：`/data/run01/scyb672/hjr/TempName`，兼容路径为`/data/home/scyb672/run/hjr/TempName`。相邻`Megatron-LM`为模型框架，`COCCL_TACO`为TACO参考实现。

| 用途 | 入口 |
| --- | --- |
| 软件环境、Megatron 固定版本与迁移 | [env.md](../env.md) |
| 单节点TP4准备、构建、验收 | [phase6/README](../../../tools/a800/phase6/README.md)；`prepare_scenario.py`、`assess.py` |
| TP4三形状冻结复跑 | [存档索引](tp4-archive.md)；`tools/a800/phase6/tp4_archive.py` |
| 双节点TP4/DP2 | [dp2/README](../../../tools/a800/phase6/dp2/README.md)；独立冻结准备与运行器 |
| 单节点TP8 BF16 | [tp8_single/README](../../../tools/a800/phase6/tp8_single/README.md) |
| 单节点TP8完整量化组合 | [tp8_quant/README](../../../tools/a800/phase6/tp8_quant/README.md)；独立计划、构建、六组对照 |
| 双节点TP8 BF16及TP4回归 | [tp8/README](../../../tools/a800/phase6/tp8/README.md) |
| 量化与共享模型入口 | [phase4/e2e](../../../tools/a800/phase4/e2e/README.md)、[phase5](../../../tools/a800/phase5/README.md) |
| 训练Python | `/data/run01/scyb672/conda_envs/flux-megatron-a800/bin/python` |
| host侧CPU检查 | `outputs/a800/venv/bin/python` |

CUDA主要修改点为`src/gemm_rs/epilogue_evt.hpp`、`taco_codec.cuh`、`taco_runtime.{h,cu}`和`src/gemm_rs/ths_op/gemm_reduce_scatter.cc`；Python入口为`python/flux/gemm_rs_taco.py`。本项目当前Phase6指组合执行成本优化，与早期路线图的阶段编号区分。

新结果写`logs/a800/`，新构建写`outputs/a800/`；这两处按[.gitignore](../../../.gitignore)忽略，不纳入Git。A800 设计文档位于`docs/design/design-a800/`，随维护代码纳入版本控制；原文中关于`design/`被忽略的说明已过时。维护代码放`tools/`、`src/`、`include/`、`python/`，保留旧日志、冻结库、兼容链接及其他未提交修改；不全局忽略JSON/CSV，也不取消Triton所需`.bc`/`.ll`文件的版本管理。提交前检查`git status`、`git diff --check`和`git check-ignore`。

