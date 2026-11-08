# 开发与分析端：H100项目接续说明

更新：2026-10-13。**本页供 `/data/run01/scyb672` 这台服务器新开的Codex对话阅读。** 仓库根目录为 `/data/run01/scyb672/hjr/TempName`。本机负责源码维护、接收H100证据、CPU分析和文档/论文整理；本任务的H100测试在另一台服务器 `gxn74` 上执行。

H100端“怎么测试、测试完交什么”统一维护在 [instruction-h100.md](instruction-h100.md)，部署命令在 [deploy-tp8.md](deploy-tp8.md)。本页负责告诉开发端在哪里找代码、数据、结论，以及哪些事情尚未验证。最新用户任务决定本轮范围，不因历史实验记录或待验证列表自动启动新测试。

**2026-10-17，新服务器消融3交接：** 给有H100服务器的新对话使用 [deploy-tp8.md](deploy-tp8.md) → [消融实验3.md](消融实验/消融实验3.md) / [README-ablation3.md](../../../tools/h100/phase6/README-ablation3.md) → [instruction-h100.md](instruction-h100.md)；公共软件版本参考 [env.md](../env.md)。本页和A800说明提供开发背景，不是新H100的安装/运行脚本。**本次只跑一轮：`prepare --blocks 9 --rounds 1`，81个正式窗口/648份正式rank；18个预检窗口不减少。** 省略轮数参数仍兼容旧两轮协议，旧已冻结目录不得改成一轮。H100部署指南同时保留仅预检分支，并补充`DEBUG=0`、共用NCCL Release开发库及本机双角色计划要求。

消融3的当前数据入口是 [base-phase6-ablation3.md](消融实验/base-phase6-ablation3.md)，原始证据在 `logs/h100/phase6/ablation3-gpt67-tp8-scope-20261016-221018-release1/`，构建登记在同名`outputs/h100/phase6/`目录。回传已有完整预检、124个完整正式窗口，其中117窗口构成13个完整九组配对块；第二轮异常，尚无完整162窗口报告。旧“尚无GPU验证”不再适用，回传的`running`状态也不代表远端仍在运行。新服务器独立生成计划、构建和日志，不延续原服务器旧PID或repeat队列；消融3任务不自动转为Phase7。

**2026-10-13 消融入口：** native＋G0–G6 已有独立实现，分组见 [消融实验.md](消融实验/消融实验.md)，命令与验证边界见 [README-ablation.md](../../../tools/h100/phase6/README-ablation.md)。两种基础顺序分别测量，G5默认三个预登记seed；默认400个完整step窗口，不套用下文原六策略的72窗口及单项4%验收。本轮只进行CPU/dry-run/无GPU编译，GPU测试尚待执行端安排。

**当前融合基线已对齐并有实测：** `build.py prepare` 的 `taco_fused` 使用 [固定 A800 源码](../../../tools/h100/phase6/fused_v2/README.md)，标识 `a800-fused-v2-cta-v1`：CTA 解码、旧融合编码转换、逐次 configure/reset、单工作区及两次 barrier；保留 H100 V2 MLP/V3 attention 的实例路由。`gpt67-tp8-fusedv2-*` 是该实现的实测会话；更早的 `gpt67-tp8-20261010-112249/compare-*` 属于 `phase6-warp-v1`。不要沿用旧文档“新基线尚无GPU结果”的判断，也不要混用两种基线结果。

## 优化起点已冻结（新对话先读）

2026-10-13按用户要求，在继续优化候选之前冻结为 **`h100-phase6-candidate-v1-20261013`**。本次仅保存源码、输入身份及核验哈希，未优化算法、重新编译或启动GPU测试。

**冻结对象明确为a–h实际复用的同一套候选构建**，包含远端组合与交替组合两个变体；权威身份是冻结manifest的 `candidate` 字段，构建提交为 `ff38f5b1ff1c80cb831f58c64b95f42b0379879a`。`source/` 中的 `944b9df` 副本用于保留维护源码与工具，不能拿这个提交号替代旧实测构建身份。后续仅优化这两种候选；`native`、`original`、`native_taco`、`taco_fused` 四个对照的实现、既有构建及统计协议保持固定，尤其不改 `tools/h100/phase6/fused_v2/`。

| 要找什么 | 固定入口 |
| --- | --- |
| 冻结说明与恢复方法 | [frozen-candidate-v1-20261013/README.md](../../../outputs/h100/phase6/frozen-candidate-v1-20261013/README.md) |
| 来源、配置、SHA256、依赖、会话映射与缺项 | [冻结 manifest](../../../outputs/h100/phase6/frozen-candidate-v1-20261013/manifest.json) |
| 当前维护源码的独立副本 | 冻结目录 `source/`，652个文件；源码提交 `944b9df06a564cabd738dc608c3d4945e98bb103`，与该提交逐文件一致 |
| 已实测候选 v1 | `outputs/h100/phase6/gpt67-tp8-20261010-112249/candidates-20261010-153517/`；构建源码提交 `ff38f5b1ff1c80cb831f58c64b95f42b0379879a` |
| 固定融合 v2 / 原生 Flux 对照 | `outputs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/base/`；构建源码提交 `d786da51e0aeafec1bb2b5d071c5c59e93e6de43`，融合标识 `a800-fused-v2-cta-v1` |
| 原选择计划 | `outputs/h100/phase6/gpt67-tp8-20261010-112249/selection-plan-20261010-153517.json`；SHA256 `2d8eee0ffaf2c86fdd1719950c7a5ed5eb0c93d5803b7deaf1e65caad752d167` |

两套已测候选分别是 `remote_arrival_selective`、`interleaved_arrival_selective`，共702个源码文件与原manifest哈希一致；最新g/h会话的worker来源和控制脚本也与当前维护入口一致。**当前源码提交、旧构建提交、数据上传提交是不同身份**：后续重建需要重新预检/测量，不能把本次归档写成新GPU验证。旧base回传快照缺少28个来源条目，冻结manifest已列出，不代表本机拥有完整运行库。

本次源码副本包括CUDA/Python实现、构建/校准/计划/运行/统计/队列工具及它们依赖的A800共享工具。GPT 6.7B、TP8/132 SM、S2048/mb1/global4、窗口64、预算1/64、10/20步协议一起固定。CUTLASS/NCCL及Megatron精确提交见manifest；环境仍由执行端环境记录和[env.md](../env.md)约束。

后续开发约定：

1. 优化继续改 `src/`、`include/`、`python/`、`tools/h100/phase6/` 等维护入口，保留优化补丁/提交。不改冻结 `source/`、校验清单、旧候选源码/库、旧计划及原始报告；需要查看旧实现时读冻结副本，需要恢复构建时按冻结README新建独立工作树。
2. 本轮后续优化仅限候选，保留 `tools/h100/phase6/fused_v2/` 的固定基线定义和四个对照组。候选 v1是新增的优化前对照身份，不是把 `taco_fused` 改名；维护脚本和共享A800代码的改动须检查是否意外影响对照或A800路径。
3. 每个新优化使用新构建、计划（需要重新校准时）、compare及run_id；交接注明 `parent_freeze=h100-phase6-candidate-v1-20261013`，列出新旧候选的实际构建路径/哈希与改动。修改维护代码不会更新旧构建，已有队列也不会自动切换到新候选。
4. 新旧性能比较需同条件新测量，历史a–h提供旧实现证据，不能代替同期对照。当前工具每次只接收一套候选构建；同时加入旧、新候选需要另行明确比较协议，不在本次冻结中修改实验接口或启动测试。

从仓库根目录可随时执行只读核对；优化维护入口之后，这两个检查仍应通过：

```bash
H100_FREEZE="$PWD/outputs/h100/phase6/frozen-candidate-v1-20261013"
(cd "$H100_FREEZE" && sha256sum --quiet -c SHA256SUMS.txt)
sha256sum --quiet -c "$H100_FREEZE/REFERENCES.sha256.txt"
```

冻结目录及清单需要随Git提交后才能在另一台服务器拉取；原运行动态库和编译缓存留在H100端。不要将源码快照完整性检查表述为原二进制核验。执行端具体保留/新建目录规则见[优化前冻结约定](instruction-h100.md#优化前冻结约定2026-10-13)。

### 新对话优化指令模板（开发端）

下面模板用于本开发服务器的代码优化；H100执行端的GPU任务另行安排。

```text
在 /data/run01/scyb672/hjr/TempName 中优化单节点8卡H100、GPT 6.7B、TP8的候选实现。

先阅读 docs/design/design-h100/instruction.md、base-phase6.md、thinking.md，
以及 outputs/h100/phase6/frozen-candidate-v1-20261013/README.md 和 manifest.json。
确认工作区已有改动并保留，核对冻结清单。

优化起点是 h100-phase6-candidate-v1-20261013 的 candidate 字段：
a–h轮共同使用的 candidates-20261010-153517 构建，包含
remote_arrival_selective 和 interleaved_arrival_selective。
a–d和e–h是同一实现的两批测试，round1/round2都是完整训练step。

只优化“基础顺序＋到达优先＋选择性量化”候选，可调整其实现；
native、original、native_taco、taco_fused及固定fused_v2源码保持不变。
保留模型/TP/batch/精度、数值正确性要求、四个对照和现有计时/验收协议。
若改动共享文件，隔离候选分支，确保不改变基线与A800路径。
先分析a–h原始报告和相关源码，区分已证实的瓶颈与待验证假设，再实施优化；
目标是提高完整optimizer step性能，沿用现有4%验收门槛，不预先宣称达标。

在维护入口修改，新候选使用新构建/会话目录；冻结源码、旧库、旧计划、
历史日志和报告保持原样。涉及tile映射、到达顺序语义等变化时说明H100重校准要求。
记录parent_freeze、新旧源码差异、构建/计划来源及潜在影响。

本轮允许CPU检查和不使用GPU的编译，不启动GPU、Slurm或自动等待队列。
完成后更新相关实现文档，给出H100端所需的构建、校准、预检、对比与回传步骤，
并明确哪些检查已通过、哪些性能结论仍需H100实测。
```

## 新对话先读什么

1. 先看工作区状态与最近提交，保留用户未提交改动；确认是否新拉取了数据，不能把下方历史快照当作远端实时进度。
2. 读本页[数据索引](#已接收数据与当前缺项)，然后读 [base-phase6.md](base-phase6.md) 的对应模型及基线版本结果。当前融合v2对齐第1–4轮、a–d、e–h分别展示，后两批均使用新监控体系及同一候选；历史`phase6-warp-v1`四轮和旧0.64B另列。
3. 性能分析见 [thinking.md](thinking.md)，已补充基线实现差异、原始 trace 与计划诊断；其统计对象是历史 `phase6-warp-v1` 会话。**旧小模型在H100上也没有相对融合v2的稳定优势，不能主要归因于6.7B变大或其OOM。**
4. 涉及方法或实现时，按下方源码地图看维护入口及对应实验快照；涉及新测试/补交数据时，将要求与执行端文档对齐。
5. 读用户当前指定的文件再工作；论文目录的实际内容以文件为准，不把此前讨论中的计划当作已完成论文。

可以先执行以下只读命令，不启动GPU：

```bash
cd /data/run01/scyb672/hjr/TempName
git status --short
git log -5 --oneline
rg --files docs/design/design-h100 tools/h100/phase6
```

## 文档和代码在哪里

| 目的 | 入口 |
| --- | --- |
| H100执行与数据交付约定 | [instruction-h100.md](instruction-h100.md) |
| 新服务器环境、八卡部署命令 | [deploy-tp8.md](deploy-tp8.md)、[公共环境](../env.md) |
| H100模型结果与证据链接 | [base-phase6.md](base-phase6.md) |
| H100收益减弱的分析 | [thinking.md](thinking.md)，涵盖旧基线版本差异及 CPU 复核 |
| 模型层数、参数量、全局/局部shape | [model-gpt-6.7b.md](../model-gpt-6.7b.md) |
| 原始方法与A800历史 | [A800接续](../design-a800/instruction.md)、[idea.md](../design-a800/idea.md) |
| A800可比数据 | [TP8](../design-a800/base-phase6-tp8.md)、[TP4](../design-a800/base-phase6-tp4.md)、[TP4优化](../design-a800/optimize/tp4/optimize2.md)；不要混入跨节点/DP实验 |
| 论文草稿与思想调研工作区 | `docs/design/paper/`；检查实际文件及完成状态 |
| H100协议、baseline定义 | [tools/h100/phase6/README.md](../../../tools/h100/phase6/README.md) |
| H100配置 | [config.json](../../../tools/h100/phase6/config.json)、[TP4配置](../../../tools/h100/phase6/config-tp4-s1024.json)；旧模型使用同目录 `config-0.64b-*.json` |
| 独立构建与注册项 | [build.py](../../../tools/h100/phase6/build.py)、[generator.cc](../../../tools/h100/phase6/generator.cc) |
| 校准、置换与选择计划 | [calibrate.py](../../../tools/h100/phase6/calibrate.py)、[plan.py](../../../tools/h100/phase6/plan.py)；复用 `tools/a800/arrival/plan.py`、`tools/a800/phase5/plan.py` 的算法 |
| 模型冻结、运行、统计 | [prepare.py](../../../tools/h100/phase6/prepare.py)、[run.py](../../../tools/h100/phase6/run.py)、[report.py](../../../tools/h100/phase6/report.py)、[common.py](../../../tools/h100/phase6/common.py) |
| 共享模型worker与适配 | `tools/a800/phase5/e2e/`；H100 prepare会生成修改后的worker快照 |
| Python算子、CUDA codec与接收端 | [gemm_rs_taco.py](../../../python/flux/gemm_rs_taco.py)、[taco_codec.cuh](../../../src/gemm_rs/taco_codec.cuh)、[taco_runtime.cu](../../../src/gemm_rs/taco_runtime.cu) |
| GEMM封装与tile调度 | [gemm_reduce_scatter.cc](../../../src/gemm_rs/ths_op/gemm_reduce_scatter.cc)、[threadblock_swizzle.hpp](../../../src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp) |

`src/`、`include/`、`python/`、`tools/`是受维护代码；`outputs/h100/phase6/<会话>/`内源码是**实验冻结证据**。修改实现应回到维护入口，再生成新构建，不直接修改旧快照。新旧源码可能不同，解释历史结果以该会话manifest及快照为准。H100工具依赖受维护的A800辅助实现，不要仅凭目录名把 `tools/a800/` 当作无关代码。

## 本机数据分布与读取顺序

所有相对路径均从仓库根目录起算：

| 想查看的内容 | 去哪里找 |
| --- | --- |
| 最近交接、环境与缺项 | `logs/h100/phase6/<会话>/handoff/`；历史会话若没有，结合结果文档、submission及状态文件，不虚构交接记录 |
| 模型、策略、种子、run_id和运行库来源 | `<compare>/submission.json`，以及 `<compare>/scripts/config.json` |
| 校准原始观察与有效计划 | `logs/.../<calibration>/`、`outputs/.../selection-plan*.json`；先确认校准完成，避免选到失败目录 |
| 构建来源、配置、源码快照及库哈希 | `outputs/.../base/manifest.json`、`outputs/.../<candidates>/manifest.json`及各变体源码 |
| 正确性和实际执行内核 | `<compare>/preflight.json`、`results/smoke-*/rank*.json`、`results/profile-*/rank*.json`、`rank*-profile.json.gz`、mapping CSV |
| 正式step耗时、loss、梯度、显存 | `<compare>/results/round*/window-*/rank*.json`；必须看全部rank |
| 启动参数与错误现场 | 各窗口的 `command.json`、`process.log`，以及失败备份、阶段状态与续跑日志 |
| 完整统计与验收 | 完成后的 `<compare>/report.json`、`report.md`；若有独立核验，连同脚本一起查看 |
| 历史A800原始数据 | `logs/a800/`、`outputs/a800/`；本地可能存在，但不保证随clone携带，缺失时明确只能依据文档摘要 |

自 `279fc32` 起，Git已携带H100日志、trace及部分构建文本证据；`.so/.a/.o`、构建缓存、Conda/Docker环境通常不在Git中。具体提交规则与完整产物树见[执行端文件规范](instruction-h100.md#生成数据与文件放置规范)。`data/`和`checkpoints/`不是当前测量入口的输入：当前为固定seed的GPU合成token，不需要另外寻找语料 `.bin/.idx` 或模型权重。

建议顺序：**交接/状态 → submission确认身份 → report了解结果 → 原始rank记录核对统计 → 校准/计划/profile解释原因 → 冻结源码解释实现**。只有摘要不足以验证性能；仅有profile也不能替代正式计时。

## 已接收数据与当前缺项

**冻结时的代码/数据提交为 `944b9df06a564cabd738dc608c3d4945e98bb103`，最新为2026-10-13两套复测，对应e/f与g/h。** 下列路径统一加前缀 `logs/h100/phase6/`；每套都有完整72窗口、576份计时rank记录及最终报告。数据接收状态不是远端实时运行状态，下一次pull后重新核对新增文件与run_id。

| 文档轮次（依次对应report的round1/round2） | 结果目录 | run_id |
| --- | --- | --- |
| 1/2 | `gpt67-tp8-fusedv2-20261011-233427/compare/` | `a7cac271fa2f407db093f20d5a584612` |
| 3/4 | `gpt67-tp8-fusedv2-20261012-005925-repeat/compare/` | `fd640c7d6fcc49d2ae203e979cb378e5` |
| a/b（新监控） | `gpt67-tp8-fusedv2-20261012-211935-repeat1/compare/` | `9d3cd8a3d15f40d78ba0f2a00ec8b87c` |
| c/d（新监控） | `gpt67-tp8-fusedv2-20261012-211935-repeat2/compare/` | `3aa54b8428aa45878c93d745c019918c` |
| e/f（新监控） | `gpt67-tp8-fusedv2-20261013-001822-repeat1/compare/` | `51f29917a2304634a6c9b6eec82fc90e` |
| g/h（新监控） | `gpt67-tp8-fusedv2-20261013-001822-repeat2/compare/` | `e5c5e72add794ee494e403096c71ec78` |

以上均使用 `a800-fused-v2-cta-v1`，复用[冻结登记](../../../outputs/h100/phase6/frozen-candidate-v1-20261013/manifest.json)中的同一套base、候选、计划与校准设备身份。原1–4轮、a–d、e–h分别展示，不能把全部轮次重新编号或混合筛选；原始report内的round编号保持不变。round1/round2均为完整前向＋反向＋优化器更新，区别是策略/配对块调度顺序反转，详见base-phase6的轮次说明。a–h使用全样本；第1–2轮及远端第4轮文档表格有明确标注的事后筛选，原始report与验收仍按全样本，不用筛选结果替代原报告。

最新证据入口：e/f的[report](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat1/compare/report.json)、[submission](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat1/compare/submission.json)，g/h的[report](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat2/compare/report.json)、[submission](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat2/compare/submission.json)。新监控方式和队列证据见 [watcher.md](watcher.md) 与各会话 `execution.json`、`handoff/repeat-inputs.json`；a–h四套的两候选均未通过完整两轮性能门槛。详细统计及此前CPU复核范围见 [base-phase6.md](base-phase6.md)。

| 固定构建/校准 | 仓库相对路径 |
| --- | --- |
| 当前融合v2基线及原生Flux构建 | `outputs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/base/` |
| 成功校准 | `logs/h100/phase6/gpt67-tp8-20261010-112249/calibration-20261010-153517/` |
| 固定选择计划 | `outputs/h100/phase6/gpt67-tp8-20261010-112249/selection-plan-20261010-153517.json` |
| 固定两套候选 | `outputs/h100/phase6/gpt67-tp8-20261010-112249/candidates-20261010-153517/` |

早先无后缀`calibration/`是失败目录，不作为有效校准；`gpt67-tp8-20261010-112249/base/`是旧基线/采样构建，不能当作当前融合v2对照。模型为32层/H4096/F16384，全局MLP shape `[2048,4096,16384]`、TP8本地K=2048。

历史 `phase6-warp-v1` 结果仍在以下目录，供分析旧基线，不作为新融合v2或新优化的实测证据：

| 模型 / 历史轮次 | 完整结果目录（加前缀 `logs/h100/phase6/`） |
| --- | --- |
| 6.7B历史第1–2轮，含续跑 | `gpt67-tp8-20261010-112249/compare-20261010-153517/` |
| 6.7B历史第3–4轮 | `gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/` |
| 旧0.64B | `tp8-20261009-deployment/compare/` |

**缺项与核验边界：** 本机有回传文本/trace及候选源码；原动态库、构建环境主要仍在H100端。此次冻结还登记了旧base四个变体共28个缺失的来源条目，见manifest；缺失文件未纳入 `REFERENCES.sha256.txt`，不能因该清单通过就声称旧base完整可运行。来源路径中的 `/workspace/TempName/...` 保持原样，不改manifest绕过核验。环境/交接缺项按[执行端交付清单](instruction-h100.md#交接说明与环境记录放在哪里)补交。

新对话若在旧副本看到“43/72、未完成”或“新CTA基线尚无H100结果”，以对应会话新回传的report和阶段状态为准。`thinking.md`的早期分析主要基于旧 `phase6-warp-v1`，不得自动套用到新的融合v2数据或未测试的优化版本。

## 本机收到新数据后怎么检查

1. 记录本次数据提交SHA，核对新增文件属于哪个会话；实验代码commit、数据上传commit分别记录。比较 `submission.json`、base/candidate manifest、计划中的模型/TP/种子和GPU身份。
2. 检查阶段状态和实际文件覆盖。默认六策略TP8应有两轮72窗口、576份计时rank记录，另有48份smoke、48份profile rank记录及48份trace；OOM目录、`*.failed-*`备份和仅有process.log的目录不算完成。
3. 按原清单检查文件哈希：submission与preflight的绑定、冻结scripts、preflight证据；然后检查全部rank的 `completed/passed`、零skip/零fallback、计时步数、初始化/RNG/token指纹、实际库哈希和模型字段。检查范围及缺项如实记录。
4. 重算时取每窗口最慢rank的elapsed除以正式step数，按策略、轮次、配对块组织数据；收益用 `100*(1-exp(mean(log(T候选/T基线))))`，按块bootstrap。不要用中位数相除替代配对收益，不混用第一轮和第二轮补空缺。
5. 更新 [base-phase6.md](base-phase6.md) 的事实与数据链接，以及本页的会话状态；机制解释写入 `thinking.md`，分别标明事实、线索和假设。若新数据改变结论，再更新相关论文表述。

`report.py`虽是CPU逻辑，却核验执行端**原绝对路径和实际库文件**。本机没有完整构建时直接运行它可能失败；拉回的 `compare/build/` 符号链接也可能指向 `/workspace/TempName`。不要通过改manifest路径/哈希绕过检查。原服务器严格核验与报告由H100端交付，本机可以按相对路径解析副本并独立复算，但要明确“原始数据统计已复核”和“原运行二进制已核验”的区别。

profile只用来解释机制：解压已有JSON即可，不需要GPU。内核duration可能含等待，多个stream可能重叠，不能把内核累计时间相加当作正式step耗时；跨rank时间戳也不能直接当作同一时钟。详细归类口径应随 `thinking.md` 的恢复核对。

## 当前结论与工作边界

- 默认新模型是6.7B，但历史约0.64B实验仍是重要独立证据。**旧小模型H100 TP8也未建立相对融合v2的稳定收益**；A800同模型TP8约2%，其TP4最佳约4%有不同batch/shape与专项优化，不能混为单纯GPU型号差异。
- 新融合v2对齐会话均已完整回传；最新新监控体系a–h的四套报告中，两候选均未通过完整两轮门槛。单轮较高收益不能当作整套达标，历史不同基线的结果分开解释；各会话的具体验收以原report和base-phase6对应小节为准。性能未达标、测量未完成、正确性失败是不同状态。
- 当前分析线索是H100全量化decode成本下降、少量化的可回收时间减少、到达计划留出命中率较低，以及局部节省未稳定进入整步关键路径。这些是此前分析线索，需对照新复测继续核查，不能宣称已证明单一硬件根因。
- 本机按当前任务做源码/文档维护、静态检查与CPU分析。不要执行拉回的 `env.sh`、watchdog、`resume_timing.py`或带 `--execute` 的GPU入口来验证数据，也不要因为文档列出待验证方向而启动远端测试。若用户安排H100实验，交给执行端按其说明运行并回传。
- 不覆盖历史会话、不修改已有观测/时间/完成状态、不移动实验运行目录来修复路径报错；用户新增数据和工作区修改先核对再处理。

## 实现边界

量化 MLP 使用重新编译为 SM90 的 Phase6 GemmV2 算法；attention 和原生 Flux 参考使用 Hopper GemmV3。该版本没有实现基于 WGMMA/TMA 的 Hopper 原生融合 TACO epilogue，不能将兼容移植称为完成 H100 性能优化。

必须 baseline 为原生 Megatron＋朴素 TACO、原始顺序 Flux＋全远端融合 TACO；参考为原生 Megatron、原生 Hopper Flux。默认候选为远端优先和交替两种基础顺序，可配置另外两种交替顺序。范围仅 MLP `linear_fc2` 前向，BF16 GEMM、本地 BF16、H128/E4M3 远端协议及 BF16 STE 反向沿用现有设计。

构建与结果分别放 `outputs/h100/phase6/`、`logs/h100/phase6/`。必须从 H100 重新采集到达观察并生成选择计划；工具拒绝 A800 计划和未完成构建。候选准备依赖真实校准，不用合成 CPU 测试输入替代 GPU 证据。

后续GPU工作由H100执行端按用户任务和预算安排，按README执行校准、独立候选编译、swizzle/模型smoke/profile及完整六组/八组两轮测量；本页是交接规范，不会自动启动或恢复这些步骤。

## 本次无 GPU 验证（2026-10-09）

以下是2026-10-09、旧0.64B配置的历史记录，未在6.7B上重跑。编译和导入进程均设置 `CUDA_VISIBLE_DEVICES=''`：

| 检查 | 结果与范围 |
| --- | --- |
| Python CPU 协议测试 | 15 项通过；覆盖 TP4/TP8、四种基础顺序、到达计划/量化预算、留出数据隔离、输入冻结和 dry-run 不启动 worker |
| 注册代码生成器 | 132 SM 和 114 SM 两种配置的主机 C++ 编译通过；完整动态库编译采用默认 132 SM 配置 |
| CUDA 交叉编译 | SM90(a) 的 TACO runtime、融合 V2、采样 V2、合成到达表候选和实际 occupancy 查询函数通过；`cuobjdump` 确认量化 V2 对象包含 `sm_90a.cubin` |
| 完整库与 Python binding | `original`、`taco_fused`、`sampler_remote_first`、`sampler_interleaved` 四套独立库完成编译、链接和 Python 扩展构建 |
| CPU 导入 | 四套库使用 CUDA SDK 驱动桩通过 Torch/Flux 导入、绑定与导出符号检查，确认没有初始化 CUDA 上下文；没有调用 occupancy 查询或任何 kernel |
| GPU mapping 检查器 | 融合 baseline 和两套采样库的检查器完成编译，未执行 |
| 最终源码生成 | 重新执行 `build.py prepare` 成功；与已编译快照核对，只有等价的链接导出符号排列顺序不同 |

本机编译环境为 Python 3.10.20、PyTorch 2.6.0+cu124、CUDA Toolkit 12.8、G++ 11.4 和 CMake 3.30；编译检查使用现有 NCCL 开发安装。编译产物与记录位于 `outputs/h100/phase6/compile-check-20261009-r2/`，汇总为其中的 `validation.json`。该目录设为 `diagnostic_only=true`，工具禁止将其直接用于正式测量；这些被 Git 忽略的本地产物不会随 clone 迁移。

修复了无 GPU 编译发现的两处问题：H100 快照避免 `setup_requires` 再次联网下载 CMake；删除没有调用方的全局 CUDA 分配，使导入发生在绑定 rank 设备之前时也不会隐式分配设备内存。检查器改为查询实际编译 kernel 的 occupancy，不再沿用 A800 的固定数值。

上述编译记录当时不包含GPU验证；后来旧0.64B以及6.7B的阶段实测见[独立记录](base-phase6.md)，当前会话及对应基线版本见本页数据索引。正式6.7B会话已有真实校准、候选构建、smoke/profile和完整报告，逐套验收以各自report为准；这些短测没有验证收敛。合成表只用于历史编译诊断，不作为真实校准计划或性能证据。
