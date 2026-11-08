# H100 新机制 baseline 汇总

更新：2026-10-27。按 [base-baseline.md](../base-baseline.md) 的指标整理，列顺序为 **MA11、MA89、MA8、MA9、CR3、R、C、A、Flux＋基础量化、Flux＋CUDA量化、Megatron＋CUDA量化、Megatron+朴素量化、原生Megatron、原始Flux**。MA8 指无新增逐调用包装的 **MA8-v2**（`F4-cache`）；MA9 指 `F4-zcount`；MA89 为两者组合（`F4-cache-zcount`）。**MA11** 在完整 MA89 上加入注意力 mask／softmax 中间态消除，策略 ID 为 `F4-cache-zcount-csoftmax`，作为本表耗时下降行的比较基准；取值对应 [ma-11.md 第 8 节](ma-11.md#8-ma11-v1-八卡实测有正增量但未达到2目标)已复核的首轮批次。CR3 为本表对原 **F4** 的新名称，指 [MA4-F 消费链融合](ma-4.md)，直接基于 CR2；原始脚本与报告中的策略标识仍为 `F4`。R 指 MA2-R；Flux＋基础量化采用 [a-baseline.md](a-baseline.md) 页首的**接收端融合 v2**，不是该文历史部分的生产端融合 v1。最后三列取自 [Attention/MLP 作用范围文档](../base-phase6-mlp-attention.md) 对应的原始数据，Megatron+朴素量化采用 Attention＋MLP 版本。

**仅测 Flux＋基础量化（2026-11-03 新入口，历史表格数值不变）：** 在 `TempName-New` 仓库根目录执行：

```bash
# 预览；不创建实验目录、不编译、不查询 GPU
bash tools/h100/phase6/run_basic_quant_only.sh

# 正式运行；默认脱离终端后台执行
bash tools/h100/phase6/run_basic_quant_only.sh --execute
```

这个入口只运行下表的 **Flux＋基础量化，接收端融合 v2，策略 `basic`，Attention＋MLP**。不运行 MA11、MA15 或 CUDA 量化对照的完整训练。正式计时固定 **4 个窗口**（两轮各 2 个），每窗口预热 10 步＋计时 50 步，共 **200 个正式 optimizer step、32 份正式 rank 记录**；另保留本组 smoke 数值检查和独立 profile。smoke 内部的 codec／接收端参考检查仍会执行，不属于另一组完整训练。生成的 worker 继承外部 CPU affinity。

默认按原流程准备／构建所需 GemmOnly 与 codec 库；已有构建可通过 `--flux-build` 和 `--codec-build` 传入并验明身份。启动后查看打印目录下的 `batch-state.json` 和 `handoff/container-supervisor.log`；完成结果位于 `compare1/report.md`、`compare1/report.json`，给出全窗口均值、中位数、吞吐和 loss，不生成同期配对加速比。需要前台运行时添加 `--foreground`。原 `run_basic_quant.sh --execute` 仍是 basic／cuda 双组比较；其单组等价命令为 `run_basic_quant.sh --basic-only --execute`。

| **组别** | **策略概括** |
| --- | --- |
| **MA11** | **MA89＋注意力中间态消除**：在 softmax 内消费原 bool mask，反向直接生成分数梯度，消除独立 mask、BF16 乘积临时态及原位 view 梯度重建；保留原归约树、BF16 舍入及 QK／PV 算术。 |
| **MA89** | **CR3＋前后向激活复用＋保序多张量统计归约**：组合原 MA8-v2 与 MA9，在保留前向完整激活的基础上融合 optimizer 零梯度统计。 |
| **MA8** | **CR3＋前后向激活复用**：QKV／FC1 的 autograd 节点独立持有前向 AllGather 的完整输入，反向直接复用，删除重复输入 AllGather；以额外激活显存换取通信消除。 |
| **MA9** | **CR3＋保序多张量统计归约**：把逐参数零梯度统计合并为分块计数与按序汇总两个 kernel，保留原 FP32 累加顺序、舍入和标量 AllReduce；梯度范数、裁剪与 Adam 不变。 |
| **CR3** | **CR2＋decoder 内融合 bias/residual 消费**：保留原归约顺序与两次 BF16 舍入，删除后继两个独立 add kernel，沿用原 LayerNorm；不叠加 CUDA Graph。 |
| **R** | **交替排序＋打包来源预取＋执行优化**：实例持有配置，返回独立输出，省去逐次配置与输出 clone。 |
| **C** | **交替排序＋到达优先＋选择性量化＋双工作区**：实例持有配置与独立输出。 |
| **A** | **交替排序** |
| **Flux＋基础量化** | **原生 Hopper Flux GemmOnly＋独立 TACO CUDA 编码＋NCCL All-to-All**|
| **Flux＋CUDA量化** | **原生 Flux GEMM＋独立 TACO CUDA 编解码＋NCCL All-to-All**。 |
| **Megatron＋CUDA量化** | **原生 Megatron GEMM＋独立 TACO CUDA 编解码＋NCCL All-to-All**。 |
| **Megatron+朴素量化** | **原生 Megatron GEMM＋PyTorch 张量版 TACO 编解码＋NCCL All-to-All** |
| **原生Megatron** | **原生 Megatron BF16 训练路径**，Attention/MLP 不量化。 |
| **原始Flux** | **原生 Hopper Flux BF16 GEMM＋Reduce-Scatter**，Attention/MLP 不量化。 |

“到达优先”使用冻结的原到达表；C/CR3 及基于 CR3 的 MA8、MA9、MA89、MA11 使用原 1/64 预算上限的选择性量化 mask，A/R、Flux＋基础量化、两组 CUDA 量化基线及 Megatron+朴素量化使用全远端量化。Flux＋基础量化覆盖 Attention＋MLP。原生Megatron、原始Flux 均不量化；所有组的本地贡献保持 BF16。MA11 消费的注意力 bool mask 与这里的量化选择 mask 是不同对象。

**本表是已有结果汇总，不是一次十四组同场实验。** MA8、MA9、MA89、MA11 各取 2026-10-27 对应实验的一个已复核 trial；MA89 的同期对照为该批自己的 F4-cache，MA11 的同期对照为该批自己的 F4-cache-zcount，均没有原始 F4 第三组。CR3 使用 2026-10-25 的 MA4-F 单轮批次；R/C 保留 2026-10-23 的 CR2 两轮批次。A 和两组 CUDA 量化基线使用原 baseline 批次。Flux＋基础量化使用 2026-10-25 的接收端融合 v2 单轮批次；原生Megatron、原始Flux 使用新监控体系 a–h，Megatron+朴素量化使用作用范围实验 a–d。这三个历史参考列每窗口计时 **20 步**，其他列为 **50 步**；量化作用范围与是否量化见策略表。

| **指标** | **MA11** | **MA89** | **MA8** | **MA9** | **CR3** | **R** | **C** | **A** | **Flux＋基础量化** | **Flux＋CUDA量化** | **Megatron＋CUDA量化** | **Megatron+朴素量化** | **原生Megatron** | **原始Flux** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 平均耗时（ms/step） | **450.662** | 459.806 | 474.004 | 489.878 | 501.433 | 515.716 | 516.784 | 536.692 | 582.318 | 610.632 | 632.905 | 1052.781 | 652.615 | 593.615 |
| 选定窗口耗时（ms/step） | **450.662** | 459.806 | 473.600 | 488.754 | 501.433 | 515.154 | 516.811 | 536.618 | 586.668 | 614.115 | 637.375 | 1052.199 | 575.367 | 542.205 |
| MA11 相对该列耗时下降（选定窗口耗时比） | +0.000% | +1.989% | +4.843% | +7.794% | +10.125% | +12.519% | +12.799% | +16.018% | **+23.183%** | +26.616% | +29.294% | +57.169% | +21.674% | +16.883% |
| 吞吐（tokens/s） | **18177.69** | 17816.21 | 17282.55 | 16722.53 | 16337.18 | 15884.70 | 15851.88 | 15263.87 | 14067.92 | 13415.61 | 12943.49 | 7781.29 | 12552.57 | 13800.19 |
| 数据来源 | MA11 首轮批次 | MA89 组合批次 | MA8-v2 批次 | MA9 批次 | MA4-F 批次 | CR2 批次 | CR2 批次 | 原 baseline 批次 | 基础量化 v2 批次 | 原 baseline 批次 | 原 baseline 批次 | 作用范围实验 a–d | 新监控体系 a–h | 新监控体系 a–h |

**选值口径：** “选定窗口耗时”行中，**MA11、MA89 与 CR3 均取全部正式窗口的算术平均值**；Flux＋基础量化、Flux＋CUDA量化、Megatron＋CUDA量化保留各自最慢窗口值；MA8、MA9 及其他列取全部窗口中位数。原有其他列的数值不变。每个窗口的步时都是该窗口正式训练步的平均值，不是单个 step 的瞬时极值；该行包含均值、最大值和中位数，并非统一的中位数口径。

所有列保留对应范围内的全部窗口；原生Megatron、原始Flux 的均值也包含历史慢窗口。三个历史参考列每窗口正式计时 20 步，其他列 50 步。

**MA8／MA9／MA89／MA11 的同期配对结果：** 下表使用各实验自己的同期对照、全部八个完整配对块和 bootstrap 95% CI，独立于上表的跨批次选值比。

| 实验 | 同期对照→候选 | 平均耗时（ms/step） | 平均减少（ms/step） | 配对收益 %［95% CI］ |
| --- | --- | ---: | ---: | --- |
| [MA8-v2](ma-8.md) | CR3→CR3＋前后向激活复用 | 502.381→474.004 | 28.377 | +5.648［+5.252, +6.035］ |
| [MA9](ma-9.md) | CR3→CR3＋保序多张量统计归约 | 503.310→489.878 | 13.431 | +2.669［+2.306, +3.062］ |
| [MA89](ma-8&9.md) | F4-cache→F4-cache＋MA9 | 474.262→459.806 | 14.456 | +3.048［+2.836, +3.253］ |
| [MA11](ma-11.md) | MA89→MA89＋注意力中间态消除 | 459.638→450.662 | 8.976 | +1.952［+1.711, +2.271］ |

MA8、MA9、MA89 三项均完成数值与实际路径检查，各自 8/8 个配对块更快，并达到各自本轮 2% 门槛；本表各收录一个独立 trial。MA8 每 rank 每步删除 256 次目标反向输入 AllGather，allocated 峰值增加约 0.995 GiB；MA9 将 rank0／其他 rank 的零统计计算 kernel 从 1945／970 次降至两次，保留原标量 AllReduce。MA89 的 trace 同时验证两项机制生效，相对同期 F4-cache 的 allocated 峰值仅增加约 0.213 MiB。

**MA11 首轮有正增量，但未达到 2% 门槛。** 本轮数值与实际路径检查通过，8/8 个配对块更快；配对点估计 1.952%、区间下界 1.711%，`increment_over_MA89_validated=true`，`target_validated=false`。每 rank 每 step 执行 128 次融合前向和 128 次融合反向，删除注意力所属的 view 梯度回放及其累计 640 次／20 GiB DtoD 复制；原 MA8 缓存、MA9 零统计与 row 投影路径保留。两组 allocated 峰值同为 17.423663 GiB，复制流量减少不等于驻留显存减少。上述 profile 路径证据不替代正式计时的 8.976 ms 净节省。

**主表 MA8 列均值为 474.004 ms，MA89 组合批次自己的 MA8 对照均值为 474.262 ms；主表 MA89 列均值为 459.806 ms，MA11 首轮自己的 MA89 对照均值为 459.638 ms。** MA89 的 +3.048% 和 MA11 的 +1.952% 分别使用各自同期对照，不能相加。主表 MA11 相对历史 MA89 的 +1.989%、相对历史 CR3 的 +10.125% 为跨批次均值比，不是同场配对收益。

**CR3 同场验证：** 本轮 CR3 / CR2 平均步时为 **501.433 / 509.852 ms**，平均减少 **8.419 ms/step**；完整块配对收益为 **+1.652%［95% CI：+1.297%, +2.034%］**，8 个配对块全部更快。前向逐位一致、完整梯度/参数更新指纹、正式训练轨迹和融合路径检查通过。详见 [MA4 实测与离线复核](ma-4.md#6-本轮-ma4-f-结果相对-cr2-有端到端增量)。

[作用范围文档](../base-phase6-mlp-attention.md#配对耗时下降) 中上述新增三组的“平均耗时”分别为 **1052.654 / 575.986 / 542.284 ms**，其口径是**八个轮次中位数的平均**。本表为保持各列一致，重新使用全部原始窗口计算算术均值，得到 **1052.781 / 652.615 / 593.615 ms**；二者不是同一统计量，这三组的中位数列于“选定窗口耗时”行。

数据来源：

- **MA11：** `ma11-v1-causal-softmax-1x-50steps-20261027-121342-060643`，按 [ma-11.md 第 8 节](ma-11.md#8-ma11-v1-八卡实测有正增量但未达到2目标)收录候选 `F4-cache-zcount-csoftmax` 的全部 **8 个正式窗口、64 份 rank 记录、400 个正式 step**；同期 MA89 另有相同工作量，两组共 16 窗口、128 份正式 rank 记录、800 步。见[原始训练报告](../../../../logs/h100/phase6/ma11-v1-causal-softmax-1x-50steps-20261027-121342-060643/fullstep/report.json)及[首轮离线复核](../../../../outputs/h100/phase6/ma11-results-audit-20261027/audit.json)。本次重新核对全部 128 份正式 rank 记录的窗口步时；MA11 的平均耗时、选定值与吞吐均取该首轮全样本。日志目录另有[较晚批次 `20261027-202038-900556`](../../../../logs/h100/phase6/ma11-v1-causal-softmax-1x-50steps-20261027-202038-900556/fullstep/report.json)，本页未与首轮合并。
- **MA8：** `ma8-v2-cache-1x-50steps-20261027-002245-655981`，取候选 `F4-cache` 的全部 **8 个正式窗口、64 份 rank 记录、400 个正式 step**；同期 F4 另有相同工作量。见[原始训练报告](../../../../logs/h100/phase6/ma8-v2-cache-1x-50steps-20261027-002245-655981/fullstep/report.json)及[机制与实测总结](ma-8.md)。不混入对照曾安装额外包装的 MA8-v1。
- **MA9：** `ma9-v1-zcount-1x-50steps-20261027-081044-826367`，取候选 `F4-zcount` 的全部 **8 个正式窗口、64 份 rank 记录、400 个正式 step**；同期 F4 另有相同工作量。见[原始训练报告](../../../../logs/h100/phase6/ma9-v1-zcount-1x-50steps-20261027-081044-826367/fullstep/report.json)及[机制与实测总结](ma-9.md)。不使用此前动态符号导出失败批次。
- **MA89：** `ma89-v1-cache-zcount-1x-50steps-20261027-104719-859382`，取候选 `F4-cache-zcount` 的全部 **8 个正式窗口、64 份 rank 记录、400 个正式 step**；同期 F4-cache 另有相同工作量。见[原始训练报告](../../../../logs/h100/phase6/ma89-v1-cache-zcount-1x-50steps-20261027-104719-859382/fullstep/report.json)、[组合实测总结](ma-8&9.md)及[离线复核](../../../../outputs/h100/phase6/ma89-results-audit-20261027/audit.json)。上述三个实验各自共 16 窗口、128 份正式 rank 记录、800 个正式步骤；本页更新重新核对了合计 384 份正式 rank 记录的窗口步时与新增候选列的 loss。
- **CR3：** `ma4-f-1x-50steps-20261025-115124-977713`，候选保留 **8 个正式窗口、64 份 rank 记录、400 个正式 step**；同期 CR2 另有相同工作量，两组共 16 窗口、128 份 rank 记录、800 步。见[原始训练报告](../../../../logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/report.json)、[离线复核](../../../../outputs/h100/phase6/ma4-f-results-audit-20261025/audit.json)及[复核脚本](../../../../outputs/h100/phase6/ma4-f-results-audit-20261025/audit.py)。四份算子/训练报告与 profile 汇总均由原始记录重算一致；CR3 与本轮 CR2 的 loss、梯度和最终参数指纹一致。
- **R/C：** `ma2-cr2-2x-50steps-20261023-153237-090898`，合并 compare1、compare2，各组保留 16 个正式窗口。见[原始报告](../../../../logs/h100/phase6/ma2-cr2-2x-50steps-20261023-153237-090898/fullstep/report.json)及[实测总结](ma-2-ablation.md#11-cr2-八卡实测结果与离线复核2026-10-23)。C 与原 baseline C 为相同实现，但这里使用新批次实测均值 516.784 ms，不混入旧批次的 517.734 ms。
- **Flux＋基础量化：** 取 [a-baseline.md 的接收端融合 v2](a-baseline.md)，批次为 `basic-quant-receiver-1x-50steps-20261025-090410-798870`，协议为 `h100-basic-receiver-fusion-fullstep-v2`，策略 ID 为 `basic`。仅取本组 **12 个正式窗口、96 份 rank 记录、600 个正式 step**，每窗口 10 步预热＋50 步正式；不混入该文历史 v1 的数据。见[原始报告](../../../../logs/h100/phase6/basic-quant-receiver-1x-50steps-20261025-090410-798870/compare1/report.json)及 [a-baseline.md 中的复核说明](a-baseline.md#回传数据复核)。本组相对其同期 CUDA 组（平均 609.744 ms）的配对收益为 **+4.498%［95% CI：+4.198%, +4.835%］**；主表 CUDA 列仍来自下面的原 baseline 批次，不能将该区间套用于主表两列的跨批次差值。
- **A/Flux＋CUDA量化/Megatron＋CUDA量化：** `baseline-taco-cuda-3x-50steps-20261021-162929-989672`，合并 compare1、compare2、compare3，各组保留 30 个正式窗口。见[原 baseline 汇总](../base-baseline.md#3-完成范围与全样本结果)及[原始审计统计](../../../../outputs/h100/phase6/baseline-taco-cuda-results-audit-20261021/audit.json)。
- **Megatron+朴素量化：** 取 [base-phase6-mlp-attention.md](../base-phase6-mlp-attention.md) 的 `native_taco_attn_mlp`，即 Attention＋MLP 同时使用朴素张量 codec。来源为 `TTention-MLP-gpt67-tp8-20261015-012238-repeat1` 至 `repeat4`：报告 [a](../../../../logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-012238-repeat1/compare/report.json)、[b](../../../../logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-012238-repeat2/compare/report.json)、[c](../../../../logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-012238-repeat3/compare/report.json)、[d](../../../../logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-012238-repeat4/compare/report.json)。每次两轮、每轮四个该策略窗口，合计 **32 窗口、256 份 rank 记录、640 正式步**；不混入 Attention-only 或 MLP-only 的朴素量化结果。
- **原生Megatron、原始Flux：** 分别为上述作用范围文档引用的新监控体系 a–h 中 `native`、`original`。来源为 [a/b](../../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat1/compare/report.json)、[c/d](../../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat2/compare/report.json)、[e/f](../../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat1/compare/report.json)、[g/h](../../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat2/compare/report.json) 四套会话，各 **48 窗口、384 份 rank 记录、960 正式步**。上述三个历史参考列均核对原始 rank 记录的完成状态、10/20 步协议、实际路由、有限 loss/gradient 和逐 rank loss 一致性；各轮中位数与报告吻合。见[重算统计](../../../../outputs/h100/phase6/historical-baselines-summary-20261023/summary.json)及[计算脚本](../../../../outputs/h100/phase6/historical-baselines-summary-20261023/summarize.py)。

各列都测完整 optimizer step，包含前后向、通信、参数更新与清梯度。每窗口预热 10 步；Megatron+朴素量化、原生Megatron、原始Flux 三个历史参考列正式计时 20 步，其他列 50 步。步时取八 rank 最大墙钟除以各自正式步数；平均耗时为全部窗口的算术均值；“选定窗口耗时”按上面的均值／最慢／中位数口径取值。每步 8192 token，吞吐为 `8192000 / 平均ms`。

“MA11 相对该列耗时下降”为 `100 × (1 − MA11平均ms / 该列选定窗口ms)`，使用未舍入原始值计算。分子为 MA11 首轮的 **450.6624015075795 ms**；MA89 分母使用原组合批次均值 **459.8060027899919 ms**，得到 **+1.989%**；CR3 分母使用均值 **501.43307737249415 ms**，得到 **+10.125%**。Flux＋基础量化、Flux＋CUDA量化、Megatron＋CUDA量化仍取最大值，MA8、MA9 及其他列取中位数；MA11 自身为零。**该行包含跨批次、混合选值口径的描述性差值，没有十四组同场配对置信区间；不能用它替代各机制的同期配对收益。历史参考列还存在量化范围、是否量化、20/50 步协议及慢窗口等差异，不能将这些百分比解释为单纯量化、排序或新增机制的净收益。** 上文 MA11/MA89 的 +1.952%、MA89/F4-cache 的 +3.048% 与 CR3/CR2 的 +1.652% 分别使用各自批次的同期对照，按完整块配对计算，与主表选值比不是同一统计口径。历史 CR2/C 两轮配对收益为 +1.027% / +1.097%，CR2/R 为 +0.630% / +1.083%，详见实测总结。


