# MA2：连续系数打包预取与 MA2-R 执行优化

更新：2026-10-23。MA2 是 [MA1](ma-1.md) 的后续优化。原版 V 优化接收端读取与寄存器使用，后续 MA2-R 保留 V decoder，进一步优化固定配置提交和输出生命周期。R 的目标为相对原始交替 A 达到至少 1% 的完整 step 收益，并在两次比较中确认优于原 MA2 V。

**最新 MA2-R 批次 `20261022-202958-492886` 已完成两次比较、72 个训练窗口和 3600 个正式 optimizer step，正确性、执行路径及性能目标均通过。** 两轮 R/A 为 **+3.761%［95% CI：+3.444%, +4.160%］**、**+3.718%［+3.413%, +4.038%］**；两轮 R/V 为 **+3.398%、+3.589%**，区间均为正。两轮 R/A 的区间下界也均超过 1%，通过单列的严格目标。最新结果见第 10 节。

**最新 MA2-R 完整训练结果。** 正值表示加速，方括号为 95% CI。A 为原交替 scalar 实现，V 为原 MA2 打包预取，R 为 V 加实例配置与独立输出优化。收益按完整块配对计算，不直接用表内算术均值相除。

| 比较 | A / V / R 平均步时（ms） | R/A 收益 % | R/V 收益 % | V/A 收益 % |
| --- | --- | --- | --- | --- |
| **MA2-R compare1** | **536.154 / 534.135 / 515.985** | **+3.761 [+3.444, +4.160]** | **+3.398 [+3.200, +3.593]** | +0.375 [−0.002, +0.771] |
| **MA2-R compare2** | **535.499 / 534.784 / 515.589** | **+3.718 [+3.413, +4.038]** | **+3.589 [+3.324, +3.835]** | +0.133 [−0.223, +0.488] |

**原版 MA2 V 的历史结果。** 下表 P 是旧 MA1 预取；复测为 `155228` 批次，首批为 `105738` 批次，各自包含 compare1/compare2。原版 V 未达到目标的结论保留，不能将 R 的收益追溯算作打包 decoder 的收益。各批统计分别报告，不合并样本。

| 比较 | A / P / V 平均步时（ms） | V/A 收益 % | V/P 收益 % | P/A 收益 % |
| --- | --- | --- | --- | --- |
| **复测 compare1** | **534.366 / 535.078 / 535.105** | **−0.138 [−0.386, +0.124]** | **−0.005 [−0.282, +0.278]** | **−0.133 [−0.398, +0.103]** |
| **复测 compare2** | **536.031 / 534.670 / 535.608** | **+0.079 [−0.201, +0.348]** | **−0.176 [−0.520, +0.226]** | **+0.254 [−0.144, +0.567]** |
| 首批 compare1 | 587.903 / 566.581 / 589.044 | −0.510 [−5.949, +3.765] | −3.926 [−10.087, +0.575] | +3.287 [−0.716, +9.547] |
| 首批 compare2 | 536.483 / 556.064 / 547.882 | −1.844 [−6.584, +0.834] | +1.178 [−6.365, +9.033] | −3.058 [−9.046, −0.040] |

原版 V 两批累计完成 144 窗口、7200 个正式 optimizer step；加上本次 R 批次，本页三个批次累计 216 窗口、10800 正式步。这是独立初始化窗口的累计工作量，不是同一模型连续训练 10800 步。

MA2-R 的设计和复测启动命令见第 9 节。原 MA2 V 及历史结果保持不变；最新收益归属于 R 的执行实现优化，不能单独归给配置绑定或 clone 去除中的某一项。

## 1. 为什么继续优化数据读取

MA1 最新完整训练中，P/A 为 +0.333%、P/W 为 +0.400%，但独立 MLP 算子的 P/W 为 −1.282%，训练 profile 中 P 的 decoder 也略慢于 W。这说明应保留完整训练中的小幅正结果，同时减少预取本身的指令和寄存器负担，不能简单增加预取深度。

旧 MA1 P 中，每个 lane 持有 `lane+32*j` 的四个系数。每个来源需四条 8-bit payload load 和两条 scale load；当前与下一来源各保留四个原始字节寄存器及两个 scale。此次改为每 lane 持有 `4*lane+j` 四个连续系数，利用一次对齐的 32-bit load 读取四个 FP8 字节，保留原来的下一来源预取。

| 接收端数据操作 | 原 MA1 P | MA2 V |
| --- | --- | --- |
| 每 lane 的四个系数 | `lane, lane+32, lane+64, lane+96` | `4*lane, 4*lane+1, 4*lane+2, 4*lane+3` |
| 每来源 payload 读取 | 4 条 `ld.global.u8` | 1 条 `ld.global.u32` |
| 两个 scale 读取 | 2 条 FP32 load | 相同的 2 条 FP32 load |
| 四个 FP8 转为 half | 四次单值转换 | 两次 FP8x2 转换，再分别转 FP32 |
| 本地 BF16 读取 / 输出写回 | 四个分散元素 | 对齐的 64-bit load / store |
| 累加算术 | 逐来源、两处 BF16 舍入 | 保持相同，不改成 FP32 来源求和 |

“打包”只改变接收端寄存器表示和访存宽度，**不改变网络 packet 布局、量化范围或通信字节数**。各源仍发送原来的 H128/E4M3 数据及两个 FP32 scale，来源顺序仍为 `rank+1,…,rank`，本地 BF16 最后加入。

## 2. Hadamard 阶段和同步保持

连续系数布局使 H128 的距离 1、2 两级在单线程的四个寄存器中执行；距离 4、8、16、32、64 则使用 warp shuffle，其 lane 距离为物理距离除以 4。每个物理系数仍严格经历 `1→2→4→8→16→32→64` 的原加减顺序。

这里不是将旧布局的最后两级直接搬到开头，而是同时改变系数归属，使每个物理元素的计算图不变。缩放乘法、归一化、非有限过滤、除以 adaptive scale、解码后的 BF16 舍入及每次来源相加后的 BF16 舍入均保持。原始来源数据只在发布屏障后读取，复用屏障继续保护工作区。

连续布局还带来一个需要显式处理的编译器问题：原来乘法与第一级加减之间隔着 shuffle，现在第一级在寄存器内执行，nvcc 默认可能将 `q*x + q*y` 的一项乘法收缩进 FMA，改变原有的乘法舍入。V 对四个量化缩放乘法使用 `__fmul_rn`，保留先乘法舍入、再加减的顺序；没有全局关闭 FMA，也没有启用 fast math。回归用例包含确实产生一位 FP32 差异的反例，最终 SASS 中循环和最后远端来源均保留四条独立 FMUL。

向量读取的地址满足所需自然对齐：payload 的 group 偏移为 `128*group` 字节，lane 偏移为 `4*lane` 字节；本地及输出的四个 BF16 对应 8-byte 对齐地址。自然对齐要求见 [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#addresses-as-operands)，FP8x2 转换语义见 [CUDA 12.8 Math API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html)。

实现见 [ma2_decode.cuh](../../../../tools/h100/phase6/ma2_decode.cuh) 与 [ma2_sources.py](../../../../tools/h100/phase6/ma2_sources.py)。使用精确形状 `TP8, M2048, N4096`，Attention/MLP 本地 K 为 512/2048。GEMM、交替映射、encoder、两次屏障、单工作区、逐次配置/复位及输出 clone 保持相同。

## 3. 同场 A/P/V 对照

| 组 | 本次实现 | 用途 |
| --- | --- | --- |
| A | 原融合 v2 scalar decoder，基础交替、全远端量化 | 原始交替参照 |
| P | 原 MA1 的预取 runtime/header，生成源码逐字保持 | 复测上一版，不使用历史计时替代 |
| V | 新连续四系数布局、打包读取/转换/写回、来源预取 | 本次优化候选 |

三组均不使用到达表或选择性量化。本轮不包含旧 MA1 W，也不包含 baseline C；V/P 检查新增打包优化的净效果，P/A 记录旧 MA1 是否在同场重现。若 V/A 提升但 V/P 没有正增量，不能认定新优化优于上一版。

[ma2_build.py](../../../../tools/h100/phase6/ma2_build.py) 为三组建立独立目录并冻结源码。P 直接复用原 `ma1_sources.sources(..., 'P')`，V 只替换接收 runtime 并新增自己的 decoder 头文件。旧 `ma1*.py`、`ma1_decode.cuh` 与已有结果保留，原实验仍可复现。

已将 MA1 成功批次 `ma1-prefetch-1x-50steps-20261021-233916-307862` 与 MA2 首批 `105738` 的 `fullstep/plan.json` 构建身份逐文件对比：去除不同批次目录前缀后，**A 的 351 个、P 的 352 个登记源码文件哈希分别全部一致**。两批模型配置、预热/正式步数、共享环境字段和 Megatron commit 也一致。最新复测与首批的 A/P/V 登记源码哈希也分别一致，核对结果见第 8.1 节。因此这里的 A/P 与 MA1 是相同实现；各批次重新构建、重新计时，不代表跨批次动态库文件或测得耗时必须相同。

## 4. 两次独立比较与 1% 判定

本次“两轮”指 **compare1、compare2 两次独立完整训练比较**；每次内部仍有顺序平衡的前后两个半程，不把这两个半程当作两次独立实验。

**compare1 与 compare2 之间没有切换机制或调参。** 两次共用本批同一套 A/P/V 构建，模型参数、模型 seed1234、token_seed4321、预热/计时步数及 12 块的 A/P/V 执行顺序均相同。变化的是 trial/run ID、结果目录和实际执行时段；compare1 完成后再执行 compare2，每个窗口均为新进程并从相同初始化开始，不承接上一轮训练权重。这是相同配置的运行重复，不是不同随机种子或不同训练数据的对照。

| 项目 | 每次比较 | 两次合计 |
| --- | ---: | ---: |
| 完整 A/P/V 块 | 12 | 24 |
| 正式训练窗口 | 36 | **72** |
| 正式 rank 记录 | 288 | 576 |
| 正式 optimizer step | 1800 | **3600** |
| 正式窗口预热步 | 360 | 720 |

每窗口独立初始化，10 步预热＋50 步正式；两次比较有不同 run ID。模型继续使用 GPT6.7B、32 层、H4096/FFN16384、32 heads、S2048、TP8、BF16、micro batch1/global batch4，每步 8192 token。`NCCL_NVLS_ENABLE=0`、`CUDA_DEVICE_MAX_CONNECTIONS=1`、初始化/RNG/token、全远端量化和反向路径均固定。

先进行算子预检、36 个算子计时窗口及 48 份算子 profile；完整训练共用一套模型 smoke 预检和独立采集的 24 份完整 step profile。所有正式窗口采用新进程，旧批次计时不混入。算子性能是否为正不作为跳过完整训练的理由，但正确性或执行证据失败会停止并保留现场。

性能指标仍为八 rank 最大窗口墙钟除以 50；按完整三组块配对 bootstrap 10000 次，seed20261021，报告全样本、负收益和区间，不选择最好一轮。

两次比较**各自**须同时满足：

1. **V/A 点估计 ≥1.0%，且其 95% CI 下界 >0。** 这是本次 1% 点估计目标。
2. **V/P 点估计及 95% CI 下界均 >0。** 确认新增优化比旧 MA1 P 更快。
3. 算子逐位正确性、模型逐 rank loss/gradient 一致性、实际调用和 profile 路径检查全部通过。

满足后，批次 `target_validated=true`。报告同时单列 **V/A 的 95% CI 下界是否也达到 1%**：`strict_one_percent_confidence_validated`。点估计达到 1% 不等于已用置信区间证明收益至少 1%；两种结论不能混用。P/A 是复测控制，只记录，不强求上一版这次必须通过。

两次的结果分别报告；只一轮达到目标也不能将整个批次写成通过。固定合成输入上的短程训练不代表真实语料长期收敛。

## 5. 原版 MA2 的复测入口

同步本次新增 `ma2*.py`、`ma2_decode.cuh`、`detach_ma2.py` 和已有工具依赖后，在原服务器 `/workspace/TempName` 执行：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_ma2.py --execute
```

默认重新准备和编译 A/P/V，CUDA12.8 编译、CUDA12.4 运行、jobs2、相邻 `../Megatron-LM`；双 fork 后台托管。容器和主机需持续运行。输出为 `logs/h100/phase6/ma2-packed-prefetch-2x-50steps-<北京时间戳>/`，其中：

```text
batch-state.json                    完成状态与两个 1% 判定字段
handoff/container-supervisor.log   后台总日志
build/                            A/P/V 隔离构建和 manifest
operator/                         算子逐位预检、计时、实际 mapping 和 profile
fullstep/                         两次比较的 plan、原始 rank 记录与 report.json/report.md
```

去掉 `--execute` 只预览，无目录创建、编译、fork 或 GPU 查询。`--foreground` 可前台诊断，`--python`、`--megatron`、`--cuda-home`、`--build-cuda-home`、`--nccl-root` 可指定环境。`--ma2-build` 只接受已完成且通过验证的本轮 A/P/V 构建，不能传入旧 MA1 的 A/W/P 构建。

**再次复测两轮时，直接重新执行上述 `detach_ma2.py --execute` 命令即可。** 默认按新的北京时间戳创建独立批次目录，重新构建 A/P/V，再完成两次比较，共新增 72 窗口、3600 正式步；模型、机制、种子和验收条件保持不变。无需修改脚本，也不要将 `--out` 指向已完成的旧批次。新结果单独报告，不覆盖或自动合并第 7 节已有结果；启动前让上一批及其他训练任务结束，避免复测任务相互竞争资源。

回传整个批次，保留冻结脚本、plan、rank JSON、mapping、trace、manifest 和状态。离线报告入口：

```bash
python3 tools/h100/phase6/ma2_bench.py report /path/to/batch/operator --offline
python3 tools/h100/phase6/ma2_bench.py profile-report /path/to/batch/operator --offline
python3 tools/h100/phase6/ma2_fullstep.py report /path/to/batch/fullstep --offline
```

## 6. 上机前本地验证与限制

2026-10-22 已完成以下本地检查：

| 检查 | 结果与证据 |
| --- | --- |
| 构建身份、数值模拟、报告、profile 解析、两轮编排与启动器 | **70 项 CPU 测试通过**；[测试登记](../../../../outputs/h100/phase6/ma2-validation-20261022/cpu-validation.json)、[完整日志](../../../../outputs/h100/phase6/ma2-validation-20261022/cpu-tests.log) |
| A/P/V 实际 runtime 编译与两种 K 的交替 mapping | 三组均通过 host mapping 和 CUDA 12.8 **SM90 目标文件编译**，保存 PTX/SASS；[编译登记](../../../../outputs/h100/phase6/ma2-validation-20261022/runtime-check/validation.json) |
| P/V 同一编译单元、相同选项的 decoder 比较 | 最终 P **40**、V **34** 个寄存器/线程；两者 stack/spill 均为 0；[指令与数值审计](../../../../outputs/h100/phase6/ma2-validation-20261022/decoder-check/validation.json) |
| FP8x2 与单值转换的主机实现 | 穷举 **65,536 对字节**（其中 64,516 对均为有限值），half 位模式全部一致；[转换检查](../../../../outputs/h100/phase6/ma2-validation-20261022/decoder-check/fp8x2-host.json) |
| 一行命令预览 | 正确生成两次比较、72 窗口、3600 正式步的计划，未创建目标目录或启动训练；[预览记录](../../../../outputs/h100/phase6/ma2-validation-20261022/launch-preview.txt) |

SASS 确认每来源读取指令从 6 条降到 3 条、FP8 转换从 4 条降到 2 条，本地读取和最终写回分别从 4 条降到 1 条。每来源仍有 20 条 shuffle；下一来源的读取确实发射在当前来源首条 shuffle 之前，并在后续迭代消费。这些是编译产物中的指令观察，不是运行时计数或耗时。

本节记录的是上机前本地检查，当时没有执行 GPU kernel，也没有完成整套 Flux 动态库链接、八卡逐位正确性或完整训练性能验证。后续服务器批次已完成运行，首批与复测结果分别见第 7、8 节。静态指令或寄存器减少不等于已经改善实际 occupancy、访存重叠或完整 step，寄存器分配粒度也可能使 occupancy 不变；不能将 MA1 的 0.3%–0.4% 与一个推测收益相加。

## 7. 首批两轮八卡实测总结（2026-10-22，105738 批次）

### 7.1 批次完成情况与证据

本批目录为 `logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-105738-798285/`。依据[批次状态](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-105738-798285/batch-state.json)与[后台总日志](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-105738-798285/handoff/container-supervisor.log)，构建、算子计时、算子 profile、完整训练计时及完整训练 profile 均执行完成，最终 `status=completed`、`stage=done`。

| 检查项 | 本批结果 |
| --- | --- |
| 正式训练工作量 | 两次比较，各 12 个 A/P/V 块；合计 72 窗口、576 份 rank 计时记录、3600 个正式 optimizer step |
| 算子正确性 | `correctness_passed=true`、`bitwise_equal=true` |
| 模型正确性 | `correctness_passed=true`、`identical_trajectories=true`；配对 loss/gradient 轨迹一致 |
| 算子执行路径 | `operator_execution_path_validated=true` |
| 完整训练执行路径 | `fullstep_execution_path_validated=true` |
| 1% 点估计目标 | 两次均未通过；批次 `one_percent_point_target_validated=false` |
| 新增打包机制收益 | 两次均未通过；批次 `packed_benefit_validated=false` |
| 最终性能验收 | `target_validated=false`、`acceptance_status=not_validated` |

已从回传原始 rank 记录离线复算完整训练报告，结果与保存的 `fullstep/report.json` 完全一致。**`completed` 和性能 `target_validated` 是两个不同状态，本批不存在因启动退出而少跑窗口的问题。** 此前 MA1 的 CPU/GPU 镜像 annotation 解析故障没有在本批复现；没有证据将本批性能波动归因于进程绑定，因此未以“修复启动器”为由改动脚本或放宽门槛。

### 7.2 完整训练：未达到 1% 目标

完整训练汇总表已置于页首，数据来自[完整训练报告](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-105738-798285/fullstep/report.md)及[原始统计 JSON](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-105738-798285/fullstep/report.json)。正值表示候选更快，负值表示更慢；方括号为配对 bootstrap 95% CI。

平均步时为窗口步时的算术平均；收益使用配对块耗时比的对数均值再转换为百分比，不直接用表中算术均值相除。两次 V/A 点估计均为负，V/A 与 V/P 的四个区间均跨零。compare2 的 V/P 虽有 +1.178% 点估计，但区间很宽且跨零，也不是相对原交替 A 的 1% 收益，不能据此判定通过。

旧 MA1 P 在本批同场对照中也未复现稳定正收益：compare1 的 P/A 区间跨零，compare2 的 P/A 为负且区间低于零。这不改写上一批 MA1 的测量结果，但说明其 +0.3%–0.4% 尚不能作为跨批次稳定收益。两批计时不合并，不选择其中最好的一轮。

### 7.3 波动与半程结果

compare1 前半程 A/P/V 平均步时为 641.272/599.859/642.128 ms，后半程为 534.533/533.302/535.960 ms。全部窗口中，`compare1/block02-P` 达到 787.549 ms/step，`compare1/block01-A` 达到 776.831 ms/step；compare2 也有 `block05-P` 的 775.848 ms/step，而该轮 A/P/V 窗口中位数分别为 534.990/537.119/534.964 ms。不同臂均出现明显较慢的窗口，与宽置信区间相对应。

即使只作诊断查看较平稳的 compare1 后半程，V/A 仍为 −0.269% [−0.995, +0.476]，V/P 为 −0.499% [−1.112, −0.007]，也没有显示新增打包机制的正收益。compare2 前、后半程 V/A 分别为 +0.644% [−0.037, +1.855] 和 −4.395% [−13.638, +0.105]，方向不一致。

这些观察说明本批计时存在明显波动，**不能单凭它们确定是频率、资源竞争、主机调度或其他原因，也不能把所有负结果都归因于噪声**。保留全部窗口，不事后删除慢窗口或仅凭中位数重新判定通过。

### 7.4 算子与 decoder profile：局部改善未转化为整步收益

独立[算子报告](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-105738-798285/operator/report.json)采用完整 `op.forward().clone()` 路径计时：

| 算子 | A / P / V 平均耗时（µs） | V/A 收益 %［95% CI］ | V/P 收益 %［95% CI］ |
| --- | --- | --- | --- |
| Attention | 192.761 / 147.686 / 146.792 | +23.848 [+23.514, +24.158] | +0.606 [+0.263, +0.902] |
| MLP | 266.845 / 227.662 / 226.541 | +15.105 [+14.404, +15.754] | +0.495 [−0.468, +1.345] |

V/A 的较大算子收益包含原 MA1 warp/预取路径相对 scalar A 的差异，不能全部归给本次新增打包机制。V/P 才是本次增量：Attention 有小幅正收益，MLP 区间跨零。

独立[完整训练 profile](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-105738-798285/fullstep/profile-summary.json)中，先取各 rank 的 decoder 时长中位数，再对八 rank 求算术平均，得到：

| decoder 时长（µs） | A | P | V |
| --- | ---: | ---: | ---: |
| Attention | 50.722 | 15.752 | 15.004 |
| MLP | 52.810 | 15.962 | 15.220 |

该 profile 支持 V 的 decoder 在本次独立采集中比 P 更快，且执行路径检查通过；它不等同于正式训练窗口的墙钟收益。**本批最终结论是：MA2 实现及正确性验证通过，局部 decoder/Attention 算子有改善，但未证明完整训练达到 1%，也未证明稳定优于 MA1。** 后续优化应结合整步关键路径和计时波动继续定位，不能仅凭寄存器、指令数或单独 decoder 时长宣称目标已达成。

## 8. 最新两轮复测总结（2026-10-22，155228 批次）

### 8.1 批次完成、同配置复测与正确性

本次目录为 `logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-155228-072061/`。[批次状态](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-155228-072061/batch-state.json)为 `status=completed`、`stage=done`，新增 72 个正式窗口、576 份 rank 计时记录、3600 个正式 optimizer step；另有每窗口 10 步、合计 720 步预热。它是使用相同机制再次运行的独立批次，没有修改 V 或降低验收条件。

对照两批[训练计划](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-155228-072061/fullstep/plan.json)，模型、种子、预热/正式步数、共享环境字段、Megatron commit、完整窗口顺序及 fullstep 控制器哈希均一致。去除批次路径前缀后，A/P/V 分别 351/352/352 个登记源码文件哈希也全部一致。该核对说明源码和预定配置相同，不等于两个运行时段的硬件负载或频率相同。

算子 `correctness_passed=true`、`bitwise_equal=true`；模型 `correctness_passed=true`、`identical_trajectories=true`。算子与完整训练的执行路径检查均通过，实际完成了 A/P/V 三组，没有因启动或 profile 解析错误中断。性能状态仍为 `acceptance_status=not_validated`、`target_validated=false`；1% 点估计目标、95% CI 下界达到 1% 的严格目标、V/P 正增量及 P/A 控制项的批次判定均为 false。

### 8.2 完整训练：波动收窄，但收益仍接近零

最新两轮的均值与配对区间已置于页首，原始结果见[完整训练报告](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-155228-072061/fullstep/report.md)及[统计 JSON](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-155228-072061/fullstep/report.json)。两轮 V/A 分别为 −0.138%、+0.079%；V/P 分别为 −0.005%、−0.176%。两轮共六项 V/A、V/P、P/A 比较的合并区间均跨零。

**本次 V/A 的 95% CI 上界分别只有 +0.124%、+0.348%，均低于 1%**，相较首批更窄的区间仍未支持 1% 目标。因此不能继续仅用首批的大幅计时波动解释整步未加速；在本次配置和样本下，新增打包机制没有显示可确认的整步净收益。这也不证明该机制在所有硬件、形状或负载上必然无效。

| 复测窗口分布（ms/step） | A 最小–最大 | P 最小–最大 | V 最小–最大 |
| --- | --- | --- | --- |
| compare1 | 532.323–537.764 | 533.170–538.609 | 531.639–538.319 |
| compare2 | 533.447–539.555 | 532.313–541.314 | 530.733–538.733 |

本次全部 72 个窗口处于 530.733–541.314 ms/step，没有首批约 695–788 ms/step 的慢窗口。该结论来自全量样本，未删除或筛选窗口；波动减小的原因尚未定位。

两轮各自前后半程的 V/A、V/P 区间也全部跨零。P/A 只有 compare2 后半程为 +0.315% [+0.063, +0.593]，但该轮合并结果为 +0.254% [−0.144, +0.567]，compare1 合并结果为 −0.133% [−0.398, +0.103]。不能用一个半程替代完整比较，旧 MA1 的稳定跨批次正收益仍未得到确认。

### 8.3 算子与 decoder profile：仍有局部改善

独立[算子报告](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-155228-072061/operator/report.json)仍测量完整 `op.forward().clone()` 路径：

| 算子 | A / P / V 平均耗时（µs） | V/A 收益 %［95% CI］ | V/P 收益 %［95% CI］ |
| --- | --- | --- | --- |
| Attention | 192.843 / 148.066 / 147.116 | +23.716 [+23.109, +24.248] | +0.645 [−0.222, +1.457] |
| MLP | 266.220 / 227.467 / 224.776 | +15.567 [+15.031, +16.093] | +1.182 [+0.616, +1.731] |

本次 MLP 的 V/P 为正且区间高于零，Attention 的 V/P 区间跨零；首批则是 Attention 通过、MLP 跨零。**这里的 MLP +1.182% 是单算子的新增机制收益，不能当作完整 optimizer step 达到 1%。** V/A 的大幅算子收益仍包含共同 warp/预取路径相对 scalar A 的差异。

独立[完整训练 profile](../../../../logs/h100/phase6/ma2-packed-prefetch-2x-50steps-20261022-155228-072061/fullstep/profile-summary.json)沿用第 7.4 节口径：各 rank 先取 decoder 时长中位数，再对八 rank 求算术平均。

| decoder 时长（µs） | A | P | V |
| --- | ---: | ---: | ---: |
| Attention | 50.648 | 15.738 | 15.040 |
| MLP | 52.946 | 15.946 | 15.242 |

复测 profile 再次观察到 V 的 decoder 比 P 短约 0.7 µs，但单独采集的 profile 不能替代正式整步墙钟结果。现有证据支持局部解码优化，尚未定位它为何没有转化为完整训练收益。

### 8.4 复核与累计结论

**离线复核通过。** 已使用回传原始记录重新计算算子与完整训练的 `report.json`，并从 48 份算子 trace、24 份完整训练 trace 重新生成两份 `profile-summary.json`；四份结果均与回传文件逐项一致。复核使用 `ma2_bench`、`ma2_fullstep` 的 offline 路径，核验冻结脚本与记录，未重新执行 GPU 训练，也未覆盖原始报告。

本批与首批分开报告，不将四次比较拼成一个更大的统计样本，不选择最好一轮，也不追溯修改预先规定的 1% 门槛。两批累计 144 窗口、7200 正式步均完成，但四次完整比较没有一次同时满足 V/A 的 1% 目标和 V/P 正增量条件。

**截至原版 V 的这次复测，MA2 正确性与执行路径通过，局部 decoder 加速得到两批 profile 的一致观察；完整训练收益仍未确认，相对 MA1 也未显示稳定优势。** 该结果不支持将原版 V 作为已验证的整步性能提升。随后执行优化 R 的实测另见第 10 节；不能把单算子 1% 或静态指令减少直接等同于整步收益。

## 9. MA2-R：优化固定配置提交与输出生命周期

### 9.1 为什么这次优化执行实现

本次选择继续优化 MA2 的实现，暂不另立 MA3。两批 profile 都观察到 V 的 decoder 比 P 约快 0.7 µs，但完整训练未确认收益。因此新实验保留 V 的解码内核，转而减少每次投影调用的固定工作。

原路径每次进入 Python 后检查 device/stream/capture 和 tensor，再通过 ctypes 配置 thread-local 状态，调用 C++，最后通过 ctypes 清空配置。C++ 的 `local_reduction` 已经用 `torch::empty` 创建独立输出，但调用方仍对返回值再做一次 `clone()`。每 optimizer step 有 256 次目标投影，这些工作反复发生。

本次复核 profile 的部分 rank 中，`GemmRSFunction` 的 CPU 中位时长约 220 µs，内部 clone 的 CPU 时长约 31 µs；decoder 则约 15–18 µs。不过 GPU 提交存在排队，profile 也有额外开销，**这些数值只能帮助寻找可减少的工作，不能证明 CPU 已是实际瓶颈，也不能将省下的 profile 时间直接乘以 256 预测整步收益**。R 是否达到 1% 仍由新批次完整训练判定。

配置实例绑定和独立输出的思路已有[历史 op4 工程方案](../到达优先/op-4-engineering.md)可借鉴，本次明确将其作为 MA2 实现优化，不称为新的量化或排序算法。R 保留单工作区与两道原屏障；没有引入历史工程包的双工作区与复用屏障省略，也不是直接改用 baseline C 的选择性量化实现。

### 9.2 R 的改动与不变量

| 项目 | 原 MA2 V | MA2-R |
| --- | --- | --- |
| decoder | 连续四系数打包预取 | 与 V 逐字相同的头文件与 kernel |
| 固定配置 | 每次 forward 经 Python/ctypes 设置和清空 | 构造时绑定到 C++ 实例，forward 用 RAII 设置并恢复线程局部上下文 |
| device/stream/shape/grad 检查 | Python 层逐次检查 | C++ 层检查准确设备、创建 stream、BF16 连续布局、精确 M/N/K、capture 与 autograd 条件 |
| 返回输出 | decoder 独立分配后，调用方再 clone | 直接返回 decoder 已独立分配的输出 |
| packet 工作区 | 单工作区 | 相同，仍为单工作区 |
| 同步 | 发布屏障＋复用屏障 | 两道屏障均保留 |
| 量化、生产与算术 | 基础交替、全远端 H128/E4M3、逐来源 BF16 舍入 | 全部保持 |

`FluxTacoConfig` 的布局不变；K 绑定只在主机侧使用。R 不缓存复用输出张量，也不返回会被下一调用覆盖的工作区视图。不同算子实例持有各自 packet 指针与创建 stream，C++ 在异常路径也恢复此前的线程局部状态；发生 forward 错误的实例禁止继续重用，防止在未完成同步的工作区上重试。

实现通过 [ma2r_sources.py](../../../../tools/h100/phase6/ma2r_sources.py) 生成隔离源码，[ma2r_build.py](../../../../tools/h100/phase6/ma2r_build.py) 校验 A 的固定基础源码、V 的原 MA2 变换以及 R 的声明改动。旧 `ma2*.py` 实验入口与原始结果保留，新文件使用 `ma2r_*` 独立协议和构建身份。

### 9.3 三组对照与两轮测试

| 组 | 实现 | 用途 |
| --- | --- | --- |
| A | 原交替＋scalar decoder | 检查相对原交替是否达到 1% |
| V | 已测过的原 MA2，保留逐次配置与输出 clone | 同场复测原版，检查新增执行优化的净收益 |
| R | V decoder＋实例配置＋直接独立输出 | 新候选 |

R/V 同时衡量配置提交与冗余 clone 去除这两项执行改动，不把该比较解释成其中某一项的单独收益。当前不混入旧 MA1 P 或 baseline C。正式模型、初始化与 token seed、全远端量化、反向路径、两屏障及统计方法沿用原 MA2。

默认仍为 **两次比较，各 12 个完整 A/V/R 块，共 72 个窗口、3600 个正式 optimizer step**；每窗口 10 步预热＋50 步正式，合计另有 720 步预热。两轮采用相同配置和顺序，在不同新进程中重复；不把它们解释为不同数据或不同随机种子。

两次比较各自要求 **R/A 点估计 ≥1%，95% CI 下界 >0；同时 R/V 点估计与区间下界均 >0**。R/A 的区间下界是否达到 1% 单列为严格结果；V/A 仅为诊断控制。所有窗口、负收益和置信区间照常报告，不因单算子收益为负而跳过完整训练。

正确性仍要求 A/V/R 输入、权重及算子输出逐位一致，模型逐 rank loss/gradient 轨迹一致，并保留延迟 rank 和连续调用的检查。针对直接输出，预检检查多个返回值的独立存储和后续调用不覆盖旧输出，并交错调用不同实例检查配置隔离。实际 profile 对 A/V 要求每次一次 clone，对 R 要求零次 clone；三组均要求实际 GEMM→发布屏障→对应 decoder→复用屏障路径成立。不能通过给 R 偷加 clone 或切回旧配置路径来通过测试。

### 9.4 新版一行启动

同步新增 `ma2r_*.py`、`detach_ma2r.py` 与原有工具依赖，在服务器 `/workspace/TempName` 执行：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_ma2r.py --execute
```

默认重新构建 A/V/R，生成独立目录 `logs/h100/phase6/ma2-submit-2x-50steps-<北京时间戳>/`，然后执行算子预检/计时/profile、完整训练预检/两轮计时及独立 profile。`batch-state.json` 登记完成状态、1% 目标和 `submission_benefit_validated`；详细两轮结果保存在 `fullstep/report.json` 与 `fullstep/report.md`。本命令与第 5 节旧版 MA2 的复测入口不同。

去掉 `--execute` 仅预览，不创建目录或启动 GPU。仍采用既有后台托管和每 worker 超时机制，容器和主机需持续运行；启动前确保其他训练任务已结束。`--ma2r-build` 仅接受本轮通过核验的 A/V/R 构建，不能复用旧 A/P/V 构建。

### 9.5 上机前验证

2026-10-22 本地检查已完成：

| 检查 | 结果与证据 |
| --- | --- |
| 构建、native 配置、输出、统计与启动编排 | **78 项 CPU 测试通过**；[验证登记](../../../../outputs/h100/phase6/ma2r-validation-20261022/cpu-validation.json)、[完整日志](../../../../outputs/h100/phase6/ma2r-validation-20261022/cpu-tests.log) |
| native 配置与异常语义 | 编译运行从实际生成源码提取的配置/前向校验代码，覆盖不同 K 的多实例、线程隔离、异常恢复、错误设备/布局/stream/capture/梯度模式 |
| A/V/R 映射与 CUDA runtime | 三组 host mapping 和实际 SM90 runtime 目标文件编译通过，保存 PTX/SASS；[编译登记](../../../../outputs/h100/phase6/ma2r-validation-20261022/runtime-check/validation.json) |
| R 的实际 PyTorch C++ wrapper | 使用 PyTorch 2.6.0+cu124 头文件及 CUDA 12.8 编译真实 `gemm_reduce_scatter.cc` 目标文件通过；不是仅编译 decoder |
| 两轮后台命令 | 预览确认 72 窗口、3600 正式步，未创建目标目录或启动 GPU；[预览记录](../../../../outputs/h100/phase6/ma2r-validation-20261022/launch-preview.txt) |

本节是上机前验证记录，当时本地未完成整套 Flux 动态库链接或八卡 GPU 执行。随后服务器已完成这些运行与门禁，本批结果见第 10 节；CPU 模拟与目标文件编译本身仍不能代替设备实测。

## 10. MA2-R 两轮实测：通过 1% 目标（2026-10-23 回传）

### 10.1 批次与完整性

批次目录：`logs/h100/phase6/ma2-submit-2x-50steps-20261022-202958-492886/`。依据[批次状态](../../../../logs/h100/phase6/ma2-submit-2x-50steps-20261022-202958-492886/batch-state.json)、[完整训练报告](../../../../logs/h100/phase6/ma2-submit-2x-50steps-20261022-202958-492886/fullstep/report.md)和[统计 JSON](../../../../logs/h100/phase6/ma2-submit-2x-50steps-20261022-202958-492886/fullstep/report.json)，本批正常完成，`status=completed`、`stage=done`。

| 检查项 | 本批结果 |
| --- | --- |
| 正式工作量 | 2 次比较×12 个完整 A/V/R 块；72 窗口、576 份 rank 计时记录、3600 正式 optimizer step |
| 预热 | 每窗口 10 步，正式窗口合计 720 步，不计入正式步时分母 |
| 算子正确性 | `correctness_passed=true`、`bitwise_equal=true` |
| 模型正确性 | `correctness_passed=true`、`identical_trajectories=true` |
| 算子与模型实际执行路径 | 两项均通过 |
| 1% 点估计目标 | `one_percent_point_target_validated=true` |
| R/A 的 95% CI 下界 ≥1% | `strict_one_percent_confidence_validated=true` |
| R/V 执行优化净增量 | `submission_benefit_validated=true` |
| 原 MA2 V/A 控制项 | `ma2_control_passed=false`，两轮合并区间均跨零；不影响 R 的验收 |
| 最终判定 | `target_validated=true`；`acceptance_status=one_percent_point_target_and_positive_submission_increment` |

与上一批 `155228` 的构建身份比较，去除目录前缀后，A 的 351 个、V 的 352 个登记源码哈希分别全部一致。模型、种子、预热/正式步数、共享环境字段和 Megatron commit 也一致。本批使用相同源码的 A/V 重新计时，不用历史耗时替代同场对照。完整训练 run ID 为 `8765426ad3574d9484ae4a02811873cd`。

### 10.2 完整训练：两轮与四个半程均支持 R 的收益

页首列出两轮全量结果。R/A 分别为 +3.761% [+3.444, +4.160] 和 +3.718% [+3.413, +4.038]；R/V 分别为 +3.398% [+3.200, +3.593] 和 +3.589% [+3.324, +3.835]。两轮均超过预定 1% 目标，并且相对原 MA2 V 有明确正增量。

按算术平均步时，compare1 的 R 比 A 少 20.169 ms、比 V 少 18.150 ms；compare2 分别少 19.910 ms 和 19.194 ms。百分比仍使用预注册的完整块配对方法，与均值差的口径不同。

| 半程诊断 | R/A 收益 %［95% CI］ | R/V 收益 %［95% CI］ |
| --- | --- | --- |
| compare1 前半 | +3.487 [+3.257, +3.766] | +3.410 [+3.063, +3.729] |
| compare1 后半 | +4.034 [+3.513, +4.679] | +3.387 [+3.169, +3.616] |
| compare2 前半 | +3.563 [+3.095, +4.103] | +3.534 [+3.103, +3.892] |
| compare2 后半 | +3.872 [+3.566, +4.173] | +3.644 [+3.340, +3.968] |

四个半程的 R/A、R/V 区间也都为正，方向与完整比较一致；它们仍只是两次比较内部的诊断，不当作四次独立复现。R 的窗口范围为 compare1 的 513.020–519.362 ms/step、compare2 的 513.720–520.645 ms/step；本批全部窗口均保留。

原 V/A 两轮为 +0.375% [−0.002, +0.771]、+0.133% [−0.223, +0.488]，均未确认正收益，与原版 MA2 未通过的历史结果相容。因此，本批约 3.4%–3.6% 的 R/V 增量应归于新增执行路径这组改动，不能归给未改变的打包 decoder。

### 10.3 独立算子与 profile 观察

[算子报告](../../../../logs/h100/phase6/ma2-submit-2x-50steps-20261022-202958-492886/operator/report.json)计时包含各臂完整独立输出路径：A/V 为 `forward().clone()`，R 直接返回原生分配的独立输出。

| 算子 | A / V / R 平均耗时（µs） | R/A 收益 %［95% CI］ | R/V 收益 %［95% CI］ |
| --- | --- | --- | --- |
| Attention | 194.165 / 149.239 / 145.881 | +24.868 [+24.233, +25.446] | +2.247 [+1.292, +3.154] |
| MLP | 268.113 / 226.490 / 223.516 | +16.644 [+15.560, +17.596] | +1.322 [+0.394, +2.255] |

两个算子的 R/V 区间均为正，支持执行优化在独立算子中也有收益；这些数值不替代整步测量。

独立[完整训练 profile](../../../../logs/h100/phase6/ma2-submit-2x-50steps-20261022-202958-492886/fullstep/profile-summary.json)按各 rank 的时长中位数再取八 rank 算术平均：

| 角色 | V / R 投影 CPU 时长（µs） | V / R decoder 时长（µs） | V / R GEMM 时长（µs） |
| --- | --- | --- | --- |
| Attention | 225.821 / 130.177 | 14.947 / 14.980 | 100.591 / 100.708 |
| MLP | 228.905 / 132.084 | 15.161 / 15.176 | 158.424 / 158.584 |

R 的投影 CPU 时长明显缩短，而 decoder 与 GEMM 没有显示相应缩短。这与 R 减少配置提交、Python 检查和冗余输出复制的实现相符，但 profile 有扰动与 GPU 排队，不能将 CPU 时间差直接乘调用数作为完整 step 的精确收益分解。

每臂 profile 覆盖八 rank 各 256 次投影，共 2048 次 rank-local 调用、4096 次原屏障。A/V 每臂各观察到 2048 次输出 clone，R 为 0；V/R 仍执行同一个 `taco_ma2_decode_kernel`。算子预检包含不同 K 的实例交错调用、独立工作区与多代输出保留，跨臂输出逐位一致；模型全部 rank 的配对 loss/gradient 轨迹也一致。R 去除的是多余复制，未以复用输出视图替代独立输出。

### 10.4 复核与结论边界

**四份离线结果均复算一致。** 本次拉取后，按回传原始 rank 记录重新生成完整训练与算子 `report.json`，并从 24 份完整训练 trace、48 份算子 trace 重新计算两份 `profile-summary.json`，均与回传文件逐项相同，原始报告未覆盖。冻结脚本与本地控制器身份一致；离线核对不等于在本机重新执行训练。回传没有包含本批 `build/` 动态库目录，不能在本机重新加载或独立复核缺失二进制；运行端登记的库身份和实际 trace 用作执行证据。

**本批支持的结论是：在当前 GPT6.7B、TP8、固定合成输入配置下，MA2-R 两次完整训练比较均达到约 3.7% 的 R/A 收益，超过 1% 目标，且相对原 MA2 V 有约 3.4%–3.6% 的正增量。** 这属于固定配置提交与独立输出路径的联合工程优化；没有单独消融配置绑定、检查下沉与 clone 去除，不能量化其中任一项的独立贡献。

本批不包含旧 MA1 P 或 baseline C 的同场比较，也不能用跨批次约 515 ms 的相近均值宣称与 C 等价。两次比较共享同一机器和批次条件，支持本批重复结果，不代表已跨机器、跨日期或真实语料长期训练验证。原版 V 的历史负结果继续保留，后续复测仍使用第 9.4 节命令并单独报告。
