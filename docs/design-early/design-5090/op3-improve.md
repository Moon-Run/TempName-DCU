# OP3：R＋CR3 融合与编码器改进的对比及组合测试

更新：2026-11-02。目标为单机 8×RTX 5090，TP8/DP1。**v1 两方案已完成 2 trial × 20 blocks 对比，结果见第 6 节；v2 组合组已完成独立算子和完整训练测试，结果见第 7 节。** 当前 v2 脚本只测“A＋MA89＋MA11＋R＋CR3 消费者融合＋OP3 编码器改进”，运行命令见第 4 节。回传批次为 1 个完整训练窗口、50 个正式 step；算子每种形状各测 1 个计时窗口。

## 1. 已完成的 v1 比较内容

这里的 A 是 [OP1](op-1-A&B&C.md) 的 `flux-basic-interleaved`，即 Flux GemmOnly＋基础全远端量化＋交替 CTA 排序；不是 [baseline.md](baseline.md) 中使用另一套 native producer 的 `a`。MA89 取 MA8 激活缓存和 MA9 零梯度统计，MA11 取精确 causal softmax，三项沿用 [OP2](op2-ma11&ma89.md) 的安装方式，每个 worker 各安装一次。

| 实验组 | 脚本 ID | 编码 | 解码与后继消费者 |
| --- | --- | --- | --- |
| **A＋MA89＋MA11＋R＋CR3 消费者融合** | `op3-r` | 原 A 的独立 H128 编码器 | R packed-load/source-prefetch 解码；写回前执行 bias 和 residual 的两次 BF16 加法 |
| **A＋MA89＋MA11＋OP3 编码器改进** | `op3-improved` | 一个 warp 负责一个 H128 组，四个组组成一个 CTA | 保留 A 的基础解码器及原两个独立 BF16 add |

两组共用原 A 的同一份交替排序 GemmOnly 二进制、全远端量化、单次 NCCL All-to-All、MA8/MA9/MA11、模型和训练输入。第二组对应用户要求的“A＋MA89＋MA11＋你的改进”，其改动是编码器；本轮直接比较两个方案。没有把历史 OP2 步时加入新批次配对统计，因此本轮只证明两组之间的差异，不能单独证明任一组相对原 OP2 A 的净增量。

## 2. 回传数据与改进依据

### 编码器改进的 profile 依据

使用 A 路径的八 rank profile，覆盖 **2048 次投影**；trace 来自 [OP2 回传批次](../../../../Load-5090/logs/5090/phase6/op2-abc-ma11-v1-1x-50steps-20261102-013244-945721/report.md)。

以下 profile 指标先求每 rank、每角色的调用中位数，再求八 rank 中位数；单位 μs。它们来自独立 profile，不能直接当作正式步时节省。

| A 的实际路径 | Attention | MLP |
| --- | ---: | ---: |
| 编码器 | **65.128** | **64.808** |
| 基础解码器 | 26.456 | 26.392 |
| 两个消费者 add 合计 | 7.616 | 7.560 |
| decoder 开始→LayerNorm 结束 | 52.536 | 52.320 |
| NCCL kernel | 660.783 | 282.671 |

按每 rank 整步 kernel 时长求和，再取八 rank 中位数，编码器约 **16.688 ms**、解码器约 **6.784 ms**、两个 add 约 **1.942 ms**、NCCL 约 **152.577 ms**。这些时间不是关键路径分解，不能相加推导整步收益。通信仍是更大的开销；当前数据支持优先减少独立编码器的 CTA 数和共享内存同步成本。

原编码器每个远端目的 rank 的 grid 为 8192，每 CTA 128 threads，总计 `8192×7=57344` 个 CTA；trace 中每 CTA shared memory 为 656 bytes。OP3 改进编码器仍使用 128 threads，但每 warp 完成一个组，每 CTA 完成四个组，grid 变成 `2048×7=14336`，消除该 kernel 的 CTA 内共享内存 butterfly/barrier。

### 为什么纳入 R 和 CR3 消费者融合

[5090 baseline 的同期数据](baseline.md#同期配对结果)中：

- R/A：977.081 / 986.816 ms，配对降幅 **0.987%［0.954%, 1.019%］**，18/18 块更快。
- CR3/C：1017.192 / 1022.978 ms，配对降幅 **0.566%［0.522%, 0.612%］**，18/18 块更快。

**CR3/C 不是单独融合的消融。** 其代码链还包含 R 的 mixed decoder 和 CR2 提交缓存，且使用选择性量化。不能把 0.566% 全归给 bias/residual 融合，也不能直接作为当前 A 的预期收益。

更直接的融合证据来自 [H100 MA4-F 的 F4/CR2 对照](../design-h100/New-Mechanism/ma-4.md#6-本轮-ma4-f-结果相对-cr2-有端到端增量)：保持相同 producer、选择策略和提交缓存，F4 为 501.433 ms、CR2 为 509.852 ms，融合新增量 **1.652%［1.297%, 2.034%］**，8/8 块更快；profile 每 rank 每步减少 512 个消费者 add kernel，并保留原 LayerNorm。它说明融合值得迁移测试，但属于 H100 历史证据。

因此，本次 **纳入 R＋CR3 的消费者融合部分**，不迁入 C 的选择性量化和分块到达路径。v1 比较了 `op3-r` 与编码器改进组，但没有同期原 A 控制，不能单独归因 `op3-r` 相对原 A 的收益。

## 3. v1 组件实现与算术约束（v2 组合复用）

### R＋CR3 组

[R 的来源](../../../tools/h100/phase6/ma2_decode.cuh)使用 packed FP8 load、下一 source 预取、末尾本地 BF16 贡献预取，以及每 warp 一个 H128 组。本次将其地址计算改为已完成 NCCL 交换后的本地 `H128-AoS-136` packet，每个 packet 包含 128 bytes FP8＋两个 FP32 scale。

解码按 `rank+1 → … → rank+7 → local` 顺序，每个解码值及每次累加均保留 BF16 舍入。R 的收益机制作用于本地 decoder；它不恢复 H100 的跨卡直接访存或 peer 到达重叠。原 A 已有实例配置和独立输出，本轮不把这些已有机制再记为新增优化。

写回执行严格的 `BF16(residual + BF16(projection + bias))`，而不是三项 FP32 求和后只舍入一次。保留原 LayerNorm。融合 wrapper 绑定当前层实际 residual 和原 bias 参数，autograd 继续返回 input、weight、bias、residual 梯度；bias 归约先于原线性 backward 的序列 AllGather、dgrad 和 wgrad。相关生成器复用已验证的 [MA4-F adapter](../../../tools/h100/phase6/ma4_f_adapter.py)，v1 由 `E2E_POLICY=op3-r` 启用，v2 的 `op3-combined` 同样显式启用。

### 编码器改进组

每个 lane 持有 `lane+32*j` 的四个原始系数。保留原来的 FP32 平方和归约树、adaptive scale、Hadamard butterfly 的距离顺序、max 归约、quant scale、E4M3 转换及 packet 布局。减少线程组数和共享同步，不改变量化算法、预算或远端字节数。

正式执行前必须证明新编码器与冻结 upstream 编码器生成完全相同的 packet，以及解码结果逐位一致。若不一致，直接停止在 probe 阶段，不允许通过放宽量化误差阈值进入计时。

### 两组共同运行规则

每个模块持久保存自己的收发 packet，使用固定 caller stream；每次投影只有一次异步 All-to-All，并调用 `Work.wait()` 建立通信完成依赖后解码。输出每次独立分配，不能覆盖仍被 autograd 使用的旧输出。提交异常会使相应进程组/GPU 的后续提交失效；不使用 clone 假装输出所有权。

每 rank 每投影发送 `7×8192×136=7798784` bytes，本地贡献不发送、不量化；这是逻辑贡献字节数，不等于 NCCL 实测链路流量。两组保持相同 `CUDA_DEVICE_MAX_CONNECTIONS=1` 和 baseline NCCL 设置。本次没有启用 CUDA Graph，也没有通过变更线程流或通信设置制造组间差异。

代码入口：

| 文件 | 职责 |
| --- | --- |
| [op3_codec.cu](../../../tools/5090/phase6/op3_codec.cu)、[op3_codec.py](../../../tools/5090/phase6/op3_codec.py) | 两条 CUDA 路径、固定流与输出生命周期 |
| [op3_build.py](../../../tools/5090/phase6/op3_build.py) | 新建小型 SM120 codec 库；只读复用原 A 的 GemmOnly 构建 |
| [op3_sources.py](../../../tools/5090/phase6/op3_sources.py) | 两组完整训练 worker、融合 adapter 和三项通用优化 |
| [op3_probe.py](../../../tools/5090/phase6/op3_probe.py)、[op3_validation.py](../../../tools/5090/phase6/op3_validation.py) | upstream 位串、融合舍入、完整模型及最终运行计数校验 |
| [op3_profile.py](../../../tools/5090/phase6/op3_profile.py) | 实际 kernel、NCCL、消费者、反向和 MA8/MA9/MA11 路径审计 |
| [op3_operator.py](../../../tools/5090/phase6/op3_operator.py) | 组合组算子梯度、更新、延迟和独立 profile |
| [op3.py](../../../tools/5090/phase6/op3.py)、[op3_protocol.py](../../../tools/5090/phase6/op3_protocol.py) | 分阶段执行、当前单组统计、报告和只读重放 |

## 4. 当前脚本：只测组合组（v2）

本节替代原两组运行命令。新组为 **A＋MA89＋MA11＋R＋CR3 消费者融合＋OP3 编码器改进**，ID 为 `op3-combined`，协议为 `rtx5090-phase6-op3-combined-v2`。将 v1 的 `op3-improved` 编码器与 `op3-r` 的解码、消费者接在同一条投影路径上；MA8/MA9/MA11 继续各安装一次。v1 结果保留在第 6 节，不能当作本次组合组结果。

只生成一个 worker 配置，只计时组合组。原生正确性参考和非融合 autograd 参考用于校验，不作为第二个性能组运行。默认 `--trials 1 --blocks 1`，只有 **1 个完整训练计时窗口、50 个正式 step**。需要手动扩展采样时仍可显式设置 `--trials`/`--blocks`，当前命令均保持单窗口。单组没有 AB/BA、同期相对收益或配对 CI，报告只给每轮全部窗口的均值、中位数、最快/最慢、吞吐和显存。

在目标机 `TempName-New` 仓库根目录运行，使用已配置的 5090 环境。下面复用本次已成功批次的 build：

```bash
op3_build=/workspace/Load-5090/logs/5090/phase6/op3-two-arm-v1-2x-50steps-20261102-022324-519378/build
```

CUDA 文件 `op3_codec.cu`、`op3_native.py` 和 `op3_build.py` 与该批次冻结源码逐字节一致，原库已经包含新编码器和 R 融合解码两个导出；本次组合不需要修改旧构建。脚本仍检查整个依赖构建、库哈希及 Python/CUDA ABI，生成新的运行时副本；MA9/MA11 小库与 softmax 自检照常准备。若该 build 已不存在，可改用原 `--abc-build <ABC build>` 新建 OP3 小库。

### 先只测算子

```bash
bash tools/5090/phase6/run_op3_operator.sh \
  --op3-build "$op3_build" \
  --foreground --execute
```

此入口等价于 `run_op3.sh --operator-only`。完成硬件/NCCL、原生码流/输出所有权、融合舍入检查后，测试 Attention 与 MLP 两种真实形状：`M=2048, N=4096, K_local=512/2048`。每 rank 输出为 `256×4096`，TP8，全远端量化、本地 BF16，逻辑传输量与 v1 相同。

算子阶段先执行三步正确性检查，调用**实际生成的** `GemmRSFunction` 与 `ConsumerGemmRSFunction`，以裸投影后的两个 PyTorch BF16 add 作为融合参考，逐位检查输出、input/weight/bias/residual 四类梯度及三步更新；也检查输入没有被覆盖。这里的“更新”是简单的诊断更新，不是完整训练的 Adam。任一结果不符就停止，不能进入性能测量。

正确性通过后，**仅组合组**每角色预热 20 次，执行 **1 个窗口×100 次 forward**。输入、weight、bias、residual 预先分配，operator 实例和工作区复用；计时包含新输出分配、GemmOnly、编码、NCCL 等待、R 解码与两次 BF16 消费加法。每窗口开始前八 rank 同步，分别记录 CUDA event 与等待完成后的墙钟时间，先逐窗口取最慢 rank，再汇总全部窗口。

性能窗口不含独立 backward、LayerNorm、optimizer 或 MA8/MA9/MA11 训练工作，不使用 CUDA Graph。MA89/MA11 在后续完整训练中验证，不能把它们算成这份投影微测的收益。

另采每角色 10 次 forward 的独立 profile，核对新编码器、R 融合解码器、唯一 All-to-All、`Work.wait`、流依赖、启动归属与次数。报告列出 GEMM、编码、NCCL kernel 合计、解码含融合、CPU 提交和 GPU 起止跨度；这些 profile 数值用于定位瓶颈，不直接相加推导正式步时收益。

结果在新批次的 `operator/report.md` / `operator/report.json`，附八份 rank 记录和 16 份 `attn/mlp-rank*-profile.json.gz`。算子成功状态为 `operator_completed`、`full_step_completed=false`。这个命令使用前台运行，返回成功后才能再启动完整实验。

### 完整单组测试：自动先测算子

```bash
bash tools/5090/phase6/run_op3.sh \
  --op3-build "$op3_build" \
  --trials 1 --blocks 1 --execute
```

完整入口自动依次执行：硬件/NCCL → 原生数值/传输 → 算子正确性、性能和 profile → 单组全模型 smoke → 八 rank 全模型 profile → 单组正式计时。无需先单独运行算子命令；若已单独运行，完整批次仍重复前置检查，避免跨批次借用验证结果。

原 `run_op3_compare.sh` 保留为兼容入口，**现在同样只测组合组**。`--transport-smoke-only` 仍可停在原生正确性阶段，它不测算子延迟。不加 `--execute` 只预览；完整实验默认后台运行，可加 `--foreground`。查看打印的 `handoff/supervisor.log` 和 `batch-state.json`，不要同时启动独立算子与完整实验。

所有新结果写入 `Load-5090/logs/5090/phase6/op3-combined-v2-<trials>x-50steps-<时间>/`，不追加到历史两组目录。完整报告为根目录 `report.md` / `report.json`；各窗口在 `results/compare1/`、`results/compare2/` 下。算子报告与整步报告分开保留。

### 只读复核新批次

算子批次与完整批次都使用基础入口复核，不能给此命令额外加 `--operator-only`：

```bash
bash tools/5090/phase6/run_op3.sh --out /path/to/completed-combined-batch --report
```

需要保留目标机绝对路径对应的构建和全部证据。复核会验证源码/库/worker 身份、原生数值、算子原始 trace；完整批次再验证模型与全部正式窗口。v1 两组批次须使用当批冻结版本重放；当前脚本不会将其解释为新的单组协议。

## 5. 组合组的验证与边界

- 全模型仍为 TP8/DP1、32 层、hidden=4096、sequence=2048、micro batch=1、global batch=4，每窗口 10 步预热＋50 步正式。未减少模型、量化范围或 MA8 缓存。
- 正式计时要求原生 upstream packet/解码逐位一致、融合两次 BF16 舍入和输出保持、独立算子四类梯度及三步更新一致、完整模型 smoke 和八 rank profile 全部通过。
- 每模块持久的收发包保持独立，完成 NCCL 后才解码，输出每次独立分配。全模型最终同步后读取 64 个模块状态，timing 每模块恰好 240 次新编码、交换、R 解码和融合消费；BDA 计数须一致。算子计数另算，每角色共 136 次投影，其中 133 次融合、3 次裸投影参考，不能混入全模型计数。
- 单组没有第二个全模型训练轨迹可比；替代检查是算子真实 autograd 与非融合参考逐位对照，加完整训练的数值预算、全部参数梯度/更新记录和消费者生命周期审计。报告明确记录 `full_training_cross_arm_comparison=false`，不声称完成两组训练等价验证。
- 全模型 profile 要求 `op3_warp_encode_kernel` 与 `op3_r_decode_kernel<true>` 同时执行，后继保持原 LayerNorm、没有两个独立消费者 add；反向、MA8/MA9/MA11 各自审计。
- 这是一次组合实验，历史 v1 两组结果不能作为同期控制，也不能将其局部收益直接相加。组合组目标机结果见第 7 节；本地没有重跑 GPU。

本地回归入口：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools/5090/phase6 -p 'test_*.py' -v
bash -n tools/5090/phase6/run_op3.sh tools/5090/phase6/run_op3_compare.sh tools/5090/phase6/run_op3_operator.sh
git diff --check
```

本次单窗口版本的 **49 项 OP3 专项测试、283 项完整 5090 CPU 回归均通过**。三个 shell 入口语法、单窗口预览、源码生成和文档链接检查通过；已用新分析器重新解析 v1 两组各自的 rank0 原始 profile，均通过实际 trace 格式与执行路径审计。该离线复核属于历史数据兼容性检查；组合组目标机验证结果见第 7 节。

## 6. OP3 回传结果（2026-11-02）

数据来自 [OP3 原始报告](../../../../Load-5090/logs/5090/phase6/op3-two-arm-v1-2x-50steps-20261102-022324-519378/report.md)、[报告数据](../../../../Load-5090/logs/5090/phase6/op3-two-arm-v1-2x-50steps-20261102-022324-519378/report.json) 和 [批次状态](../../../../Load-5090/logs/5090/phase6/op3-two-arm-v1-2x-50steps-20261102-022324-519378/batch-state.json)。该批次状态为 `completed`，使用 **2 trial × 20 blocks × 2 组**，共 **80 个正式窗口、640 份正式 rank 记录、4000 个正式 optimizer step**。每两个 block 采用 AB/BA 顺序；每窗口 10 步预热＋50 步正式计时。两个 trial 分别报告，未与 OP2 历史窗口合并。

| trial | 指标 | R＋CR3 消费者融合 | OP3 编码器改进 |
| --- | --- | ---: | ---: |
| compare1 | 平均 ms/step | 855.544 | **854.444** |
| compare1 | 中位数 / 最慢 ms/step | 855.230 / 858.047 | 854.322 / 857.537 |
| compare1 | tokens/s | 9575.19 | 9587.53 |
| compare2 | 平均 ms/step | 855.393 | **855.132** |
| compare2 | 中位数 / 最慢 ms/step | 855.318 / 857.690 | 854.721 / 857.011 |
| compare2 | tokens/s | 9576.88 | 9579.80 |

| 同期配对：编码器改进 / R＋融合 | 配对降幅及 95% CI | 改进组更快的块数 |
| --- | ---: | ---: |
| compare1 | +0.129% ［+0.067%, +0.191%］ | 19/20 |
| compare2 | +0.031% ［-0.024%, +0.083%］ | 13/20 |

正降幅表示编码器改进组更快；配对估计使用块内对数耗时比和 10000 次 bootstrap。两次 trial 的平均耗时差分别约 **1.100 ms/step** 和 **0.261 ms/step**。改进组在两次的均值都较低，但第二次置信区间跨过零，不能把首次的收益幅度视为稳定复现。两个方案是不同改动的直接比较，并非单独编码器消融；本批没有同期原 OP2 A 对照，不能从与 OP2 的跨批次差值推断任一方案相对原 A 的净收益。

R＋融合和改进组的最大 Torch allocated 分别为 **19.423 / 19.412 GiB**；报告的 allocated＋native 与各自 allocated 相同，仍不含 CUDA context、NCCL 等内部显存，不代表整卡峰值。报告记录 `correctness_passed=true`、`execution_path_validated=true`、`native_staged_runtime_validated=true`；裸投影 packet/输出要求逐位一致，smoke 要求 loss、gradient norm 及全部参数梯度和更新轨迹一致。报告中后续数值观测的 loss 和 gradient norm 最大绝对差均为 0，但不能由此推断长期收敛或模型质量。两组 smoke 采样输出的位置不同，融合输出和裸投影的哈希不能直接比较。

以上为对回传报告和批次状态的整理；此处没有在本地重跑 GPU 或完整 `--report` 重放。

## 7. OP3 v2 组合组回传结果（2026-11-02）

数据来自 [完整训练报告](../../../../Load-5090/logs/5090/phase6/op3-combined-v2-1x-50steps-20261102-100813-883313/report.md)、[报告数据](../../../../Load-5090/logs/5090/phase6/op3-combined-v2-1x-50steps-20261102-100813-883313/report.json)、[算子报告](../../../../Load-5090/logs/5090/phase6/op3-combined-v2-1x-50steps-20261102-100813-883313/operator/report.md)和[批次状态](../../../../Load-5090/logs/5090/phase6/op3-combined-v2-1x-50steps-20261102-100813-883313/batch-state.json)。批次状态为 `completed`：**1 trial × 1 block × 1 组**，共 1 个正式窗口、8 份正式 rank 记录、50 个正式 optimizer step，另有 10 步预热。

| 组合组完整训练指标 | 实测值 |
| --- | ---: |
| 正式步时（ms/step） | **850.195** |
| 吞吐（tokens/s） | **9635.44** |
| 最大 Torch allocated（GiB） | 19.422869 |
| 最大 allocated＋native（GiB） | 19.422869 |

只有一个窗口，均值、中位数、最快和最慢均为 850.195 ms/step；它不能反映窗口间波动。显存合计不含 CUDA context、NCCL 等内部占用，不代表整卡峰值。本批没有同期性能对照，不能计算相对 v1 两组或原 A 的配对降幅与置信区间，也不能证明两项改动的收益可以叠加。

### 独立投影算子

每个角色预热 20 次，测量 1 个窗口的 100 次 forward；报告分别取八 rank 最大 CUDA event 和同步墙钟平均延迟。算子计时包括 GemmOnly、编码、NCCL 等待、R 解码和融合 bias/residual，不包括 LayerNorm、backward、optimizer、MA8/MA9/MA11 的训练工作或 CUDA Graph。

| 角色 | GPU 平均延迟（μs） | 同步墙钟平均延迟（μs） |
| --- | ---: | ---: |
| Attention | 664.285 | 664.685 |
| MLP | 726.075 | 726.338 |

独立 profile 先取每 rank 的调用中位数，再取八 rank 中位数；与上表计时窗口分开采集。

| 角色 | GEMM（μs） | 编码（μs） | NCCL kernel 合计（μs） | 解码含融合（μs） | CPU 提交（μs） | GPU 起止跨度（μs） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention | 59.816 | 39.040 | 785.738 | 13.808 | 854.721 | 899.763 |
| MLP | 213.367 | 38.640 | 277.702 | 13.432 | 369.166 | 547.148 |

profile 与正式计时的采样方式不同，kernel 时间用于诊断，不能相加推导整步耗时或收益。算子报告只对组合路径计时，没有同期算子性能对照。

### 验证和判读

完整报告记录 `correctness_passed=true`、`execution_path_validated=true`、`native_staged_runtime_validated=true` 和 `operator_gradients_validated=true`。算子正确性检查中，融合路径与非融合 autograd 参考的输出、input/weight/bias/residual 四类梯度和三步诊断更新逐位一致。完整训练的 profile 汇总记录八 rank 路径检查通过；单组没有跨组完整训练轨迹对照，报告中的 `full_training_cross_arm_comparison=false` 不支持声称完整训练跨组等价或长期模型质量等价。

以上根据回传报告和状态整理；本地没有重跑 GPU 或执行完整 `--report` 重放。
