# Flux 基础量化 v2：仅融合接收端反量化与求和

更新：2026-10-25。**新版基础量化 v2 的一次八卡完整比较已经完成**：覆盖 Attention＋MLP，共 **24 个正式窗口、1200 个正式 optimizer step**。基础量化平均 **582.318 ms/步**，CUDA 组 **609.744 ms/步**；配对耗时下降 **4.498%［95% CI：4.198%, 4.835%］**，12 个配对块全部更快。协议为 `h100-basic-receiver-fusion-fullstep-v2`，与下方生产端融合 v1 分开记录。数据来自 [本批原始报告](../../../../logs/h100/phase6/basic-quant-receiver-1x-50steps-20261025-090410-798870/compare1/report.json)。

| 指标 / 策略 | **Flux＋基础量化 v2** | **Flux＋CUDA量化** |
| --- | --- | --- |
| 策略 ID | `basic`（新协议） | `cuda` |
| GEMM | 原生 Hopper Flux GemmOnly | 相同源码、同一个构建和动态库 |
| 编码与发送 | 独立 TACO CUDA 编码＋NCCL All-to-All | 完全相同 |
| 接收端 | **一个 kernel 完成七来源反量化＋按序 BF16 求和＋加入本地贡献** | 独立反量化 kernel＋八次 PyTorch BF16 相加 |
| 反量化后的全量临时张量 | 不落地 `[7,256,4096]` BF16 张量 | 写入该张量后逐来源读取 |
| 量化与数值语义 | 全远端 H128/E4M3；本地 BF16；逐次 BF16 舍入 | 相同 |
| 计时范围 | 前向、反向、梯度处理、optimizer 的完整训练步 | 相同 |
| **平均步时（ms/步）** | **582.318** | **609.744** |
| 全部窗口中位步时（ms/步） | 582.321 | 609.011 |
| 吞吐（tokens/s，由平均步时换算） | 14067.918 | 13435.146 |
| 平均 loss | 3.821911153 | 3.821911153 |
| 第 50 正式步 loss | 0.019168688 | 0.019168688 |
| 相对 CUDA 的配对耗时下降及 95% CI | **+4.498%［+4.198%, +4.835%］** | 对照 |
| 正式窗口 / 正式步数 | 12 / 600 | 12 / 600 |

## 本批实测结论

批次为 `basic-quant-receiver-1x-50steps-20261025-090410-798870`，run ID 为 `1d72fad1053f4396b304796da9596430`。[批次状态](../../../../logs/h100/phase6/basic-quant-receiver-1x-50steps-20261025-090410-798870/batch-state.json) 与 [训练状态](../../../../logs/h100/phase6/basic-quant-receiver-1x-50steps-20261025-090410-798870/compare1/all-state.json) 均为 `completed`；后台 supervisor 的 PPID 为 1。此次没有启动绑定失败或未完成窗口。

运行环境为 `gxn75` 的 8×H100 80GB HBM3，每卡 132 SM；PyTorch 2.6.0+cu124、运行 CUDA 12.4、构建 CUDA 12.8、NCCL 2.21.5，`NCCL_NVLS_ENABLE=0`。两组共享相同 GemmOnly 动态库、codec 动态库、模型初始化和输入数据。

| 比较范围 | 配对块数 | 基础 v2 平均步时（ms） | CUDA 平均步时（ms） | 基础相对 CUDA 耗时下降 | 配对 bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| 内部 round1 | 6 | 582.693 | 610.570 | **+4.564%** | [+4.137%, +5.064%] |
| 内部 round2 | 6 | 581.942 | 608.918 | **+4.431%** | [+4.016%, +4.883%] |
| **合并** | 12 | 582.318 | 609.744 | **+4.498%** | [+4.198%, +4.835%] |

平均每步减少 **27.426 ms**。主结果按配对块的对数耗时比计算；直接用两组算术均值计算为 **4.497985%**，与配对估计 **4.497566%** 接近，但不能把均值比与配对置信区间当成同一种计算。

基础组 12 个窗口的步时范围为 **577.932–586.668 ms**，CUDA 组为 **606.250–616.571 ms**。全部 **12/12** 配对块基础组更快，没有删除慢窗口；两个内部 round 的区间均为正。这里仍然只有 **一次独立比较**，不能写成两次独立复测。

本批均值 **582.318 ms** 比事先提出的 585–600 ms 区间下沿低 **2.682 ms**，即略快于预期下沿。本次结果支持“朴素接收端融合可以降低完整训练步耗时”。与历史约 533–534 ms 的改进量化对照，顺序符合原先的方向性预期；**改进量化没有在本批运行，不能给出其相对新基础版的同场收益或显著性结论**。

### 数值与执行路径

[预检报告](../../../../logs/h100/phase6/basic-quant-receiver-1x-50steps-20261025-090410-798870/compare1/preflight.json) 记录的 `correctness_passed`、`execution_path_validated` 和 `bitwise_equal_within_pairs` 均为 `true`；正式报告的训练轨迹一致检查也通过：

- 两组各 8 个 rank、各 4 类输入（zeros、small、outliers、完整形状），共 **64 项融合接收端用例**均与原 CUDA 解码＋按序 BF16 求和逐位一致。
- 每组 **512 项投影前向检查**通过，合计 1024 项；两组最大相对 BF16 L2 均为 **2.545994%**，最大相对参考 codec L2 均为 **0.017685%**，低于原 5% / 1% 门槛。
- 两组首个投影输出指纹一致；所有对应窗口的 loss 与 grad_norm 轨迹完全一致，各组自身 12 个正式窗口的轨迹也一致。每个 rank 的两类投影各保留 6 个输出，独立存储和后续调用不改写检查均通过。
- 每个投影调用，基础组实际执行一个 `basic_quant_decode_reduce_kernel`；CUDA 组执行原 `fused_ash_decompress_kernel` 和八次独立 `aten::add`。双方均有相同类型的 SM90 GemmOnly、原编码和 NCCL All-to-All 路径，没有换成旧 GEMM-RS 融合库。

下表取八个 rank 各自逐调用中位数的中位数；单位为 μs，来自独立 profile，不混入正式训练计时：

| 角色 | 基础 / CUDA GEMM | 基础：反量化＋求和融合 kernel | CUDA：仅独立反量化 kernel | 每个投影额外独立 BF16 add（基础 / CUDA） |
| --- | ---: | ---: | ---: | ---: |
| Attention | 17.744 / 17.728 | 27.696 | 36.288 | 0 / 8 |
| MLP | 46.928 / 46.944 | 27.584 | 36.256 | 0 / 8 |

**CUDA 的反量化统计不包含后续八次相加，不能把它当作完整接收端耗时。** 每个 rank、每种角色的 profile 各有 128 次投影调用，对应基础组 0 次、CUDA 组 1024 次独立 BF16 add。相近的 GEMM 时间与明确减少的接收端操作支持收益来自接收端融合；其中包含省去 BF16 中间张量读写、相加中间结果及算子提交的综合作用。现有记录尚未把每步节省的 27.426 ms 精确拆分到 GPU 访存、kernel 执行和 CPU 提交，不能用上述单 kernel 时差直接外推全部整步收益。

两组第 48–50 个正式步 loss 均为 `0.0000321670 → 0.001038385 → 0.019168688`，最终 grad_norm 均为 `0.625838742`，存在相同的末段回升。因此本批没有观察到由融合引入的数值差异，**也不能据此宣称长期训练收敛已经验证**；固定合成数据、50 步轨迹的边界仍然存在。

### 回传数据复核

离线从 **192 份正式 rank 记录**重新生成报告，除离线标记外与服务器 `report.json` 完全一致；另外检查了 32 份 smoke/profile rank 记录、16 份 trace 的哈希，并重新解析两组 rank0 的实际 profile，结果与冻结预检记录一致。复核记录见 [audit.json](../../../../outputs/h100/phase6/basic-quant-results-audit-20261025/audit.json)，复核脚本见 [audit.py](../../../../outputs/h100/phase6/basic-quant-results-audit-20261025/audit.py)。原始状态、报告和预检文件未改写；本地没有重新启动 GPU 训练，也没有重新加载服务器二进制。

## 为什么重做基础版

旧基础版 v1 虽然拆开了接收端解码和归约，仍保留 **GEMM epilogue 内编码、直接向目标 GPU 写入、生产与通信的融合路径**，已经包含较大范围的工程优化。因此它接近改进量化，不能代表“只把相邻两个操作合成一个 kernel”的朴素基线。

已回传的 v1 `report.json` 记录如下；它们是旧实现的历史数据，不作为 v2 的同场对照：

| v1 历史批次 | 改进量化（ms） | 旧基础量化（ms） | CUDA 量化（ms） |
| --- | ---: | ---: | ---: |
| [20261025-001305](../../../../logs/h100/phase6/a-baseline-fusion-1x-50steps-20261025-001305-210375/compare1/report.json) | 533.129 | 535.689 | 608.937 |
| [20261025-012216](../../../../logs/h100/phase6/a-baseline-fusion-1x-50steps-20261025-012216-521143/compare1/report.json) | 533.959 | 536.286 | 610.493 |

新基础版从 CUDA 组出发，仅融合接收端反量化与求和，实验差异更集中：

```text
两组共同路径：
  Flux GemmOnly → 完整 BF16 partial → 整理远端数据
  → 原作者 TACO 编码 kernel → NCCL All-to-All

CUDA 接收端：
  解码七来源 → BF16 临时张量 → 零初始化 → 八次 BF16 相加 → 新输出

基础 v2 接收端：
  一个朴素 CTA kernel：逐来源解码 → BF16 舍入 → 按序累加
  → 加入本地 BF16 → 写新输出
```

585–600 ms 是开跑前提出的预期区间，**不是验收门槛或实现约束**。本批已经测得 582.318 ms，并验证了相对同场 CUDA 组的正收益。融合减少中间读写和 kernel 提交，也使同一输出组的七来源解码串行化；最终净收益以本批完整训练结果为准。代码没有按目标步时插入等待、重复计算或结果筛选。

## 朴素融合的实现边界

实现位于 [basic_quant_receiver.cuh](../../../../tools/h100/phase6/basic_quant_receiver.cuh)，Python 路径见 [basic_quant_codec.py](../../../../tools/h100/phase6/basic_quant_codec.py)。

1. 每个 CTA 使用 128 个线程，只负责输出中的一个 H128 组；沿用原作者解码中的 FP8 转换、warp shuffle、跨 warp 共享内存 Hadamard、缩放和异常值处理。
2. 收到的七个 packet 按 NCCL 来源编号升序排列、去掉本 rank。kernel 按 `(rank+1)%8, …, (rank+7)%8` 读取，对应 packet 下标为 `source - int(source > rank)`，最后加入本地贡献。
3. **每个远端解码值先舍入到 BF16，每次相加再舍入到 BF16**，匹配 CUDA 组原来“写 BF16 张量后逐次相加”的计算。不能改成 FP32 累加到最后才舍入。
4. 每次调用分配独立输出；临时 packet、通信和 current stream 规则沿用 CUDA 组。没有 GEMM 内编码、直接 IPC 写入、交替排序、到达优先、选择性量化、来源预取、双工作区或 CUDA Graph。

两组共用一个 GemmOnly 构建和一个 codec 动态库。codec 保留冻结的原作者编码、解码函数，另加基础融合入口；`cuda` 分支继续调用原独立解码函数与八次 BF16 相加。原作者 device 源码来自固定提交 `ca717e05b6dd6d21fff4d2617d209c95be63f3ef`，没有修改其 device 函数体。基础融合 kernel 是本项目新增实现，不能称为作者原版。

这次能够隔离“接收端融合及其省去的临时张量/算子提交”的总体影响。旧改进量化仍采用不同 GEMM 与通信路径，本次不测它，也不据此给出新基础版相对改进版的同场收益。

## 一轮的测试规模与检查

| 项目 | 本次执行 |
| --- | --- |
| 独立比较 | **1 次，仅 `compare1`** |
| 正式组别 | **基础 v2、CUDA** |
| 内部排程 | 两个方向平衡的 round，每个 6 个两组配对块 |
| 正式窗口 | **24**，每组 12 |
| 每窗口 | 10 步预热＋50 步正式计时 |
| 累计正式 optimizer step | **1200**，每组 600；另有 240 步预热 |
| 正式 rank 记录 | 192 |
| 独立 smoke / profile | 4 个八卡窗口、16 份 profile trace |

这里的一轮是一次完整比较，内部 `round1/round2` 用于平衡先后顺序，不会启动第二次 `compare2`。每个窗口都重新初始化训练，1200 步是累计工作量，不是一条连续的 1200 步训练轨迹。

保留此前 GPT-6.7B、TP8、32 层、序列 2048、micro batch 1/global batch 4、Attention＋MLP 的完整步协议。正式窗口用八 rank 最大墙钟除以 50，按全部 12 个完整块做配对 bootstrap（10000 次），报告均值、中位数、loss 和 95% CI。**负收益或没有落入 585–600 ms 都照常出报告**。

执行前自动验证：

- 每个 rank 的 zeros、small、outliers、真实 `[7,256,4096]` 远端形状，与“原 CUDA 解码＋八次 BF16 相加”逐位比较；八个 rank 覆盖全部来源排列。
- 两组的 64 个投影首输出指纹、smoke/profile/正式窗口中的 loss 与 grad_norm 轨迹一致；不将 grad_norm 相同描述成每个参数梯度逐位相同。
- 保留输出不被后续调用覆盖；原有相对 BF16 5%、相对 codec 1% 的前向误差门槛继续生效。
- profiler 通过实际 CUDA launch correlation 检查 GemmOnly、编码、NCCL 和声明的接收 kernel；基础组不能出现独立解码/八次 add，CUDA 组不能出现新融合 kernel。profile 与数值预检不混入正式计时。

## 启动一次比较

在 `/workspace/TempName` 中运行，推荐使用新包装脚本，它会读取容器恢复时生成的 `env.sh`（若存在）：

```bash
bash tools/h100/phase6/run_basic_quant.sh --trials 1 --execute
```

已经加载恢复环境时，也可以直接启动：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_basic_quant.py --trials 1 --execute
```

新入口默认只测一次，并且只允许 `--trials 1`；不构建、不测试改进量化。去掉 `--execute` 只打印计划，不创建实验目录、不查询 GPU。`run_a_baseline.sh` / `detach_a_baseline.py` 仍然对应下方旧三组实验，**本次使用 `run_basic_quant.sh` / `detach_basic_quant.py`**。

执行时新建 `logs/h100/phase6/basic-quant-receiver-1x-50steps-<时间>/`，先构建 GemmOnly 与新 codec，再完成预检和正式计时。继续使用原双重 fork 与 PID 1 绑定检查，启动成功后可关闭终端，容器需要保持运行；没有自动改为前台执行。状态见 `batch-state.json` 和 `handoff/container-supervisor.log`，最终结果为 `compare1/report.json` / `report.md`。`--worker-timeout` 默认 1800 秒。

`--flux-build` 可以显式复用校验通过的旧 GemmOnly 构建；旧三组实验的 `codec-build` 不含新融合入口，不能传给 `--codec-build`。默认全部重新构建，避免误用旧基础库。报告离线复核命令：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/basic_quant.py report <批次目录>/compare1 --offline
```

## 本地验证与当前状态

CUDA 12.8 / SM90 的新 codec 动态库已编译、链接通过。CPU 检查覆盖冻结源码、两组调度、真实生成的 adapter、来源/数值预检记录、输出保存、profile 路径替换、报告缺失与负收益、单次启动和只读预览。详见 [本地验证记录](../../../../outputs/h100/phase6/basic-quant-validation-20261025/validation.json)。

**本批新增的八卡数值预检、profile 与完整训练已通过，实际步时与配对收益见页首。** 上述本地验证记录描述开跑前的 CPU / 编译检查，不是本批 GPU 性能数据；本批的 GPU 证据及离线复核位于“回传数据复核”一节。目前仍缺少第二次独立重复和真实语料的长期质量验证。

## 历史 v1：生产端融合的三组方案

以下保留旧实现说明与命令，用于解释已有数据。其 `basic` 是旧生产端融合实现，与本页上方新协议中的 `basic` 不同。

历史设计日期：2026-10-24。v1 三组实验，**统一覆盖 Attention＋MLP 输出投影**。本页的“Flux＋基础量化”仅保留生产端融合：GEMM 内编码并向目标 GPU 写入；接收端先独立解码，再独立归约。“Flux+改进量化”保留既有融合 v2 的生产端融合与接收端解码归约融合。“Flux＋CUDA量化”沿用原生 Hopper GemmOnly＋作者独立 TACO CUDA codec＋NCCL All-to-All。

**以下保留 v1 开跑前的方案与占位表，不代表当前测试状态。v1 已有回传结果，见本页上方历史结果表；这些数字不能填入 v2 的新基础量化列。**历史“改进量化”只覆盖 MLP，本次将其相同实现用于 Attention＋MLP，不能把历史步时当成本次对照。

| 指标 | **Flux+改进量化** | **Flux＋基础量化** | **Flux＋CUDA量化** |
| --- | --- | --- | --- |
| 策略 ID | `improved` | `basic` | `cuda` |
| 融合范围 | GEMM 内编码；解码与归约融合 | 仅 GEMM 内编码；解码与归约分开 | GEMM、编码、通信、解码、归约分开 |
| 作用范围 | Attention＋MLP | Attention＋MLP | Attention＋MLP |
| 量化覆盖 | 全远端 H128/E4M3 | 同左 | 同左 |
| 平均步时（ms/step） | 待实测 | 待实测 | 待实测 |
| 全部窗口中位步时（ms/step） | 待实测 | 待实测 | 待实测 |
| 平均 / 最终训练 loss | 待实测 | 待实测 | 待实测 |
| 吞吐（tokens/s） | 待实测 | 待实测 | 待实测 |

用户希望检验的耗时关系为：

```text
T(Flux+改进量化) < T(Flux＋基础量化) < T(Flux＋CUDA量化)
```

这是实验假设，不是必须实现的排序。基础版的来源解码可并行，CUDA 组使用不同 GEMM 后端；中间访存、GPU 并发和主机提交之间存在权衡，不能仅凭融合数量保证上述关系。报告保留全部窗口及负收益，性能排序不成立也继续完成所选次数的比较；正确性或执行证据不通过则停止。

### 1. 为什么这样定义“基础版”

原历史“改进量化”是冻结的 `a800-fused-v2-cta-v1`，它已经使用较朴素的 scalar CTA codec，并没有 MA1/MA2 来源预取、CR2 提交缓存或 CUDA Graph。若仅复制并改名，就无法得到一个具有明确区别的基础版。

本实验按融合边界定义三个实现层次：

```text
Flux＋CUDA量化：
  原生 Hopper GemmOnly → 整理远端 BF16
  → 作者独立编码 kernel → NCCL All-to-All
  → 作者独立解码 kernel → 逐来源 BF16 相加 → 输出

Flux＋基础量化：
  [原 GEMM＋编码＋远端写入] → 原发布屏障
  → [各来源独立解码到 BF16 中间张量]
  → [按固定来源顺序归约] → 原复用屏障 → 独立输出 clone

Flux+改进量化：
  [相同 GEMM＋编码＋远端写入] → 相同发布屏障
  → [解码＋固定顺序归约融合] → 相同复用屏障 → 独立输出 clone
```

“基础版”与“改进版”都沿用原始生产顺序、单工作区、逐次配置与原输出 clone。没有加入交替排序、到达优先、选择性 mask、打包来源预取、双工作区、持久配置优化、提交缓存或 Graph。这些历史优化不是本实验的组成部分。

### 2. 基础融合 kernel 的实现

实现由 [a_baseline_sources.py](../../../../tools/h100/phase6/a_baseline_sources.py) 注入隔离构建；新增接收端计算体见 [a_baseline_decode.cuh](../../../../tools/h100/phase6/a_baseline_decode.cuh)。

基础版与改进版共享相同 GEMM、epilogue 编码、packet 格式和 codec 数学代码。基础版仅在接收时增加必要的分阶段数据流：

1. 分配七个远端来源的 BF16 临时张量，形状为 `[7,256,4096]`。
2. 每个 CTA 负责某个来源的一个 H128 组，调用原 scalar CTA 解码过程，将结果舍入 BF16 后写入中间张量。
3. 独立归约 kernel 按 `(rank+1)%8, …, (rank+7)%8, rank` 的原顺序相加。最后一个是本地 BF16 贡献，不经过量化解码；每次相加后保留原 BF16 舍入。
4. 沿用原来的工作区复用屏障和输出所有权规则。

临时张量大小为 `7×256×4096×2 = 14,680,064` 字节，即 **14 MiB/调用**。它由 PyTorch 分配器管理，计入正式计时；可由缓存分配器复用，不表示每个模型必须永久持有 64 份。没有增加重复解码、延时循环或额外跨 rank 屏障。

改进版直接在同一个 kernel 中解码并累加，不生成这份完整的 BF16 中间张量。基础/改进的直接比较主要回答：**在同一生产端下，去掉接收端中间数据写回和独立归约提交，是否改善完整训练步。**

两版必须保持解码后舍入、来源顺序和逐次累加舍入，不能将基础版改为 FP32 跨来源求和。由于原始数学相同，要求基础/改进的实际投影输出及训练 loss/gradient 轨迹一致。

### 3. 三组固定条件与对比边界

| 项目 | 基础 / 改进 | 独立 CUDA |
| --- | --- | --- |
| GEMM | 相同 SM90 编译的 GemmV2 | 原生 Hopper GemmOnly |
| 编码 | 相同冻结 epilogue 内编码 | 作者原 CUDA 编码 kernel |
| 远端传输 | 原融合路径直接写目标工作区 | NCCL All-to-All |
| 接收 | 基础分开解码/归约，改进融合两者 | 作者 CUDA 解码后逐来源 BF16 相加 |
| 排序/选择 | 原始顺序，无到达表、无选择性 mask | 原生 GemmOnly，无上述策略 |
| 本地贡献 | BF16 | BF16 |
| 反向 | 原 BF16/STE 适配 | 原 BF16/STE 适配 |

基础/改进是一对有明确融合差异的直接对照；与 CUDA 组比较还包含 GEMM 后端和通信路径差异，不能把其全部差距归因于一个 kernel 的融合。

独立 CUDA 组复用 [baseline_codec.py](../../../../tools/h100/phase6/baseline_codec.py) 及固定提交 `ca717e05b6dd6d21fff4d2617d209c95be63f3ef` 中的作者 CUDA 编解码计算体。它是作者 codec 的移植基线，未使用完整 COCCL 通信调度与重叠，不代表原论文完整 TACO 系统。

训练固定 GPT6.7B：32 层、H4096、FFN16384、32 heads、S2048、TP8，micro batch1/global batch4，BF16，seed1234/token_seed4321。每个 optimizer step 为四个微批、8192 token，覆盖 32 个 Attention 输出投影及 32 个 MLP 输出投影；本地 K 分别为 512、2048。统一 `NCCL_NVLS_ENABLE=0`、`CUDA_DEVICE_MAX_CONNECTIONS=1`。

### 4. 完整训练次数与结果判定

`--trials 1` 只执行 `compare1`；默认 `--trials 2`，依次执行 `compare1`、`compare2`。每次完整比较内部仍包含两轮、每轮六个完整三组块，组内顺序平衡，第二轮反转；每个窗口独立初始化，10 步预热＋50 步正式计时。**这里的 trials 是独立比较次数，不是每次比较内部的 rounds；选择一次不会取消内部第二轮的顺序平衡。**

| 项目 | 单次（`--trials 1`） | 两次合计（默认） |
| --- | ---: | ---: |
| 完整三组块 | 12 | 24 |
| 正式窗口 | 36 | **72** |
| 正式 rank 记录 | 288 | **576** |
| 正式 optimizer step | 1800 | **3600** |
| 正式窗口的预热 step | 360 | 720 |
| 独立 smoke/profile 窗口 | 6 | 12 |
| 训练 profile trace | 24 | 48 |

1800 / 3600 步分别是一次 / 两次比较中独立窗口的累计工作量，不是同一个模型连续训练这些步数。profile 单独执行，不混入正式计时。每个窗口以八 rank 最大墙钟除以 50 得到步时。只运行一次时，结果支持本次比较，不声称已完成两次独立重复验证。

完整报告三项配对：

- **improved/basic：** 接收端解码归约融合的增量。
- **basic/cuda：** 基础融合执行路径相对独立 CUDA 量化的整体差距。
- **improved/cuda：** 双端融合执行路径相对独立 CUDA 量化的整体差距。

收益使用完整三组块配对的对数耗时比和 bootstrap 95% CI；同时报告算术均值、中位数、吞吐、平均/末步 loss、各轮结果和全部窗口。预期顺序的点估计与置信区间确认情况分别列出，不使用历史耗时，也不因为某一轮不符合期望而只保留另一轮。

smoke/profile 检查需确认：

- 三组初始化、输入、模型与设备身份一致，实际覆盖全部 64 个目标投影，无 skip/fallback。
- 基础/改进的投影输出及训练轨迹一致；CUDA 组通过自己的独立 codec 参考和原误差门槛，不要求跨 GEMM/codec 实现逐位等价。
- 返回输出不被后续调用覆写，正确性检查在正式计时外。
- 基础版实际出现独立解码及归约 kernel；改进版出现原融合解码归约 kernel；CUDA 组出现实际 GemmOnly、作者编解码和 NCCL 通信。
- 分开统计 CPU annotation 与 GPU 镜像 annotation，避免仅凭重复同名标记误判；编译或真实路径错误保留现场并停止。

### 5. 一行启动

若重建容器后出现 `Build CUDA Toolkit missing`，先按 [新容器环境恢复](../restore-container.md) 运行恢复脚本。它复用持久化 conda、补齐缺失 Toolkit 与构建依赖，并生成加载固定环境变量的实验启动包装脚本。

将本次新增 `a_baseline*`、`detach_a_baseline.py` 与现有 `tools/h100/phase6` 依赖同步到原服务器，在 `/workspace/TempName` 仓库根目录执行。**只测一次完整比较，36 个窗口、1800 个正式 optimizer step：**

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_a_baseline.py --trials 1 --execute
```

不传 `--trials` 时仍默认测两次，共 72 个窗口、3600 个正式步；原命令保持有效：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_a_baseline.py --execute
```

默认重新构建基础/改进两份隔离 Flux 库、独立 Hopper GemmOnly 库和作者 CUDA codec 库，CUDA12.8 编译、CUDA12.4 运行、jobs2；使用相邻 `../Megatron-LM` 的固定提交。无需复用旧 C 校准或到达表。已有库可用 `--fused-build`、`--flux-build`、`--codec-build` 指定，仍须通过本协议的来源与库身份检查。

**去掉 `--execute` 只打印计划**，不创建目录、不编译、不 fork、不查询 GPU。路径可以通过 `--python`、`--megatron`、`--build-cuda-home`、`--cuda-home`、`--nccl-root`、`--out` 指定；输出目录必须不存在。单个 worker 默认超时 1800 秒，可通过 `--worker-timeout` 调整；超时保留日志并终止该 worker 进程组，停止后续比较。

沿用已有双 fork 容器后台托管流程，确认监督进程由容器 PID1 接管后才开始工作。`--foreground --execute` 可用于有意前台诊断。启动成功只表示后台进程建立，实验完成需要检查批次状态；容器和主机必须保持运行。

默认输出目录随所选次数使用 `1x` 或 `2x`：

```text
logs/h100/phase6/a-baseline-fusion-<1x或2x>-50steps-<北京时间戳>/
  batch-state.json
  handoff/container-supervisor.log
  handoff/supervisor.json
  fused-build/                    基础/改进的隔离源码与库
  flux-build/                     原生 Hopper GemmOnly
  codec-build/                    作者独立 CUDA codec
  compare1/
    submission.json
    scripts/
    results/                      每窗口的进程日志、八 rank JSON 和 profile
    preflight.json
    report.json / report.md
  compare2/                       仅 --trials 2 时生成，第二次独立比较
```

各阶段命令与日志同时保存在批次根目录。`status=completed` 表示所选次数的完整比较及正确性检查完成，不表示预期速度排序一定成立。离线复核入口为：

```bash
python3 tools/h100/phase6/a_baseline.py report /path/to/batch/compare1 --offline
## 仅测两次时执行：
python3 tools/h100/phase6/a_baseline.py report /path/to/batch/compare2 --offline
```

### 6. 本地验证状态

新增 `--trials 1` 后，启动器 **7 项 CPU 回归通过**，覆盖一次比较完成状态、36/1800/288 计数、无 compare2、默认两次流程和失败停止规则；一次比较的命令预览未创建批次目录。见 [单次比较验证](../../../../outputs/h100/phase6/a-baseline-validation-20261024/single-trial-validation.json)、[测试日志](../../../../outputs/h100/phase6/a-baseline-validation-20261024/cpu-single-trial-tests.log)及[命令预览](../../../../outputs/h100/phase6/a-baseline-validation-20261024/launch-preview-single-trial.txt)。本次只调整启动次数，以下保留原实现的完整验证记录。

本次已完成以下检查，汇总见 [validation.json](../../../../outputs/h100/phase6/a-baseline-validation-20261024/validation.json)：

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 新构建、BF16 舍入/来源顺序、训练适配、完整报告与启动门禁 | **22 项 CPU 测试全部通过，无跳过** | [测试日志](../../../../outputs/h100/phase6/a-baseline-validation-20261024/cpu-tests.log) |
| 两版真实接收 runtime 的 SM90 对象、PTX、SASS 编译 | CUDA12.8 编译通过；未启动 GPU | [编译登记](../../../../outputs/h100/phase6/a-baseline-validation-20261024/runtime-agent/validation.json)、[编译日志](../../../../outputs/h100/phase6/a-baseline-validation-20261024/runtime-agent/compile.log) |
| 启动命令预览 | 两次比较、72 窗口、3600 正式步；未创建批次目录 | [命令预览](../../../../outputs/h100/phase6/a-baseline-validation-20261024/launch-preview.txt) |
| 新 profile 解析器对已回传真实 trace 的兼容性 | 原融合 A 与独立 Flux CUDA 的 rank0 trace 均通过，每角色 128 调用 | 汇总登记中的原文件路径及 SHA256 |

CPU 模拟测试包括完整 36 窗口/288 rank 报告，并刻意构造与预期相反的速度顺序，确认仍会正常记录完成；也检查首轮报告不完整会阻止第二次训练、source-only 不能启动 GPU、返回输出别名或覆盖会失败，以及 CPU/GPU 镜像 annotation 不重复计数。

上述历史 trace 只用于验证解析器兼容性，融合 A 的交替生产顺序不作为本实验的原始顺序证据。**当前没有完整链接本次 Flux 动态库，也没有运行基础版的新 GPU kernel 或八卡训练。** 启动命令会在原环境完成完整构建、实际 mapping、数值及路径预检，再进行正式计时。本页没有填入推测性能。
