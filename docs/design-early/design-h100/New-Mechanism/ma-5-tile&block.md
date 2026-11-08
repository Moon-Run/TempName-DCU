# MA5：在 CR2 上调节 tile 与 block

更新：2026-10-25，已纳入 **MA5-R 按角色选择**批次 `ma5-role-split-1x-50steps-20261025-125018-954486` 的回传结果。**MA5-R 未验证相对 CR2 的整步正增量，也未达到 500 ms；本批存在明显计时突变。** 保留全部窗口后，T 平均 **2423.968 ms/step**、同期 CR2 平均 **2444.915 ms/step**，完整块配对收益 **+1.947%，95% CI [−0.456%, +5.854%]**，区间跨零。

**前四个配对块约 505–512 ms，最后三个块两组都升至约 5.6 秒/step。** 批次正常完成，数值、缓存与按角色执行路径通过，但不能把全样本的正点估计写成稳定加速，也不能删掉慢窗口后重新宣称达标。最新实测、逐块耗时与离线复核见[第 8 节](#8-ma5-r-八卡实测计时突变下未验证增量2026-10-25)。

最新 MA5-R 中 Attention 使用 `<<<528,256>>>`，MLP 使用原 CR2 的 `<<<2048,128>>>`；两种角色的 GEMM 均为 K32、`avail_sms=-1`。独立算子的 Attention 收益点估计为 **−3.468%**、MLP 为 **+0.001%**，两者区间都跨零。`increment_over_CR2_validated`、`absolute_500ms_observed`、`target_validated` 均为 `false`，表示性能目标未通过，并非运行失败。

**上一轮统一配置 MA5 的结果保留在第 6 节，与本轮分开统计。** 旧批次 `ma5-tile-block-1x-50steps-20261025-104210-136736` 的 T/CR2 平均为 **509.705 / 510.058 ms**，配对收益 **+0.069% [−0.108%, +0.260%]**；Attention 约快 0.811%、MLP 约慢 0.889%。第 1–5 节保留原方案，第 7 节保留 MA5-R 实现及 `run_ma5_role_split.sh` 入口。两批均未测四配置 sweep、K64 或 SM128，也没有同期 MA5-R/旧 MA5 对照。

## 1. 目标与边界

[baseline.md](baseline.md) 中 CR2 的全部窗口平均为 **511.297 ms/step**、中位数为 **510.450 ms/step**。从该均值降到 500 ms，需要节省约 **11.297 ms（2.21%）**。新实验使用本次重新构建、同场运行的 CR2 作为对照；历史 511.297 ms 只用于说明目标，不参与新配对收益计算。

本轮保持 GPT6.7B、TP8、Attention＋MLP、原冻结到达表、原 1/64 selective mask、双工作区、独立输出、原生提交缓存及每调用一次发布屏障。量化覆盖仍为 Attention **53/3584**、MLP **50/3584**。不改变 H128 编解码、来源累加顺序或 BF16 舍入规则。

CR2 上存在三个不同概念，调参必须落在实际执行路径上：

| 参数 | CR2 当前值 | MA5 的处理 |
| --- | --- | --- |
| GEMM CTA tile / warp tile | `128×128×32` / `64×64×32` | 默认保持；可选 K tile 同时改为 64 |
| decoder 每 block 线程数 / 一次处理的输出元素 | `128` / `512` | 支持 64、128、256、512 线程，即 256、512、1024、2048 元素 |
| decoder grid 的 block 数 | `2048` | 独立指定，通过 grid-stride 循环覆盖全部输出 |
| GEMM `avail_sms` | `-1`，由 CUTLASS 使用设备默认 | 可选 128 或 132，改变 Stream-K 调度输入；**不是直接指定 GEMM grid 数** |
| 旧 RS option 的 `num_blocks` | `6` | CR2 走直接 decoder 路径，单改该值不能调节实际 decoder grid，且会触发固定配置检查 |

CR2 历史训练 profile 中 decoder 合计约 2.8 ms/step，远小于 11.3 ms 的目标差距；profile 还存在观测开销与重叠。因此 decoder 调参是局部机会，不能据此保证达到 500 ms。GEMM K tile 和 SM 调度候选用于继续探索生产端收益。

## 2. 已实现的改动

### 2.1. Decoder：tile 大小与 grid 解耦

原 decoder 为 `<<<2048,128>>>`，每 warp 处理一个完整 H128 组，每 block 4 个 warp。每 rank 输出 `[256,4096]`，共 **8192 个 H128 组**。

MA5 将原逐组计算原样提取为设备函数，再使用 warp 粒度的 grid-stride 循环：

```cpp
group = blockIdx.x * warps_per_block + threadIdx.x / 32;
group < 8192;
group += gridDim.x * warps_per_block;
```

一个 warp 内的循环条件一致，原 full-mask shuffle 的参与线程不变。纯 BF16 路径的提前 `return` 留在逐组函数内部，不会漏掉同一 warp 后续负责的组。量化组继续使用原来源预取、原解码与逐次舍入；每个输出组只被一个 warp 写入。

默认候选为 **256 线程、528 blocks**：单次 CTA tile 从 512 增至 1024 个输出元素，grid 从 2048 减至 528。528 是目标 H100 SM132 的四倍，但不意味着硬件一定同时驻留四个 CTA/SM；驻留情况仍取决于资源使用。该配置大部分 warp 处理两个 H128 组，尾部按边界退出。

`--sweep` 顺序运行以下四组，每组都有自身同场 CR2 对照：

| 候选 | threads/block | 每次 CTA 输出 tile | decoder blocks | GEMM tile K / avail_sms |
| --- | ---: | ---: | ---: | --- |
| `t128-b528-k32-s-1` | 128 | 512 元素 | 528 | 32 / -1 |
| `t256-b1024-k32-s-1` | 256 | 1024 元素 | 1024 | 32 / -1 |
| `t256-b528-k32-s-1`（默认） | 256 | 1024 元素 | 528 | 32 / -1 |
| `t256-b264-k32-s-1` | 256 | 1024 元素 | 264 | 32 / -1 |

这组比较分别探查更大的 CTA tile、较少的 CTA 数量及二者组合。候选也可能因寄存器、驻留或循环开销而变慢。四组都保留，不依据算子微测的速度提前淘汰。

### 2.2. GEMM：可选 K64 与调度 SM 数

`--gemm-tile-k 64` 同时将 CTA tile 改为 **128×128×64**、warp tile 改为 **64×64×64**，保持 4 warp、128 线程和一个 K partition。不能只改 CTA K：原自动 warp K=32 会产生两个 K partition，破坏融合 encoder 的四 warp 布局假设。

K64 的 epilogue 使用精确匹配的新 thread map，并以编译期断言核对其基础输出布局和 `ThreadMapShape` 与 K32 相同。M/N tile 保持 128×128，因此原 **512 个物理 tile** 的 mask 和到达表可以继续使用。该候选增加 mainloop 的共享内存需求，可能降低 occupancy；也可能改变 Stream-K 分解及浮点计算轨迹，收益与数值一致性都必须实测。

`--gemm-sms 128` 将 CR2 缓存的 `GemmReduceScatterArguments.avail_sms` 从 -1 改为 128。512 个物理 tile 与 128 的整除关系提供了探索动机，但实际 grid 仍由 CUTLASS、K、occupancy 和 Stream-K 策略共同决定。报告记录真实 GEMM grid/block，不把 SM 预算冒充实际 block 数。

### 2.3. 来源与执行身份

代码入口：

- [ma5_sources.py](../../../../tools/h100/phase6/ma5_sources.py)：decoder 循环、真实 launch 参数与 GEMM 调度参数。
- [ma5_gemm.py](../../../../tools/h100/phase6/ma5_gemm.py)：可选 K64 generator/epilogue 变换和实际 GEMM 编译探针。
- [ma5_build.py](../../../../tools/h100/phase6/ma5_build.py)：从原 CR2 重建 `T/CR2` 两组，冻结 tuning、源码、mask 和构建输入。
- [ma5_bench.py](../../../../tools/h100/phase6/ma5_bench.py)、[ma5_fullstep.py](../../../../tools/h100/phase6/ma5_fullstep.py)：算子检查、完整训练及独立 profile。
- [detach_ma5.py](../../../../tools/h100/phase6/detach_ma5.py)、[run_ma5_tile_block.sh](../../../../tools/h100/phase6/run_ma5_tile_block.sh)：预览、后台启动和批次检查。

`T` 表示本次 tuning 候选，`CR2` 为逐文件核对的原实现。构建后的 C ABI 返回候选 tuning；算子与完整训练均读取它，并检查两组的 CR2 缓存初始化次数、调用次数、K、packet 地址及失败状态。Profile 进一步核对真实 decoder grid/block、GEMM K tile 与 128 线程，以及 GEMM→发布屏障→decoder 的执行顺序。

## 3. MA5 原候选测试命令

在八卡 H100 服务器的仓库根目录（如 `/workspace/TempName`），先运行默认 decoder 候选：

```bash
bash tools/h100/phase6/run_ma5_tile_block.sh --trials 1 --execute
```

启动方式与 `run_basic_quant.sh` 一致，优先读取 `outputs/h100/container-recovery/env.sh`，复用恢复环境中的 Python 和 CUDA 路径。默认后台托管；`--foreground` 可在当前终端顺序执行。**去掉 `--execute` 只打印计划，不创建批次目录、不编译、不查询 GPU。** 本地只预览时可用 `MA5_PYTHON=/usr/bin/python3 bash tools/h100/phase6/run_ma5_tile_block.sh --trials 1` 显式覆盖服务器 Python 路径。

一次完整 decoder 扫描：

```bash
bash tools/h100/phase6/run_ma5_tile_block.sh --trials 1 --sweep --execute
```

四个候选依次完成独立构建、算子检查和完整训练，不并发占用八张 GPU。每个候选都重新测量 CR2；首次扫描会编译四套候选/对照，耗时较长。`--sweep` 固定 decoder 参数，不能与单独的 decoder 覆盖参数混用。

单独检查 GEMM K64，decoder 保持原线程数和 grid：

```bash
bash tools/h100/phase6/run_ma5_tile_block.sh --trials 1 \
  --decoder-threads 128 --decoder-blocks 2048 --gemm-tile-k 64 --execute
```

单独检查 GEMM SM 调度，K tile 和 decoder 保持原值：

```bash
bash tools/h100/phase6/run_ma5_tile_block.sh --trials 1 \
  --decoder-threads 128 --decoder-blocks 2048 --gemm-sms 128 --execute
```

得到有希望的配置后，以明确参数在新批次复测两轮。例如，若默认 decoder 候选值得复测：

```bash
bash tools/h100/phase6/run_ma5_tile_block.sh --trials 2 \
  --decoder-threads 256 --decoder-blocks 528 --execute
```

如需合并 K64、SM128 和 decoder 调参，可同时提供三个选项，但要用同场训练结果验证组合收益。不同独立实验的加速不能直接相加。不同批次不要同时运行；`--execute` 返回只表示后台启动成功，完成状态以报告为准。

常用选项：`--out <新目录>`、`--jobs 2`、`--worker-timeout 1800`、`--cuda-home`、`--build-cuda-home`、`--nccl-root`。`--ma5-build <已有MA5构建>` 可复用已验证的两组构建，但必须显式给出与该构建一致的 tuning；不能用于 `--sweep`。线程数必须为 64/128/256/512，blocks 为正整数且不得超过 `8192/(threads/32)`；例如 512 线程最多 512 blocks，不能直接搭配默认 528。

## 4. 工作量、正确性与判定

每个候选、每个 trial 为 **8 个平衡 T/CR2 块、16 个正式窗口、128 份 rank 记录、800 个正式 optimizer step**。每窗口独立进程，先预热 10 步，再正式计时 50 步。`--trials 1` 只产生 compare1；`--trials 2` 为 32 个正式窗口、1600 正式步。四候选单轮 sweep 合计 64 窗口、3200 正式步。另有算子测试、训练 preflight 及独立 profile，不混入正式计时。

保持原完整 optimizer step 计时，包括前后向、通信、参数更新和清梯度；窗口步时为八 rank 最大墙钟 / 50。报告保留全部窗口，给出算术均值、中位数、前后半程，以及完整块配对 log 耗时比的 10000 次 bootstrap 95% CI。

必须满足：

1. 两种 K、多个 seed、零值/小值/离群值/随机值、输出保留、双槽复用、实例隔离和慢 rank 压力检查通过；T/CR2 输出逐位一致。
2. 完整训练的对应 rank loss/gradient 轨迹一致，无跳过、fallback 或未覆盖模块；两组都真实走 CR2 缓存。
3. 算子和完整训练 profile 的实际 launch 与冻结 tuning 一致，屏障、工作区和输出路径符合定义。

如果 K64 或 SM 调度改变了数值轨迹，该候选按当前协议拒绝，不静默放宽误差预算。性能退化、区间跨零和未达 500 ms 均为有效结果，仍完成报告。

| 报告字段 | 含义 |
| --- | --- |
| `increment_over_CR2_validated` | 每个请求的 trial 中，T/CR2 收益点估计与 95% CI 下界均 >0 |
| `absolute_500ms_observed` | 每个 trial 的 T 全部窗口算术均值 ≤500 ms；不是最低窗口或中位数阈值 |
| `target_validated` | 上述两项同时成立；批次完成还要求正确性与独立 profile 通过 |

`absolute_500ms_observed` 只描述本批均值，并不等于以置信区间证明长期均值 ≤500 ms。单轮结果与 sweep 属于探索；从多个候选中挑出最快者之后，应在新批次用两轮复测，不能把挑选后的点估计当作独立确认。

## 5. 查看结果与本地验证

默认批次目录：`logs/h100/phase6/ma5-tile-block-1x-50steps-<北京时间戳>/`。

```text
batch-state.json                       # 状态、候选列表、完成情况与训练结论
handoff/container-supervisor.log        # 后台总日志
t256-b528-k32-s-1/
  build/tuning.json                    # 实际构建参数
  operator/report.json                 # 正确性与独立算子时间
  operator/profile-summary.json        # 算子实际 launch
  fullstep/report.md                   # 易读的整步结果
  fullstep/report.json                 # 配对统计、全部窗口与数值摘要
  fullstep/profile-summary.json        # 训练期间的执行证据
```

失败时保留阶段日志和原始记录，不覆盖旧批次；`batch-state.json` 的 `status=completed, stage=done` 才表示所有请求的候选已完成。`handoff/controllers.json` 冻结控制器及依赖源码，每个阶段前后核对，运行期间不要修改工具源码。回传整批目录即可按原始 rank 记录重新汇总。

本地 CPU 检查命令：

```bash
python3 -m unittest discover -s tools/h100/phase6 -p 'test_ma5*.py' -v
bash -n tools/h100/phase6/run_ma5_tile_block.sh
```

上机前本地 **37 项 CPU 检查通过**，覆盖完整输出组覆盖、候选参数边界、原 CR2 源码冻结、K64 thread map、launch 身份、缓存状态、单/双 trial 工作量、缺失 rank 与轨迹漂移拒绝、负收益正常完成、只预览不执行和控制器漂移拒绝。shell 语法及带空格参数转发也已检查。见 [CPU 日志](../../../../outputs/h100/phase6/ma5-validation-20261025/cpu-tests.log)、[默认命令预览](../../../../outputs/h100/phase6/ma5-validation-20261025/launch-preview.txt)、[扫描预览](../../../../outputs/h100/phase6/ma5-validation-20261025/sweep-preview.txt)。

最终冻结源码已完成两组 host mapping、SM90 decoder runtime、GPU mapping 程序编译、真实 K32 CUTLASS/encoder 探针，以及候选 PyTorch C++ wrapper 和 pybind 目标文件编译。见 [最终编译验证](../../../../outputs/h100/phase6/ma5-validation-20261025/final/compile-default/validation.json)及[编译日志](../../../../outputs/h100/phase6/ma5-validation-20261025/final/compile-default/compile.log)。默认 decoder 的编译器报告为 40 registers、零 spill，原 CR2 为 44 registers、零 spill；这是编译诊断，不是性能结论。

可选 K64 的真实 CUTLASS/encoder 探针也已通过 SM90 编译，包含实际 `StoreD::kTacoSupported` 与 128 线程断言；最终源码与该探针逐文件核对一致，并编译了当前 K64/SM128 的 mapping 程序。见 [K64 编译记录](../../../../outputs/h100/phase6/ma5-validation-20261025/k64-probe-supported/validation.json)与[最终源码和 mapping 复核](../../../../outputs/h100/phase6/ma5-validation-20261025/final/k64-validation.json)。

**上述上机前检查没有运行 GPU、完整 Flux 动态库链接或训练性能实验，mapping 程序当时只编译未执行。** 随后服务器已完成默认候选的完整构建与八卡运行，结果见下节；本地编译结果本身不构成性能证据。

## 6. 默认候选八卡实测结果与离线复核（2026-10-25）

### 6.1. 完成范围与本次实际调参

批次为 `ma5-tile-block-1x-50steps-20261025-104210-136736`，唯一候选目录为 `t256-b528-k32-s-1`。[批次状态](../../../../logs/h100/phase6/ma5-tile-block-1x-50steps-20261025-104210-136736/batch-state.json)为 `completed/done`，[启动记录](../../../../logs/h100/phase6/ma5-tile-block-1x-50steps-20261025-104210-136736/handoff/launch.json)确认 `trials=1, sweep=false`。

实际完成 **8 个完整 T/CR2 配对块、16 个正式窗口、128 份正式 rank 记录、800 个正式 optimizer step**；每窗口 10 步预热＋50 步正式，全部窗口保留。另有 16 个算子计时窗口、32 份算子 trace 和 16 份完整训练 trace，profile 与正式计时分开。

运行环境为 **8×H100 80GB HBM3、SM132、PyTorch 2.6.0+cu124、CUDA runtime 12.4、NCCL 2.21.5**。模型仍为 GPT6.7B、32 层、hidden 4096、FFN 16384、sequence 2048、TP8、global batch 4。

本次只将 decoder 从 CR2 的 **2048 blocks × 128 threads** 改为 T 的 **528 blocks × 256 threads**，每次 CTA 输出 tile 从 512 增至 1024 个元素。两组 GEMM tile 都为 **128×128×32**、`avail_sms=-1`，实际 GEMM launch 都为 **512 blocks × 128 threads**。这些值来自真实训练 trace；本批没有测试 GEMM tile 或 SM 调度调整。

### 6.2. 完整训练：未确认正增量，未达到 500 ms

下表来自[完整训练报告](../../../../logs/h100/phase6/ma5-tile-block-1x-50steps-20261025-104210-136736/t256-b528-k32-s-1/fullstep/report.json)。每窗口步时为八 rank 最大墙钟 / 50；收益使用完整块配对对数耗时比与 10000 次 bootstrap，不以展示均值直接相除替代。

| 范围 | 配对块 | T 平均 ms/step | CR2 平均 ms/step | T/CR2 收益 %［95% CI］ |
| --- | ---: | ---: | ---: | --- |
| 前半程 | 4 | 510.717 | 510.878 | +0.032 [−0.183, +0.211] |
| 后半程 | 4 | 508.692 | 509.238 | +0.107 [−0.177, +0.439] |
| **完整一轮** | **8** | **509.705** | **510.058** | **+0.069 [−0.108, +0.260]** |

T 的算术均值比同期 CR2 少约 **0.353 ms/step**，8 个块中 **5 个更快、3 个更慢**。两组全部窗口中位数分别为 **509.640 / 509.467 ms**，T 中位数反而略高。完整一轮及前后半程的收益区间均跨零，因此本批不能确认正增量，也不能据此认定一定退化。

T 的全部窗口范围为 **505.043–512.925 ms**，平均值距离 500 ms 仍有 **9.705 ms**；CR2 范围为 **506.674–513.574 ms**。三个字段均为 `false`：`increment_over_CR2_validated`、`absolute_500ms_observed`、`target_validated`。本批没有剔除慢窗口，也不将历史 CR2 的 511.297 ms 与本次 T 作配对比较。

### 6.3. 正确性、缓存与执行路径通过

- [算子报告](../../../../logs/h100/phase6/ma5-tile-block-1x-50steps-20261025-104210-136736/t256-b528-k32-s-1/operator/report.json)确认 T/CR2 输出逐位一致，原 mapping、selective mask、保留输出、双槽复用、实例隔离和慢 rank 检查通过。
- 完整训练的对应 rank 全部 **loss/grad_norm 轨迹一致**，零 skip/fallback，64 个目标模块均覆盖。两组正式步平均 loss 都为 **3.793363792**，末步 loss 为 **0.000305734604**，末步 grad norm 为 **0.018002458919**。这些是固定合成输入上的数值一致性证据，不能代替长期收敛评价。
- 两组各有 **64 份正式 rank JSON、4096 个模块记录、8192 个槽状态**。每槽初始化一次、成功调用 **120 次**，绑定 K 为 512 或 2048，`failed=false`。两个槽每模块共 `(10+50)×4=240` 次，包含预热；实际运行的是 CR2 缓存路径。
- [训练 profile](../../../../logs/h100/phase6/ma5-tile-block-1x-50steps-20261025-104210-136736/t256-b528-k32-s-1/fullstep/profile-summary.json)中，每 rank、每角色均有 **128 次投影、128 道发布屏障、零输出 clone、零原生输出 slice/view**；公共 adapter 的 256 次 view 保留。T 的 decoder 实际为 `<<<528,256>>>`，CR2 为 `<<<2048,128>>>`，两组 GEMM tile、grid 和 block 如第 6.1 节所述。

因此，本次目标未通过并非候选未启用或路径回退；线程数和 block 数调整确实生效，但没有形成可确认的整步收益。

### 6.4. 局部结果：Attention 略快，MLP 略慢

以下为独立算子完整 API 的无 profiler 计时，含独立输出管理与最终同步；来自[算子报告](../../../../logs/h100/phase6/ma5-tile-block-1x-50steps-20261025-104210-136736/t256-b528-k32-s-1/operator/report.json)。每个角色保留 8 个配对块。

| 角色 | T 平均 μs | CR2 平均 μs | T/CR2 收益 %［95% CI］ |
| --- | ---: | ---: | --- |
| Attention | 91.539 | 92.288 | **+0.811 [+0.468, +1.188]** |
| MLP | 140.686 | 139.447 | **−0.889 [−1.058, −0.717]** |

两个角色方向不同，不能只展示 Attention 的正结果，也不能把这两个百分比直接相加推算整步收益。

训练 profile 先取每 rank 的角色中位数，再取八 rank 中位数，得到下表（**T / CR2，μs**）：

| 角色 | 投影 CPU 区间 | GEMM | decoder | 发布屏障 |
| --- | --- | --- | --- | --- |
| Attention | 98.728 / 99.005 | 65.192 / 65.048 | **10.080 / 10.736** | 465.792 / 459.735 |
| MLP | 99.068 / 99.696 | 111.496 / 111.288 | **10.968 / 11.136** | 219.520 / 216.071 |

该 profile 中 decoder 的中位数下降约 **0.656 / 0.168 μs**，但 MLP 的无 profiler 完整算子计时仍退化，完整训练区间也跨零。这说明局部 decoder 时长、CTA 数量或寄存器数减少均不能替代完整路径的收益判断。Profile 含观测开销，CPU、GPU 和通信会重叠；不能将这些中位数差相加或乘以 256，解释为已经实现的步时节省，也不能只凭屏障时长确定退化根因。

### 6.5. 离线复核与结论范围

已用与回传冻结脚本 SHA256 一致的控制器，离线重算**算子报告、完整训练报告、算子 profile 汇总、训练 profile 汇总**，四份结果均与回传 JSON 完全一致；同时核对启动配置、全部窗口、正确性门槛、缓存计数、真实 launch 和目标判定。原始计划、报告、前置检查和批次状态的哈希在复核前后不变。

证据见[复核脚本](../../../../outputs/h100/phase6/ma5-results-audit-20261025/audit.py)、[复核 JSON](../../../../outputs/h100/phase6/ma5-results-audit-20261025/audit.json)与[复核日志](../../../../outputs/h100/phase6/ma5-results-audit-20261025/audit.log)。本地未重跑 GPU，也未重新加载或重验服务器动态库；运行身份由回传记录与真实 trace 交叉核验。

**本批结论仅适用于默认 decoder 候选：正确性与执行路径通过，整步正增量和 500 ms 目标均未验证，继续保留 CR2 为已验证基线。** 四配置 sweep、GEMM K64、SM128 及其组合没有本批实测结果，不能据此认定全部 tile/block 调优无效，也不能将其列为已验证收益。

## 7. MA5-R：按角色固定 decoder 的新候选（2026-10-25）

以下保留 MA5-R 的实现与上机前方案；该单轮批次现已运行完成，实测见第 8 节。

### 7.1. 根据实测收窄改动

上一批同一 decoder 配置在 Attention 上为 **+0.811%**，在 MLP 上为 **−0.889%**，区间分别完全为正和为负。训练中两种角色各调用 128 次，继续强制使用一套参数缺乏实测支持。MA5-R 因此保留 Attention 的候选配置，并使 MLP 回到原 CR2 decoder。

| 实现 | Attention decoder grid / block | MLP decoder grid / block | GEMM |
| --- | --- | --- | --- |
| CR2 对照 | `2048 / 128` | `2048 / 128` | 原 `128×128×32`、`avail_sms=-1` |
| 已测 MA5 | `528 / 256` | `528 / 256` | 同上 |
| **新 MA5-R** | **`528 / 256`** | **`2048 / 128`，原 decoder 函数** | 同上 |

这项改动直接针对本批 MLP 退化配置。Attention 的算子正收益仍不能保证在混合角色的训练中保留；新的主机选择接口也有自身开销，都会计入正式计时。当前没有证据表明这一项改动就能补足距 500 ms 的约 9.705 ms 差距。

### 7.2. 原生实例固定选择，保留数值与生命周期

实现见 [ma5r_sources.py](../../../../tools/h100/phase6/ma5r_sources.py)。从原 CR2 源码生成新候选，保留原 `taco_ma2c_decode_kernel` 与原 `taco_decode_reduce`，为 Attention 追加已测 MA5 的逐组计算和 grid-stride kernel。

每个原生工作区实例在创建时依据自身固定 `cr2_k` 绑定 decoder 函数：**K=512 使用 Attention 候选，K=2048 使用原 CR2 函数**。函数指针为实例内常量；forward 不重新查询 TLS 中的 K、不逐次设置环境变量，也不经过 Python 的角色分支。这样不同角色、不同模块和双工作区交错调用时，选择仍属于各自实例。原 CR2 对固定 K 和错误形状的拒绝检查保留。

MLP 的原 CUDA decoder 源码和 launch 保留；Attention 的每组算术与已测 MA5 一致。两组仍使用原冻结 mask/到达表、H128 codec、逐源 BF16 舍入、双工作区、独立输出和每调用一次发布屏障。所有 GEMM 参数及反向路径保持原设置。两种角色共享新增的固定选择接口，因此不能将整个 MLP 主机调用称为与 CR2 完全相同。

[ma5r_build.py](../../../../tools/h100/phase6/ma5r_build.py)使用独立协议 `h100-ma5-role-split-v1`，生成 **T/CR2** 两组；这里 T 表示新 MA5-R，旧 MA5 的报告和工具保留。构建配置显式记录 `attn`、`mlp` 两份参数，拒绝改变 MLP decoder 或 GEMM 设置。

[算子测试](../../../../tools/h100/phase6/ma5r_bench.py)和[完整训练测试](../../../../tools/h100/phase6/ma5r_fullstep.py)均按角色读取原生参数，核对两槽绑定的 K；profile 必须确认 Attention 使用新 kernel、MLP 使用原 CR2 kernel，且各自 grid/block 正确。即使误用的 kernel 数值恰好相同或 grid 相同，也不能通过角色路径检查。

### 7.3. 新一轮启动命令

在八卡服务器的 `TempName` 仓库根目录运行：

```bash
bash tools/h100/phase6/run_ma5_role_split.sh --trials 1 --execute
```

默认候选为 `attn-t256-b528-mlp-cr2`，与本轮重新构建的原 CR2 同场比较。依次完成独立构建、算子正确性/计时/profile、完整训练和训练 profile；每 trial 仍为 **8 个配对块、16 个正式窗口、800 个正式 optimizer step**，每窗口 10 步预热＋50 步正式。性能负收益仍完整保留，正确性或角色路径错误会终止并保留现场。

默认后台启动，查看：

```text
logs/h100/phase6/ma5-role-split-1x-50steps-<北京时间戳>/
  batch-state.json
  handoff/container-supervisor.log
  attn-t256-b528-mlp-cr2/operator/report.json
  attn-t256-b528-mlp-cr2/fullstep/report.md
  attn-t256-b528-mlp-cr2/fullstep/report.json
  attn-t256-b528-mlp-cr2/fullstep/profile-summary.json
```

去掉 `--execute` 只预览。服务器 Python、CUDA 路径继续沿用恢复环境；本地只预览可设置 `MA5R_PYTHON=/usr/bin/python3`。`--trials 2` 在新批次做两轮确认；`--foreground` 可前台运行。保留 `--out`、`--jobs`、`--worker-timeout` 和 CUDA/NCCL 路径选项；`--attn-threads`、`--attn-blocks` 只调 Attention，MLP 固定。复用构建使用 `--ma5r-build`，只接受参数一致的新协议构建。

报告继续分开列出 **相对同期 CR2 的正增量**与**候选均值 ≤500 ms**，不能把旧 MA5/CR2 的历史点估计加入新收益。本轮没有同期旧 MA5 组，因此只能检验新 MA5-R/CR2，不能直接给出 MA5-R/旧 MA5 的配对收益。

### 7.4. 上机前验证

验证覆盖两种角色的原生 K、参数与 kernel 身份、原 MLP CUDA 代码保留、Attention 算术保留、缓存/双槽状态、数值轨迹、完整配对工作量及预览不执行。本节记录上机前检查，不构成新候选已加速的证据；服务器实测见第 8 节。

最终 **63 项 CPU 检查通过**，其中 26 项为 MA5-R、37 项为旧 MA5 回归。两组 host mapping、真实 SM90 runtime/PTX、mapping 程序，以及候选的真实 K32 GEMM、PyTorch C++ wrapper 和 pybind 目标文件编译通过。候选中保留的 MLP decoder 与 CR2 对照的对应 **PTX kernel 内容完全一致**；Attention 逐组代码则与上一批 MA5 保持一致。

证据见 [CPU 日志](../../../../outputs/h100/phase6/ma5r-validation-20261025/cpu-tests.log)、[编译验证](../../../../outputs/h100/phase6/ma5r-validation-20261025/compile-check/validation.json)、[验证汇总](../../../../outputs/h100/phase6/ma5r-validation-20261025/validation.json)和[启动预览](../../../../outputs/h100/phase6/ma5r-validation-20261025/launch-preview.txt)。当时本地未运行新 GPU 实验或完成整套动态库链接；随后服务器已通过新入口完成构建与八卡运行。

## 8. MA5-R 八卡实测：计时突变下未验证增量（2026-10-25）

### 8.1. 批次完成、实际角色与正确性

批次为 `ma5-role-split-1x-50steps-20261025-125018-954486`，候选为 `attn-t256-b528-mlp-cr2`。[批次状态](../../../../logs/h100/phase6/ma5-role-split-1x-50steps-20261025-125018-954486/batch-state.json)为 `completed/done`，[启动记录](../../../../logs/h100/phase6/ma5-role-split-1x-50steps-20261025-125018-954486/handoff/launch.json)确认只运行一轮 T/CR2。

完成 **8 个完整配对块、16 个正式窗口、128 份正式 rank 记录、800 个正式 optimizer step**，每窗口仍为 10 步预热＋50 步正式；另有 16 个算子计时窗口、32 份算子 trace、16 份完整训练 trace。环境为 **8×H100 80GB HBM3、SM132、PyTorch 2.6.0+cu124、CUDA runtime 12.4、NCCL 2.21.5**，模型和输入协议保持 GPT6.7B、TP8、32 层、sequence 2048、global batch 4。

算子输出逐位一致，完整训练对应 rank 的全部 **loss/grad_norm 轨迹一致**，64 个目标模块均覆盖、零 skip/fallback。两组正式步平均 loss 都为 **3.793363792**、末步 loss 为 **0.000305734604**、末步 grad norm 为 **0.018002458919**。两组各 64 份正式 rank JSON、4096 个模块记录、8192 个槽状态；每槽初始化一次、成功调用 120 次、`failed=false`，绑定 K 为 512 或 2048。

[训练 profile](../../../../logs/h100/phase6/ma5-role-split-1x-50steps-20261025-125018-954486/attn-t256-b528-mlp-cr2/fullstep/profile-summary.json)确认：T 的 Attention 使用新 decoder `<<<528,256>>>`，MLP 使用原 CR2 decoder `<<<2048,128>>>`；CR2 两种角色均为 `<<<2048,128>>>`。两组 GEMM tile 都是 **128×128×32**、实际 launch 都是 **512 blocks × 128 threads**。每 rank、每角色有 128 次投影和 128 道发布屏障，零输出 clone、零原生输出 slice/view。**角色拆分确实执行，性能目标未通过不能解释为候选未启用。**

### 8.2. 正式整步统计：保留所有慢窗口

下表来自[完整训练报告](../../../../logs/h100/phase6/ma5-role-split-1x-50steps-20261025-125018-954486/attn-t256-b528-mlp-cr2/fullstep/report.json)。计时仍取八 rank 最大墙钟 / 50；收益按完整块配对对数比及 10000 次 bootstrap 计算，不用均值相除替代。

| 范围 | 配对块 | MA5-R 平均 ms/step | CR2 平均 ms/step | MA5-R/CR2 收益 %［95% CI］ |
| --- | ---: | ---: | ---: | --- |
| 前半程（block00–03） | 4 | 510.557 | 508.675 | **−0.371 [−1.003, +0.284]** |
| 后半程（block04–07） | 4 | 4337.379 | 4381.156 | **+4.212 [−0.196, +11.266]** |
| **完整一轮** | **8** | **2423.968** | **2444.915** | **+1.947 [−0.456, +5.854]** |

全部窗口中位数为 **511.446 / 553.470 ms**；MA5-R 范围为 **508.299–5668.111 ms**，CR2 为 **505.231–5653.558 ms**。MA5-R 在 8 个配对块中 **4 个更快、4 个更慢**。完整一轮和前后半程的区间均跨零；正增量、均值 ≤500 ms 和综合目标均未通过。

前半程是原计划已有的诊断项，不能事后将其提升为替代正式结论。全样本均值也不能与上一轮约 510 ms 的均值直接比较，解释为角色拆分导致数倍退化：本批同期 CR2 同样出现大幅变慢。

### 8.3. 逐块耗时显示运行中发生明显变化

以下保留实际执行顺序及全部窗口，单位 ms/step；数据由原始 rank 记录重算。

| 配对块 | 执行顺序 | MA5-R | CR2 |
| --- | --- | ---: | ---: |
| block00 | T → CR2 | 511.204 | 510.444 |
| block01 | CR2 → T | 511.037 | 505.231 |
| block02 | T → CR2 | 511.688 | 507.335 |
| block03 | CR2 → T | 508.299 | 511.688 |
| block04 | T → CR2 | **509.151** | **595.252** |
| block05 | CR2 → T | **5600.805** | **5653.558** |
| block06 | T → CR2 | **5571.449** | **5648.221** |
| block07 | CR2 → T | **5668.111** | **5627.592** |

block04 的两组已经明显不在同一耗时水平，该块单独的时间比对应 **+14.465%**；随后三个块两组均达到约 5.6 秒/step。这个变化使全样本点估计容易被跨状态的配对影响，不能把 +1.947% 解释为稳定机制收益。两组各有三个超过 1000 ms 的窗口，全部保留，没有按慢窗口筛选或重新配对。

原始记录的 `ms_per_step` 与 `elapsed_seconds × 1000 / 50` 一致，各窗口取八 rank 最大值与报告一致，慢窗口的八个 rank 也共同变慢。因此这不是单纯的汇总单位或步数换算错误。**现有记录确认了计时突变，但尚不能确定其原因**；不能仅凭这些数字认定是 GPU 争用、降频、通信故障或某个新 kernel 导致。后续若复测，应先核查并记录运行期间的资源状态，保持原协议与同场对照，再评估稳定收益。

### 8.4. 独立算子与 profile：未复现可确认的 Attention 收益

来自[算子报告](../../../../logs/h100/phase6/ma5-role-split-1x-50steps-20261025-125018-954486/attn-t256-b528-mlp-cr2/operator/report.json)；每个角色 8 个完整配对块，保留全部样本。

| 角色 | MA5-R 平均 μs | CR2 平均 μs | MA5-R/CR2 收益 %［95% CI］ |
| --- | ---: | ---: | --- |
| Attention | 95.885 | 92.105 | **−3.468 [−12.337, +0.927]** |
| MLP | 139.538 | 139.539 | **+0.001 [−0.259, +0.228]** |

Attention 的 T 在 block07 为 **126.284 μs**，其余七个 T 窗口约 91–92 μs；该慢窗口同样保留。Attention 的负点估计没有得到区间确认，本批也没有复现上一轮约 +0.811% 的可靠正收益。MLP 与同期 CR2 基本持平，未显示上一批的明确负收益；但没有同期旧 MA5 组，不能给“恢复 MLP”单独分配净贡献。

训练 profile 的各 rank 角色中位数再取八 rank 中位，单位 μs（**MA5-R / CR2**）：

| 角色 | 投影 CPU 区间 | GEMM | decoder | 发布屏障 |
| --- | --- | --- | --- | --- |
| Attention | 102.814 / 104.344 | 64.968 / 64.928 | 13.920 / 13.872 | **3384.666 / 3130.804** |
| MLP | 102.229 / 103.718 | 110.544 / 110.489 | 12.968 / 12.616 | **2890.043 / 2178.066** |

本批发布屏障的可见时间达到毫秒量级，GEMM 中位时长仍接近此前的约 65/111 μs；这一现象支持继续检查等待和调度状态，但不能仅靠 profile 确定根因。Profile 与正式计时分开且有观测开销，各阶段时长不能相加推算未观测的步时节省。

### 8.5. 离线复核与最终结论

已用与回传冻结脚本 SHA256 一致的控制器，离线重算**算子报告、完整训练报告、算子 profile 汇总、训练 profile 汇总**，四份均与回传 JSON 完全一致；核对了配置、全部 16 个训练窗口、原生缓存计数、按角色 launch、正确性与目标判定。原始计划、报告、前置检查和批次状态的哈希在复核前后不变。

证据见[复核脚本](../../../../outputs/h100/phase6/ma5r-results-audit-20261025/audit.py)、[复核 JSON](../../../../outputs/h100/phase6/ma5r-results-audit-20261025/audit.json)、[复核日志](../../../../outputs/h100/phase6/ma5r-results-audit-20261025/audit.log)。本地只复核回传记录与 trace，未重跑 GPU，也未重新加载或重验服务器动态库。

**MA5-R 功能与执行路径通过，本批整步正收益和 500 ms 目标均未验证；明显计时突变进一步限制了性能归因。** 继续保留 CR2 为已验证基线，MA5-R 作为待稳定环境复测的候选；不将本批 +1.947% 登记为加速，也不将共同出现的秒级慢窗口归咎于角色拆分。
