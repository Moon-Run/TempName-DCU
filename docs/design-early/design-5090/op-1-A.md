# OP1-A：面向 RTX 5090 的连续 packet 实现与三组对比

更新：2026-11-01。目标为单机 **8×RTX 5090、TP8/DP1**。本次新增独立的 A 实现和[三组测试入口](../../../tools/5090/phase6/run_op1_a.sh)，不改写原九组 baseline 或 H100 实现。

**状态：三组八卡 full-step 测试已完成。** 批次 `op1-a-direct-wire-v1-1x-50steps-20261101-185132-330063` 中，新 A 平均 **978.184 ms/step**，同期 Flux＋基础量化为 **967.327 ms/step**；新 A 慢约 **1.122%**。Flux 加交替排序后平均 **967.302 ms/step**，相对原排序的配对区间跨零，本轮没有观察到明确的端到端收益。两个 Flux 组的训练轨迹不逐位相同，数值差异和结果边界见第 6 节。

## 1. 为什么原 A 在 5090 上失去优势

[5090 baseline](baseline.md) 的完成批次中，A 平均 **986.816 ms/step**，Flux＋基础量化平均 **967.474 ms/step**，A 慢约 **2.00%**。原始记录为 [report.json](../../../../Load-5090/logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-1x-50steps-20261101-124230-640474/report.json)。这只是该历史批次的对比，不是下面新 A 的结果。

[H100 baseline](../design-h100/New-Mechanism/baseline.md) 中的 A 依靠交替排序，让不同目的 rank 的 GEMM tile 交替产出；其原始传输可使用跨卡地址。当前 5090 机器的跨卡 peer access 为 **0/56**，现有迁移实现必须先完成整个 producer，再通过 NCCL All-to-All 交换数据。交替排序保留下来，但原来依赖 tile 逐步到达的计算通信重叠已不存在。

旧 5090 A 的真实路径是：

```text
每次调用重新配置
  → 原 A GEMM＋融合量化，写本地 SoA packet
  → pack：SoA packet 转成 tile wire
  → NCCL All-to-All
  → unpack：tile wire 转回 SoA packet
  → 原 A decoder，按固定 source 顺序进行 BF16 归约
  → adapter clone 输出
```

tile wire 是九组共同支持选择性量化而采用的格式。A 对所有远端贡献都量化，发送端和接收端本来可以直接使用完整的 SoA packet，因此这里的两次转换并非 A 算法所需。Flux＋基础量化已经直接发送连续 packet，不承担这两次转换。

## 2. 新 A 的实现

新 A 保留原 A 的 GEMM 注册、融合量化 epilogue、CTA 交替排序、原 scalar decoder 和 BF16 舍入，只改 packet 寻址和生命周期：

```text
实例初始化固定配置
  → 原 A GEMM＋融合量化，直接写 NCCL tx_wire
  → NCCL All-to-All
  → 原 A decoder 直接读 rx_wire，返回独立输出
```

具体改动在 [op1_a_native.py](../../../tools/5090/phase6/op1_a_native.py)，以已验证的旧 `selective/a` 源码为输入，生成独立变体，旧构建不被覆盖。

### 2.1 连续 packet，消除 pack/unpack

每个目的 rank 的输出为 `256×4096`，共有 8192 个 H128 group。每个 packet 保留三个连续区段：FP8 payload、FP32 quant scale、FP32 adaptive scale；大小仍为 `8192×136 = 1,114,112` 字节。只传七个远端 packet，每 rank 发送 **7,798,784 字节**，self split 为 0。

对本 rank `r` 和远端 rank `d`，连续 packet 下标为：

```text
packet_slot(r, d) = d − int(d > r)，要求 d != r
```

producer 的目的 packet 指针直接指向 `tx_wire + packet_slot(r,d)×packet_bytes`；接收后 decoder 的 source packet 指针使用同一 compact 规则。NCCL peer 顺序仍为升序；decoder 仍以 `(rank+1)%8` 到 `(rank+7)%8` 的原顺序累加，最后加入本地 BF16 贡献。**传输存储顺序与数值归约顺序是两个不同概念**，本次没有改变后者。

`tx_packets` 与 `tx_wire` 是同一块存储，`rx_packets` 与 `rx_wire` 也是同一块存储；不再运行 `rtx5090_pack_tiles_kernel` 或 `rtx5090_unpack_tiles_kernel`。原 A decoder 的运算代码只替换 source packet 地址表达式，量化公式、Hadamard 变换、FP8 转换和每次 BF16 累加舍入保持不变。

### 2.2 固定配置与输出所有权

- 配置在实例初始化时完成。前向不再调用 `rtx5090_local_reconfigure`；误调用会失败，不能悄悄退回旧路径。
- 每次前向直接返回新分配的输出，adapter 根据 `owns_output=True` 删除额外 clone。六次输出在第七次复用之后仍必须保持原值且地址不同。
- 同一 group/device/stream/shape 的实例共享 `tx_wire` 和本地 BF16 scratch；接收 packet、native handle、GEMM workspace 和返回输出保持各实例独立。提交锁覆盖 producer→NCCL→decoder，失败会禁止继续使用相关共享 scratch。
- 保留 NCCL `Work.wait()` 在当前 CUDA stream 上建立的依赖。下一次 producer 必须排在上次 decoder 之后；不把 CPU 返回或 GPU kernel 的提交当作传输已完成。

**本轮不叠加 R 的 packed-prefetch decoder、CR3 的提交缓存或 bias/residual 融合。** GEMM 初始化计数仍等于前向次数；全远端量化、本地 BF16、梯度计算、optimizer、模型尺寸和计时循环均保持原协议。这样可将新 A 的效果限定在连续 packet、实例配置和输出所有权这一组改动上。

新 native ABI 使用版本 2；新 A 内部传输标识为 `rtx5090-op1-a-direct-packet-v1`，拒绝错误加载旧库。外围硬件/NCCL 环境仍复用现有 5090 staging 检查。它仍然是整 producer 完成后再交换，**不声称恢复 H100 的跨卡 tile overlap**。

## 3. 三组定义与交替排序对照

实验协议为 **`rtx5090-phase6-op1-a-direct-wire-v1`**。

| 展示组 | 脚本 ID | GEMM 与排序 | 量化和接收端 |
| --- | --- | --- | --- |
| A（5090 direct-wire） | `a` | 原 A SM80/A100 V2 算法注册，实际编译 SM120；原 CTA 交替排序 | GEMM epilogue 内量化，直接连续 packet；原 A scalar decoder；实例配置及独立输出 |
| Flux＋基础量化 | `flux-basic` | 原 5090 SM89/RTX5090 GemmOnly V2 / Stream-K，实际编译 SM120；原 Flux 排序 | 原独立 CUDA 编码＋NCCL All-to-All＋接收端融合 v2 |
| Flux＋基础量化＋交替排序 | `flux-basic-interleaved` | 保留同一 Stream-K 配置，仅替换目标 GEMM 的 CTA tile 映射；允许由此改变部分输出的归约分配 | 与上一组相同的 codec、packet、NCCL、decoder 和输出所有权 |

两种目标形状均为 `M=2048, N=4096`，本地 `K=512`（Attention）或 `2048`（MLP）。128×128 tile 共 512 个。对线性 tile 编号 `i`：

```text
destination = (i % 8 + rank) % 8
within      = i // 8
tile_m      = destination * 2 + within // 32
tile_n      = within % 32
```

这与 A 的目的 rank 交替映射一致。实现见 [op1_a_swizzle.hpp](../../../tools/5090/phase6/op1_a_swizzle.hpp)：rank 在主机端构造 kernel 参数时固定，仅改变实际 GemmOnly kernel 的 tile 坐标映射。没有额外的输入/输出行重排，也没有修改 NCCL peer 顺序。这里是 CTA 逻辑映射，GPU 的实际 CTA 启动先后仍由硬件调度决定。

测试会从实际注册的 GEMM kernel 参数导出 mapping，检查八个 rank、两种 K 的 512 个 tile 全覆盖、无重复、与上述公式相同且区别于原 Flux；profile 再核对实际执行的 `Op1AInterleavedSwizzle` kernel 类型。只记录一个配置名称不能通过这些门禁。

**按本次要求，允许归约方式随排序变化，不要求两个 Flux 组逐位同轨迹。** 原注册的 `Auto` 实际解析为 `StreamkSK`，目标形状存在拆分 K 的 tile；交替映射会改变哪些输出使用哪些拆分/归约任务，可能产生不同浮点舍入。本实现保留原 Stream-K 基线，不将两个 Flux 组另行改为 DP，也不改变 NCCL 的 source 归约顺序。

两个 Flux 组的实验差异是“加入交替映射及其引起的 Stream-K 归约分配变化”，不能声称只是数值完全相同的执行顺序变化。各组仍须独立通过相同数值误差预算和完整训练门禁。报告额外保留逐窗口 loss、gradient norm 的最大绝对/相对差，projection 输出和最终参数是否逐位相同的观测；不相同不会单独阻止计时。

新 A 与任一 Flux 组同时存在 GEMM 后端、编码融合和 decoder 的差异，属于端到端组合比较；不能把全部差值解释为排序收益。新旧 A 只改地址和生命周期，因此二者的独立 packet/输出 oracle 仍要求逐位一致。

## 4. 一键执行

在目标 5090 环境的 `TempName-New` 仓库根目录执行，沿用 `env/setup_5090.sh` 和 `env/activate_5090.sh` 已配置的 Python/CUDA/Megatron 路径。更新整个 `tools/5090/phase6/` 目录，不能只复制 shell 文件。

```bash
# 只预览：不创建批次、不查询 GPU、不构建、不启动训练。
bash tools/5090/phase6/run_op1_a.sh

# 完整三组测试，默认后台运行。
bash tools/5090/phase6/run_op1_a.sh --execute
```

首次运行会准备冻结的九组基准构建，再独立编译新 A 和交替排序 GemmOnly；没有必要重新跑九组训练。**已有正确的九组构建时，建议指定 `--baseline-build`，只编译两个新变体：**

```bash
bash tools/5090/phase6/run_op1_a.sh --execute \
  --baseline-build /workspace/TempName-New/logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-1x-50steps-20261030-151053-241177/build
```

上面的路径沿用 baseline 文档已使用的目标机构建；若目标机存放位置不同，替换为实际的、完成构建的目录。回传日志中没有二进制的归档目录不能冒充可复用构建。源文件、环境、依赖或二进制身份不符会停止；旧注册前端的已知修复沿用原入口的验证与备份机制。

先做独立门禁，或在当前终端运行：

```bash
# 只做硬件、Apex 和真实八 rank NCCL 预检，不编译或训练。
bash tools/5090/phase6/run_op1_a.sh --execute --preflight-only

# 构建后只做新旧 A 逐位检查及实际 CTA mapping，不创建完整训练模型。
bash tools/5090/phase6/run_op1_a.sh --execute --foreground \
  --transport-smoke-only --baseline-build /path/to/verified-baseline/build

# 完整实验，当前终端显示阶段；18 块可与原九组每组窗口数对齐。
bash tools/5090/phase6/run_op1_a.sh --execute --foreground --blocks 18 \
  --baseline-build /path/to/verified-baseline/build

# 复用已经编译好的新 OP1 构建，重新执行门禁与新批次计时。
bash tools/5090/phase6/run_op1_a.sh --execute \
  --op1-build /path/to/previous-op1-batch/build
```

`--baseline-build` 与 `--op1-build` 二选一。`--op1-build` 仍需要其记录的原基准构建，用于控制组和新旧 A oracle。每次启动使用新的输出目录；不能向旧批次追加或混合不同源码的窗口。`--python`、`--cuda-home`、`--build-cuda-home`、`--megatron`、`--jobs`、`--worker-timeout`、`--out` 与原入口用途相同。

新结果默认保存到：

```text
../Load-5090/logs/5090/phase6/op1-a-direct-wire-v1-1x-50steps-<时间>/
```

状态看 `batch-state.json`，后台日志看 `handoff/supervisor.log`，完成后看 `report.md` 和 `report.json`。后台任务需要目标容器持续运行。只读重放完整批次：

```bash
bash tools/5090/phase6/run_op1_a.sh --report --out /path/to/completed-op1-batch
```

## 5. 必须通过的门禁与统计口径

执行顺序如下，任一步失败均不进入正式计时：

1. 源码生成与冻结训练/计时 AST 检查；硬件、Apex 实际运算和八 rank NCCL 集合通信预检。仍固定 `NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=0`、`NCCL_NVLS_ENABLE=0`。
2. 验证旧构建并生成隔离的新变体；记录源码、依赖、SM120 二进制与编译环境身份。SM80/SM89 是算法注册标签，不是输出 cubin 的架构。
3. **新旧 A 独立 oracle：** 分别在新进程组加载旧 A 与新 A，在非默认 stream 上，对 Attention/MLP 两形状及零值、随机、离群输入检查数学参考；逐 rank 比较新旧输出与七个远端发送/接收 packet 的 SHA256，要求逐位一致。互证 56 条有向链路，检查延迟 rank、七次复用、六个保留输出、两个共享 scratch 实例交替调用及关闭其中一个后另一个仍可用。
4. 原 A mapping 检查及新 Flux 实际 GEMM 参数的 CTA mapping 检查。
5. 三组完整模型 smoke：保留原 BF16/codec 数值误差预算、全部目标模块覆盖、无 fallback、有限 loss/梯度和完整参数轨迹检查。允许两个 Flux 组因 Stream-K 归约变化而出现不同训练轨迹，差异如实写入报告，不把它标成已通过逐位等价验证。
6. 三组各八 rank profile：A 必须实际执行 direct producer→NCCL→direct decoder，pack/unpack、逐次 configure、输出 clone 均为零；两个 Flux 组必须执行各自真实 GEMM 类型及原独立 codec。profile 的 launch 归属和依赖关系检查通过后才计时。

模型保持 **32 层、hidden 4096、FFN 16384、32 heads、sequence 2048、micro batch 1、global batch 4、BF16、TP8/DP1**；每 step 为 **8192 tokens**。计时包含前向、反向、通信、optimizer 和清梯度，不启用 CUDA Graph。

默认 `--trials 1 --blocks 6`，每六块穷举三组全部六种顺序，随机种子固定；每组在每个位置出现两次，两两前后关系平衡。`--blocks` 支持 6 的正整数倍，`--trials` 支持 1 或 2，各 trial 单独汇总。

| 默认正式工作量 | 数量 |
| --- | ---: |
| 每组窗口 | 6 |
| 每窗口预热 / 正式 optimizer step | 10 / 50 |
| 每组正式 step | 300 |
| 三组正式窗口 / rank 记录 | 18 / 144 |
| 三组正式 optimizer step | 900 |

主指标仍为 `max(8 rank elapsed_seconds)×1000/50`，组均值使用全部正式窗口；同时报告中位数、最慢窗口、吞吐、Torch allocated/reserved 和独立 TACO workspace。allocated＋native 逐 rank 相加再取最大，不含 CUDA context/NCCL 内部占用。

报告提供三项同期配对：**A/Flux＋基础量化、A/Flux＋基础量化＋交替排序、Flux＋基础量化＋交替排序/Flux＋基础量化**。使用完整 block 的对数耗时比和 10,000 次 bootstrap 给出 95% CI；均值比与配对估计分开，负收益和慢窗口保留。区间只描述该固定负载，三项比较没有多重比较校正。

## 6. RTX 5090 八卡实测与回传复核

数据来自 `op1-a-direct-wire-v1-1x-50steps-20261101-185132-330063`：**1 trial × 6 个完整 block × 3 组，共 18 个正式窗口、144 份正式 rank 记录、900 个正式 optimizer step**。每组有 6 个窗口、300 个正式 step；每窗口另有 10 步预热。全部窗口保留，窗口步时取八 rank 最大 elapsed 除以 50，平均值为各组六个窗口的算术均值，每 step 为 8192 tokens。见[原始报告](../../../../Load-5090/logs/5090/phase6/op1-a-direct-wire-v1-1x-50steps-20261101-185132-330063/report.md)、[报告数据](../../../../Load-5090/logs/5090/phase6/op1-a-direct-wire-v1-1x-50steps-20261101-185132-330063/report.json)和[批次状态](../../../../Load-5090/logs/5090/phase6/op1-a-direct-wire-v1-1x-50steps-20261101-185132-330063/batch-state.json)。

| 指标 | A（5090 direct-wire） | Flux＋基础量化 | Flux＋基础量化＋交替排序 |
| --- | ---: | ---: | ---: |
| 平均耗时（ms/step） | 978.184 | 967.327 | **967.302** |
| 中位数（ms/step） | 978.640 | 967.481 | 967.512 |
| 最快 / 最慢窗口（ms/step） | 976.006 / 979.205 | 964.130 / 968.949 | 964.347 / 969.549 |
| 吞吐（tokens/s） | 8374.70 | 8468.69 | 8468.91 |
| A 相对该组的均值耗时下降 | +0.000% | -1.122% | -1.125% |
| 最大 Torch allocated（GiB） | 16.893 | 17.473 | 17.473 |
| 最大 Torch reserved（GiB） | 17.0625 | 17.6348 | 17.6348 |
| 最大 allocated＋native workspace（GiB） | 18.221 | 17.473 | 17.473 |

负降幅表示 A 更慢。显存合计先逐 rank 将 Torch allocated 与独立 `cudaMalloc` 的 native workspace 相加，再在所有窗口取最大值；不与 reserved 相加，也不含 CUDA context、NCCL 等内部占用，不能视为整卡峰值。

### 同期配对

按六个完整 block 配对，使用对数耗时比及 10,000 次 bootstrap；区间描述本次固定负载，三项比较未做多重比较校正。平均耗时差与配对点估计是不同统计量。

| 同期比较（前者 / 后者） | 平均耗时差（前者 − 后者，ms/step） | 配对降幅 %［95% CI］ | 前者更快的块数 |
| --- | ---: | --- | ---: |
| A / Flux＋基础量化 | +10.857 | -1.122［-1.300, -0.985］ | 0/6 |
| A / Flux＋基础量化＋交替排序 | +10.882 | -1.125［-1.304, -0.971］ | 0/6 |
| Flux＋基础量化＋交替排序 / Flux＋基础量化 | -0.025 | +0.003［-0.059, +0.078］ | 2/6 |

**结果解读：** 新 A 在六个配对块中均慢于两组 Flux。两个 Flux 均值仅差 0.025 ms/step，配对区间同时包含正负值，且交替排序仅在 2/6 块更快，不能据此认定它有稳定的加速或减速。A 与 Flux 同时存在 GEMM、融合编码及 decoder 的差别，不能把全部差值归因于某一个环节。完整条件下没有恢复 H100 的跨卡 tile 到达重叠。

此前九组 baseline 的旧 A 平均 **986.816 ms/step**，新 A 本次平均 **978.184 ms/step**，数值相差 **8.632 ms/step**；旧批同期 Flux 为 967.474 ms/step，本批为 967.327 ms/step。**这两次 A 不在同一个配对实验中**，不能将 8.632 ms 或约 0.875% 作为本改动的同场净收益或置信区间，也不能用同期 Flux 接近就排除其他批次因素。新旧 A 的逐位 native probe 只证明所覆盖的 packet 和输出一致，不代替同场训练速度对比。

### 正确性与数值差异

目标机报告 `correctness_passed=true`、`execution_path_validated=true`、`a_native_old_new_bitwise=true`。新旧 A probe 覆盖 Attention/MLP 两种形状和零值、随机、离群三类输入；八 rank 的输出和 packet 哈希逐位一致，56 条有向链路互证通过。三组模型 smoke 和 24 份 profile 均通过；A 的 profile 记录验证了 direct producer→NCCL→decoder，pack/unpack、逐调用配置和输出 clone 均为零。交替排序 Flux 的实际 GEMM CTA mapping 检查通过。

两个 Flux 组**不满足逐位轨迹等价**：报告的 8 个配对窗口包含 1 个 smoke、1 个 profile 和 6 个正式窗口，最终参数逐位一致的窗口为 **0/8**；smoke 中投影输出和参数轨迹也不逐位相同。整个比较中的最大 loss 绝对差为 **1.060479**，最大 gradient norm 绝对差为 **4.307620**。例如 block00 的第 18 个正式 step，rank0 loss 为 **4.177876 / 3.117397**；第 20 个正式 step 的 gradient norm 为 **14.641653 / 18.949272**（原排序 / 交替排序）。两组使用相同初始化、输入和训练协议；这些差异不应仅凭本报告归因于某个具体舍入步骤。上述门禁允许排序引起的归约变化，两组各自通过数值预算及有限 loss/梯度检查，但这些结果**不能证明长期收敛或模型质量等价**。

最大相对 loss 差为 **9.584 倍**，发生在原排序 loss 约 **0.000143** 的位置；最大相对 gradient norm 差为 **25.299 倍**，发生在原排序 norm 约 **0.001615** 的位置。相对差的分母接近零，不宜单独作为整体精度变化比例；上面的绝对差示例反映了两个轨迹在其他位置也存在可见差异。

### 环境和复核范围

目标主机为 `SL3060`，8×RTX 5090、SM120/170 SM、TP8/DP1，驱动 `595.84`，PyTorch `2.8.0a0+5228986c39.nv25.06`、CUDA `12.9`、NCCL `2.27.3`、Python `3.12.3`，跨卡 peer access 为 0/56。单机固定合成输入结果不直接代表其他拓扑、负载或真实数据训练表现。

回传复核使用本批冻结 controller 源码：重新验证 24 个 smoke、profile 和正式窗口的 **192 份 rank 记录**及其哈希，重算全部 18 个正式窗口、三组统计和 Flux 数值差异，与报告一致；八卡 NCCL、native 新旧 A packet/输出和实际 mapping 证据也重新校验通过。三组共 **24 份原始 GPU profile trace** 均在本地重新解析，并与保存的分析结果逐项一致。回传目录没有完整 build 与 CUDA 动态库，因此未在本地重新校验二进制或执行包含 build 身份检查的完整 `--report` 重放。

历史开发阶段的本地验证：`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools/5090/phase6 -p 'test_*.py' -v` 共 **163 项测试通过**；后续目标机实测证据以本节批次为准。

相关代码：[控制器](../../../tools/5090/phase6/op1_a.py)、[协议与统计](../../../tools/5090/phase6/op1_a_protocol.py)、[隔离构建](../../../tools/5090/phase6/op1_a_build.py)、[实际 CTA 排序](../../../tools/5090/phase6/op1_a_swizzle.py)、[新旧 A 与 mapping 探针](../../../tools/5090/phase6/op1_a_probe.py)、[worker 生成](../../../tools/5090/phase6/op1_a_sources.py)、[记录校验](../../../tools/5090/phase6/op1_a_validation.py)、[profile 审计](../../../tools/5090/phase6/op1_a_profile.py)。
