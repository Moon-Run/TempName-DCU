# RTX 5090 九组 NCCL staging v2 baseline：MA11 / MA89 / MA8 / MA9 / CR3 / R / C / A / Flux＋基础量化

## 慢 GPU 的 MLP GEMM 差异：完整步时情景估算（2026-11-03）

**若保持 TP8，并假设 GPU 2–6 的 MLP 输出投影 GEMM 均达到当前快卡水平，OP3 组合组的情景估算为约 841.573 ms/step，OP4 两窗口平均约 823.460 ms/step。** 对应实测分别为 850.195 和 832.249 ms/step。这是仅消除所观察到的 MLP GEMM 差异、且假定它全部体现在整步关键路径上的简化估算，**不是排除慢卡后的实测结果，也不是已验证的性能收益或误差界限**。

这里的“排除慢卡影响”指用同等快卡替换这些卡的相关 GEMM 性能，仍假定八个 rank、TP8、模型和其他阶段保持原样。直接删掉 GPU 2 等 rank 的计时记录并不能得到去除慢卡后的训练时间：其余 rank 在实测中仍要参与跨卡同步、等待慢卡；实际减少 GPU 数量也会改变张量切分和通信，不能用原 TP8 数值推算。

估算口径：每个配置分别从完整训练 profile 的 GPU 0、1、7 中取 **每 rank 的 128 次 MLP GEMM 均值中的最大值**作为快卡参考，从 GPU 2–6 中取对应最大值作为慢卡参考。每个 optimizer step 有 32 层 × 4 个微批，即每 rank 128 次 MLP 输出投影；假定每次慢卡相对快卡的差值都无重叠地落在整步关键路径上，计算：

```text
单步假设差额 = (慢卡参考 GEMM μs − 快卡参考 GEMM μs) × 128 / 1000
情景估算 ms/step = 实测 ms/step − 单步假设差额
```

| 配置和正式计时口径 | 快卡参考 μs | 慢卡参考 μs | 假设差额 ms/step | 实测 ms/step | 情景估算 ms/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flux＋量化，4 个完整块均值 | 221.903（GPU 1） | 320.894（GPU 4） | 12.671 | 994.003 | 981.332 |
| Flux＋量化2，4 个完整块均值 | 208.202（GPU 7） | 301.199（GPU 4） | 11.904 | 992.012 | 980.108 |
| Flux＋量化3，4 个完整块均值 | 150.943（GPU 1） | 218.420（GPU 3） | 8.637 | 975.476 | 966.839 |
| OP3 R，compare1 的 20 块均值 | 151.136（GPU 1） | 218.979（GPU 3） | 8.684 | 855.544 | 846.860 |
| OP3 改进，compare1 的 20 块均值 | 150.826（GPU 1） | 218.955（GPU 3） | 8.721 | 854.444 | 845.723 |
| OP3 组合，1 个窗口 | 150.976（GPU 1） | 218.334（GPU 3） | 8.622 | 850.195 | 841.573 |
| OP4，block00 | 150.829（GPU 1） | 219.469（GPU 3） | 8.786 | 833.486 | 824.700 |
| OP4，block01 | 150.640（GPU 7） | 219.323（GPU 3） | 8.791 | 831.011 | 822.220 |
| OP4，两个窗口均值 | — | — | 8.789 | 832.249 | 823.460 |

各行的 GEMM 均值来自各自配置的独立 profile；正式步时来自相应配置的实际计时窗口，两者并非同一次采样。profile 只采样了该阶段，不能证明差值在每个正式 step 中相同，也不能证明八卡每次 GEMM 同时开始、同一张卡始终处在关键路径上。通信等待可能吸收一部分差额，替换 GPU 也可能改变其他 GEMM、反向、通信及系统状态；本表未估算这些影响，因此实际变化可能小于或大于该情景差额。

三组 Flux 数值取自 [11 月 1 日 21:42 批次的原始 profile 索引](../../../../Load-5090/SL3060/flux-quant-four-arm-v1-1x-50steps-20261101-214204-981363/profile-summary.json)和同批 [samples.json](../../../../Load-5090/SL3060/flux-quant-four-arm-v1-1x-50steps-20261101-214204-981363/samples.json) 的 block00–03；该批次后来中断，四块仅为阶段性数据，不是完整实验结果。OP3 双组来自 [v1 报告](../../../../Load-5090/SL3060/op3-two-arm-v1-2x-50steps-20261102-022324-519378/report.md)及其 profile；OP3 组合来自 [v2 报告](../../../../Load-5090/SL3060/op3-combined-v2-1x-50steps-20261102-100813-883313/report.md)及其 profile；OP4 两窗口来自 [OP4 报告](../../../../Load-5090/SL3060/op4-nccl-channels-single-v1-1x-50steps-20261102-105103-438654/report.md)和各自 profile。这些不同批次不构成同期配对对照，OP3“改进”与 Flux“改进量化”也不是同一配置。下方九组 baseline 属于另一个批次，不使用本表的差额直接修正。

更新：2026-11-01。目标机器为**单机 8×RTX 5090，TP8、DP1**。本页汇总九组同场 full-step 对比，并提供 [5090 入口](../../../tools/5090/phase6/run_baseline.sh)。

**当前状态：九组 RTX 5090 同场实验已完成。** 批次 `baseline-nine-arm-nccl-staged-v2-1x-50steps-20261101-124230-640474` 的九组 smoke、九组 profile 和 **162 个正式计时窗口**均通过，完成 **8100 个正式 optimizer step**。MA11 平均 **911.458 ms/step**、**8987.79 tokens/s**；相对同期 MA89 平均耗时下降 **3.406%**，相对同期 Flux＋基础量化下降 **5.790%**。原始证据见[批次报告](../../../../Load-5090/logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-1x-50steps-20261101-124230-640474/report.md)、[报告数据](../../../../Load-5090/logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-1x-50steps-20261101-124230-640474/report.json)和[批次状态](../../../../Load-5090/logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-1x-50steps-20261101-124230-640474/batch-state.json)。

本轮实验协议为 `rtx5090-phase6-nine-arm-nccl-staged-v2`，传输协议为 `rtx5090-nccl-staged-v2`；旧的直接访存 v1 构建与结果不能复用。目标机的 56 条跨卡有向 peer-access 链路均不可用，因此投影使用 **本卡 producer 写入本地 packet → NCCL All-to-All → 本卡 decoder**。这是经过迁移的机制对照，不保留 H100 的跨卡 tile 到达及计算通信重叠；原有 OOM 和 profile 检查器兼容问题的历史记录见下文。

## 1. 对比目标与 H100 参考

[H100 baseline](../design-h100/New-Mechanism/baseline.md) 中 MA11 的完整 optimizer step 平均耗时为 **450.662 ms/step**，原始值为 `450.6624015075795 ms`。该页取 MA11 首轮已复核批次 `ma11-v1-causal-softmax-1x-50steps-20261027-121342-060643`；本页保留机制来源和量化算术，使用新的 staging 传输实现，不把 MA19、MA20 或 MAV1 算作 MA11。

| 指标 | MA11 | MA89 | MA8 | MA9 | CR3 | R | C | A | Flux＋基础量化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 历史平均耗时（ms/step） | **450.662** | 459.806 | 474.004 | 489.878 | 501.433 | 515.716 | 516.784 | 536.692 | 582.318 |
| 5090 同场平均耗时（ms/step） | **911.458** | 943.601 | 965.172 | 995.407 | 1017.192 | 977.081 | 1022.978 | 986.816 | 967.474 |
| 5090 同场中位数（ms/step） | **911.603** | 943.621 | 965.721 | 995.406 | 1017.265 | 977.195 | 1023.035 | 986.831 | 967.859 |
| 5090 同场最慢窗口（ms/step） | **912.119** | 943.930 | 966.544 | 995.997 | 1017.859 | 978.014 | 1024.961 | 988.477 | 969.195 |
| 5090 吞吐（tokens/s） | **8987.79** | 8681.63 | 8487.60 | 8229.80 | 8053.54 | 8384.16 | 8007.99 | 8301.44 | 8467.41 |
| 5090 MA11 相对各组均值耗时下降 | +0.000% | +3.406% | +5.565% | +8.434% | +10.395% | +6.716% | +10.902% | +7.636% | +5.790% |
| 5090 最大 Torch allocated（GiB） | 18.728 | 18.728 | 18.728 | 17.730 | 17.730 | 17.008 | 17.730 | 17.008 | 17.473 |
| 5090 最大 allocated＋TACO workspace（GiB） | 21.384 | 21.384 | 21.384 | 20.386 | 20.386 | 18.337 | 20.386 | 18.337 | 17.473 |

H100 行来自历史多批次汇总，5090 行来自本次 **1 trial × 18 块 × 9 组**同场实验。每组均保留 18 个正式窗口、144 份正式 rank 记录和 900 个正式 step；窗口步时取八 rank 最大 elapsed 除以 50，每步处理 8192 tokens。H100 原表的“选定窗口耗时”混合了均值、中位数和最慢值，**本页九组一律用全部正式窗口的算术均值计算主表降幅**。跨机器差异同时包含 GPU、互联、CUDA/PyTorch、GEMM 和传输后端差异，不表示某一机制的净收益。结果显式记录 `original_mechanisms_not_identical=true` 和 `original_peer_overlap_preserved=false`。

最大显存为所有正式窗口、八 rank 中的最大值；allocated＋workspace 先逐 rank 相加再取最大，不与 reserved 相加。该合计不含 CUDA context、NCCL 等内部占用，不能当作整卡峰值。所有窗口均保留；例如 MA8 最快窗口为 954.171 ms，而中位数为 965.721 ms，主表均值没有剔除该窗口。

### 同期配对结果

下表按 18 个完整 block 配对，以对数耗时比和 10,000 次 bootstrap 计算；配对估计与主表的算术均值比是不同统计量。正值表示前者更快。区间仅描述本次固定负载，36 个两两比较未作多重比较校正。

| 同期比较（前者 / 后者） | 平均耗时（ms/step） | 平均减少（ms/step） | 配对降幅 %［95% CI］ | 更快的完整块 |
| --- | --- | ---: | --- | ---: |
| MA11 / MA89 | 911.458 / 943.601 | 32.143 | +3.406［+3.384, +3.428］ | 18/18 |
| MA89 / MA8 | 943.601 / 965.172 | 21.571 | +2.235［+2.090, +2.317］ | 18/18 |
| MA8 / CR3 | 965.172 / 1017.192 | 52.020 | +5.114［+5.029, +5.253］ | 18/18 |
| MA9 / CR3 | 995.407 / 1017.192 | 21.785 | +2.142［+2.105, +2.175］ | 18/18 |
| CR3 / C | 1017.192 / 1022.978 | 5.786 | +0.566［+0.522, +0.612］ | 18/18 |
| R / A | 977.081 / 986.816 | 9.736 | +0.987［+0.954, +1.019］ | 18/18 |
| MA11 / Flux＋基础量化 | 911.458 / 967.474 | 56.016 | +5.790［+5.686, +5.856］ | 18/18 |

**结果解读：** MA11 在本批九组中平均最快；MA89、MA8、Flux＋基础量化、R、A、MA9、CR3、C 依次排在其后。MA11/MA89、MA89/MA8、MA8/CR3 和 MA9/CR3 使用相同选择性量化策略，适合观察本 staged 后端中相应机制组合的同期差异；各项收益不应直接相加。MA11 与 Flux＋基础量化的量化策略及 GEMM 后端不同，5.790% 是端到端组合对比，不能归因于单一优化。

**实际环境与验证：** 目标主机 `SL3060`，8×RTX 5090（SM120、每卡 170 SM）、TP8/DP1，驱动 `595.84`，PyTorch `2.8.0a0+5228986c39.nv25.06`、CUDA `12.9`、NCCL `2.27.3`、Python `3.12.3`。八卡 peer access 为 0/56；拓扑记录显示 GPU0–3 与 GPU4–7 分属两个 NUMA 域。报告标记 `correctness_passed=true`、`execution_path_validated=true`、`equivalent_policy_trajectories_identical=true`：同策略等价组的 loss、梯度和参数轨迹检查通过，九组各八 rank 的 profile 路径检查通过；不同量化策略之间不强求相同轨迹。归档中的完整 36 组配对结果见[报告](../../../../Load-5090/logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-1x-50steps-20261101-124230-640474/report.md)。

**回传数据复核：** 使用该批次冻结的 controller 源码，对全部 180 个 smoke、profile 和正式窗口的 **1440 份 rank 记录**重新验证身份、硬件、机制、数值和同策略等价组，并与报告的 rank 文件哈希一致；从正式 rank 记录重算 162 个窗口和所有统计，与 `report.json` 一致。八卡 NCCL 预检、native mapping、八组 TACO 原生 smoke 证据及 72 份 profile trace 的哈希也通过核对。profile 的实际路径分析通过是目标机报告的结果，本次本地复核核对了 trace 哈希和已保存的 72 份分析结果，**没有重新解析全部 trace**。回传目录不含 MA11 兼容 `.so` 和复用的完整构建，且记录使用目标机绝对路径，因此此处不是在本地执行完整 `--report` 重放；没有将缺失的二进制验证记为通过。

## 2. 九组实现与量化范围

| 展示名称（均为 staged 迁移版） | 脚本组名 | 原机制 ID | 新后端中保留的比较内容 |
| --- | --- | --- | --- |
| MA11 | `ma11` | `F4-cache-zcount-csoftmax` | 完整 MA89，加注意力 mask/softmax 中间态消除；softmax 对齐目标 PyTorch 的归约与 BF16 舍入，QK/PV 算术不变。 |
| MA89 | `ma89` | `F4-cache-zcount` | CR3，加 MA8-v2 前后向激活复用和 MA9 保序多张量零梯度统计。 |
| MA8 | `ma8` | `F4-cache` | CR3，加 MA8-v2 激活缓存；QKV/FC1 反向复用前向完整输入，减少重复 AllGather。 |
| MA9 | `ma9` | `F4-zcount` | CR3，加分块计数与保序汇总两个零统计 kernel；不改梯度范数、裁剪和 Adam。 |
| CR3 | `cr3` | `F4` | decoder 内融合 bias/residual，保留两次 BF16 舍入和原 LayerNorm；使用本地 producer 的提交缓存。 |
| R | `r` | `R`（MA2-R） | 保留原 decoder 与全远端量化；由新前端持有实例配置和独立输出。 |
| C | `c` | `C` | 保留冻结 swizzle、选择性量化 mask、双工作区和实例配置；不保留原跨卡到达时序。 |
| A | `a` | `A` | 全远端量化；复用持久 CUDA 工作区，但每次调用重新配置 native producer，作为新后端中的提交方式对照。 |
| Flux＋基础量化 | `flux-basic` | `basic` | Flux GemmOnly、独立 TACO CUDA 编码、NCCL All-to-All、接收端融合解码/归约 **v2**。 |

量化均覆盖 32 层 Attention 输出投影与 MLP 输出投影，每 rank 共 64 个目标模块；本地贡献保持 BF16。MA11/MA89/MA8/MA9/CR3/C 使用原 `1/64` 预算上限的选择性量化 mask，R/A/Flux＋基础量化使用全远端量化。MA11 的注意力 bool mask 与量化选择 mask 是两个不同对象。

选择性量化各组共用**冻结的 H100 到达表和 mask**，用于固定算法条件下的跨架构移植对照；这些表不代表 5090 校准结果。若后续在 5090 上重新采样到达顺序或选择 mask，须新建协议和结果批次，不能与本次混算。

原 H100 的 A/R/C/CR3 主机提交、IPC 屏障和工作区生命周期与新后端不同；上表是新后端中的对应消融，不能把 R/A 等差值解释为原 IPC wrapper 的逐调用成本。MA8 激活复用、MA9 保序零统计、MA11 softmax 及 decoder BF16 算术继续接受独立的数值和实际执行路径检查。

原 MA11 Python helper 锁定 PyTorch 2.6；5090 生成副本改为绑定预检记录中的精确 PyTorch 版本，包括目标 NGC 25.06 的 `2.8.0a0+5228986c39.nv25.06`。原 Megatron 类源码身份、native softmax self-test，以及完整模型 smoke 中每次前后向的逐位对照均保留。版本门禁调整不代表新 PyTorch 的 softmax 已证明等价；任何精确检查失败都会阻止正式计时。

## 3. 本地 packet、NCCL 传输与同步

H100 Phase6 的量化投影使用 CUTLASS V2，内部注册键为 SM80/A100。5090 构建将机器码编译为 **`sm_120`**，本地 producer 使用冻结 GEMM/epilogue 算术和 swizzle，将按目的 rank 打包的内容写入本 GPU 的发送缓冲，不构造 CUDA IPC peer 地址或跨卡 GroupBarrier。NCCL 交换完成后，decoder 只访问本 GPU 的接收缓冲和本地 BF16 贡献，保留原 source 归约顺序与舍入。

生产、交换、消费按 CUDA stream 依赖排序：先完成本地 producer 的 packet 写入，再提交字节 All-to-All，等待通信的 stream 依赖后启动本地 decoder；工作区再次写入前必须排在上次消费之后。临时 scratch 按进程组、GPU、固定 stream、native arm、形状、mask 和 slot 共享，持有同一提交锁直到 decoder 提交完成；下一实例的 producer 排在该 stream 上的 decoder 之后。任何失败会同时禁止共享该 scratch 的实例继续提交。**这一版本不尝试恢复原 GEMM tile 尚在产生时跨卡发送和到达优先消费的重叠**，通信开始时间与源端排序的性能意义都已改变。

全部九组冻结 `NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=0`、`NCCL_NVLS_ENABLE=0`，保留并记录其他 NCCL/NIC 设置。这里允许 NCCL 使用主机中转，不把环境变量本身作为“实际采用 SHM”的证据；具体 transport 由 NCCL/目标系统选择。

Flux＋基础量化使用适合 5090 的 **GemmOnly V2** 后端；H100 该列的 Hopper V3/GMMA 后端不能直接用于 5090。因此这一列保留基础量化 v2 的算法定义，但不能解释为与 H100 使用了同一个 GEMM kernel。

执行前置检查要求：

- 恰好暴露 8 张不同的 RTX 5090，计算能力为 `(12, 0)`，每卡 170 SM，且没有占用 GPU 的其他计算任务。
- CUDA Toolkit 至少 12.8，PyTorch CUDA 构建能够实际执行 SM120 运算。Apex 等 Megatron 所需扩展须在相同 PyTorch/CUDA 环境中重新编译并可导入。**不要用 `env/setup_h100.sh` 把 5090 环境安装回 H100 的旧版 cu124 环境。**
- 不要求 CUDA peer access；预检记录完整矩阵，并启动真实八 rank NCCL 检查。BF16 AllGather、AllReduce、ReduceScatter，以及非等长 `uint8` All-to-All（self split 为 0）必须全部正确。探针在非默认 CUDA stream 上循环三次并等待异步 Work，记录每个 rank 的 GPU、软件环境和通信配置。
- 仓库中的冻结源码、选择表、相应依赖以及指定的 Megatron 源码完整可用。构建器按源文件身份校验，在新的输出目录生成 SM120 二进制，不复用 H100 的 `.so`。

旧本地 staging 为每个 slot 分别分配约 77 MiB，64 个目标模块、双 slot 的张量约占每卡 9.6 GiB。当前 `fixed-stream-shared-scratch-private-packets-v1` 复用五项临时缓冲：`tx_bf16`、`tx_packets`、`rx_bf16`、`tx_wire`、`rx_wire`。接收 `rx_packets`、选择 mask、偏移表、native handle、GEMM workspace/提交缓存及返回输出保持每实例独立，双 slot 继续轮转且不互相共享 scratch。当前两种形状、一个 stream 对应四套共享 scratch；按冻结 mask 逐 rank 计算，这部分张量总驻留从 **9.5935–9.6091 GiB 降至 1.3305–1.3310 GiB，减少 8.2630–8.2781 GiB**。这是分配尺寸计算，不是实测峰值；还需叠加模型、optimizer、激活缓存和临时张量。32 GiB 5090 上的完整 smoke 仍是门禁，模型层数、batch、MA8 缓存及逐位数值检查均不降低。

GEMM 私有 workspace 使用独立 `cudaMalloc`，不在 PyTorch allocator 峰值中。每个 rank 另记所有 TACO slot 的 `native_workspace_bytes`，报告同时展示 Torch allocated 峰值及逐 rank 加上 TACO workspace 后的最大值；该合计仍不包含 CUDA context、NCCL 等内部显存，不能视为整卡总占用。

环境/Apex 检查通过尚不能完成预检，必须取得八个成功的 NCCL rank 记录。随后依次执行 native mapping、八组 TACO 原生传输 smoke、九组完整模型 smoke、九组 profile，再开始正式计时。原生传输 smoke 不依赖模型执行，覆盖 Attention/MLP 两个真实形状、零值/随机/离群输入、56 条有向数据包 SHA256 互证、pack/unpack 位串检查、消费者融合与普通解码后两次 BF16 add 的逐位一致，以及 rank 延迟和七次复用后的六个输出存储保持。

### 目标机回传检查（2026-10-30）

证据来源为用户在目标机执行临时容器命令后粘贴的终端输出，不是本地执行记录或完整训练报告。

| 检查项 | 回传结果 |
| --- | --- |
| 镜像 | `nvcr.io/nvidia/pytorch:25.06-py3`，NVIDIA build `177567386` |
| Python | `3.12.3`，`/usr/bin/python` |
| PyTorch | `2.8.0a0+5228986c39.nv25.06` |
| CUDA | PyTorch linked CUDA `12.9`；nvcc `V12.9.86` |
| cuDNN | `91002`（9.10.2） |
| GPU 架构 | PyTorch 编译架构包含 `sm_120` 和 `compute_120`；8 张 RTX 5090 均为 `(12, 0)` / 170 SM |
| 跨卡 peer access | **0/56 可用**；矩阵仅对角线为 1，对角线不计作跨卡链路 |
| Apex 导入 | `apex_C`、`amp_C`、`fused_layer_norm_cuda` 均成功；尚未验证实际 GPU kernel 执行 |

NVIDIA 在 [RTX 5090 P2P 问题的公开答复](https://forums.developer.nvidia.com/t/p2p-issue-using-two-rtx-5090-gpus/326776/8)中说明 GeForce RTX 50 系列不支持 GPU 间 P2P；本机回传结果与之相符。这份证据解释了旧版直接访存方案的阻塞，是本次新增 staging 路径的原因，不是新版 NCCL 探针或训练已通过的证据。

普通 NCCL 通信可在不能使用 P2P 时使用主机共享内存等传输方式；`NCCL_P2P_DISABLE=1` 仅禁用 NCCL 的 P2P transport，不会改写应用中自定义 CUDA kernel 的远端指针访问。[NCCL 环境变量说明](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2276/user-guide/docs/env.html#nccl-p2p-disable)

新的统一入口已把八卡 NCCL 集合通信作为正式前置检查。`Flux＋基础量化` 保留 GemmOnly＋独立编码＋NCCL All-to-All＋接收端融合 v2；其余八组使用本地 fused producer 和 staging 传输。修复 `_NativeRuntime.rank` 后，用户回传批次 `baseline-nine-arm-nccl-staged-v2-1x-50steps-20261030-163211-193828` 已依次完成 MA11、MA89、MA8、MA9、CR3、R、C、A，打印 `TRANSPORT SMOKE PASSED`；证据位置为该批次的 `local-smoke-summary.json`。这是八组 TACO 算子的传输检查通过，尚不代表九组完整模型或 Flux＋基础量化模型 smoke 已通过。

后续批次 `baseline-nine-arm-nccl-staged-v2-1x-50steps-20261030-163809-528885` 的完整模型 MA11 worker 在第 38 行 `assert all(torch.cuda.can_device_access_peer(...))` 失败。这是 H100 模板残留的 P2P 必需检查，不是 GPU 忙碌或显存耗尽错误；独立原生传输 smoke 不执行这段模型启动代码。5090 生成器现为九组统一移除该断言，仍通过 `runtime.check_topology` 检查八张独立 RTX 5090、SM120/170 SM 及冻结通信配置，并记录 P2P 矩阵。源码预检会拒绝重新引入 worker 直接 P2P 查询。此次仅修改生成 worker 的代码，可继续复用已编译构建；更新 `tools/5090/phase6/baseline_sources.py` 后新启动批次即可，历史批次不改写。

2026-10-31 拉回的 `20261031-195751-850613` 和 `20261031-200655-868659` 两次批次都通过了 NCCL 前置检查和八组 TACO 原生传输 smoke，随后在 `smoke-ma11` 失败。两次的八个 rank 都报告 `MA11 native arithmetic differs from installed ATen, case 0`，日志中没有 CUDA OOM。该自检比较 MA11 native masked softmax 与当前 PyTorch ATen 的前向和反向结果是否逐位一致，旧版错误信息没有区分是哪一项不同。现有预检记录 GPU 总容量，但不记录当时的可用显存；此次失败日志不支持把数值自检失败归因于显存占用，也不能证明后续完整训练显存一定足够。两个批次均未进入正式计时。

核查上游实现发现，原 H100 MA11 使用 PyTorch 2.6 的 2048 列 block 归约；[目标版本对应的 PyTorch 代码](https://github.com/pytorch/pytorch/blob/5228986c39/aten/src/ATen/native/cuda/SoftMax.cu)将 BF16 前向的 2048 列送入 persistent warp softmax，并为对齐的 2048 列反向使用 shared-memory/warp 归约。归约顺序变化会改变浮点舍入，是现有逐位失败的重要兼容性原因；实际镜像的结果仍以目标 GPU 自检为准。

对 `case 0` 的第 318 行做 CPU 浮点顺序示例：旧 block 顺序与新 warp 顺序得到的分母分别为 `16.51946258544922` 和 `16.519460678100586`，进而有 2 个 BF16 输出不同。此示例使用 CPU `expf`，用于说明舍入次序可改变结果，并非目标 GPU 的数值记录。

新增 `tools/5090/phase6/ma11_softmax_5090.cu`，前向使用 32 线程 warp、每线程 64 个元素与 XOR 归约，反向使用 128 线程的两级 warp 归约，并保留 BF16 中间乘积舍入与 mask。它是 5090 的独立 CUDA 库，不修改 H100 实现或原有 Flux/TACO 构建。完整测试会编译该小库，先在 GPU 0 上独立运行 6 组前向及反向逐位自检；全部通过后继续原生传输和完整模型检查。自检失败会分别报告 `forward_bitwise`、`backward_bitwise`，没有放宽比较门禁。MA89 profile 同步检查目标版本的 warp softmax 路径。

后续 `20261031-210455-861085/ma11-compat/self-test/probe-result.json` 已记录目标 RTX 5090 上 6 组前向及反向逐位检查全部通过，RNG 未改变；这解决了先前独立 softmax 自检的阻塞，完整模型和正式性能验证仍未完成。

### 历史 OOM 定位与修复（2026-10-31）

失败证据均在 `Load-5090/logs/5090/phase6/` 的对应批次中，原始文件保持不变：

| 批次 | 失败证据 | 结论 |
| --- | --- | --- |
| `20261031-210050-693717` | `results/smoke-ma11/process.log`：例如 rank1 申请 16 MiB，整卡仅剩 14.75 MiB，而该进程占用约 17.56 GiB | 存在未被该进程占用解释的显存用量；旧预检没有 free-memory 快照，无法仅凭日志归因到具体进程。 |
| `20261031-210455-861085` | 同文件：rank7 等在 `ma11_adapter.py` 的 `value.masked_fill(mask, -10000).softmax(-1)` 申请 32 MiB 失败；进程占用 31.32 GiB、Torch allocated 26.34 GiB、可用 30.19 MiB | 已是本进程容量不足。该 32 MiB 是触发分配；删除自检不代表完整 optimizer step 能装下。 |

修复覆盖八组 TACO 前端，保留 A 逐调用配置、R/C 实例配置、CR3 家族提交缓存、MA8/MA9/MA11 机制与选择表。Flux＋基础量化继续原有独立 codec。构建迁移仅更新六份 Python 前端和哈希清单，沿用已编译 CUDA 二进制；完整 smoke、跨组 loss/gradient/parameter 检查、profile 和正式计时均重新执行。工作区策略及每 slot 的实际共享地址/字节数写入训练记录；报告拒绝同组模块未共享或不同 role/slot/stream 错误别名。

原生 smoke 新增两个实例的连续交替提交检查：核对五项 scratch 共享、双 slot 分离、接收 packet/mask/native handle 私有、先前输出不被覆盖，以及关闭一个实例后另一实例仍能正确执行。逐卡预检新增 CUDA free/total/used 和本进程 Torch allocated/reserved；后续预检失败仍保留已采集快照。`nvidia-smi` 的计算进程列表为空不能单独证明显存空闲，快照也只是当时观测。

该次共享 scratch 修复在无 GPU 的开发环境执行 `python3 -m unittest discover -s tools/5090/phase6 -p 'test_*.py' -v`，**93 项测试全部通过**，另通过入口预览与 `git diff --check`。测试覆盖冻结训练/计时 AST、九组报告、SM120 softmax 编译、共享 scratch 生命周期以及旧构建迁移/回滚。当时尚未实机执行；后续 `20261031-215027-387404` 已通过共享 scratch 原生 smoke 和九组完整模型检查，结果如下。

### 历史 profile 检查器兼容性故障（2026-10-31）

证据目录为 `Load-5090/logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-1x-50steps-20261031-215027-387404/`。`preflight.json` 的 `completed=true` 且九组均为 `true`；九组 smoke 共 72 个 rank 记录和 MA11 profile 的 8 个 rank 记录全部 `passed=true`、`completed=true`。离线重新执行逐 rank 校验、初始硬件/软件/权重/RNG/输入身份检查及同量化策略组的 loss、梯度、参数轨迹比较，也全部通过。没有正式 timing 窗口。

`handoff/supervisor.log` 在 `STAGE profile-ma11` 后报 `ValueError: Decoder/consumer does not immediately feed LayerNorm`。实际 trace 中，每 rank 的 256 次投影后有 252 次接 Apex `cuApplyLayerNorm`；另外 4 次是每个 microbatch 最后一层 MLP，接 Transformer Engine `transformer_engine::normalization::ln_fwd_tuned_kernel`，实际 CPU owner 为 `_OperationFuserAutogradFunction`。旧检查器只认识 Apex 的 kernel 名和 `FusedLayerNormAffineFunction` owner，所以在 GPU worker 正常完成后拒绝了最终 LayerNorm。这是检查器兼容问题，不是 decoder 少算 LayerNorm，也不是显存分配失败。

继续离线重放还发现两项旧 H100 检查条件不适用于当前目标环境：

- **同步 NCCL 的 stream：** 目标 PyTorch `5228986c39` 将 `asyncOp=false` 的集合通信提交到当前计算 stream，异步调用仍使用独立 NCCL stream；这与本批 trace 一致。[对应版本 ProcessGroupNCCL 源码](https://github.com/pytorch/pytorch/blob/5228986c39/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L3186) 因此仅在 5090 私有分析器中，要求同步 forward/head/row AllGather 和零统计 AllReduce 与计算同 stream，保留异步 backward AllGather、ReduceScatter 和 staged All-to-All 的独立 stream、实际 launch 归属与完成顺序检查。
- **cuBLAS split-K：** 本批 Attention row backward 的 dgrad GEMM 带有 `cublasLt::splitKreduce_kernel` 后处理；旧分析器按 `reduce_kernel` 子串把它误计为第二个 bias 归约，并假定每次反向只有四个 kernel。5090 分析器单独核对该归约属于对应 `aten::mm`，验证 GEMM→归约→后续计算的依赖并记录数量，不忽略额外 GPU 工作。

这批实测显存如下，均为八个 rank 的最大值，单位 GiB；来自各 `results/smoke-<arm>/rank*.json`，不是正式计时窗口：

| 组 | Torch allocated 峰值 | Torch reserved 峰值 | TACO native workspace | allocated＋native |
| --- | ---: | ---: | ---: | ---: |
| MA11 | 18.751 | 18.904 | 2.656 | 21.408 |
| MA89 | 18.711 | 18.904 | 2.656 | 21.367 |
| MA8 | 18.712 | 18.896 | 2.656 | 21.368 |
| MA9 | 17.715 | 17.900 | 2.656 | 20.372 |
| CR3 | 17.715 | 17.900 | 2.656 | 20.371 |
| R | 17.018 | 17.213 | 1.328 | 18.346 |
| C | 17.716 | 17.893 | 2.656 | 20.372 |
| A | 17.018 | 17.213 | 1.328 | 18.346 |
| Flux＋基础量化 | 17.481 | 17.686 | 0 | 17.481 |

MA11 profile 的 allocated＋native 最大为 **21.384 GiB**。最后一列逐 rank 相加后取最大值，不把 allocated 和 reserved 重复相加，也不包含 CUDA context、NCCL 等内部占用。它与全部 worker 正常完成一起说明这批已越过先前 OOM 阶段；该历史批次没有完成正式计时，后续 2026-11-01 的完成批次见第 1 节。

修复后的验证：在无 GPU 的开发环境执行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools/5090/phase6 -p 'test_*.py' -v`，**116 项测试全部通过**；入口预览、shell 语法和 `git diff --check` 均通过。另对本批 `results/profile-ma11/rank0-profile.json.gz` 至 `rank7-profile.json.gz` 逐一执行完整 `analyze()`，八个 rank 的投影、激活缓存、反向、零统计、softmax 五项审计全部通过。每 rank 均确认 252 次 Apex＋4 次 TE LayerNorm、128 次 Attention dgrad split-K 归约。这里是对历史批次原始 GPU trace 的离线复核，未修改原 trace 或失败状态；后续 2026-11-01 新批次已完成全部九组 profile 和正式计时。

这次仅修复 5090 的 profile 分析及回归测试，保留训练代码、模型尺寸、数值门禁和全部九组机制。更新整个 `tools/5090/phase6/` 后，可继续使用原 `--baseline-build` 路径，不需重编译 Flux/TACO；入口仍会编译独立 MA11 兼容小库并重复完整门禁。使用原命令启动新批次，不向失败批次追加结果；`--report` 只适用于完整批次，不能续跑这次未完成的实验。

## 4. 一键执行

在 `TempName-New` 仓库根目录、已准备好的 5090 Python 环境内执行：

新启动的 5090 批次默认写入同级数据仓库 **`../Load-5090/logs/5090/phase6/<批次名>/`**；目标容器中的完整路径为 `/workspace/Load-5090/logs/5090/phase6/<批次名>/`。硬件预检、传输 smoke 和完整实验共用这个默认位置，后台日志、原始 rank 记录、profile、汇总报告及复核快照都保存在批次目录中。已有 `TempName-New/logs/5090/` 的结果和构建不搬迁，`--baseline-build` 仍可指向原位置；显式 `--out` 可以覆盖新默认值，历史批次仍按原路径执行 `--report`。

`Load-5090/.gitignore` 保留实验 `.log`、JSON/CSV/Markdown、压缩 profile 和源码快照，排除本机 `build/`、CUDA 编译产物、Python 缓存及虚拟环境。这只改变后续 5090 结果的默认存放位置，不调整 H100 的输出路径，也不自动暂存或推送数据。

官方 25.06 镜像沿用自身的 CUDA 12.9、Python 和 PyTorch；为运行新协议，需要更新完整仓库中的脚本和原生源码，不能只复制入口 shell。

已有 `hjr/TempName-New` 和 `hjr/Megatron-LM` 时，在容器内的 `TempName-New` 根目录先运行环境配置脚本。它复用现有源码，创建继承镜像包的轻量 venv，补齐缺失依赖并执行八卡预检，不重装 PyTorch/CUDA/Apex，也不启动训练：

```bash
bash env/setup_5090.sh
source env/activate_5090.sh
```

测试入口会自动读取生成的 Python、CUDA 和 Megatron 路径；参数、日志、软件预配置模式见 [5090 环境说明](../../../env/README_5090.md)。已配置环境无需再手动重设 `RTX5090_PYTHON` 为镜像系统解释器。

先在目标机独立验证修复后的 MA11，不创建完整模型、也不重编译 Flux/TACO；结果保存到 `Load-5090/logs/5090/phase6/ma11-softmax-probe-<时间>/`。脚本默认只预览，传入 `--execute` 才执行 GPU 自检：

```bash
bash tools/5090/phase6/run_ma11_softmax_probe.sh --execute
```

成功时显示 `MA11 SOFTMAX PROBE PASSED`；失败信息及 `probe-result.json` 会说明前向和反向的逐位比较结果。单卡探针通过后，下面的完整测试仍会在预检和全模型 smoke 中重复检查，不以单次探针替代正式正确性验证。

```bash
# 仅打印执行计划；不查询 GPU、不构建、不启动训练。
bash tools/5090/phase6/run_baseline.sh

# 只做硬件、Apex 实际运算和八 rank NCCL 检查；不编译 Flux 或训练模型。
bash tools/5090/phase6/run_baseline.sh --execute --preflight-only

# 构建并检查新的原生传输；通过后停止，不运行完整模型和正式计时。
bash tools/5090/phase6/run_transport_smoke.sh --execute

# 九组完整批次：预检查、构建、原生/模型 smoke、profile、正式计时与报告。
bash tools/5090/phase6/run_baseline.sh --execute
```

`--preflight-only` 无需 Megatron 源码；`--transport-smoke-only` 使用统一构建器，首次构建仍需实验指定的 Megatron 源码及依赖，也可以通过 `--baseline-build` 复用已完成的新协议构建。每次使用新的输出目录。仅硬件/NCCL 预检成功并不会生成原生构建。

若旧版脚本在 Python 3.12 的 `prepare-workers` 阶段报 `Original training functions changed`，原因是 [Python 3.12 为函数和类 AST 新增了 `type_params` 字段](https://docs.python.org/3.12/library/ast.html#ast.FunctionDef)，普通函数也会带 `type_params=[]`，无法直接匹配 Python 3.10 生成的冻结 AST 哈希。5090 的 `baseline_sources.py` 已在局部解析树中统一这个空字段；原训练函数和计时循环哈希保持不变，非空类型参数及实际代码修改仍会被拒绝。无需更换镜像或重装环境。

**已有 20261030-151053 构建的一次性重试：** 同步更新整个 `tools/5090/phase6/` 目录后，在容器内执行下面一条命令。脚本会自动识别已知的旧 Python 前端，备份并修复六份前端和相关哈希清单，完整校验旧/新构建，复用 CUDA 二进制，再运行八组原生传输 smoke；无需单独运行修复脚本或重新编译。

```bash
bash tools/5090/phase6/run_transport_smoke.sh --execute \
  --baseline-build /workspace/TempName-New/logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-1x-50steps-20261030-151053-241177/build
```

若希望一条命令继续完成九组完整实验，使用下面的入口；它包含同样的自动修复和传输 smoke，所有正确性和 profile 检查通过后才计时：

```bash
bash tools/5090/phase6/run_baseline.sh --execute --foreground \
  --baseline-build /workspace/TempName-New/logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-1x-50steps-20261030-151053-241177/build
```

正式执行先检查九组 worker 源码，再进行硬件检查和构建，避免源码错误在长时间编译后才被发现。`_NativeRuntime` 在设备检查前绑定传入的 rank，保持 rank/device 一致性门禁。

自动迁移只接受两个冻结旧前端：缺少 rank 初始化的 `61bb8c847e41774648629cc1cffcb35729601bd6ad64e6fa9f73a8d6c0ebab22`，以及已修 rank、仍为每实例完整分配的 `ba668cc182904040bce8592d5d98840e0a286dde9dbcec4742874ddbd65eca8c`。两者均升级到共享 scratch 前端；其他原生源码、库、环境或构建身份变化仍拒绝复用。同步目录时须包含 `legacy_nccl_transport_v2.txt`，供旧源码精确校验。旧文件备份位于构建目录的 `repairs/`，新批次保存 `build-frontend-repair.json`，校验失败会恢复原文件。迁移持有与测试相同的仓库锁；已升级构建再次使用时只校验。原失败结果不改写，也不与新工作区策略的正式结果混算。

`run_transport_smoke.sh` 固定使用前台传输检查；不加 `--execute` 时仅预览，不修复构建、不使用 GPU。传输 smoke 子进程失败时，入口会直接打印 rank 记录中的原始错误，并保留 `process.log`。

与 H100 的 `run_ma19.sh` 类似，默认执行模式交给后台控制进程；使用 `--foreground` 可在当前终端运行。后台任务持续运行要求容器保持运行；如果容器的主进程是交互式 bash，退出该 bash 会停止容器及其子进程。可在持续运行的容器中通过 `docker exec` 启动测试，并查看批次状态与阶段日志。

显式指定 Python、CUDA 和 Megatron 路径：

```bash
RTX5090_PYTHON=/path/to/5090-env/bin/python \
bash tools/5090/phase6/run_baseline.sh \
  --execute \
  --cuda-home /usr/local/cuda \
  --build-cuda-home /usr/local/cuda \
  --megatron /path/to/Megatron-LM \
  --out ../Load-5090/logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-01
```

`--python` 也可指定 worker 解释器；`RTX5090_PYTHON` 决定入口解释器，建议二者使用同一个环境。默认 `--trials 1 --blocks 18`；可使用 `--trials 2` 独立重复，或将 `--blocks` 设置为 18 的正整数倍。`--jobs` 控制编译并发，`--worker-timeout` 控制单个八 rank worker 窗口的超时。

只有新 native 协议 `rtx5090-nine-arm-native-nccl-staged-sm120-v2` 已完成且验证通过的构建才能复用；原 direct-peer v1、H100 和只准备源码的构建都会被拒绝：

```bash
bash tools/5090/phase6/run_baseline.sh \
  --execute --baseline-build /path/to/previous-batch/build \
  --out ../Load-5090/logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-02
```

离线复核原始结果、重新分析 profile 并打印报告（不改写原结果文件）：

```bash
bash tools/5090/phase6/run_baseline.sh \
  --report --out ../Load-5090/logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-01
```

## 5. 统一计时、正确性与结果文件

模型保持与 H100 对齐：32 层、hidden 4096、FFN 16384、32 heads、sequence 2048、micro batch 1、global batch 4、BF16、TP8/DP1；固定模型和 token seed，每个 optimizer step 为 **8192 token**。计时包含前向、反向、通信、optimizer 更新与清梯度，不启用 CUDA Graph。

每组每窗口 **10 步预热＋50 步正式计时**。一个完整块包含九组；默认采用 9 次位置轮转加对应逆序的 **18 块平衡设计**，预先确定执行顺序。每组在每个执行位置出现两次，任意两组的前后关系平衡。每个 trial 共：

| 工作量 | 数量 |
| --- | ---: |
| 每组正式窗口 | 18 |
| 每组正式 optimizer step | 900 |
| 九组正式窗口 | 162 |
| 正式 rank 记录 | 1296 |
| 正式 optimizer step | 8100 |

以上数量不含原生传输 smoke、模型 smoke、预热和 profile；模型 smoke 每组 2 步预热＋3 步检查，profile 独立运行并在正式计时前通过，不混入正式步时。

- **窗口耗时：** `max(8 个 rank 的 elapsed_seconds) × 1000 / 50`。
- **主表耗时：** 每组全部 18 个窗口的算术均值；另报中位数、最慢窗口及显存峰值，保留所有有效慢窗口。
- **吞吐：** `8192000 / 平均 ms`。
- **MA11 主表降幅：** `100 × (1 − MA11 平均 ms / 对照组平均 ms)`；正数表示 MA11 更快，负数照常保留。
- **同期配对收益：** 按完整块配对，使用对数耗时比及 10,000 次 bootstrap 给出 95% CI；与主表算术均值比单独报告。区间描述该固定工作负载，36 个两两比较未作多重比较校正。

同一 trial 中各组保持相同初始参数、输入和运行环境。MA11/MA89/MA8/MA9/CR3/C 在组内检查 loss、梯度和参数更新一致性，R/A 在另一组内检查；不同量化策略不强行要求逐位相等。基础量化 v2 单独检查 codec 和输出所有权。各组还检查实际模块覆盖、无 fallback、完整梯度/参数轨迹及相应 profile 路径，防止只改组名而优化未启用。

每个批次保留 `environment-preflight.json`（硬件/Apex/NCCL 检查汇总）、`preflight-communication.json`、`nccl-preflight/rank*.json`（八 rank 集合通信原始记录）、`local-smoke/<arm>/rank*.json` 与 `local-smoke-summary.json`（八组 TACO 原生传输检查）、`preflight.json`（九组模型 smoke）、构建清单、worker 配置、逐 rank 训练记录、profile 证据及阶段日志；最终生成 **`report.json` 和 `report.md`**。后台状态看 `batch-state.json`，控制进程日志看 `handoff/supervisor.log`。`--transport-smoke-only` 成功状态为 `transport_smoke_completed`，不生成训练性能报告。离线复核检查 NCCL/原生传输探针证据、逐 rank 完成状态、软件/硬件身份及固定通信环境；失败或缺失的窗口不能计入已完成报告，收益为负不会提前停止。第 1 节已填写本次完成批次，并记录环境和 P2P 拓扑；后续批次应单独复核和注明来源。

相关实现：[入口](../../../tools/5090/phase6/run_baseline.sh)、[九组协议与统计](../../../tools/5090/phase6/baseline_protocol.py)、[八 rank NCCL 探针](../../../tools/5090/phase6/baseline_collectives.py)、[隔离构建](../../../tools/5090/phase6/baseline_build.py)、[worker 生成](../../../tools/5090/phase6/baseline_sources.py)。
