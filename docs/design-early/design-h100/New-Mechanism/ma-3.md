# MA3：来源并行解码与有序归约

更新：2026-10-23，已纳入八卡回传结果。针对[原始思想](../idea.md)、[baseline](../base-baseline.md)和已有消融，本轮尝试**将独立来源的反变换并行执行，将必须保持顺序的 BF16 归约留到最后**。交替排序继续负责生产顺序，新机制作用于接收端的数据依赖拆分。它不同于 [MA1](ma-1.md)、[MA2](ma-2.md) 的下一来源预取，也不重复 MA2-R 的实例配置与输出 clone 优化。

**批次 `20261022-222545-811304` 已完成两次比较、72 个窗口和 3600 个正式 optimizer step，正确性与执行路径全部通过，但性能目标未通过。** S/A 完整训练收益分别为 **−0.087%［95% CI：−0.374%, +0.199%］**、**−0.078%［−0.413%, +0.305%］**；S/D 分别为 −0.128%、−0.257%，区间也均跨零。独立算子中 S 比 D 慢约 0.76%–0.80%，两种 shape 的区间均完全为负。**本批没有证实来源并行机制带来完整训练收益。** 详细结果和原始证据见[第 8 节](#8-八卡实测结果与离线复核2026-10-23)。

复测入口保持：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_ma3.py --execute
```

在服务器 `/workspace/TempName` 下执行；新脚本及其依赖需要同步到该目录。启动器默认重新编译并后台执行两次独立完整训练比较，保留负结果，不以单算子加速替代完整 step 结论。

## 1. 从已有结果选择另一个方向

到达优先尝试提前生产拖尾贡献，但当前完整发布屏障仍然存在，静态到达标签也不一定能预测正式运行中的关键来源。[baseline 的三次 B/A 对照](../base-baseline.md#4-三次独立配对比较)为 +0.516%、+0.236%、−0.391%，第三次区间跨零；它支持“额外收益有限且不稳定”，不支持“完全没有作用”。

这也不是把所有失败都归因于发布屏障：历史 op3 已尝试提前消费，完整训练仍退化；op4 合并逆变换还触发了梯度轨迹门禁。MA3 因此保留发布协议、每来源独立反变换和原 BF16 累加顺序，仅考察来源计算的并行分工。

MA2 打包预取 V 的 `155228` 复测中，V/A 完整 step 收益为 −0.138% 与 +0.079%，区间均跨零。与此同时，训练 profile 中 A 的 Attention/MLP decoder 中位时间约为 50.648/52.946 μs，V 为 15.040/15.242 μs。局部 decoder 变快没有形成可靠完整 step 收益，因此本轮需要同时验证局部机制和端到端效果，不能根据静态指令数推算已获得收益。

MA3 选择解码依赖拆分，原因是当前一个 warp 连续处理七个远端来源：

```text
读来源1 → 解码1 → BF16折叠1 → 读来源2 → 解码2 → BF16折叠2 → …
```

来源 2 的逆 Hadamard 变换不需要来源 1 的结果。真正需要原顺序的是逐来源 BF16 累加。预取尝试提前发射下一来源的读操作；MA3 则把多个来源的独立计算分配给不同 warp。

这里的“串行依赖”包括执行顺序所造成的串行工作，并不声称每条来源间指令都存在 RAW 数据依赖。已有跨 group 并行本来就很充足；来源并行是否改善 GPU 吞吐，必须实测。

## 2. 机制：并行解码，按原顺序折叠

对一个 H128 group，令 `decode_s` 是来源 s 的反变换，`BF16()` 表示该步骤的舍入。保留原计算图：

```text
v_s = BF16(decode_s(packet_s))        # 七个远端来源彼此独立
acc = BF16(0)
依次处理 s = rank+1, …, rank+7：
    acc = BF16(FP32(acc) + FP32(v_s))
acc = BF16(FP32(acc) + FP32(local))   # 本地 BF16 最后加入
```

rank 运算取模 8。没有将来源相加移入 Hadamard 变换，没有改成 FP32 来源求和，也没有树形归约。这样避免用改变数值语义换取性能。

S 使用 128-thread CTA，每 CTA 负责一个 H128 group，四个 warp 的任务为：

| warp | 第一个来源偏移 | 第二个来源偏移 |
| --- | ---: | ---: |
| 0 | 1 | 5 |
| 1 | 2 | 6 |
| 2 | 3 | 7 |
| 3 | 4 | 8，即本地 BF16 |

每个 warp 最多执行两次远端反变换。每 lane 持有四个连续系数；来源完成解码和 BF16 舍入后，将贡献写入 CTA shared memory。所有线程无条件参加一次 `__syncthreads()`，然后 128 个线程分别负责一个系数，按照偏移 1→8 做 BF16 折叠并写回。

shared 使用 `uint32_t contributions[8][4][32]`，低 16 位存 BF16 位模式，占 4 KiB。写入时 `source/j` 固定、lane 连续；读取时 warp 固定为 j，系数索引为 `4*lane+j`。两阶段每条 warp 访问都落到连续 32-bit bank，避免将 BF16 紧密转置引入 bank 冲突。这里的 CTA 同步仅用于本 CTA 的 shared 交换，不能替代跨 GPU 的发布协议。同步语义和 bank 映射分别见 [CUDA 12.8 Programming Guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#synchronization-functions) 与 [NVIDIA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks)。

所有 packet 读取仍发生在原发布屏障后；原工作区复用屏障也保留。本轮没有 readiness flag、跨 CTA 自旋、额外 stream、双缓冲或新的跨 GPU 同步协议。

## 3. A/D/S 同场对照

| 组 | 实现 | 作用 |
| --- | --- | --- |
| A | 原始交替排序＋scalar decoder | 对齐 baseline A 和 MA1/MA2 的共同参照 |
| D | 连续四系数打包解码，逐来源执行，关闭显式预取 | 隔离 packed/warp 布局的影响 |
| S | 与 D 相同的 packet 解码数学，四 warp 分担来源，shared 交换后有序归约 | MA3 候选 |

**这里“单纯交替排序”指原 baseline A：融合 V2、基础交替、全远端 TACO 量化。** 不指全 BF16、不指原生 Hopper V3，也不指带到达表和稀疏 mask 的 baseline C。D 是本轮新控制组，不冒称旧 MA2 V；D/S 都不启用下一来源的显式预取。

三组固定：交替 mapping、GEMM/encoder、全远端量化、H128/E4M3 packet 布局、通信量、两道屏障、单工作区、逐次配置/复位、输出 clone、模型与反向路径。构建器冻结 A 的源文件哈希；D/S 只替换接收 runtime 并加入 decoder 头文件。旧 MA1、MA2、MA2-R 文件和原始结果不修改。

关键比较为 S/A 和 S/D。若 S/A 为正而 S/D 不成立，只能说明组合实现相对旧 A 有优势，不能确认来源并行提供了额外收益。D/A 是诊断对照，不作为来源并行成功的证据。

## 4. 性能假设与成本

本轮假设是：将一个 group 的七次解码分摊到四个 warp，有机会降低单个 group 的串行解码时间，同时让归约只携带 BF16 贡献。**“每 warp 最多两次”不意味着算子获得 3.5 倍加速。** 总反变换工作没有减少。

| 项目 | D | S |
| --- | ---: | ---: |
| CTA 数量 | 2048 | 8192 |
| 每 CTA 线程 | 128 | 128 |
| 每 CTA 输出 group | 4 | 1 |
| 每 warp 远端解码次数 | 7 | 最多 2 |
| shared 暂存 | 0 | 4 KiB/CTA |
| CTA 内同步 | 0 | 1 次 |

S 增加 CTA 调度、shared 读写和同步成本，最终写回布局也不同。D 已有 8192 个独立 group warp，足以提供大量并行；如果硬件吞吐或内存带宽已饱和，S 可能更慢。资源减少、更多 warp 或更高 occupancy 都不是性能保证，需结合实际 profile 和完整 step 判断。[NVIDIA 执行配置建议](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#thread-and-block-heuristics)也要求在延迟隐藏与资源消耗之间实测权衡。

试验前仅作量级估计：将 MA2 历史 profile 的 decoder 中位数乘以每类 128 次调用，A 约为 13.26 ms、V 约为 3.88 ms，相对约 535 ms 步时分别约 2.48% 和 0.72%。这些来自 profile 的条件预算，不是无采样完整 step 的实际关键路径占比，也不是严格收益上界；D/S 的本轮实测见第 8 节。不能把 MA1、MA2 的历史局部收益与 MA3 推测收益相加。

## 5. 实验与验收

沿用 GPT6.7B：32 层、H4096、FFN16384、32 heads、S2048、TP8、BF16，micro batch1/global batch4，每 step 8192 token。Attention/MLP 本地 K 为 512/2048。固定模型 seed1234、token seed4321，`NCCL_NVLS_ENABLE=0`、`CUDA_DEVICE_MAX_CONNECTIONS=1`。

先检查两种 shape 的实际 GPU mapping、算子逐位输出和工作区复用，再执行独立算子计时/profile；完整训练进行两次比较，每次 12 个顺序平衡的完整 A/D/S 块，每个窗口重新初始化，10 步预热＋50 步正式计时：

| 工作量 | 每次比较 | 两次合计 |
| --- | ---: | ---: |
| 完整 A/D/S 块 | 12 | 24 |
| 正式训练窗口 | 36 | 72 |
| 正式 rank 记录 | 288 | 576 |
| 正式 optimizer step | 1800 | 3600 |

3600 是独立初始化窗口的累计步数，不是一次连续训练。两次比较使用同配置、同种子、不同 run ID 和执行时段。算子耗时为负不跳过完整训练；正确性或执行路径不符则停止。

每个窗口取八 rank 最大墙钟时间除以 50，以完整 A/D/S 块为独立样本做配对 bootstrap，10000 次、seed20261022。报告两次比较的全部样本和 95% CI；不删除慢块、不选择最好半程、不合并旧批次。

两次比较**各自**必须满足以下条件，才认定 MA3 达到“交替＋新机制更快”：

1. S/A 点估计 >0，且 95% CI 下界 >0。
2. S/D 点估计 >0，且 95% CI 下界 >0，确认来源并行有净增量。
3. 算子逐位正确性、完整训练逐 rank loss/gradient 轨迹一致性，以及实际执行路径检查通过。

另外报告 S/A 是否满足“点估计 ≥1% 且 CI 下界 >0”、以及 CI 下界是否达到 1%；这些是附加结果，不把“更快”擅自改为必须达到 1%。一次成功、另一次失败时，批次仍未通过。固定合成输入的短程数值一致不代表真实语料长期收敛验证。

## 6. 启动、状态与报告

执行页首命令后，默认输出目录为：

```text
logs/h100/phase6/ma3-source-parallel-2x-50steps-<北京时间戳>/
  batch-state.json
  handoff/container-supervisor.log
  build/                            A/D/S 构建与源码清单
  operator/                         算子正确性、计时和 profile
  fullstep/                         两次完整训练比较与原始记录
```

默认 CUDA12.8 编译、CUDA12.4 运行、jobs2，Megatron 源码使用相邻的 `../Megatron-LM`。可用 `--python`、`--megatron`、`--cuda-home`、`--build-cuda-home`、`--nccl-root` 指定环境；`--foreground` 用于前台排查，`--ma3-build` 复用已完成且身份验证通过的 MA3 构建。容器与主机需持续运行。

去掉 `--execute` 只打印计划，不创建目录、编译、fork 或查询 GPU。再次复测直接重跑命令，让默认时间戳创建新批次；不要复用已有结果目录。八卡计时应在其他实验结束后执行。

离线复算：

```bash
python3 tools/h100/phase6/ma3_bench.py report /path/to/batch/operator --offline
python3 tools/h100/phase6/ma3_bench.py profile-report /path/to/batch/operator --offline
python3 tools/h100/phase6/ma3_fullstep.py report /path/to/batch/fullstep --offline
python3 tools/h100/phase6/ma3_fullstep.py profile-report /path/to/batch/fullstep --offline
```

启动成功、`status=completed`、`target_validated=true` 分别表示已启动、实验完成、性能与执行证据验收通过，不能互换。回传时保留完整批次的 plan、冻结脚本、manifest、rank JSON、mapping、trace 和状态文件。

## 7. 实现与本地验证

实现入口：

- [ma3_decode.cuh](../../../../tools/h100/phase6/ma3_decode.cuh)：D/S 共用解码数学与来源并行 kernel。
- [ma3_sources.py](../../../../tools/h100/phase6/ma3_sources.py)、[ma3_build.py](../../../../tools/h100/phase6/ma3_build.py)：精确替换、隔离构建、源码身份和 SM90 编译检查。
- [ma3_bench.py](../../../../tools/h100/phase6/ma3_bench.py)：算子数值、复用、计时与执行路径核验。
- [ma3_fullstep.py](../../../../tools/h100/phase6/ma3_fullstep.py)：完整训练比较与严格报告。
- [detach_ma3.py](../../../../tools/h100/phase6/detach_ma3.py)：一行后台入口。

以下保留 2026-10-22 上机前的本地验证记录。当时环境未暴露 GPU，但提供 CUDA12.8 编译工具：

| 检查 | 结果与证据 |
| --- | --- |
| 数值、源码变换、构建身份、完整报告与启动器 | **73 项 MA3 CPU 测试及 9 项沿用的 MA2 数学测试通过，共 82 项**；[数值/构建日志](../../../../outputs/h100/phase6/ma3-validation-20261022/cpu-tests.log)、[实验控制器日志](../../../../outputs/h100/phase6/ma3-validation-20261022/harness-tests.log) |
| A 与旧 MA2 A 的一致性 | **351 个登记源文件哈希全部相同**；[对照记录](../../../../outputs/h100/phase6/ma3-validation-20261022/baseline-comparison.json) |
| 一行启动预览 | 生成 A/D/S、两次比较、72 窗口和 3600 正式步计划；未创建训练目录或启动 GPU；[预览记录](../../../../outputs/h100/phase6/ma3-validation-20261022/harness-preview.log) |
| A/D/S 实际 runtime 编译 | **三组均通过 CUDA12.8 SM90 对象编译及两种 K 的 host mapping**，保存 PTX/SASS；[编译登记](../../../../outputs/h100/phase6/ma3-validation-20261022/compile-check/validation.json) |
| 编译资源与指令 | D：30 registers、0 shared/stack/spill；S：31 registers、4096 B shared、0 stack/spill，SASS 包含一次 `BAR.SYNC`；[资源登记](../../../../outputs/h100/phase6/ma3-validation-20261022/cpu-validation.json) |

CPU 数值模型覆盖所有 rank、不同来源完成顺序、shared 生产/消费覆盖和 BF16 非结合性反例；报告测试包含落盘的合成原始记录复算及篡改拒绝。合成测试记录不是 H100 测量结果。

SASS 还确认 D 使用 64-bit 打包写回，S 使用每线程 16-bit 写回；S 的静态 shared load/store 数量分别为 8/8。它们是编译后的指令观察，不是运行时吞吐测量。D/S 的对象都包含共享头文件定义的两个入口，runtime 各自只发射所属 kernel；上机 profile 必须核对实际发射入口。

上述本地检查当时没有完成八卡运行或完整 Flux 动态库链接，不构成 GPU 性能证据；其源码身份见[上机前验证汇总](../../../../outputs/h100/phase6/ma3-validation-20261022/validation.json)。随后服务器批次已完成运行，结果如下。

## 8. 八卡实测结果与离线复核（2026-10-23）

### 8.1. 批次完整性与正确性

回传目录为 `logs/h100/phase6/ma3-source-parallel-2x-50steps-20261022-222545-811304/`。依据[批次状态](../../../../logs/h100/phase6/ma3-source-parallel-2x-50steps-20261022-222545-811304/batch-state.json)和[后台日志](../../../../logs/h100/phase6/ma3-source-parallel-2x-50steps-20261022-222545-811304/handoff/container-supervisor.log)，构建、算子计时/profile、完整训练计时/profile 均执行完成，最终 `status=completed`、`stage=done`。

| 检查项 | 本批结果 |
| --- | --- |
| 完整训练工作量 | compare1/compare2 各 12 个完整 A/D/S 块；共 72 窗口、576 份正式 rank 记录、3600 个正式 optimizer step |
| 算子正确性 | `correctness_passed=true`、`bitwise_equal_to_A=true`；改变输入、跨次复用与保留输出检查通过 |
| 模型正确性 | `correctness_passed=true`、`identical_trajectories=true`；配对的逐 rank loss/gradient 轨迹一致 |
| 实际执行路径 | 算子与完整训练 profile 均通过，确认各臂所属 decoder、两道屏障和输出 clone |
| 相对原交替 A 的收益 | 两次 `baseline_benefit_validated` 均为 false |
| 来源并行净增量 | 两次 `source_parallel_benefit_validated` 均为 false |
| 批次性能验收 | `target_validated=false`、`acceptance_status=not_validated`；两个附加 1% 判定也均为 false |

这里是**完整实验的负结果**，不是启动失败、少跑窗口或数值门禁阻止计时。3600 步仍为独立初始化窗口的累计工作量；固定合成数据上的轨迹一致不替代真实语料长期收敛评价。

### 8.2. 完整训练：未确认优于 A，也未确认来源并行增量

数据来自[完整训练报告](../../../../logs/h100/phase6/ma3-source-parallel-2x-50steps-20261022-222545-811304/fullstep/report.md)及[原始统计 JSON](../../../../logs/h100/phase6/ma3-source-parallel-2x-50steps-20261022-222545-811304/fullstep/report.json)。正数表示候选耗时下降，方括号为 95% CI。

| 比较 | A / D / S 平均步时（ms） | S/A 收益 % | S/D 收益 % | D/A 收益 % |
| --- | --- | --- | --- | --- |
| compare1 | 536.331 / 536.107 / 536.795 | **−0.087 [−0.374, +0.199]** | **−0.128 [−0.367, +0.094]** | +0.042 [−0.178, +0.298] |
| compare2 | 535.247 / 534.296 / 535.667 | **−0.078 [−0.413, +0.305]** | **−0.257 [−0.646, +0.091]** | +0.178 [−0.121, +0.453] |

平均步时是各窗口步时的算术平均，收益则是配对完整块耗时比的对数均值转换为百分比，不直接用表中算术均值相除。两个 S/A、两个 S/D 点估计均为负，但区间均跨零；结论是**没有证实加速**，也不能据此声称完整训练已出现统计上明确的退化。D/A 虽然两次点估计为正，区间同样跨零。

compare1 所有窗口为 531.475–541.780 ms/step，compare2 为 529.717–540.162 ms/step；按每次比较、每个臂分别计算，最大窗口与本臂窗口中位数之比最高约为 1.011。没有出现 MA2 首批那样明显偏离的大慢窗口，但这不意味着没有测量波动。全部 24 个完整块均保留。

compare2 后半程 S/D 为 −0.562% [−1.174%, −0.003%]，D/A 为 +0.397% [+0.122%, +0.668%]；这些仅是半程诊断。主结论仍使用预注册的各次完整 12 块，不能选择一个半程替代完整比较，也不与旧 MA1/MA2 批次合并。

### 8.3. 独立算子：S 快于旧 scalar A，但慢于串行控制 D

以下是完整 `forward().clone()` 加最终同步的独立算子计时，不是单独 decoder，也不是 optimizer step。每个角色有 12 个完整 A/D/S 配对块，来源见[算子报告](../../../../logs/h100/phase6/ma3-source-parallel-2x-50steps-20261022-222545-811304/operator/report.json)。

| 角色 | A / D / S 平均耗时（μs） | S/A 收益 % | S/D 收益 % | D/A 收益 % |
| --- | --- | --- | --- | --- |
| Attention | 192.563 / 146.384 / 147.554 | +23.374 [+22.916, +23.750] | **−0.799 [−1.283, −0.336]** | +23.981 [+23.632, +24.349] |
| MLP | 267.776 / 224.535 / 226.234 | +15.509 [+14.885, +16.240] | **−0.758 [−1.223, −0.193]** | +16.145 [+15.611, +16.843] |

两种 shape 的 S/D 区间均完全为负，说明本批独立算子中来源并行候选确实慢于串行控制。S/A 的较大正收益不能归因于来源并行：D 已经获得略大的收益。本批证据支持 packed/warp 串行实现相对旧 scalar 的局部改善，不支持在此基础上增加来源并行。

### 8.4. 真实训练 profile：新增机制使 decoder 略慢

完整训练另采集 24 份 profile trace，即三组各八 rank。每份 trace 覆盖 Attention/MLP 各 128 次投影，并验证每次调用有一个对应 decoder、两道屏障和一次 clone。下表对每个 rank 的角色中位时长再取八 rank 中位数，单位 μs，来自[训练 profile 汇总](../../../../logs/h100/phase6/ma3-source-parallel-2x-50steps-20261022-222545-811304/fullstep/profile-summary.json)。

| 角色 | A decoder | D decoder | S decoder | A / D / S GEMM |
| --- | ---: | ---: | ---: | --- |
| Attention | 48.912 | 14.496 | 15.792 | 98.568 / 98.624 / 98.560 |
| MLP | 51.464 | 14.736 | 15.896 | 155.160 / 155.152 / 155.248 |

S 的 decoder 比 D 分别多约 1.296 μs、1.160 μs，与独立算子 S/D 为负的方向一致。独立算子另有 48 份 trace；其 decoder 八 rank 中位数也表现为 D 的 14.296/14.664 μs 小于 S 的 15.592/15.847 μs，见[算子 profile 汇总](../../../../logs/h100/phase6/ma3-source-parallel-2x-50steps-20261022-222545-811304/operator/profile-summary.json)。

这说明“把单个 group 的七次串行解码拆到四个 warp”没有在本配置下转化为更短的 decoder。增加的 CTA 数量、shared 交换、同步和分散写回是可能原因，但当前 trace 没有逐项消融或硬件计数器证据，不能断言具体哪项主导。Profile 采集独立于正式计时，不能将其局部时长差乘以调用数作为已获得的完整 step 收益。

### 8.5. 复核方法、证据边界与结论

2026-10-23 已使用与回传冻结脚本哈希一致的控制器，从原始 rank JSON 和 trace **离线重新生成算子报告、完整训练报告以及两份 profile 汇总，四份均与回传 JSON 完全一致**。复核同时检查了批次完成状态、两次独立 run ID、72 窗口/576 rank/3600 步、逐位输出、模型轨迹、实际路径及收益判定。原始 plan、preflight 和 report 文件哈希在复核前后不变。

审计材料：[脚本](../../../../outputs/h100/phase6/ma3-results-audit-20261023/audit.py)、[审计 JSON](../../../../outputs/h100/phase6/ma3-results-audit-20261023/audit.json)、[运行日志](../../../../outputs/h100/phase6/ma3-results-audit-20261023/audit.log)。本次是对回传数据的 CPU 离线复核，没有重新运行 GPU 或重新加载、哈希验证服务器动态库；运行身份由保存的记录和实际 kernel trace 交叉核验。

本批的研究结论是：**来源并行＋有序 BF16 归约保持了数值语义，但未达到“交替＋新机制比单纯交替更快”的完整训练目标；在隔离控制 D 上，来源并行还使独立算子和 profile decoder 略慢。** 应保留这一负结果，不能把相对旧 scalar A 的算子收益写成 MA3 机制的成功，也不能因区间跨零而把完整训练点估计写成确定退化。
