# H100 上相对融合 v2 的优势为何缩小

更新：2026-10-11。基于当前工作区 `98bd193` 的源码、冻结构建文本和已回传实验，仅进行 CPU 数据分析。本次不修改实现、不编译、不启动 GPU 实验，也不改动历史报告与验收口径。

后续实现状态（同日）：用户已要求将新 H100 融合 v2 对齐 A800 原基线，构建入口现采用 [a800-fused-v2-cta-v1](../../../tools/h100/phase6/fused_v2/README.md)。以下分析及所有性能数字仍针对变更前的冻结实验，不是新基线的测试结果；新实现尚待 H100 GPU 验证。

## 结论

**目前最有依据的解释是：H100 的融合 v2 基线已经吸收了 A800 候选的一部分执行成本优化，候选可额外回收的编解码开销因而减少；同时，到达计划在 H100 上的留出命中率较低，局部 MLP 节省又没有稳定转化为完整 step 收益。** GPU 架构变化可能进一步影响这些成本，但现有实验不能把收益差异全部归因于 H100 硬件。

尤其需要澄清：两边都叫 `taco_fused`，但**不是完全相同的源码仅换编译架构**。A800 必须基线被冻结在旧版本；H100 基线由较新的维护源码构建，已使用 warp 解码、优化的 E4M3 编码和实例持有配置。A800 候选相对旧基线的收益包含这些执行优化，H100 候选相对其基线则不再独享它们。这是本次核查中最明确、应优先控制的跨平台变量。

| 判断 | 证据强度 | 当前能够支持的表述 |
| --- | --- | --- |
| 两平台的融合 v2 基线实现不同 | 已核实源码与两平台 manifest | 不能把收益百分比之差直接解释为硬件差异 |
| H100 小模型的全量解码成本及选择性解码节省更小 | 已核对双方八 rank trace | 编解码优化的可回收空间缩小；硬件和软件分别贡献多少尚未分离 |
| H100 计划的留出命中率较低 | 已核对三套选择计划 | 静态尾部预测的可迁移性较弱；尚不能证明它独立导致了整步收益消失 |
| 发布 barrier 与完整 step 关键路径限制收益兑现 | 源码事实，trace 提供线索 | 提早若干 tile、减少 kernel 数，并不保证更早完成整步 |
| Hopper 原生特性没有充分用于量化 MLP | 已核实生成器、冻结 wrapper 与 trace | 当前结果反映 SM90 编译的 V2 移植，不能代表 Hopper 原生 TACO 的上限 |
| 更快互连、CPU 发射或运行环境造成收益减弱 | 尚待控制实验 | 合理假设，不能以规格或单份 profile 作为已证实根因 |

## 1. 先确定比较的对象

主要比较应为 **A800 0.64B TP8 window64 首测与 H100 0.64B TP8 window64 首测**：两者均为 12 层、H2048/FFN8192、S2048/mb1/global4、实际 padded vocab 9216、每步 4 个微批和 8192 token，MLP 全局 shape `[2048,2048,8192]`、本地 K=1024。它们仍存在基线版本、attention 后端和环境差异，因此只是工作负载最接近的已有对照，不是严格的硬件单变量实验。

A800 TP4 的 4%–5% 结果还涉及 M8192、更大的 micro batch、较少的累积次数及 TP4 专用接收端优化；不能直接作为 H100 TP8 的预期收益。A800 window16 优化结果也不能替换 window64 首测。对应边界见 [A800 TP8](../design-a800/base-phase6-tp8.md)、[TP4](../design-a800/base-phase6-tp4.md) 和 [TP4 接收端优化](../design-a800/optimize/tp4/optimize2.md)。

下表是相对同轮 `taco_fused` 的**配对完整 step 耗时下降百分比**，正值表示更快；括号为原报告 95% 区间。每轮独立统计，不合并四轮，也不用中位数相除代替配对比值。

| 场景 / 轮次 | 远端组合 % [95% CI] | 交替组合 % [95% CI] |
| --- | --- | --- |
| A800 0.64B TP8 / 1 | +2.199 [+2.010,+2.425] | +2.515 [+2.269,+2.907] |
| A800 0.64B TP8 / 2 | +2.073 [+1.646,+2.638] | +2.440 [+1.851,+2.988] |
| H100 0.64B TP8 / 1 | +0.532 [−0.271,+1.583] | +0.485 [−0.237,+1.495] |
| H100 0.64B TP8 / 2 | −0.368 [−0.934,+0.084] | −0.197 [−0.761,+0.299] |
| H100 6.7B TP8 / 1 | −0.882 [−2.611,+0.545] | −0.143 [−0.916,+0.760] |
| H100 6.7B TP8 / 2 | +3.003 [−0.876,+9.568] | +0.223 [−8.957,+9.567] |
| H100 6.7B TP8 / 3 | −0.521 [−5.065,+2.413] | −4.265 [−10.074,−0.348] |
| H100 6.7B TP8 / 4 | +0.724 [−0.904,+2.149] | −0.727 [−2.223,+0.346] |

来源：[A800 首测报告](../../../logs/a800/phase6/tp8-quant-measure-20261006/report.md)、[H100 小模型报告](../../../logs/h100/phase6/tp8-20261009-deployment/compare/report.json)、[H100 6.7B 第1–2轮](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/report.json)、[第3–4轮](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/report.json)。完整绝对耗时与轮次映射见 [base-phase6.md](base-phase6.md)。

事实是“小模型上优势已经缩小，6.7B 四轮也没有建立稳定优势”，不是“6.7B 才开始失效”。6.7B 的模型扩大和前次 OOM 因此不能成为主要通用解释。对 `native_taco` 的约三成收益则反映另一套更高开销的朴素量化路径，不等于相对融合 v2 的收益。

## 2. 最关键的差异：融合 v2 的比较基线已经变化

核对的是实际实验冻结源码，而非仅凭当前维护源码或策略名判断。A800 首测的 [build manifest](../../../outputs/a800/phase6/tp8-quant-measure-20261006/build/manifest.json) 与所检查的 codec/runtime 文件哈希一致。H100 两模型、基线及两候选的 codec/runtime 和 C++/Python wrapper 共 24 项源码哈希也与各自 manifest 一致；以下两份 codec/runtime 文件在 H100 各版本间相同，也与当前维护入口相同。

| 实现细节 | A800 冻结 `taco_fused` | A800 候选 | H100 `taco_fused` 与候选 |
| --- | --- | --- | --- |
| H128 接收解码 | 一个 128-thread CTA 处理一组；使用共享 scratch 和跨 warp 同步 | 一个 warp 处理一组，一个 CTA 四组 | 两者都使用该 warp 版本 |
| 融合编码的 E4M3 转换 | `__nv_cvt_float_to_fp8` | `encode_e4m3_bounded` | 两者都使用 `encode_e4m3_bounded` |
| Python/C++ 配置 | 每次 forward 经 ctypes configure/reset | C++ 实例持有配置 | 两者都支持实例持有配置 |
| MLP 发布 / 复用同步 | 发布后解码，解码后另有复用 barrier | 双完整工作区，保留发布 barrier | 基线仍有解码后 barrier；候选通过双工作区避免该次 barrier |
| `taco_runtime.cu` SHA256 前12位 | `bd89ecb31462` | `5565114002e2` | 两者均 `5565114002e2` |
| `taco_codec.cuh` SHA256 前12位 | `e5b7c704974b` | `f9fbba685d7e` | 两者均 `f9fbba685d7e` |

可直接查阅：[A800 基线 runtime](../../../outputs/a800/phase6/tp8-quant-measure-20261006/build/taco_fused/overlay/gemm_rs/taco_runtime.cu)、[A800 基线 codec](../../../outputs/a800/phase6/tp8-quant-measure-20261006/build/taco_fused/overlay/gemm_rs/taco_codec.cuh)、[A800 候选 runtime](../../../outputs/a800/phase6/tp8-quant-measure-20261006/build/remote_arrival_selective/overlay/gemm_rs/taco_runtime.cu)、[H100 基线 runtime](../../../outputs/h100/phase6/tp8-20261009-deployment/base/taco_fused/src/gemm_rs/taco_runtime.cu)、[H100 基线 codec](../../../outputs/h100/phase6/tp8-20261009-deployment/base/taco_fused/src/gemm_rs/taco_codec.cuh)。配置与双缓冲分派见两实验的 `scripts/taco_support.py` 和冻结 `python/flux/gemm_rs_taco.py`。

因此，A800 的组合收益包含“旧全量 codec → 新的稀疏 codec/执行封装”的变化；H100 则是在同一个较新 codec/封装基础上比较全量与稀疏量化。**H100 基线的执行实现更完善，是收益空间缩小的直接解释之一。** 当前没有同硬件、同源码的完整消融，不能给它分配一个确定的百分比贡献，也不能把它与 H100 原生 FP8 转换的效果混在一起计算。

这不使任一平台已有的同轮比较失效，也不应回头削弱基线或重写历史结果；它限制的是“跨平台收益差异由硬件导致”的因果表述。

## 3. Trace：可回收的解码成本确实下降

本次解压并汇总四套场景中 `taco_fused`、远端、交替、`original` 的全部八 rank profile，共 128 份 trace。每份是独立的一个 profile step。下表先在每个 rank 内统计对应 kernel 的 `dur`，再对八 rank 的统计量取中位数；不是把所有 rank 的时间加起来，也不是正式 timing 数据。

### 3.1 同小模型 TP8 的解码比较

每个 profile step 有 12 层 × 4 微批 = 48 次 MLP 解码；两平台都核对到 48 次。

| 平台 | 全量融合解码 μs/次 | 远端组合 μs/次 | 交替组合 μs/次 | 全量融合解码累计 ms/step | 相对全量少花的解码累计时间：远端 / 交替 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A800 0.64B | 47.292 | 22.007 | 19.922 | 2.270 | 1.214 / 1.314 ms |
| H100 0.64B | 16.615 | 13.220 | 11.078 | 0.797 | 0.163 / 0.266 ms |

这里的累计差为各组八 rank 中位数之差，单位为一个 profile step，不能解释为正式整步的因果收益。

**全量解码成本下降得更多，选择性解码的绝对优势随之缩小。** 相对正式融合基线约 173 ms/step 与 206 ms/step，仅作数量级比较，解码差从约 0.70%–0.76% 的整步量级降至约 0.08%–0.13%。这不解释所有 A800 收益，但说明为什么不能把 A800 的收益原样迁移到 H100。

这组 47.3 → 16.6 μs 同时包含上一节的 CTA→warp 实现变化和硬件变化，**不能称为同一解码器在 H100 上获得了约 2.8 倍纯硬件加速**。

### 3.2 FP8 转换提供了一个具体的硬件机制

`decode_warp()` 仍调用 `__nv_cvt_fp8_to_halfraw`。本机 CUDA runtime 头文件的该函数在 `__CUDA_ARCH__ >= 890` 时走 FP8x2 转换分支；SM80 则以位操作、特殊值处理及次正规数归一化实现转换。FP8 转换指令的架构支持也可查 [NVIDIA PTX ISA 的 cvt 说明](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt)。

这是“H100 上 FP8 解码可能更便宜”的源码级依据，但本次未对正式 H100 二进制做 SASS/指令计数消融，不能断言它贡献了全部或主要的 47.3 → 16.6 μs 差值。H128 Hadamard、scale、BF16 舍入与归约仍需执行。

还需区分编码：本次 H100 融合路径的 `encode_warp()` 调用的是自定义 `encode_e4m3_bounded()`，不能把它也直接称为自动使用了硬件 FP8 编码。GEMM 本身仍为 BF16；这里没有使用 FP8 Tensor Core 矩阵乘或 Transformer Engine。

### 3.3 6.7B 仍有局部节省，但量级不足以直接承诺整步收益

6.7B 每个 profile step 有 32 × 4 = 128 次量化 MLP。其量化策略的 V2 kernel 仅对应 MLP，attention 单独走 V3，因此可以分开计数。

| Profile 来源 / 策略 | MLP V2 μs/次 | 解码 μs/次 | MLP V2 累计 ms | 解码累计 ms |
| --- | ---: | ---: | ---: | ---: |
| 第1–2轮预检 / 全量融合 | 156.051 | 32.122 | 19.975 | 4.112 |
| 第1–2轮预检 / 远端 | 109.971 | 22.981 | 14.076 | 2.942 |
| 第1–2轮预检 / 交替 | 112.441 | 22.051 | 14.392 | 2.822 |
| 第3–4轮预检 / 全量融合 | 155.784 | 31.842 | 19.940 | 4.076 |
| 第3–4轮预检 / 远端 | 109.503 | 22.953 | 14.016 | 2.938 |
| 第3–4轮预检 / 交替 | 112.239 | 22.207 | 14.367 | 2.842 |

两次独立 profile 中，远端组合的 MLP V2 累计差都约 5.9 ms，解码累计差约 1.1–1.2 ms；说明局部节省可复现，不能简单判断为“选择性量化完全不起作用”。但这些局部差额加起来也只是约 7 ms 的量级，相对约 540 ms 正式 step 约为 1.3%；4% 目标对应约 21.6 ms。这个换算只用于判断量级，既不是性能预测，也不是理论上限：同步、重叠和关键路径改变还可能增减整步收益。

小模型的 H100 V2 均值则为全量 66.609 μs、远端 56.167 μs、交替 74.466 μs，交替甚至在该 profile 中更慢。因此不能概括为“所有排序在 H100 上都能让 GEMM 更快”。V2 的时间也包含融合 epilogue 和远端写回，不是纯 Tensor Core 计算时间。

## 4. 为什么局部节省没有稳定兑现：同步与完整 step

冻结实现的实际依赖关系为：

```text
全量融合：GEMM + 全远端编码/写回 → 全 rank 发布 barrier → 解码/归约 → 复用 barrier
候选组合：GEMM + 稀疏编码/混合写回 → 全 rank 发布 barrier → 混合解码/归约
                                                  双完整工作区保护后续复用
```

H100 的 [冻结 wrapper](../../../outputs/h100/phase6/tp8-20261009-deployment/base/taco_fused/src/gemm_rs/ths_op/gemm_reduce_scatter.cc) 使用 `phase6_arch()` 选择 V2 算法，包括 `forward_barrier()` 的分支；量化 MLP **没有**套用 Hopper V3 的单节点免发布 barrier 分支。当前维护源码中的 `get_arch()` 条件不能脱离这个构建时改写去解释 H100 历史实验。

候选双缓冲的约束见 [冻结 Python wrapper](../../../outputs/h100/phase6/tp8-20261009-deployment/candidates/remote_arrival_selective/python/flux/gemm_rs_taco.py)：两套 BF16/FP8 完整工作区交替，每次调用的发布 barrier 仍然保留，下一次调用的同步参与保护旧工作区。它减少了 rendezvous 次数，但没有实现接收端按 tile 到达即解码、即消费。

因此，提前一个来源的几个 tile 不一定能提前接收端启动；只有它缩短了最后完成的生产者、实际等待或后继依赖，才可能进入关键路径。发布 barrier 的等待还可能吸收先前减少的工作，使剩余 barrier 更长。

| H100 场景 / 策略 | IPC barrier 次数/step | 八 rank 中位累计 duration，ms |
| --- | ---: | ---: |
| 0.64B / 全量融合 | 96 | 9.338 |
| 0.64B / 远端 | 48 | 10.938 |
| 0.64B / 交替 | 48 | 8.900 |
| 6.7B 复测 profile / 全量融合 | 256 | 24.677 |
| 6.7B 复测 profile / 远端 | 128 | 24.158 |
| 6.7B 复测 profile / 交替 | 128 | 24.252 |

**次数减半并不意味着同步 duration 减半。** 但上述不同进程的单步 profile 不足以精确定位等待从哪里转移，也不能把所有 barrier duration 当作可消除开销。

同样，H100 小模型的 `taco_fused` profile 首末 kernel 跨度约 349 ms，而正式 step 中位数约 206 ms；6.7B 复测对应约 863 ms 与 536–542 ms。profile 明显扰动了执行。NCCL、V3 等 kernel 的 duration 可能包含等待且跨 stream 重叠；不能把累计时间相加当作 step，也不能用它直接宣称网络饱和、算力不足或某一个 rank 的硬件较慢。跨 rank 时间戳也不能直接当成同一时钟。

## 5. 到达计划在 H100 上更难预测后续尾部

三套计划均用 BF16 基础顺序的 pass 0/1 拟合、pass 2 留出；window64、每来源远端 tile 预算上限 1/64。以下命中率是**选中来源在留出观察中仍属于近临界集合的比例**，并非全部 tile 排序准确率，更不是量化后的净收益预测。

| 平台 / 模型 / 基础 | 有效训练 tile | 每来源选中数 | 留出命中数 / 有效选中观察 | 命中率 | 采样中位扰动 |
| --- | --- | --- | --- | ---: | ---: |
| A800 0.64B / 远端 | 256/256 | `[3,3,3,3,3,3,3,3]` | 20/24 | 83.33% | 22.96% |
| A800 0.64B / 交替 | 255/256 | `[3,3,3,3,3,3,3,3]` | 19/24 | 79.17% | 34.01% |
| H100 0.64B / 远端 | 254/256 | `[0,3,0,3,0,3,0,3]` | 4/12 | 33.33% | 21.52% |
| H100 0.64B / 交替 | 125/256 | `[3,3,3,3,3,3,3,3]` | 10/24 | 41.67% | 10.21% |
| H100 6.7B / 远端 | 512/512 | `[2,6,7,3,4,7,7,3]` | 11/39 | 28.21% | 3.37% |
| H100 6.7B / 交替 | 512/512 | `[7,7,7,0,7,7,7,3]` | 7/45 | 15.56% | 5.97% |

来源：[A800 计划](../../../outputs/a800/phase6/tp8-quant-adapt-20261006/selection-plan.json)、[H100 小模型计划](../../../outputs/h100/phase6/tp8-20261009-deployment/selection-plan.json)、[H100 6.7B 计划](../../../outputs/h100/phase6/gpt67-tp8-20261010-112249/selection-plan-20261010-153517.json)。6.7B 两次 compare 复用同一计划，不能视为两次独立校准。

这表明 H100 计划中“训练时看起来靠近尾部”的选中项，在留出观察中较少保持该性质。6.7B 已有完整训练覆盖且采样扰动下降，命中率仍低，因此问题不能只归于旧小模型交替计划缺失了约一半训练 tile。另一方面，A800 采样扰动反而更高，不能写成“H100 因为采样扰动更大而失败”。观察样本少、近临界阈值随轮询精度定义，这些百分比也不能视为精确的总体泛化准确率。

规划规则本身有两个限制：

1. [arrival/plan.py](../../../tools/a800/arrival/plan.py) 的 `priority()` 给训练两遍均近临界的来源赋予接收端 join-time 分数；[phase5/plan.py](../../../tools/a800/phase5/plan.py) 再按分数选取有限 tile。它没有估算“节省通信 − 编码/解码 − 调度代价”的净收益，也没有直接优化整步关键路径。
2. 计划来自未量化基础执行，没有在重排与量化之后联合重新校准。重排和混合写回都可能改变原有尾部关系，离线分数与最终执行之间存在偏差。

H100 更短的 kernel/传输时间、132 SM 下不同的调度及 rank 启动偏斜，都可能改变可预测性；目前不能从命中率或零选择 rank 反推某一 GPU、链路或拓扑存在故障。零选择也不等于实现漏做量化：1/64 是上限，规则允许没有合格项的来源保持 BF16。

另一个几何事实是：小模型每来源只有 256 tile、每目标分区 32 个合法槽位，window64 已覆盖整个目标分区；6.7B 对应 512 tile、每目标 64 槽。`permute()` 保留每个目标的槽位，只在合法窗口内置换；它不是允许任意 tile 提前到任意位置的全局调度。

## 6. 量化收益与硬件规格应该怎样解释

H128 对 128 个 BF16 数使用 256 字节；TACO 包含 128 字节 FP8 和两个 FP32 scale，共 136 字节。因此选中部分的逻辑数据降幅为 46.875%。全量融合将全部远端贡献压缩，稀疏候选则只压缩少数贡献：

| 场景 / 策略 | 选中远端 tile 比例 | 相对全 BF16 的远端逻辑字节下降 |
| --- | ---: | ---: |
| 全量融合 | 100% | 46.875% |
| A800 0.64B 两候选 | 1.3393% | 0.6278% |
| H100 0.64B 远端 / 交替 | 0.6696% / 1.3393% | 0.3139% / 0.6278% |
| H100 6.7B 远端 / 交替 | 1.0882% / 1.2556% | 0.5101% / 0.5886% |

按该口径，候选发送的远端逻辑 payload 约为全量融合的 **1.87 倍**。这些数不包含控制流量、事务放大或缓存影响，不能等同硬件链路计数。相对融合 v2，候选主要是在交换“少做 codec”和“多传 BF16”的成本，而不是进一步降低总通信量。

H100 SXM 公布的 HBM 带宽为 3.35 TB/s、NVLink 为 900 GB/s 双向；A800 80GB SXM 的对应规格为 2.039 TB/s、400 GB/s。来源为 [NVIDIA H100 规格](https://www.nvidia.com/en-us/data-center/h100/) 与 [Lenovo A800 SXM 产品规格](https://lenovopress.lenovo.com/lp1813-thinksystem-nvidia-a800-pcie-gpu)。它们是设备峰值，不能除以 7 当作实测单 peer 带宽，也不能直接代入 step 收益。

更快互连可能减少可供到达排序优化的暴露尾部，但也会降低候选额外 BF16 流量的代价，后者反而可能有利于候选。因此“NVLink 更快，所以相对融合 v2 必然收益更小”并不是完整推理。当前较强证据是 codec 成本与基线版本变化；互连的净效应还需同源码、同协议和实际流量/等待证据分离。

Hopper 还提供 TMA、更大的片上存储等能力，见 [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)。但本项目 [generator.cc](../../../tools/h100/phase6/generator.cc) 的量化 MLP 仍为 V2 的 `128×128×32`、3-stage 注册路径，经 SM90 编译；原生参考和 attention 使用 V3。重新编译不会自动将自定义 V2 epilogue 重写为 WGMMA/TMA 流水线。由于融合基线和候选都用 V2，这一共同限制也不能单独证明两者差距为何缩小；它更直接限制了对“H100 方法上限”的外推。

## 7. 模型规模、环境与噪声的作用

6.7B 增加了层数、隐藏维度与本地 K，前向/反向 GEMM、attention、optimizer 等共同开销也改变。只优化 `linear_fc2` 前向，必然受到其他 step 工作的稀释；增加模型规模不是自动提高收益的办法。不过旧小模型已无稳定收益，因此规模只能是附加因素。

还有一条值得单独审查的环境线索：相同小模型工作负载下，A800 的 `native` 约 179 ms、`original` 约 170 ms；H100 分别约 220 ms、209 ms，H100 的正式整步没有体现通常预期的硬件加速。这涉及 host、软件环境、后端、同步、kernel 发射和可能的资源竞争等变量，不能凭现有记录确定是 CPU 瓶颈，也不能宣称 H100 硬件更慢。由于两个原生参考都出现这一现象，仅讨论 TACO 内核还不足以解释平台差异。

6.7B 正式窗口还有明显长尾：

- 第2轮融合 v2 为 526.983–659.164 ms/step，交替组合也出现 641.158 ms 窗口。
- 第3轮融合 v2 为 533.754–646.244 ms，远端为 530.468–623.309 ms，交替为 533.828–680.787 ms。
- 第4轮融合 v2 为 532.119–551.547 ms；该轮远端为 +0.724%，交替为 −0.727%，区间仍跨零。

这些观察解释了为什么某些点估计和区间变化较大，但不证明长尾的具体来源。既定配对协议应继续保留所有有效窗口；不能事后删除慢窗口，把第2轮远端 +3.003% 或第3轮的中位数排序当作稳定收益。第3轮远端中位数低于基线，却得到 −0.521% 配对下降，正说明中位数相除会改变问题。

前次 OOM 目录不参与统计，最新复测状态未设置 resumed；现有文件未提供足以确认全过程独占 GPU、时钟稳定和 CPU 调度条件一致的完整证据。也没有证据支持把所有负收益都归咎于外部干扰。

## 8. 后续最有区分力的验证方向（本次未执行）

建议优先回答“收益来自哪里”，再决定是否修改 Hopper kernel：

| 优先级 | 后续控制实验 | 能区分的问题 |
| --- | --- | --- |
| 1 | 在原有基线之外新增同源码的全量融合诊断组：使两平台及候选共享 codec、封装、tile 参数；同步策略另做显式消融，保留原基线结果 | 先剥离旧基线与新执行实现的差异，再估计硬件因素 |
| 2 | 候选相同执行协议下比较零 mask、当前 mask；同 mask 下比较基础顺序与到达置换，保持双缓冲和解码器一致 | 分离稀疏量化、到达重排、执行协议三者的贡献 |
| 3 | 核对环境与独占性；低扰动捕获 CPU 发射、MLP 开始至输出可用、发布 barrier、后继 NCCL/attention 的依赖 | 判断局部收益被等待、共同工作还是 host 发射限制吸收 |
| 4 | 在最终执行协议下独立校准与留出验证，评估选中项是否真正缩短临界依赖；校准数据与正式收益验证分离 | 检查低命中是否来自校准到部署的偏差，以及是否值得自适应计划 |
| 5 | 控制相同 codec 后查看正式 cubin 的转换指令、访存和占用；再评估 Hopper 原生 epilogue/流水线方向 | 分离 FP8 转换、内存系统与 V2 移植限制，避免用架构峰值替代实测 |

这些是后续实验设计，不是本次已启动的任务，也不承诺任何方案能恢复 4% 收益。特别是同源码的新诊断组应新增命名与产物，不能替换冻结必须基线。

## 9. 本次核查范围与复算方法

本次读取了两平台接续说明、结果文档、A800 首测报告、三套 H100 报告、三套选择计划、维护入口和相关冻结源码；核对了 A800 基线/候选 codec 与 runtime 的 manifest 哈希，以及 H100 两模型共 24 项相关源码的 manifest 哈希与版本差异。没有在本机重新核验 H100 执行端原绝对路径的动态库，也没有重新运行远端预检。

对三套 H100 正式结果，本次 CPU 检查覆盖 216 窗口、1728 份 rank 计时记录：每窗口八 rank 齐全、completed/passed、零 skip/fallback、20 个正式 step，并按最慢 rank 重算本文三种量化策略的每轮中位数与候选对融合 v2 的配对点估计，均与报告一致。95% 区间引用原报告；本次未重新执行完整 bootstrap/二进制验收流程。既有更全面核验的范围见 [instruction.md](instruction.md)。

Trace 原始目录如下，每个策略目录内使用全部 `rank0`–`rank7` 的 `rank*-profile.json.gz`：

- [A800 0.64B window64](../../../logs/a800/phase6/tp8-quant-measure-20261006/results/preflight/)
- [H100 0.64B](../../../logs/h100/phase6/tp8-20261009-deployment/compare/results/)
- [H100 6.7B 第1–2轮预检](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/results/)
- [H100 6.7B 第3–4轮预检](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/results/)

可在仓库根目录用以下只读 CPU 代码复算本文 kernel 表；它不导入 Torch、不启动 GPU、不写入原始产物：

```python
import gzip
import json
import statistics as st
from pathlib import Path

roots = {
    "A800-small": "logs/a800/phase6/tp8-quant-measure-20261006/results/preflight",
    "H100-small": "logs/h100/phase6/tp8-20261009-deployment/compare/results",
    "H100-large-first": "logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/results",
    "H100-large-repeat": "logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/results",
}
policies = ["taco_fused", "remote_arrival_selective", "interleaved_arrival_selective"]
for tag, root in roots.items():
    for policy in policies:
        stats = {k: [] for k in ("decode", "v2", "barrier")}
        for rank in range(8):
            path = Path(root) / ("profile-" + policy) / f"rank{rank}-profile.json.gz"
            with gzip.open(path, "rt") as f:
                events = [e for e in json.load(f)["traceEvents"] if e.get("cat") == "kernel"]
            for kind, needle in [("decode", "taco_decode"), ("v2", "gemmv2"),
                                 ("barrier", "cudaipcbarrierallkernel")]:
                durations = [e["dur"] for e in events if needle in e["name"].lower()]
                if durations:
                    stats[kind].append((len(durations), st.mean(durations), sum(durations) / 1000))
        for kind, rows in stats.items():
            if rows:
                print(tag, policy, kind, "counts", sorted({r[0] for r in rows}),
                      "median-rank mean-us", round(st.median(r[1] for r in rows), 3),
                      "median-rank sum-ms", round(st.median(r[2] for r in rows), 3))
```

A800 的 V2 汇总同时包含 attention 和 MLP，本分析没有把该项与 H100 的 MLP-only V2 直接比较。解码与 barrier 按实际 kernel 名单独统计；编码已融合在 GEMM 内，不能重复计时。所有机制解释最终都应回到独立的完整 step 测量检验。
