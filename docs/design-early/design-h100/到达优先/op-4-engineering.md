> 历史工程方案：已退出本轮主方案。新的算法性方案见 [op-4.md](op-4.md)。以下保留原记录。

# H100 op4：停止扩展到达优先，验证固定开销优化

更新：2026-10-21。本文完成原因判断、替代方案实现及另一台服务器的启动入口。**本轮没有运行新的GPU实验，尚不声称提速。** 用户所指 `op3.md` 当前文件名为 [op-3.md](op-3.md)，最新实测为 [base-op3.md](base-op3.md)；已有重命名及历史结果保持原样。

## 1. 判断：当前路线缺少净收益，不是开关没有生效

**建议在当前双投影、TP8、8×H100配置上停止把到达优先作为主要优化方向，改为减少每次投影的固定成本。** 这不是证明所有到达感知调度都不可行，而是现有几代实现已经分别验证了重排、真正提前消费，却没有验证完整训练收益；继续调整排序门槛或扩大try范围缺乏实验支持。

需要区分三件事：

| 路线 | 实际做了什么 | 已有结果及可支持的判断 |
| --- | --- | --- |
| 原离线到达排序 | Attention/MLP改变2491/4096、2422/4096个物理槽位 | 旧B/A为−0.185%、−0.135%，区间跨零。映射确实生效，但整阶段发布屏障使中间tile早到不能触发早消费 |
| 消融7残差排序 | 两角色均0交换，回退identity | 三次B/A为−0.151%、+0.119%、−0.096%，均未确认收益。这测到了回退行为，不能作为非identity排序有效或无效的直接证据 |
| op1、op2进一步排序 | op1改变119/4096、42/4096槽位；op2改变3072/4096、3584/4096槽位 | 两次完整训练均无稳定正收益，不能继续解释为“映射没用上” |
| op3 v2实际提前消费 | 64次诊断调用中43次有有效提前消费 | 两次O/A分别−0.928%、−0.844%；O/S分别−0.726%、−0.746%，95%区间均在退化方向 |

依据：[消融7定义与机制分析](../消融实验/消融实验7-改进到达优先.md)、[消融7实际结果](../消融实验/base-phase6-ablation7.md)、[op1结果](base-op1.md)、[op2结果](base-op2.md)、[op3 v2结果](base-op3.md)。**消融7定义页开头“尚未进行新H100实测”的状态已陈旧，应以实际结果页的三次完整实验为准。** 本页不把identity结果误写成一次成功的到达优化。

原思想 [idea.md](../idea.md) 中的 `J(t)=max_s A_s(t)` 对解释某个tile何时齐备有意义，但不等于实际算子或训练步的结束时间。当前排序保持总计算、总字节数和目的地轮转不变；有阶段屏障时，主要能改变的是局部性、争用及最终生产尾部。提前一个中途tile的最后来源，未必改变阶段结束。旧评分又混入原排位趋势，标签来自稀疏BF16校准，实际全远端量化路径不同，不能靠降低准入门槛消除这些偏差。

op3已越过“屏障后才消费”的限制，但出现另一项成本：新发布协议对每个16行fragment增加同步/发布，每rank每次投影共有4096个fragment发布；接收端还增加try、stop、join及tail。v2虽然显著缩短扫描拖尾，仍未使O/S转正。其生产kernel在带诊断profile中明显长于A/D，提示成本存在，但**不能用profile中位数直接分摊正式训练的退化量**。

因此，当前证据更接近“优化目标与执行结构不匹配，协议成本超过可兑现收益”，而非某一个可修正的开关或索引错误。没有发现足以解释历次负结果的映射失效或输出数值错误。新一轮不再追求制造更多早到/重叠，而验证一个可直接减少工作量的方向。

## 2. 替代思路：减少每次投影必付的固定成本

op4保持基础交替顺序、全远端H128/E4M3量化、本地BF16、原生产epilogue及归约数值，仅改变工作区生命周期与解码执行方式。没有新到达表、采样器、选择性mask、ready标志、辅助stream或在线队列。

具体依据来自A的实际代码，而不是性能外推：

1. **输出已经独立，外层clone是冗余复制。** A的 `local_reduction()` 本来就通过 `torch::empty()` 创建独立输出，再让decoder直接写入。实验adapter因历史所有权约定仍调用clone。L/F公开这一所有权并省去额外clone；不是把可覆盖的工作区视图直接返回给训练。
2. **配置可绑定实例。** A每次Python forward经ctypes configure/reset，并重复Python侧输入验证。L/F在构造时绑定packet、rank、完整shape与K；C++ forward用RAII恢复实例配置，并验证设备、固定stream、dtype、布局及准确shape，异常时恢复原线程局部配置。必要验证仍保留，配置不在实例之间串用。
3. **双工作区让下一次发布屏障承担复用证明。** L/F在两个完整GemmRS/packet工作区之间轮转，保留每次GEMM后的发布屏障，删除本次decode之后单独的复用屏障。依赖关系见下一节。
4. **快速解码单独作为一个因素。** D/F复用op3 v2已经实现的warp H128 decoder，保持逐来源BF16舍入和ring累加顺序。L保留A的标量CTA decoder，避免把生命周期效果与decoder混为一谈。

原A源码见 [fused_v2 wrapper](../../../../tools/h100/phase6/fused_v2/src/gemm_rs/ths_op/gemm_reduce_scatter.cc)、[frontend](../../../../tools/h100/phase6/fused_v2/python/flux/gemm_rs_taco.py)；op4变换见 [arrival_op4_sources.py](../../../../tools/h100/phase6/arrival_op4_sources.py)。

双工作区和实例配置在已有候选实现中已有先例，warp decoder来自op3 v2；**本轮是从固定fused-v2基线独立组合、隔离和验证固定成本优化，不把这些已有机制包装成新发明，也不把C的历史优势当作op4收益。** 尤其D曾在独立算子快13.145%/9.729%，完整训练却未确认改善，这次仍必须完成完整训练。

### 2.1 是否改了量化实现，是否等同于消融实验3的C

**改了执行实现，没有改量化规则，也没有直接换成消融实验3的C。** D/F把原标量CTA解码换成op3 v2的warp解码；L/F改变配置、缓冲及输出生命周期。H128/E4M3编码数学、全远端量化对象、本地BF16和归约舍入规则保持op4基线A的定义。

这里有两个不同的“A”，需要明确：[消融实验3](../消融实验/消融实验3.md) 的A/B/C全部共用优化候选实现，原本就有warp解码、双工作区及相同输出生命周期；**op4的A则是旧固定fused-v2实现，带标量CTA解码、单工作区和外层clone。** 这些执行优化并非消融3的C独有。

| 项目 | op4组合组F | 消融实验3的C |
| --- | --- | --- |
| 到达优先 | 关闭，基础交替顺序 | 开启，冻结角色到达表 |
| 远端量化范围 | 全部远端，每来源448个tile | 定向选择，每来源最多1/64，即最多7个tile，其余BF16 |
| 双工作区、实例配置、独立输出 | 从固定fused-v2单独改造 | 已有候选实现的共同能力，消融3 A/B/C均具备 |
| 解码来源 | op3 v2的全远端warp decoder | 候选的BF16/FP8混合warp decoder |
| 构建来源 | 固定fused-v2源树派生的新库 | 已冻结的候选计算源树 |

**从“交替＋关闭到达＋全远端量化”的策略定义看，F更接近消融3的A-双模块，而不是C；但两者不是同一份计算实现，不能直接视作相同库或拼接性能结果。** op4是在把旧封装的一部分执行结构向已有候选结构靠近，并拆开验证生命周期与decoder的效果。

因此，op4的定位是**执行层工程优化与收益归因实验**，不是新的量化算法、选择性量化策略或到达调度思想。如果讨论的是对 `idea.md` 的算法创新替代，op4本身尚未提供这样的新算法。


## 3. 双工作区为什么可以少一道屏障

对同一公开实例，所有rank按相同顺序调用，调用k使用 `slot=k mod 2`：

```text
同一个CUDA stream：
GEMM(k, slot0) → 发布屏障(k) → decode(k)
→ GEMM(k+1, slot1) → 发布屏障(k+1) → decode(k+1)
→ GEMM(k+2, slot0) → …
```

任一rank能够开始 `GEMM(k+2)`，说明它已经通过 `发布屏障(k+1)`。这道全rank屏障要求其他rank都到达；每个rank的 `decode(k)` 又先于它进入该屏障。因此，覆盖slot0时，所有rank均已读完slot0的上一代数据。两slot使用独立屏障存储并不破坏此证明，跨slot顺序来自同一stream。

代码继续使用原IPC全rank屏障实现；构建关闭NVSHMEM。CUDA对同一stream的命令顺序提供基础保证，见 [CUDA12.8 Streams](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#streams)。跨GPU发布/汇合仍由项目的全rank屏障承担，不能仅凭stream有序就删除它。

边界必须明确：

- 仅支持当前TP8、M2048、N4096、K_local512或2048，固定创建设备/stream；禁止CUDA graph capture。
- 使用公开 `GemmRSTaco.forward()`，不能直接调用内部slot，或改用C++的 `profiling/forward_barrier/forward_reduce_scatter` 分拆入口绕过配置和轮转。
- 所有rank构造、调用顺序一致；前向失败后实例标记不可重用，整项实验停止，不重试半次collective。
- 工作区必须存活到所有rank完成最后一次消费并进行collective rendezvous；不能某个rank单独提前销毁。算子工具在同步、全rank rendezvous后释放实例；完整训练持有全部实例直至窗口结束。
- 增加一套完整通信工作区是代价，不能只报告少一道屏障而忽略显存。原worker保留各rank的 `peak_allocated_gib`、`peak_reserved_gib`；回传时应同时检查这些值。两套工作区只复用通信缓冲，返回的输出仍各自独立。

每步两类投影各128次，共256次投影。路径上从每次两道屏障减为一道、从一次冗余clone减为零，对应每rank每step少256次屏障kernel和256次输出clone。这是**执行次数变化**，不是已测出的耗时下降；不会将它乘一个估计kernel延迟后宣布达到1%。

## 4. 四组对照及归因

| 组别 | Decoder | 通信工作区 | 配置及外层输出 | 每投影屏障 |
| --- | --- | ---: | --- | ---: |
| A | 原标量CTA | 1 | 原逐次配置/复位＋clone | 2 |
| D | warp H128 | 1 | 同A | 2 |
| L | 原标量CTA | 2 | 实例绑定＋独立输出直接返回 | 1 |
| F | warp H128 | 2 | 同L | 1 |

四套库独立构建，A逐项核验原登记源码，D只改runtime并新增decoder helper；L只改header/runtime/wrapper/frontend，F为两项组合。所有组的GEMM生产代码、量化对象、编码算法和identity交替映射一致。运行期核验真正加载的动态库及Python扩展，不接受op3或旧C二进制冒充op4。

| 比较 | 回答的问题 | 报告字段 |
| --- | --- | --- |
| F/A | 组合候选能否取得完整训练至少1%净收益？ | `target_validated` |
| L/A | 生命周期优化包在标量decoder下是否有效？ | `lifecycle_benefit_validated` |
| F/D | 生命周期优化包在warp decoder下是否还有增量？ | `lifecycle_increment_validated` |
| D/A | 单独warp decoder在原生命周期下是否有效？ | `decoder_control_passed` |
| F/L | warp decoder在新生命周期下是否还有增量？ | `decoder_increment_validated` |

主验收沿用op3的1%标准：**两次独立比较都要求F/A点估计与配对95%区间下界至少1%。** 四项归因分别要求两次比较的对应点估计和区间下界大于0，互不替代。L把配置、clone、双工作区一起作为一个生命周期优化包，当前四组不能进一步精确分摊这三项各自贡献；也不报告“到达优先收益”。

## 5. 正确性、执行证据与完整训练

入口不会依据独立算子是否提速决定进入训练。数值、运行身份和执行路径检查通过后，即使算子估算为零或负，也继续两次完整训练；错误、证据缺失及运行失败仍终止。

算子阶段默认8个完整四组块，共32个fresh-process TP8窗口；每角色100次预热、200次正式调用，所有组均交付独立输出，计时包含各自实际API成本。包括：

- 3个seed×4次改变输入/权重的复用检查、相同输入重复调用，rank0/rank7延迟生产；各组输出哈希逐项一致，并通过独立codec检查。
- 每角色连续6次不同输入调用，中间不插入reference collective、hash或同步；保留全部输出，跨越两工作区多次复用后再核验。这用于捕获过早覆盖及把输出工作区错误暴露给上层的问题。
- 独立算子profile验证A/D每4次调用有8个屏障、4次clone，L/F有4个屏障、零clone，并检查正确decoder及同stream执行次序。

完整训练仍是GPT6.7B、32层、H4096/FFN16384、S2048、TP8、BF16、micro batch1/global batch4、seed1234/token_seed4321，每步8192token。所有组统一 `CUDA_DEVICE_MAX_CONNECTIONS=8`、`FLUX_GEMM_RS_GEMM_ONLY=1`、`NCCL_NVLS_ENABLE=0`，Megatron使用独立源码副本及共同连接数检查补丁。

两次独立比较，每次12个四组块，共 **96个正式窗口、768份正式rank记录、4800个累计optimizer step**；每窗口另有10步预热，50步正式计时。每次前后各6块，后半反转块序和组序，四组在每个位置均出现3次。每窗口重新初始化，累计4800步不等于一个模型连续训练4800步。

窗口时间取八rank最大墙钟除以50，包含前后向、通信、optimizer及清梯度。配对下降率为 `100×[1−exp(mean(log(T_X/T_Y)))]`，正值表示更快；按完整四组块bootstrap10000次，保留全部窗口、负收益与跨零区间。八rank和50个窗口内step不当作独立样本。

正式训练之后另跑四组真实模型profile，各rank每角色应有128个投影，核验屏障、clone及decoder，避免只在微基准中验证优化路径。profile不计入50步耗时。训练验收同时要求初始化、输入、loss/grad_norm、64个目标模块、调用次数、工作区指针及运行库一致，skip/fallback为零。短程合成数据的一致性不解释成长期收敛。

## 6. 另一台服务器启动

同步本轮新增工具以及它们依赖的现有仓库源码和CUTLASS/NCCL子模块。在此前运行op3的8×H100服务器/容器内，沿用原Python和编译器环境，在仓库根目录直接使用一行启动命令：

```bash
cd /workspace/TempName
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_arrival_op4.py --execute
```

这里使用 **`detach_arrival_op4.py`**；`detach_arrival_op3_v2.py`仍对应旧op3 v2实验。新命令默认后台完成“构建 → 算子/正确性/profile → 两次完整训练 → 模型profile”，fresh构建四套库，不依赖旧校准、旧父构建或回传日志。

默认CUDA12.8编译、CUDA12.4运行、`jobs=2`、Megatron为仓库相邻目录 `../Megatron-LM`，因此原 `/workspace/TempName` 环境无须在命令中重复这些参数。`CC/CXX/CUDAHOSTCXX`继承当前容器的编译器配置；环境路径不同时可用 `--build-cuda-home`、`--cuda-home`、`--megatron`、`--jobs`覆盖。去掉 `--execute`即只读预览，不创建目录、编译或启动GPU：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_arrival_op4.py
```

默认通过双fork交给容器PID1托管；关闭终端后继续执行，容器和主机须保持运行。`--foreground --execute`可前台诊断。每个torchrun任务默认1800秒超时；需要时用 `--worker-timeout` 显式调整。任一任务失败即终止该worker进程组并保留现场，不自动重试，不重跑已有窗口。

默认在新目录构建NCCL；已有可用开发库时可增加：

```bash
--nccl-root /path/to/nccl-release
```

该目录须包含 `include/nccl.h` 和 `lib/libnccl_static.a`。可用 `--op4-build /path/to/completed/op4/build` 复用同协议完整A/D/L/F构建；不接受只编译了对象的验证目录、source-only计划、op3 A/D/O或旧C。用 `--out /path/to/new/batch` 指定全新输出目录。

默认输出 `logs/h100/phase6/arrival-op4-2x-50steps-<北京时间戳>/`，以终端打印的 `Batch:` 为准：

```text
batch-state.json                         阶段、完成状态、主验收及四项归因
handoff/container-supervisor.log         后台总日志
build/arrival-op4-manifest.json           四套源码与变换身份
operator/report.json                     算子耗时、逐位正确性、诊断估算
operator/profile-summary.json            算子实际屏障/clone/decoder证据
fullstep/report.json                     两次完整训练配对结果
fullstep/profile-summary.json            正式计时之外的真实模型执行证据
fullstep/results/                        各rank的loss、grad_norm、显存及原始记录
```

只有 `batch-state.json` 的 `status=completed` 且两项执行路径字段为true，才表示全流程完成；`target_validated=false`可以表示正常完成但没有达到1%。若最后profile失败，正式计时结果仍保存，但批次为failed，不能只看已有target字段就认定完整验收通过。

回传整个批次目录；如不便传大体积构建，可保留全部plan、scripts、rank JSON、trace、report及handoff记录，省略build二进制。离线重算不声称重新核验未回传动态库：

```bash
python3 tools/h100/phase6/arrival_op4_bench.py report /path/to/batch/operator --offline
python3 tools/h100/phase6/arrival_op4_fullstep.py report /path/to/batch/fullstep --offline
```

## 7. 实现文件与本地验证

| 文件 | 内容 |
| --- | --- |
| [arrival_op4_sources.py](../../../../tools/h100/phase6/arrival_op4_sources.py) | A派生的生命周期变换，复用原warp decoder |
| [arrival_op4_build.py](../../../../tools/h100/phase6/arrival_op4_build.py) | 四套独立源码、来源核验、完整构建与CPU/SM90编译入口 |
| [arrival_op4_bench.py](../../../../tools/h100/phase6/arrival_op4_bench.py) | 算子数值、异步复用与输出保留检查、计时及执行路径profile |
| [arrival_op4_fullstep.py](../../../../tools/h100/phase6/arrival_op4_fullstep.py) | 独立训练快照、两次配对统计、模型profile及离线报告 |
| [detach_arrival_op4.py](../../../../tools/h100/phase6/detach_arrival_op4.py) | 一键后台入口、失败停止、完整验收 |

2026-10-21本地验证：

- **74项op4 CPU测试＋6项既有warp decoder数学回归通过。** 包含双工作区异步复用模型、单工作区反例、真实C++配置函数编译执行、四组源码边界及host mapping、输出保留/工作区别名拒绝、两次训练验收与归因隔离、丢rank/轨迹/运行库漂移拒绝、profile执行次序、离线重算及Python虚拟环境路径。见 [CPU验证登记](../../../../outputs/h100/phase6/arrival-op4-validation-20261021/cpu-validation.json)、[测试日志](../../../../outputs/h100/phase6/arrival-op4-validation-20261021/cpu-tests.log)、[decoder回归](../../../../outputs/h100/phase6/arrival-op4-validation-20261021/decoder-tests.log)。
- **36条真实编译/检查命令、16个对象全部通过**：四组各自的host mapping、SM90 mapping/runtime、完整GEMM模板实例与C++ wrapper。使用CUDA Toolkit12.8及PyTorch2.6.0+cu124头文件；`completed=true,gpu_used=false,linked_runtime=false`。见 [SM90编译登记](../../../../outputs/h100/phase6/arrival-op4-validation-20261021/compile/validation.json)、[编译日志](../../../../outputs/h100/phase6/arrival-op4-validation-20261021/compile/compile.log)、[四组源码登记](../../../../outputs/h100/phase6/arrival-op4-validation-20261021/build/arrival-op4-manifest.json)。
- 新入口无副作用预览、Python语法、本文本地链接检查通过。见 [启动预览](../../../../outputs/h100/phase6/arrival-op4-validation-20261021/launch-preview.txt)。
- 独立审查将新的解析器用于32份历史A/D算子trace和8份历史A模型trace，原样解析通过；这仅说明解析器兼容已有实际trace，不能当成op4的GPU执行证据。

任何CPU检查或SM90对象编译都不等同于八卡运行、完整动态库链接或性能验收。
