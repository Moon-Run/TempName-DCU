# H100 op4 v2：按归约组分配量化预算

更新：2026-10-21。**本轮的新机制是“归约组闭包量化”：把有限量化预算集中到同一接收tile的多个来源，使它们共享解码中的逆变换。** 选择单位从独立的 `[source,tile]` 改为接收端归约组，不预测谁先到，也不增加提前消费协议。代码与一键入口已实现。**最新回传：算子收益的95%区间均跨零；模型在正式计时前未通过数值门禁，尚无整步性能结论。详见 [base-op4.md](base-op4.md)。** 下文保留原设计与预注册验收要求。

上一版op4只优化双工作区、配置和clone，不符合本次“寻找可归纳的新机制”的要求，已归档到 [op-4-engineering.md](op-4-engineering.md)。旧实现及验证产物保留，但不再作为当前方案；新入口为 **`detach_arrival_op4_v2.py`**。

## 1. 为什么换方向，以及新的研究问题

现有结果不支持继续把到达优先作为当前配置的主要收益来源：旧离线重排确实执行，但全阶段发布屏障使中间tile提前到达未必缩短算子；op3 v2已产生43/64次有效提前消费，O/A两次却分别慢0.928%、0.844%，O/S两次也显著为负。见 [最新实测](base-op3.md)、[旧调度分析](../消融实验/消融实验7-改进到达优先.md)。消融7实际已完成三次，且B回退identity，应以 [结果页](../消融实验/base-phase6-ablation7.md) 为准。

这不能证明所有到达感知算法不可行，但说明在本路径上，继续改变到达时刻的可兑现空间有限。新问题改为：

> 在量化贡献数、各来源预算、各链路字节数相同的条件下，应该把预算分散到许多输出tile，还是集中到少数接收端归约组，才能减少实际解码工作？

原选择性量化根据每个贡献的到达评分独立选择，容易让少量FP8来源散布到很多tile。它只约束“压缩多少贡献”，没有约束“有多少输出组因此启动解码变换”。op4把后者纳入决策目标。这是预算组织策略的变化，能够与交替生产顺序组合；不是把warp实现、双缓冲或删clone重新命名。

## 2. 可归纳的核心：把解码固定成本集中摊销

令接收tile `t` 被量化的远端来源集合为 `Q_t`，数量为 `k_t`。TP8中 `0≤k_t≤7`，本地来源始终BF16。固定量化贡献预算为：

\[
B=\sum_t k_t.
\]

如果逐来源解码，逆H128的数量正比于B；如果同一输出组的量化来源可以共享一次逆变换，数量正比于：

\[
G=\sum_t\mathbf 1[k_t>0],\qquad G\ge\left\lceil B/7\right\rceil.
\]

**闭包选择令 `k_t∈{0,7}`，在预算可整除时达到这个工作量下界。** “闭包”指一个被选中的tile的七个远端来源全部进入量化集合；并不表示七源闭包是数学必要条件，任何同tile的多个量化来源都可以共享逆变换。

本轮固定每来源量化56/448个远端tile，即 **1/8预算**，不用历史到达分数，不与原C的1/64预算混用。每个接收rank拥有64个tile，共56份量化来源贡献：

| 同预算布局 | 接收rank的tile分布 | 逐来源解码所需tile逆变换 | 合并解码所需tile逆变换 |
| --- | --- | ---: | ---: |
| 散布S | 56个tile各量化1个来源，另8个tile全BF16 | 56 | 56 |
| 闭包C | 8个tile各量化7个远端来源，另56个tile全BF16 | 56 | **8** |

一个128×128 tile有128行独立H128组，因此每rank每投影从 **7168次逆H128降到1024次**，相应部分工作量减少6/7。两种布局的编码次数、FP8载荷和BF16载荷相同。**这不是算子或完整step七倍加速的预测**：GEMM、通信、来源读取、尺度处理和未量化归约仍然存在，工作量下界也不是并行执行时间的下界。

简化成本模型可写为：

\[
T_{consume}\approx B\,c_{load/scale}+G\,c_{inverse}+c_{BF16}+c_{other}.
\]

本轮验证的是改变G是否真正有价值，而不是默认这几个成本能够完全线性相加。

## 3. 为什么多个来源可以共享一次逆变换

沿用TACO的H128/E4M3编码。令H为归一化Hadamard矩阵，`z_s`为FP8解包后的系数，`q_s`和`a_s`为该来源、该H128组的两个标量尺度。理想实数下：

\[
\widehat x_s=H(q_sz_s)/a_s,
\qquad
\sum_{s\in Q_t}\widehat x_s
=H\!\left(\sum_{s\in Q_t}(q_s/a_s)z_s\right).
\]

不同来源可以有不同尺度，但必须先按各自 `q_s/a_s` 缩放，再合并系数；不能直接累加裸FP8值。未量化远端及本地BF16贡献在物理域累加，最后与量化部分相加。

**线性变换后移不是本轮声称的新数学发现。** [SDP4Bit（NeurIPS2024，系统优化部分）](https://proceedings.neurips.cc/paper_files/paper/2024/file/10826a1a80f816ea98d559d7c7a97973-Paper-Conference.pdf) 已利用Hadamard线性性减少重复变换；[TACO（HPDC2026，第4节）](https://arxiv.org/html/2604.24088v1#S4) 是这里复用的自适应尺度与FP8编码来源。op4要研究的是**选择性量化预算的跨来源组织结构**，以及该结构怎样激活可共享的解码成本。当前只说明这是相对本项目已有方案的新机制，不把有限文献检索写成“已经证明世界首创”。

### 数值语义明确改变

旧路径每来源解码后转BF16，每次相加再转BF16；共享逆变换不能保持这些中间舍入。新四组统一采用：

```text
量化来源：FP8→FP32，先乘各自q/max(a,1e-12)
P：逐来源逆H128 → FP32累加
M：先FP32累加系数 → 对非空集合做一次逆H128
未量化来源：BF16→FP32，按固定来源次序累加
两部分相加 → 最后一次BF16输出
```

P与M使用相同的尺度前移和最终舍入，避免只给候选换成更高精度。它们在实数上等价，在浮点上仍可能不同；S与C量化对象也不同。因此，**不能继续要求四组输出、loss逐位相同，也不能把新结果直接当作原A数值语义下的纯调度收益。** GEMM和原BF16/STE反向适配保留，前向归约协议的变化独立登记。

## 4. 严格同预算的散布控制

简单把每来源的量化tile错开，可能同时改变生产侧编码工作的集中程度。这里用一个GF(8) Latin方阵构造散布控制，固定更多边际条件。

记来源为s、目标为d、目标内tile编号为w，预先冻结相位p：

```text
r = (w + p) mod 8
闭包C：s != d 且 r == d
散布S：s != d 且 r == L(s,d)
L(s,d) = mul2(s) XOR mul3(d)
mul3(d) = mul2(d) XOR d
mul2 = [0,2,4,6,3,1,7,5]  # GF(8)，多项式x³+x+1
```

L每行、每列都是0到7的排列，且 `L(s,s)=s`。因此两种策略同时满足：

- 每来源恰量化56个远端tile；每来源—目标链路恰量化8个tile。
- 每来源每个w轮的编码tile数相同：`r==s`时为0，其他时候恰为1。基础交替每轮有8个目标槽位，所以不是通过减少某组生产侧编码工作获得优势。
- 每来源每投影逻辑远端载荷均为 **13,819,904字节**，相对全BF16的14,680,064字节少5.859375%。这是逻辑字节，不是实测NVLink计数。
- S的每个目的地只有0/1个量化来源，C只有0/7个；这正是要改变的跨来源结构。

具体轮内目标、物理数据位置与访存局部性仍然不同，不能据此声称生产时间完全相同。P解码控制和独立producer profile用来辨别这些剩余影响。相位p取0到7，每次比较每个相位出现两块，轮换所有列余数；不根据性能或误差挑一个最有利相位。

实现与完整预算核验见 [arrival_op4_v2_plan.py](../../../../tools/h100/phase6/arrival_op4_v2_plan.py)。

## 5. 四组消融：把策略和共享解码分开

| 组 | 量化布局 | 逆变换组织 | 每rank每投影逆H128 |
| --- | --- | --- | ---: |
| SP | 散布 | 每量化来源分别逆变换 | 7168 |
| CP | 闭包 | 每量化来源分别逆变换 | 7168 |
| SM | 散布 | 同输出组先合并、后逆变换 | 7168 |
| CM | 闭包 | 同输出组先合并、后逆变换 | **1024** |

四组从同一固定fused-v2源树派生，均保留**同一个warp线程布局、单通信工作区、逐次配置/复位、两道屏障及外层clone**。没有把上一版生命周期优化放入CM，也没有采用不同的GEMM形状、精度或连接数。

| 比较 | 用途 |
| --- | --- |
| **CM/SM** | 主比较：同合并归约器、同预算，闭包布局是否优于散布布局？ |
| CP/SP | 不共享逆变换时，分组布局本身的影响，包含剩余生产/访存差异 |
| CM/CP | 相同闭包mask，共享逆变换的增量 |
| SM/SP | 每tile最多一个压缩源时的零复用控制，不应被当作闭包收益 |
| `(CM/SM)/(CP/SP)` | 对数域factorial交互诊断：共享逆变换下的闭包优势是否超过P路径已有的布局效应？ |

两次独立比较都要求CM/SM点估计和95%区间下界大于0，才记为 `mechanism_validated=true`；同时单列沿用的1%目标 `target_validated`。交互、分组控制、共享收益分别报告，不能用一个更快的组替代主比较。若只有CP/SP有收益、交互不支持共享优势，应将结果解释为布局效果，不能宣称已证实完整的解码摊销解释。

本轮没有原生Megatron、旧C、旧op3或全量化旧A组，不能据此声称优于它们。每组都是新的1/8预算实验，而非复现历史C。

## 6. 正确性与研究风险

闭包可能把量化误差集中到少数token/channel tile。相同量化数量不保证相同模型误差，也不能假定不同来源的误差相互独立。当前必须同时看性能与数值证据：

1. 每组用自己的实际mask，对独立BF16 GEMM贡献进行参考重建；独立参考通过显式128×128 FP64 Hadamard矩阵逐来源求和，不复制CUDA蝶形实现。保持整投影相对BF16 L2≤5%、相对独立codec参考L2≤1%的门槛，并记录最大绝对误差、99%绝对误差和最坏tile相对误差。
2. 算子检查另外读取**实际选中packet**的FP8及双尺度，独立FP64重建完整输出；未选中packet区域不读取。此项归一化L2门槛0.5%，分母考虑来源相消。它与重新编码的参考检查都必须通过。
3. 同mask的P/M编码packet哈希必须相同；另比较固定512点输出采样的相对L2≤0.5%。不同mask不强求输出相同。全输出独立oracle、重复调用和保留输出检查共同覆盖，采样不能替代全输出正确性。
4. 真实模型smoke覆盖全部八相位、四组，共32个窗口、256份rank记录；每rank的64个目标投影都进行独立参考检查，按相位分别比较四组数值，全部通过后才正式计时；正式记录必须完整、有限、零skip/fallback，初始参数/RNG/token及硬件软件身份一致。
5. 同块跨mask训练轨迹固定门槛为 `|Δloss|≤0.01+0.02×max(|loss_X|,|loss_Y|)`，`|Δgrad_norm|≤0.001+0.05×max(|g_X|,|g_Y|)`；同mask P/M更严格，loss为`atol=1e-4,rtol=1e-3`，grad_norm为`atol=1e-4,rtol=1e-2`。逐step误差与容限写入报告，失败停止，不根据结果放宽。

这些轨迹阈值是本次预先固定的短程数值筛查，不是证明与旧实现等价，也不是证明长期收敛。真实语料长程训练、误差敏感的归约组选择以及其他预算/TP规模属于后续工作；当前不预填其结论。默认只实现TP8、M2048、N4096、K_local512/2048、1/8预算。

## 7. 执行规模与统计

算子默认16个完整四组块，共64个fresh-process TP8窗口；每角色100次预热、200次正式调用，计时四组统一包含 `forward().clone()`。包含多seed、输入/权重改变、rank0/rank7延迟生产、连续6次调用保留输出、同mask packet检查。独立profile验证每次一GEMM、一正确decoder、两barrier及一次clone，并报告producer和decoder耗时。逆H128次数来自已核验mask及算法的解析计数，不冒称硬件实测计数。

完整训练保持GPT6.7B、32层、H4096/FFN16384、S2048、TP8、BF16、micro batch1/global batch4，seed1234/token_seed4321。每step两个角色各128次目标投影，共8192token。四组统一 `CUDA_DEVICE_MAX_CONNECTIONS=8`、`NCCL_NVLS_ENABLE=0` 与同一隔离Megatron源码副本。

默认 **2次独立比较×16块×4组＝128个正式窗口、1024份rank记录、6400个累计正式optimizer step**。每窗口10步预热＋50步计时；每相位每次比较出现两块，各组执行位置均衡。所有窗口独立初始化，不是同一模型连续训练6400步。

窗口时间取八rank最大墙钟除以50，覆盖完整前后向、通信、optimizer及清梯度。下降率为 `100×[1−exp(mean(log(T_X/T_Y)))]`，按完整同相位四组块bootstrap10000次，保留所有正式样本和负结果。profile、编译、预热与参考检查不混入50步计时。算子性能估算无论正负都不拦截完整训练，只有数值、来源和执行证据失败才停止。

正式训练之后在phase0独立运行四组真实模型profile，每rank每角色128次投影，核验实际decoder、两屏障和clone，避免把微基准路径当作训练路径。

## 8. 另一台服务器的一行启动命令

同步新增工具及已有仓库依赖，在原8×H100服务器/容器的 `/workspace/TempName` 下，沿用已验证的Python和编译器环境，执行：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_arrival_op4_v2.py --execute
```

默认fresh构建SP/CP/SM/CM四库；CUDA12.8编译、CUDA12.4运行、jobs2、Megatron取相邻目录 `../Megatron-LM`。后台双fork托管，关闭终端后继续；容器及主机需保持运行。去掉 `--execute`只读预览，加 `--foreground`可前台诊断；路径不同时使用 `--build-cuda-home`、`--cuda-home`、`--megatron`、`--python`覆盖。编译器仍继承容器的 `CC/CXX/CUDAHOSTCXX`。

默认新建NCCL开发库；已有库可传 `--nccl-root`，需同时有 `include/nccl.h` 和 `lib/libnccl_static.a`。`--op4-v2-build`只接受本协议的完整四库构建，旧op4/旧op3及仅对象编译目录均被拒绝。每个torchrun任务默认1800秒超时，可用 `--worker-timeout`设置；失败保留现场并停止，不自动重试或复用已有窗口。

默认输出 `logs/h100/phase6/arrival-op4-v2-2x-50steps-<北京时间戳>/`，以实际打印的 `Batch:` 为准：

```text
batch-state.json                     完成状态、主比较、数值/预算及归因字段
handoff/container-supervisor.log     后台总日志
build/arrival-op4-v2-manifest.json    来源、四库mask/归约实现身份
operator/report.json                数值、实际packet、同预算算子比较
operator/profile-summary.json       算子执行路径和诊断耗时
fullstep/report.json                两次完整训练、各相位、轨迹容限及交互
fullstep/profile-summary.json       正式计时之外的真实模型执行证据
```

`status=completed`表示全流程完成，`target_validated=false`可以是正常完成但未达到1%；不能把未通过数值检查或最后profile失败的批次当作完整验收。回传整个批次；若省略大体积二进制，仍需保留plan、scripts、原始rank JSON、trace和report，离线复核不会声称重新加载缺失的库：

```bash
python3 tools/h100/phase6/arrival_op4_v2_bench.py report /path/to/batch/operator --offline
python3 tools/h100/phase6/arrival_op4_v2_fullstep.py report /path/to/batch/fullstep --offline
```

旧 `detach_arrival_op4.py`仅启动已归档的工程方案；`detach_arrival_op3_v2.py`仅启动旧提前消费实验。本次使用带 **`op4_v2`** 的新入口。

## 9. 实现及本地验证

| 文件 | 内容 |
| --- | --- |
| [arrival_op4_v2_plan.py](../../../../tools/h100/phase6/arrival_op4_v2_plan.py) | 闭包与Latin散布mask、相位、预算及工作量证明的穷举核验 |
| [arrival_op4_v2_sources.py](../../../../tools/h100/phase6/arrival_op4_v2_sources.py)、[decoder](../../../../tools/h100/phase6/arrival_op4_v2_decode.cuh) | 按mask选择BF16/FP8生产、逐源/合并两种归约，同一工程路径 |
| [arrival_op4_v2_build.py](../../../../tools/h100/phase6/arrival_op4_v2_build.py) | 独立四库、冻结基线与变换核验、SM90编译 |
| [arrival_op4_v2_reference.py](../../../../tools/h100/phase6/arrival_op4_v2_reference.py) | 独立FP64矩阵参考、实际packet校验及误差诊断 |
| [arrival_op4_v2_bench.py](../../../../tools/h100/phase6/arrival_op4_v2_bench.py) | 同预算算子、精度控制及执行证据 |
| [arrival_op4_v2_fullstep.py](../../../../tools/h100/phase6/arrival_op4_v2_fullstep.py) | 八相位两次训练、固定数值门槛、主比较与交互 |
| [detach_arrival_op4_v2.py](../../../../tools/h100/phase6/detach_arrival_op4_v2.py) | 新的一键后台入口 |

2026-10-21本地验证完成：

- **76项CPU测试全部通过，无跳过。** 使用PyTorch2.6.0+cu124环境并设置 `CUDA_VISIBLE_DEVICES=''`；包括四组×八相位预算/生产轮级公平性、真实C++ mask/configure函数、FP64参考与舍入反例、实际packet布局及未选中区域NaN毒值、同mask数值控制、八相位模型预检、失败拒绝及一键入口。见 [CPU登记](../../../../outputs/h100/phase6/reduction-closure-validation-20261021/cpu-validation.json) 与 [完整日志](../../../../outputs/h100/phase6/reduction-closure-validation-20261021/cpu-tests.log)。
- **33条真实编译/检查命令、16个对象通过。** 四组分别覆盖host mapping、SM90 mapping/runtime、完整GEMM模板实例及C++ wrapper；`gpu_used=false,linked_runtime=false`。见 [编译登记](../../../../outputs/h100/phase6/reduction-closure-validation-20261021/compile/validation.json)、[日志](../../../../outputs/h100/phase6/reduction-closure-validation-20261021/compile/compile.log)、[四组源码登记](../../../../outputs/h100/phase6/reduction-closure-validation-20261021/build/arrival-op4-v2-manifest.json)。
- 新入口的无副作用预览、Python语法和文档链接检查通过；预览不导入Torch/Flux或启动GPU。见 [启动预览](../../../../outputs/h100/phase6/reduction-closure-validation-20261021/launch-preview.txt)。

以上是运行前的本地验证，不等于八卡GPU数值验证、完整动态库链接或训练性能验收。2026-10-21回传实测已整理至 [base-op4.md](base-op4.md)：算子阶段完成，模型数值门禁失败，正式训练计时未开始，仍无已验证的整步加速结论。
