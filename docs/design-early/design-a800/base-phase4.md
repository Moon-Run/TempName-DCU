# 阶段4：原始 Flux + TACO 全量远端压缩 baseline

## 融合改进v2：warp并行编码与紧凑暂存（2026-10-03）

已实现并在179147的独立内部step `179147.12`中验证，原驻留step `179147.1`保留。仅MLP前向远端贡献量化、attention保持BF16原始Flux映射，无tile重排；算法、packet协议、解码和反向近似均沿用下文定义。

- 原来一个128线程CTA逐个编码fragment的16组，改为每个warp负责一组128元素、每线程持有4个元素；4个warp同时处理4组，共4轮。
- Hadamard前5级使用warp shuffle，最后2级在每线程4个寄存器之间完成蝶形运算。统计也在warp内归约；编码函数不使用CTA同步或共享scratch。
- fragment暂存从128×128 BF16（32 KiB）缩为16×128（4 KiB），取消active-row数组；每个回调只保留暂存完成和复用前的2次CTA同步。host布局检查验证所有8个fragment的压缩行坐标与逆映射。
- 编码公式和尾部补零语义相同，统计求和顺序变化不承诺逐位一致。保留独立codec参考、BF16误差预算和生命周期检查。

新构建：[taco-fused-warp-20261003](../../../outputs/a800/phase4/taco-fused-warp-20261003/)。9项CPU检查及80项GPU算子检查通过；最大relative-L2：相对BF16为0.022803、相对codec参考为0.000668，预算仍为0.05/0.01。实际选用GEMM仍为255寄存器，编译spill stores/loads由旧版16字节变为28字节，不能仅据暂存减少就承诺加速。

### v2五组完整step结果

2026-10-03T16:25:45.741445+08:00至2026-10-03T16:59:13.673328+08:00完成。每轮10个位置平衡块、五组策略、每窗口10步预热＋20步计时，第二轮反序。共100窗口、400份rank记录、2000个全局optimizer计时step，每策略400步。两轮在同一分配内独立进程，不是独立Slurm作业。

[完整报告](../../../logs/a800/phase4/taco-warp-e2e-tp4-20261003/report.md)、[核验记录](../../../logs/a800/phase4/taco-warp-e2e-tp4-20261003/verification.json)、[profile分解](../../../logs/a800/phase4/taco-warp-e2e-tp4-20261003/profile-components.csv)。

| 策略 | 第一轮ms/step | 第二轮ms/step |
| --- | ---: | ---: |
| 原始Flux BF16 | 203.404 | 203.471 |
| 新融合构建关闭量化 | 204.002 | 204.025 |
| 独立编码 | 212.465 | 212.375 |
| 旧融合 | 231.867 | 231.935 |
| 新融合v2 | 211.950 | 211.725 |

正值表示配对耗时下降，区间按每轮10个完整配对块bootstrap；不以两个中位数相除替代配对统计。

| 新融合v2相对基线 | 第一轮下降% [95%区间] | 第二轮下降% [95%区间] |
| --- | --- | --- |
| 旧融合 | +8.622 [+8.567, +8.683] | +8.715 [+8.665, +8.764] |
| 独立编码 | +0.282 [+0.217, +0.362] | +0.318 [+0.248, +0.384] |
| 原始Flux BF16 | -4.177 [-4.243, -4.093] | -4.089 [-4.151, -4.032] |
| 新融合构建关闭量化 | -3.858 [-3.946, -3.763] | -3.787 [-3.852, -3.722] |

**结论：** 融合v2相对旧融合的完整step耗时下降8.622% / 8.715%，相对独立编码下降0.282% / 0.318%；相对原始Flux仍慢4.177% / 4.089%。上述方向在两轮中重复，区间见表。改进了融合实现，但没有证明当前量化方案优于BF16原始Flux。

五组模型smoke/profile通过；每rank均为12个attention BF16目标模块和12个MLP目标模块。量化组每完整step有48次DQ，独立组另有48次编码，新旧融合均无独立编码launch。全部计时记录loss/梯度有限、零跳步、零fallback；跨组/跨轮初始化、token、GPU及脚本/构建哈希核验通过。反向近似及真实数据收敛限制仍适用。

单次profile取同一个rank 0，GEMM累计包含attention和MLP共96次调用；下表只是kernel时间和，不是完整step关键路径。

| 策略 | GEMM（融合组含编码）ms | 独立编码ms | 解码归约ms | GEMM launch总共享内存bytes |
| --- | ---: | ---: | ---: | ---: |
| 独立编码 | 7.430 | 7.301 | 2.527 | 49152 |
| 旧融合 | 34.217 | 0.000 | 2.595 | 82976 |
| 新融合v2 | 14.097 | 0.000 | 2.469 | 53248 |

原驻留step `179147.1`和外层batch保留；新旧构建均未覆盖，数据/构建继续被Git忽略。


## 已完成：MLP-only分离/融合量化端到端（2026-10-03）

用户确认仅MLP量化，attention保持BF16原始Flux；本实验完全不加入远端优先、交替、到达优先或其他tile重排。具体量化位置为12层模型`mlp.linear_fc2`前向GEMM-RS的全部远端贡献，本地贡献保留BF16。

原文的“原始Flux＋TACO全量远端baseline”本身就是融合实现。为区分用户要求的两条路径，本轮新增独立编码对照，而非把同一个库换名计时：

| 策略 | MLP前向 | attention | 作用 |
| --- | --- | --- | --- |
| `original` | stock Flux BF16 | stock Flux BF16 | 未量化基线 |
| `original_matched` | 融合TACO构建，关闭量化 | 同构建BF16原始映射 | 检查编译后kernel资源与分支成本 |
| `taco_separate` | 原始Flux GEMM产生本地BF16贡献→独立TACO编码/FP8 scatter→DQ+ring归约 | BF16原始Flux映射，不量化 | 分离Q/DQ对照 |
| `taco_fused` | GEMM epilogue内TACO编码/FP8 scatter→DQ+ring归约 | BF16原始Flux映射，不量化 | 融合对照 |

两条TACO路径均使用本文相同的E4M3/128、自适应scale、归一化Hadamard、packet布局、归约顺序与同步协议。分离路径复用源rank自己的BF16工作区，GEMM完成后才启动独立编码scatter kernel，远端BF16贡献不会先跨卡传输。没有先运行完整BF16 RS再对最终输出量化。

attention无量化指保持其BF16算法与原始tile映射；融合编译可能改变共享kernel的资源占用，不能宣称它与stock Flux具有完全相同的机器码或occupancy。因此同时报告stock Flux与同构建关闭量化控制，避免把整个编译变化都归因于量化。分离/融合对比仍包含各自实际编译资源、访存、配置、Q/DQ和两次barrier成本。

模型接入使用显式autograd适配器：量化只在前向通信发生，反向仍采用相同BF16 AllGather和本地dgrad/wgrad，是对codec的直通近似，不是量化/scale选择的精确导数。输出clone和原有适配成本各组均保留。有限loss/梯度不替代配对梯度误差预算或真实数据收敛验证。

四组均重新计时，12层/H=2048/FFN=8192/S=2048/TP4/global batch=4/BF16；每轮8个位置平衡块，每窗口10步预热＋20步完整optimizer step，第二轮反序，取四rank最慢墙钟。两轮都在179147内，不是独立Slurm作业。运行数据：[taco-e2e-tp4-20261003](../../../logs/a800/phase4/taco-e2e-tp4-20261003/)，[进度](../../../logs/a800/phase4/taco-e2e-tp4-20261003/campaign-state.json)；代码：[e2e入口](../../../tools/a800/phase4/e2e/)。结果统一在logs，构建缓存在outputs，均已gitignore；原外层驻留循环保持存活。

### 本轮已通过的检查

9项CPU检查通过。原融合构建先行在GPU通过80项检查；新分离与新融合构建又分别通过四rank共80项检查，覆盖零/随机/尖峰/重复调用、N=128/136/256、实际模型shape及输出生命周期。两者相对BF16参考的最大relative-L2均约0.022803，相对独立codec参考最大约0.000668，低于预设0.05/0.01预算。见[融合预检](../../../logs/a800/phase4/taco-e2e-tp4-20261003/results/codec-fused/)、[分离预检](../../../logs/a800/phase4/taco-e2e-tp4-20261003/results/codec-separate/)。

四组模型smoke/profile均通过。每个量化策略每rank的12个attention目标模块标记为BF16、12个MLP为对应codec；完整step profile实际检出96次Flux GEMM、48次TACO DQ，分离路径另有48次独立编码，融合路径独立编码次数为0。stock和关闭量化控制的TACO kernel次数为0。

分离构建新增`FLUX_TACO_SEPARATE`：epilogue将远端贡献暂存源rank的原BF16工作区，`forward_gemm_impl`在同一stream上追加`taco_encode_scatter_kernel`，原group barrier随后发布packet。融合路径保留原内联编码；正常未启用实验宏的Flux路径不变。`taco_placement()`导出让Python检查实际加载的是哪种实现。新增源码、构建和模型脚本均有独立快照与哈希。

### 单次profile诊断（不替代正式计时）

从独立模型preflight的完整step trace提取。每列先在单rank内累加对应kernel持续时间，再取四rank最大值；各列最大值可能来自不同rank，不能相加当作完整step时长或严格关键路径分解。所有组每rank均为96次Flux GEMM；量化组另有48次DQ，分离组另有48次编码。

| 策略 | GEMM kernel时间和，ms | 独立编码时间和，ms | DQ时间和，ms |
| --- | ---: | ---: | ---: |
| 原始Flux | 8.980 | 0 | 0 |
| 融合构建关闭量化 | 9.908 | 0 | 0 |
| 分离TACO | 7.485 | 7.408 | 2.530 |
| 融合TACO | 34.180 | 0 | 2.604 |

融合组的GEMM时间已包含内联编码，不能据此说纯矩阵乘法慢了相同倍数。该诊断支持重点检查融合epilogue的逐行编码、同步与资源成本，但不是缓存/occupancy瓶颈的独立证明。原始数据见[profile-components.csv](../../../logs/a800/phase4/taco-e2e-tp4-20261003/profile-components.csv)。

## 本轮四卡量化端到端结果

2026-10-03 15:04:16–15:26:20（北京时间）完成，内部step `179147.11`。两条codec各80项GPU检查、四组模型smoke/profile和两轮计时均通过。正式计时共64窗口、256份rank记录、1280个全局optimizer step，每策略320步；两轮同一分配内独立进程，第二轮反序。

[完整报告](../../../logs/a800/phase4/taco-e2e-tp4-20261003/report.md)、[记录核验](../../../logs/a800/phase4/taco-e2e-tp4-20261003/verification.json)、[第一轮原始汇总](../../../logs/a800/phase4/taco-e2e-tp4-20261003/results/model-1/summary.csv)、[第二轮原始汇总](../../../logs/a800/phase4/taco-e2e-tp4-20261003/results/model-2/summary.csv)。

| 策略 | 第1轮ms/step | 第2轮ms/step | 第1轮tokens/s | 第2轮tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 原始Flux（BF16） | 204.127 | 203.487 | 40,132 | 40,258 |
| 融合构建关闭量化 | 205.367 | 204.634 | 39,890 | 40,033 |
| 原始顺序Flux＋独立TACO | 213.263 | 212.470 | 38,413 | 38,556 |
| 原始顺序Flux＋融合TACO | 232.486 | 232.008 | 35,237 | 35,309 |

正值表示配对耗时下降，负值表示耗时增加；95%区间按每轮8个完整配对块重采样，不能以表中中位数相除重算。

| 对比 | 第1轮下降% [95%区间] | 第2轮下降% [95%区间] |
| --- | --- | --- |
| 融合构建关闭量化 / 原始Flux（BF16） | -0.572 [-0.667, -0.451] | -0.537 [-0.615, -0.463] |
| 原始顺序Flux＋独立TACO / 原始Flux（BF16） | -4.447 [-4.494, -4.404] | -4.395 [-4.477, -4.311] |
| 原始顺序Flux＋融合TACO / 原始Flux（BF16） | -13.853 [-13.929, -13.768] | -13.947 [-14.076, -13.804] |
| 原始顺序Flux＋融合TACO / 融合构建关闭量化 | -13.205 [-13.279, -13.133] | -13.337 [-13.414, -13.256] |
| 原始顺序Flux＋融合TACO / 原始顺序Flux＋独立TACO | -9.005 [-9.058, -8.952] | -9.149 [-9.199, -9.092] |

**结论：** 原始顺序Flux＋独立TACO相对原始Flux（BF16）为-4.447% / -4.395%，两轮区间均为负，重复退化。原始顺序Flux＋融合TACO相对原始Flux（BF16）为-13.853% / -13.947%，两轮区间均为负，重复退化。原始顺序Flux＋融合TACO相对原始顺序Flux＋独立TACO为-9.005% / -9.149%，两轮区间均为负，重复退化。

全部计时窗口的loss/梯度有限，optimizer跳步与fallback均为0。逐rank模块路由、跨轮初始化/输入/GPU一致性和实际反序已核验；损失、梯度范数及显存见完整报告。逻辑远端字节减少46.875%不能等同于模型加速。codec的独立误差预算通过不等于真实数据收敛或完整梯度精度预算通过。

所有策略固定原始tile顺序，量化仅在MLP前向远端贡献发生。attention保留BF16原始Flux映射，但各编译构建的kernel资源可能不同；同构建关闭量化控制用于拆分此项，不能忽略该控制而把编译差异全部算作量化收益。反向采用BF16直通近似。

179147外层batch和原驻留step `179147.1`保留，内部测试结束不释放GPU分配。

### 为什么初版融合比独立编码慢

两轮完整step中，初版融合相对独立编码版分别增加约9.005%和9.149%的配对耗时。这是初版实现的重复退化，不能推导为融合编码本身没有优化空间；改进版见页首v2。

诊断profile取同一个rank 0比较，避免拼接不同rank的最大值：独立版GEMM累计7.258 ms、编码7.328 ms、解码2.530 ms；融合版GEMM（包含编码）累计34.180 ms、解码2.596 ms。GEMM统计包含attention和MLP共96次调用，编码/解码各48次；这些是插桩窗口的kernel累计时间，不是完整step耗时或严格关键路径分解。主要退化集中于融合GEMM＋编码区域，解码差异很小。原始数据见 [profile-components.csv](../../../logs/a800/phase4/taco-e2e-tp4-20261003/profile-components.csv)。

源码和编译记录给出以下原因线索：

- **编码调度粒度变粗。** 独立版由一个128线程CTA处理一组128元素，各组可以独立调度。融合版 `taco_end_step` 先暂存fragment，再遍历active rows；布局检查确认每个fragment含16组，它们由同一个GEMM CTA逐组编码。不同CTA仍可并行，但组内工作绑定到资源较重的GEMM CTA，并延长其占用时间。
- **融合后仍有布局转换和多次同步。** 当前声明128×128 BF16共享暂存数组（32 KiB），另有active-row标记和codec scratch。每组编码包含两次统计归约、128点Hadamard和FP8转换；沿源码路径约有10次CTA同步，16组逐个执行约160次，另有外层暂存同步。这是源码同步次数，不等于实测stall次数。
- **编译资源压力增大。** 本轮选用的128×128×32、3-stage融合kernel使用255个寄存器并出现16字节spill stores/loads，独立版GEMM为254个寄存器且无spill。尚无硬件计数器证据，不能仅据254→255就断言occupancy下降，也不能给spill分配确定的毫秒开销。同一融合构建关闭量化仅慢约0.54%–0.57%，不足以单独解释启用量化后的全部退化。

融合省下中间BF16写回/读取和每step 48次独立编码launch，但两版使用相同codec、packet字节量和解码流程；统计、Hadamard和量化计算没有消失。目前证据支持优先检查串行编码、布局转换和同步成本。具体贡献仍需硬件计数器或受控消融验证。

据此提出并在页首v2实施的改进：在保持128元素分组及数值语义的前提下，让多个warp并行处理不同组，减少跨warp/整块同步；将32 KiB完整tile暂存改为当前fragment所需的约4 KiB。直接适配epilogue寄存器布局仍未实现。新融合与旧独立编码的对比包含编码调度及暂存实现变化，不是仅改变kernel启动数量的单因素消融；是否获益以各自完整step实测为准。

## 初始融合原型的设计与历史构建记录

更新日期：2026-10-03。按用户最新要求，等待4卡/8卡资源期间先实现融合代码。本阶段固定为 **BF16 GEMM + 原始 Flux 映射 + TACO 全量远端通信压缩**，不加入远端优先、交替、到达优先或选择性量化。已有排队实验的入口、冻结脚本和二进制不变。

## 1. Baseline 的确切范围

策略名为 `original_taco_all_remote`。所有远端输出贡献都走相同的 TACO 编解码，本地贡献保留 BF16；这对应 [base.md](base.md) 中的“全量远端压缩”对照，不按预测、tile位置或数值大小选择是否压缩。

参考代码是 [COCCL_TACO 的 taco.cu](../../../../COCCL_TACO/src/device/compress/taco/taco.cu) 和 [ReduceScatter 配置](../../../../COCCL_TACO/src/device/compress/configs/taco/taco_RS.config)。复用的是128元素分组的自适应缩放、归一化 Hadamard 和 E4M3 编解码公式；通信仍走 Flux 的 NVLink/IPC scatter，不替换成 COCCL 的集合通信算法，也不将 NCCL、SDP4Bit 或 TAHQuant 当作 TACO。

固定参数：E4M3、有限值饱和、group_size=128、target_range=448、lambda=1e-6、Hadamard开启。本版没有E5M2、格式自适应或在线参数选择。来源与BSD许可证保存在 [TACO_LICENSE.txt](../../../tools/a800/phase4/TACO_LICENSE.txt)，构建manifest记录参考源码哈希。

对一组输入 `x`，先算 `s_a=clip(448/sqrt(mean(x²)+1e-6), 1e-3, 1e3)`，再做 `z=H128(s_a*x)/sqrt(128)`；取 `s_q=clip(max(max(abs(z)),1e-12)/448, 1e-12, 1e6)`，把 `z/s_q` 饱和舍入为E4M3。解码为 `H128(q*s_q)/sqrt(128)/s_a`，转回BF16。接收端按原 `ring_reduction=True` 的来源顺序 `rank+1,…,rank` 做BF16逐次加法。

## 2. 融合在哪里

1. GEMM仍为BF16输入、FP32 accumulator，epilogue先完成原有BF16输出转换。
2. 远端分支在 **同一个 GEMM/Stream-K epilogue** 内完成自适应统计、Hadamard、量化，直接通过peer指针写FP8 payload和两份FP32 scale；不再写远端BF16副本。本地分支保留原BF16 store。
3. 原有GEMM完成后的group barrier保证贡献写入完成。接收端用一个 `taco_decode_ring_kernel` 完成读取、解码、逆Hadamard和最终归约，不产生完整的DQ中间矩阵。
4. 解码归约后再做一次group barrier，保证所有接收者读完旧packet后，下一次调用才能重用它。这个同步属于完整算子成本，不可从正式计时中剔除。

这不是“先运行完整BF16 GEMM-RS，再压缩已经通信完成的输出”，也不是只加入独立Q/DQ kernel的分离方案。

实现文件：

| 文件 | 作用 |
| --- | --- |
| [epilogue_evt.hpp](../../../src/gemm_rs/epilogue_evt.hpp) | 在 `FLUX_TACO_BASELINE` 开关下接入远端融合编码；未定义宏时保留原路径 |
| [taco_codec.cuh](../../../src/gemm_rs/taco_codec.cuh) | 128线程协作的统计、Hadamard、E4M3编码和解码 |
| [taco_runtime.h](../../../src/gemm_rs/taco_runtime.h)、[taco_runtime.cu](../../../src/gemm_rs/taco_runtime.cu) | 精确shape的packet协议、线程局部配置与DQ+ring归约kernel |
| [gemm_reduce_scatter.cc](../../../src/gemm_rs/ths_op/gemm_reduce_scatter.cc) | 宏开关下的配置校验、接收归约和缓冲区重用同步 |
| [gemm_rs_taco.py](../../../python/flux/gemm_rs_taco.py) | 独立 `GemmRSTaco` 前向入口，拥有IPC packet并在每次调用后复位配置 |
| [build.py](../../../tools/a800/phase4/build.py) | 单独编译CUDA注册、运行时和C++ wrapper，复用已验证的IPC修复对象；不覆盖历史库 |

配置只在调用线程中临时启用，并按值进入CUDA kernel参数。Python入口绑定创建时的device/stream，检查shape、dtype、连续布局和加载库，拒绝CUDA Graph及隐式autograd使用；异常时也复位配置。它是阶段4的前向算子入口，完整模型autograd接入现已在本页上方的MLP-only实验中实现，反向近似与验收范围以上方说明为准。

## 3. Fragment 与packet布局

首版限定A800/SM80、单节点NVLink、TP=2/4/8、BF16、`ring_reduction=True`、无bias、无fuse_reduction、非转置权重接口。要求 `M % (128*TP) == 0`、`N % 8 == 0`、`K_local % 32 == 0`。

只接受128×128×32 tile、64×64×32 warp、128线程、8元素向量访问的实际CUTLASS epilogue布局。若实际dispatch选择其他布局，host参数构造直接报错，不静默回退。原始swizzle文件必须与冻结的 `original` 构建逐字节一致。

[check_layout.cc](../../../tools/a800/phase4/check_layout.cc) 使用实际CUTLASS坐标划分验证：一个tile的8个fragment互不重叠，每个fragment包含16条完整128元素行，总计覆盖16384元素一次。因此每个Stream-K归约回调可以独立压缩其16行，不等待其他CTA产生完整tile。编码用共享内存暂存当前fragment涉及的行；新增共享内存、寄存器和同步都算融合成本。

每个接收rank为每个source预留独立slot：`[G×128 bytes FP8][G×float quant_scale][G×float adaptive_scale]`，其中 `G=(M/TP)*ceil(N/128)`。本地source的packet slot不使用。当前保留原BF16工作区分配，新增packet；不能宣称峰值显存下降。

满组远端传输从256字节变成136字节，逻辑字节下降 **46.875%**，不是50%。N尾块补零后仍发送完整128个Hadamard系数，DQ后再裁剪；原TACO代码按原始count截掉补零位置的系数会破坏逆变换，本实现没有照搬这一行为。分组按tile的行定义；N不整除128时不与COCCL扁平分组流逐字节兼容，padding成本也计入。例如N=136时每行两组272字节，等于该行BF16的272字节，不能声称这种shape节省了字节。

## 4. 检查、构建与GPU预检

已完成8项CPU检查，其中6项数值/布局参考检查覆盖Hadamard正交/可逆、零值/脉冲、跨尺度随机误差、非有限值拒绝、尾组系数不能丢失、含两份scale和padding的字节账。另外2项入口检查覆盖非法shape和子组rank拒绝。实际CUTLASS布局的host检查通过；CUDA helper已按SM80编译。完整融合CUDA注册、device link和C++ wrapper现已编译/链接通过，产物位于 [taco-build](../../../outputs/a800/phase4/taco-build/)，命令和源码/库哈希见 [manifest.json](../../../outputs/a800/phase4/taco-build/manifest.json)。已核对TACO导出符号、原始swizzle一致性，以及排队TP4/TP8快照哈希均未改变。

编译资源检查显示：128×128融合GEMM使用255个寄存器、33824字节静态共享内存，另有16字节spill stores/loads；DQ归约使用47个寄存器、536字节静态共享内存。GEMM原有动态共享内存另计。这些是初次编译的资源记录；它们可能影响occupancy与性能，本轮已加入同构建关闭量化控制并开始GPU实测。

CPU检查与首次构建（输出目录必须不存在；重建请使用新目录）：

```bash
outputs/a800/venv/bin/python -m unittest discover -s tools/a800/phase4 -p 'test_*.py'
python3 tools/a800/phase4/build.py --out outputs/a800/phase4/taco-build
```

登录节点没有 `libcuda.so.1`，因此直接加载完整Flux Python包会因驱动库缺失而失败；编译/链接通过不等于GPU可运行。当时尚未进行GPU验收；本轮A800数值、生命周期与同步预检已完成，见页首。使用驱动链接stub的host导入尝试在45秒后超时，也未记为通过；汇总见 [validation-summary.json](../../../outputs/a800/phase4/validation-summary.json)。

当前常规CMake构建不默认打开实验宏。阶段4必须使用该独立入口；它复用冻结基础对象，只重编译本策略必需的部分，并保存overlay、命令、源码/库哈希、原始swizzle哈希和许可证。默认Flux库不能直接使用 `GemmRSTaco`。

GPU资源可用后，先在明确分配的作业内做预检，**本次代码工作没有提交新的GPU作业或接管4卡/8卡已有实验**：

```bash
# 在已分配的Slurm作业内，从TempName根目录执行。
TACO_BUILD="$PWD/outputs/a800/phase4/taco-build/taco"
export PYTHONPATH="$TACO_BUILD/python"
export LD_LIBRARY_PATH="$TACO_BUILD:/data/apps/cuda/12.8/lib64"
export FLUX_FORCE_NVLINK=1
outputs/a800/venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  tools/a800/phase4/validate.py outputs/a800/phase4/preflight-tp4
```

[validate.py](../../../tools/a800/phase4/validate.py) 覆盖零值、随机值、尖峰、N=128/136/256和重复调用，核对旧输出生命周期；分别比较BF16参考和独立TACO tensor参考，并单独profile确认GEMM与DQ归约kernel。预先设定relative-L2上限：相对BF16为0.05，相对codec参考为0.01；max-abs全部记录，有限值必须通过。这是压缩原型预检预算，不沿用BF16重排的逐元素0.02容差，也不代表训练收敛预算已通过。

参考实现返回的饱和计数是tensor参考中的统计，不是融合kernel实测计数。GPU预检必须收齐全部rank并通过，之后再做包含额外同步、packet访存和Q/DQ的完整算子计时。独立Q/DQ成本、与GEMM并发干扰、全量FP32误差对照、模型梯度和训练误差预算仍待资源到位后补齐。

## 5. 与后续阶段的边界

初始阶段只提供 `original_taco_all_remote` 融合baseline；现已补入分离路径及MLP-only模型入口，见页首。远端/交替/到达优先映射、选择性FP8和联合策略均未接入；阶段5仍未开始。已有tile重排作业只加载各自冻结产物，不会自动获得本次压缩代码。

在GPU预检与无插桩完整算子对照完成前，不宣称融合已经运行正确、性能提高或模型精度通过。若Q/DQ、共享内存和同步抵消字节收益，应保留负结果，不把通信字节下降等同于加速。
