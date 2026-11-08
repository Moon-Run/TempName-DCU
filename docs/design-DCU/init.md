# BW1000 五组：Flux、TACO、交替顺序、到达优先与选择性量化

更新：2026-11-08。当前为五组v2协议，统一入口 `run_init.sh`。执行“环境/通信检查 → 目标机校准并冻结计划 → 五组算子正确性与计时 → GPT 6.7B TP8完整训练步 → 配对报告”。**本地完成CPU验证；尚未在BW1000编译HIP库或运行GPU性能实验。** 不填推测的加速率。

## 1. 本轮实现的范围

用户给出的 `flux` 是CUDA/CUTLASS实现，没有现成HIP后端。因此本轮的 **Flux-DCU是依据Flux计算通信重叠设计实现的DCU适配后端**：使用目标DTK PyTorch的BF16 BLAS生产大块，通过RCCL分块传输，使用HIP原生归约/codec。它不是直接运行原CUDA Flux二进制，也不是把CUTLASS模板机械替换为HIP。

协议：`bw1000-flux-dcu-five-arm-arrival-selective-v2`；执行后端：`dcu-blas-rccl-block-lanes-v2`。结果明确登记 `original_cuda_flux=false`、`cta_level_swizzle=false`，不可将本轮速度称为上游Flux在BW1000上的原生性能。

这是对 [thinking.md](../../../thinking.md) / [ideal.md](../../../ideal.md) 中问题的第一步实现：**每个生产块完成后即可进入通信，随后继续生产其他块；改变的是实际GEMM调用顺序。** 没有将交替顺序放在完整GEMM之后重排输出，也没有用串行标量HIP GEMM代替矩阵库。大块拆分与Python提交仍有成本，因此同时记录独立完整GEMM耗时，最终收益以实机为准。

| 展示组 | arm | 实际生产顺序 | 远端表示 | 接收端 |
| --- | --- | --- | --- | --- |
| A：Flux-DCU | `flux` | 按目的分区连续生产 | BF16 | 按来源0…W−1归约，每次加后BF16舍入 |
| B：Flux-DCU＋TACO | `flux_taco` | 与Flux-DCU相同 | 全远端H128/E4M3FN，本地BF16 | 先解码远端，再采用相同归约 |
| C：Flux-DCU＋TACO＋交替 | `flux_taco_interleaved` | 全rank共同的目的分区交替 | 全远端TACO | 单consumer按逻辑顺序等待 |
| D：再加到达优先 | `flux_taco_interleaved_arrival` | 交替槽位内、同目的地实测排序 | 全远端TACO | 两lane独立消费本地就绪块 |
| E：再加选择性量化 | `flux_taco_interleaved_arrival_selective` | 与D完全相同 | 选中输出块的全部远端来源TACO，其余BF16 | 与D完全相同 |

新增D组“交替＋到达优先”和E组“交替＋到达优先＋选择性量化”。五组共用块尺寸、BF16 GEMM、两条通信lane、输出所有权及来源归约次序。B/C仅改变生产顺序；D/C改变实测选择的生产顺序及分lane就绪消费；E/D保持到达计划与消费协议，只改变物理量化集合。B/C/D全量TACO必须保持相同训练轨迹，E按自己的mask独立校验。没有加入MA8/MA9/MA11。

**v2让前五组在同一套双lane基础设施下重测**，不能把旧三组单lane结果与新增两组作配对。旧v1日志不改写，原目录中的冻结源码仍可用于其历史复核。

## 2. 目标环境与自动预检

[env.md](../../../env.md)确认的是单节点8张BW系列DCU、每卡65520 MiB、80 CU、device ID `0x6320`。它没有给出实际PyTorch/HIP版本、gfx架构或peer-access矩阵；代码不把80 CU当成SM80，也不假设warp32。

用户提供的已验证镜像作为部署参考：

```text
42.228.13.241:5000/jenkins/model_test_env/vllm:0.28.1rc1-ubuntu22.04-dtk26041-py3.10-20261019-1402
```

在已分配的单节点八卡容器内运行。沿用镜像自带的DTK、HIP版PyTorch、RCCL和NumPy，**不要安装NVIDIA版PyTorch来满足依赖**。镜像标签不是运行时版本证据，脚本会保存实际版本及设备属性。无须安装Megatron、Apex、Triton或原CUDA Flux；默认直接从本仓库源码运行。

执行预检会检查/采集：

- `torch.version.hip`、实际可见设备、device属性、gfx架构、Python/PyTorch版本、现有通信与设备可见性环境；拒绝CPU/NVIDIA执行回退。
- 实际BF16 GEMM和BF16 ReduceScatter；BF16与uint8的非等长、含零split All-to-All，所有目的地都验证。HIP版PyTorch仍通过`torch.cuda`及`backend='nccl'` API调用设备和RCCL。
- 每rank可查询的peer-access信息；rank0尝试保存 `hy-smi` 的设备、PCI bus、拓扑和驱动输出。当前后端通过RCCL工作，不以P2P可用为前提，也不声称采用特定物理transport。
- 自动寻找`hipcc`，根据实际设备 `gcnArchName` 编译。若运行时不提供架构，须显式设置已确认的 `FLUX_DCU_HIP_ARCH`；不会猜测一个gfx型号。显式架构与可见设备不一致时拒绝运行。
- HIP codec原生结果与独立CPU参考，以及输入、输出、packet payload和全部scale的有效性。

`hy-smi`不可用时保留采集失败信息；设备实际身份以保存的记录核对，不把镜像名称当成硬件检测结果。软件或集合通信能力不满足时保留错误并停止，不切换CPU/FP32或另一传输协议继续计入结果。

## 3. 生产调度和通信依赖

设每rank输出行数为 `M/W`，生产块为 `chunk_rows × N`，每目的分区有 `L=M/(W×chunk_rows)` 个块。默认 `M=2048,N=4096,W=8,chunk_rows=128`，因此每目的地2块、每投影16次GEMM和16次稀疏All-to-All。

```text
基础：destination = slot // L，within = slot % L
交替：destination = slot % W， within = slot // W
物理首行 = destination × (M/W) + within × chunk_rows
```

交替公式来自 [Entwine v1第4页§3.1公式(1)](https://arxiv.org/pdf/2609.11562v1)：**所有rank共用同一映射，没有加rank偏移。** 默认目的地顺序从基础的 `0,0,1,1,…,7,7` 变为 `0,1,…,7,0,1,…,7`，将各来源的本地保留块分散在远端块之间。这次复用的是论文的共同调度思想，论文的CUDA tile通信内核和资源预算机制没有直接移植。

执行见 [runtime.py](../../python/flux_dcu/runtime.py)：

```text
producer stream：当前块BF16 GEMM → 远端块可选TACO编码 → ready事件
                     ↓ 下一块GEMM可继续
communication：      等ready → RCCL本块交换 → received事件
consumer：                                等received → 解码／固定次序归约
caller：                                                    等最终done → 使用完整输出
```

每rank共享1条producer、2条communication和2条consumer stream，共5条。两条通信lane分别使用独立RCCL communicator，在全rank相同顺序创建/销毁；64个实例不会各建一套。每实例保留独立scratch，返回输出每次独立分配；下一调用的producer等待前次done，保护packet与partial不被过早覆盖。生产映射与数据物理身份分别保存，量化始终作用于物理远端贡献。

每个块的All-to-All只有当前目的rank接收其他W−1个来源，本地贡献不经链路。lane固定为物理块的 `within % 2`，不随排序或量化改变。所有rank仍按同一个冻结块序发起collective，每次异步Work建立同步后再提交下一次；不能按各rank本地到达顺序随意发集合通信。每个lane的BF16/uint8及零split行为均在预检中验证。

A/B/C的consumer0保留逐逻辑slot等待；D/E在块所属lane消费本地块，省去非本地块和另一lane的消费等待。每块仍等待本块全部来源，来源累加次序不变。最终done显式汇合两条通信与两条消费流，之后才允许caller使用完整输出或下一调用覆盖scratch。**这是有界的分lane就绪消费，不是全局任意ready-first调度，也不自动保证RCCL能并发。** 多进程组仍遵守[PyTorch的同序提交和同步要求](https://docs.pytorch.org/docs/2.6/distributed.html#torch.distributed.new_group)。

**这是BLAS大块调度，不是CTA swizzle。** 它避免完整GEMM后才发第一笔通信，但每块仍有GEMM/codec/集合通信的提交成本。`--chunk-rows`可调整粒度；要求整除每rank输出行数，且每目的地至少两块，否则交替与基础映射无法区分。不同粒度必须用新的批次，不能混合统计。

## 4. TACO来源、算术与许可证

用户提供的 `COCCL_TACO` 当前main并不含实际TACO代码；本轮从本地已有Git对象只读提取了：

```text
Flux参考：19831ca2d820e3e782ed1d15d8b52d0898b78b26
TACO参考：ca717e05b6dd6d21fff4d2617d209c95be63f3ef（taco-new-feature）
```

没有切换或修改这两个来源仓库。TACO源码、配置、许可证与SHA256保存在 [third_party/taco](../../third_party/taco/)，Flux设计/调度参考保存在 [third_party/flux](../../third_party/flux/)。HIP实现保留BSD-3-Clause归属，见 [NOTICE](../../NOTICE)。

[codec.hip](../../src/taco/codec.hip)固定H128/E4M3FN：先按均方值计算adaptive scale，做归一化Hadamard，按最大绝对值计算quant scale，再做E4M3FN RNE饱和编码；解码执行反变换与尺度恢复，最后转BF16。使用共享内存完成H128运算和软件FP8转换，不依赖CUDA `__nv_*` 指令、warp32或目标硬件FP8格式。

```text
每来源packet = [全部FP8 payload][全部FP32 quant scale][全部FP32 adaptive scale]
128个BF16值：256 B → 128 + 4 + 4 = 136 B
```

默认每rank每次投影远端逻辑载荷：A为14,680,064 B，B/C/D均为7,798,784 B；E取决于实际物理mask，逐来源单独记账。这个46.875%减少只针对目标投影远端载荷，不包含其他训练通信和元数据；不是整步收益承诺。

## 4.1 到达优先：完整投影校准与分lane消费

新增 [planning.py](../../python/flux_dcu/planning.py) 和 [calibration.py](../../python/flux_dcu/calibration.py)。每个新批次在正式窗口前，以实际DTK、TACO、RCCL路径分别校准Attention/MLP两种shape，不读取旧H100到达表。

生产排序保持每个逻辑slot的目的地不变，只置换该目的地的物理块。候选包含基础交替、各目的地内反序、奇/偶目的地反序及循环移位，去掉重复表。**优先级由真实完整投影的净耗时选择，不把晚join时间直接当作收益分数。** 基础顺序和候选使用相同输入做AB/BA成对比较；数据生成、数值参考和校准操作不计入正式训练时间。

默认每次候选测量2次预热、5次计时，3个训练pair；只将训练集最佳候选带到另外3个确认pair。输入seed与正式训练不同，训练/确认seed区间分离。每个确认pair都须比identity快超过0.2%，否则保持identity，不再用确认结果挑第二名。这个小样本门槛是保守准入规则，不是95% CI或长期稳定性证明。

D即使生产表回退identity，仍采用两条lane独立消费，因此报告分别记录 `arrival_order_changed` 与 `consumer_policy`。当前策略对应“在实际codec上选择生产顺序＋让另一lane的就绪本地块继续消费”，没有实现无限粒度在线调度，也没有声称离线角色级计划对32层每层都最优。

## 4.2 选择性量化：输出块闭合选择与净成本校准

E使用D同一份到达顺序及同一消费协议。mask按**物理输出macroblock**绑定：选中块的全部W−1个远端来源都用TACO，未选块的全部远端来源都用BF16，本地贡献始终BF16。这既保持同一集合通信各rank的格式/split一致，也避免按来源零散量化使大量输出块进入混合来源格式路径。

选择过程固定如下：

1. 在D执行协议下建立Q0（全BF16远端）参考，分别实测只量化一个物理块的完整投影；只有正收益块进入候选排序。
2. 按实测收益排序构造若干前缀mask，再测它们的完整投影，不将singleton收益直接相加。
3. 单独测全远端量化控制组，即使预算不允许选择全量也保留它。任何候选的数值失败都会停止，不把失败隐藏为未选中。
4. 训练样本最佳mask再用独立确认pair检验；未通过门槛则回到Q0。确认样本不反过来参与选mask。

默认 `--quant-budget 0.25` 是全局远端贡献/输出块的预算上限。默认16个macroblocks，最多选4块，实际可为0—4块；每块包含全部远端来源，所以输出块比例等于全局远端贡献比例，各来源字节数可能不同。**1/64在当前16块粒度下会取整为0块，不能沿用旧实验的细粒度标签。** 预算0只允许Q0；预算1允许全量控制胜出，实际结果都必须报告。

每个rank的字节按其实际远端块计算。每选中一个不是该rank本地的块，节省 `chunk_rows×N/128×120` B；Q0和Qall的默认每来源载荷分别为14,680,064 B与7,798,784 B。E相对BF16省多少、相对Qall多多少都会同时报告，不把稀疏量化描述成比全量量化传得更少。

`selection-plan.json`保存两个角色的共同到达表、物理mask、所有候选与确认数据。计划在五组正式算子和训练之前冻结；模型64个实例按角色装入该计划，运行中不继续学习。每窗口登记整体计划hash和实际实例role hash。报告将原始各rank耗时取最大值，并用冻结规划器在CPU重放选择，要求复现同一计划；不会只相信组名或名义预算。

## 4.3 人工错峰诊断：区分“具备越过能力”和“通常更快”

D/E的算子检查另建独立实例，保持相同物理mask，以基础交替顺序执行一次正常调用和一次人工错峰调用。后者只在rank0首个lane0通信前插入有界HIP `clock64` 等待，给lane1制造可先完成的机会；普通forward、校准及正式计时绝不插入该延迟。

检查要求错峰前后输出逐位一致，并记录是否有**后提交的lane1本地块，已经完成消费，而较早的lane0通信仍未完成**。同一rank上的事件顺序可检验这一行为，不比较不同GPU的绝对时钟。未观察到越过会如实记录，仍继续真实性能测量；运行错误、死锁超时或输出错误则停止。

人工错峰不代表正常负载有同样长尾，更不代表D/E一定更快。正式报告同时保留这个诊断和未注入延迟的算子/step耗时，用于区分执行机制是否可用以及净收益是否成立。

## 5. GPT 6.7B完整训练定义

默认模型为32层、hidden4096、FFN16384、32 heads、sequence2048、micro batch1、累积4个微批、vocab33792、TP8/DP1、BF16，每步8192 token。覆盖每层Attention输出投影与MLP输出投影，共64实例；每正式step每rank各角色调用128次。

[model.py](../../python/flux_dcu/model.py)使用原生PyTorch实现训练：

- 词表分片embedding/LM head并绑权，分布式cross entropy；位置嵌入、pre-LayerNorm、因果SDPA、GELU，无dropout。
- residual/LN沿序列分片；QKV/FC1前向AllGather，反向ReduceScatter输入梯度。
- 两种row投影前向走本轮后端，反向沿用BF16 GEMM/STE，并AllGather输出梯度。
- FP32梯度累积、复制参数梯度同步、去重global grad norm及clip、FP32 master AdamW、回写BF16权重和清梯度。

实际逻辑参数 **6,590,963,712**，每rank **831,905,792**，其中复制参数9,183,232。绑权、bias、位置嵌入和优化器定义写入记录。此“GPT 6.7B”负载是本轮独立训练实现，**不是之前Megatron执行图的逐位复刻**，不得拿绝对步时跨实现比较机制收益。前后向TP数学已用双rank CPU小模型与独立dense模型验证；DCU实际算子仍待服务器检查。

默认每窗口10步预热、50步正式计时。首次预热的64个投影分别检查实际输入/partial/输出及所有packet合法性，此检查不进入正式计时；因此完整训练要求至少1步预热。正式计时包含前向、反向、梯度累积/同步、范数/裁剪、更新、回写和清梯度，不含构建、独立算子参考、profile及参数哈希。

每窗口独立初始化并使用相同固定合成微批；loss下降反映这个小数据集的训练轨迹，不代表真实语料长期收敛。B/C/D三个全量TACO组的初始化、输入、loss/grad norm轨迹和最终参数哈希必须相同；A和E分别使用自己的量化策略，不强求它们与全量TACO同轨迹。E以同mask独立参考以及相同mask不同生产顺序的逐位结果检查正确性。强制确定性算法，不支持的目标DTK算子会报错，不能悄悄放宽轨迹门槛。

## 6. 服务器执行命令

在已准备好上述DTK镜像的八卡计算节点内，进入仓库；不在登录节点发起GPU训练，不复用env.md中的历史Slurm job ID。

```bash
cd /path/to/TempName-DCU

# 只打印模型、任务和输出计划；不导入torch、不编译、不查询GPU、不写结果。
bash tools/dcu/phase6/run_init.sh --trials 1 --blocks 1

# 一键五组短测：预检 → 校准 → 五组算子 → 五组完整训练 → 报告。
bash tools/dcu/phase6/run_init.sh --trials 1 --blocks 1 --execute

```

首次迁移可先分阶段检查：

```bash
# 只验证设备、BF16、RCCL、HIP codec，完成状态为preflight_completed。
bash tools/dcu/phase6/run_init.sh --preflight-only --execute

# 先校准，再测两种真实shape的五组算子，并另录独立profile；不建完整模型。
bash tools/dcu/phase6/run_init.sh --operator-only --profile --trials 1 --blocks 1 --execute

# 小模型的完整接线诊断；报告明确不是GPT6.7B标准负载。
bash tools/dcu/phase6/run_init.sh --small-model --trials 1 --blocks 1 \
  --warmup-steps 1 --steps 2 --operator-warmup 2 --operator-iters 3 --execute

# 每trial十块构成完整位置/先后平衡；也可--trials 2独立重复。
bash tools/dcu/phase6/run_init.sh --trials 1 --blocks 10 --execute
```

校准成本或量化上限可显式调整；每次变更必须新建批次：

```bash
# 默认校准门槛/预算；通常无需显式重复这些参数。
bash tools/dcu/phase6/run_init.sh --trials 1 --blocks 1 \
  --quant-budget 0.25 --calibration-warmup 2 --calibration-iters 5 \
  --calibration-pairs 3 --calibration-min-gain 0.002 --execute

# 允许选择0至全量；与默认预算属于不同实验，不混算。
bash tools/dcu/phase6/run_init.sh --quant-budget 1 --trials 1 --blocks 1 --execute
```

校准只在每批开始运行一次，所有trial共用这份冻结计划；若需要检验校准的可重复性，请运行两个独立批次。校准sample不是正式统计样本，`--preflight-only`不会运行校准。

指定解释器、编译器和输出位置：

```bash
DCU_PYTHON=/path/to/dtk-python \
bash tools/dcu/phase6/run_init.sh --hipcc /opt/dtk/bin/hipcc \
  --trials 1 --blocks 1 --execute --out /path/to/new-dcu-batch
```

使用镜像中实际路径；无需按示例强行改成`/opt/dtk`。入口也支持 `--python`。默认继承可见设备和现有RCCL/NCCL环境，不写入NVIDIA专属配置。若需要改变通信设置、块大小或训练参数，启动新批次。

脚本前台执行，需保持Slurm分配与容器存活；可以在持久会话中运行。Ctrl-C、SIGTERM和超时只清理该controller创建的worker进程组。每个执行目录必须不存在，失败留存现场，不覆盖/续写旧批次。

## 7. 检查、计时和报告口径

每个arm/窗口使用独立torchrun进程组。先完成全部算子窗口，再进入完整训练窗口。算子默认100次预热、200次正式调用，同时保存完整GEMM-only控制耗时，便于观察大块拆分的潜在代价；这项控制不作为另一个五组arm，也不是完整投影性能。

算子门禁包含：

- 原生codec对独立CPU FP64参考；实际packet的所有scale与payload合法。
- 完整shape对独立完整GEMM＋通信＋归约参考；Flux相对BF16 L2≤1%，TACO相对BF16≤5%，相对独立codec参考≤1%。
- 输入改变与scratch复用、旧输出保留；同输入下B/C/D全量TACO输出逐位一致；E与相同mask、基础交替顺序的独立实例逐位一致。
- 完整训练的实际模块数/调用数、有限loss和grad norm、初始化/输入身份及B/C/D轨迹相同。

窗口耗时取所有rank最大墙钟：算子 `max(elapsed)/iterations`，训练 `max(elapsed)/steps`。正式计时不启用profiler，所有负收益和慢窗口保留；不同trial分别报告。

五组使用5次循环位移及其逆序，共10块；每组在每个位置出现两次，任意两组前后各5次。`--blocks 1`运行完整五个arm，但只有一个五组块，主要用于检查，不给CI。只有完成十块平衡周期（10的正整数倍）才提供完整块bootstrap的95% CI，不把八rank或窗口内step当独立样本。配对下降为 `100×[1−exp(mean(log(T候选/T参考)))]`，正值表示候选更快。

`--trials 1 --blocks 1`默认有5个完整训练窗口、250个正式optimizer step、40份正式训练rank记录，另有1个预检、1个校准worker和5个算子窗口，共12个torchrun worker。完整十块对应50个训练窗口、2500个正式step、400份训练rank记录。校准自己的算子调用不计入正式性能窗口。每窗口从同一初始化重新开始，不是单模型连续训练这些总步数。

`--profile`记录独立调用的torch trace及事件区间。通信事件区间可能包含等待，**事件区间相交不是通信kernel实际重叠的证明**；记录中 `actual_kernel_overlap_validated=false`，需结合trace查看实际kernel。不能仅凭使用五条stream或两个communicator声明重叠生效。

默认输出：

```text
logs/dcu/phase6/init-five-arm-<北京时间>/
  plan.json / selection-plan.json / source-manifest.json / completed-jobs.json
  batch-state.json
  sources/                  # 本批实际执行的源码快照
  build/                    # 实际HIP库、编译命令/日志、源码/库哈希
  results/preflight/
  results/calibrate/          # 双角色候选、训练/确认样本与原始rank时间
  results/operator/trial1/block000-<arm>/
  results/train/trial1/block000-<arm>/
    config.json / command.json / process.log / rank0.json ... rank7.json
  report.json / report.md
```

失败状态不会写成completed。报告包含各组算子/step均值、中位数、配对差异、吞吐、loss、梯度范数与显存；数据和独立源码保留用于复核。Torch峰值已经包含本后端的张量scratch，不重复相加；设备context/RCCL内部占用不等于Torch allocator记录。

```bash
# 只读重新验证结果与统计。保留sources、build、结果和可选trace，不能只拷贝report。
bash tools/dcu/phase6/run_init.sh --report --out /path/to/completed-dcu-batch
```

## 8. 本地验证与尚待实机确认的事项

2026-11-08五组v2集成验证：在已有PyTorch 2.6.0环境下设置 `CUDA_VISIBLE_DEVICES=''`，**83项CPU测试全部通过，无skip**。覆盖规划、独立确认、闭合mask、多lane依赖、模型计划接线、人工错峰门禁及校准/报告重放。此前三组v1的54项记录不作为本次新增机制的验收结果。

CPU验证覆盖：论文共同映射及实际生产调用序列、双射/目的地/字节量、稀疏All-to-All split逐对匹配、真实runtime的事件依赖模型与输出生命周期、H128参考、FP8舍入边界、报告对缺rank/错误计时/身份漂移/轨迹变化的拒绝，以及纯预览无副作用。

实际HIP源码中的标量BF16/E4M3转换被抽取后用g++编译，与独立穷举参考交叉检查；这不等于HIP kernel编译通过。使用已有PyTorch环境并隐藏全部GPU，完成CE及column/row梯度检查、双rank Gloo小模型与独立dense模型的loss/全部参数梯度/global norm对照。CPU回环socket测试仅用于数学验证。

本地命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py' -v
```

没有PyTorch时，Torch数学检查明确skip，其余标准库测试照常运行。运行全部CPU数学检查需可用CPU PyTorch及本机进程间通信。**本机没有HIP编译器/DCU，目标gfx编译、RCCL异步进展、DTK确定性SDPA、完整训练显存和性能均由上述服务器入口验证。** 当前没有承诺五组排序优劣，也不把人工错峰结果或校准胜出当作完整训练收益。

本次另通过Python/shell语法、文档本地链接、diff空白检查，以及五组 `--trials 1 --blocks 1` 无副作用预览。该预览确认12个worker任务与实际五组列表，不导入Torch、不构建、不查询GPU、不创建实验输出目录。

主要实现：[调度映射](../../python/flux_dcu/mapping.py)、[执行后端](../../python/flux_dcu/runtime.py)、[HIP codec](../../src/taco/codec.hip)、[模型](../../python/flux_dcu/model.py)、[worker](../../python/flux_dcu/worker.py)、[controller](../../tools/dcu/phase6/init.py)、[统计与复核](../../tools/dcu/phase6/report.py)。
