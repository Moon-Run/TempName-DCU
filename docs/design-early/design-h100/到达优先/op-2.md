# op2：保持 A 的量化实现，只优化生产顺序

更新：2026-10-20。本版按用户澄清重新实现：**A 固定为消融7的“交替＋全远端量化”；候选与 A 使用完全相同的编码、解码、mask、归约、工作区、输出生命周期和跨 rank 屏障，只改变生产 tile 的顺序。**

上一版 warp 解码方向偏离了这个目标，已从当前实验入口撤回；其源码与说明仅存于 [withdrawn-decoder-v1](../../../../outputs/h100/phase6/arrival-op2-cpu-20261020/withdrawn-decoder-v1/)。旧版 D/A 的潜在收益不能用于证明到达调度有效，也不计入本次1.5%目标。当前协议为 **`h100-gpt67-tp8-arrival-op2-order-v2`**，拒绝旧版 decoder 构建和选择结果。

**本轮选择用“保持分区交替的成对行调度”替代不稳定的到达残差排序，并用同量化实现的 GPU 算子测量选择参数。** 本机没有GPU，新代码只完成CPU及编译检查；GPU测试由用户执行，目前没有新的加速率，不能保证达到1.5%。

## 1. 对旧到达优先的判断

[消融4](../消融实验/base-phase6-ablation4.md)、[消融7](../消融实验/base-phase6-ablation7.md)及[最新op1结果](base-op1.md)说明，主要问题不是表未生效：

| 实验 | Attention / MLP 改变槽位数，每角色4096 | 同实现 B/A 完整step耗时下降 |
| --- | ---: | --- |
| 消融4，两次60步 | 2491 / 2422 | −0.185%、−0.135%，区间跨零 |
| 旧消融7 / op1的6ε规则 | 0 / 0 | 实际identity，不能据此判断新排列效果 |
| op1 v3/1ε，两次50步 | 119 / 42 | −0.111% [−0.435%, +0.192%]；−0.098% [−0.379%, +0.160%] |

原始数据复核见 [audit.json](../../../../outputs/h100/phase6/arrival-op2-cpu-20261020/evidence/audit.json)。大幅旧重排和少量新重排都没有稳定正收益，继续放宽门槛不能直接解决问题。

现有执行为：

```text
GEMM＋融合编码/远端写入 → 全rank发布屏障 → 解码归约 → 全rank复用屏障 → 输出clone
```

对最新A/B的8份trace检查，2048次投影全部同stream串行，GEMM/decode重叠为0。提前一个tile的贡献，不会让该tile立即开始消费；只有影响整个生产阶段、发布等待或后继关键路径，才有整步收益。

此外，旧标签来自BF16稀疏采样，而A执行全远端量化；各tile也来自不同调用。“槽位趋势＋tile残差”是否在换序后仍成立，没有稳定实测支持。op2不再用这些标签为新候选评分，不改变 ε，也不把合成尾部模型改善当作GPU提速。

## 2. 替代机制：让共享权重块的两个行块靠近执行

当前每个来源的输出为2048×4096，共16×32个128×128 tile。每个目的地有两个M行块、每行32个N列块，共64个tile。A在目的地内先遍历第一行32列，再遍历第二行32列：

```text
目的地内物理offset：0,1,...,31,32,33,...,63
对应行/列：        行0的全部列，随后行1的全部列
```

两个M行块计算同一N列块时，会使用同一个权重块。A让这两次使用相隔32个目的地内位置，即256个全局逻辑槽位。新顺序把每行的少量连续列配成一组，例如跨度 `g=4`：

```text
0,1,2,3, 32,33,34,35, 4,5,6,7, 36,37,38,39, ...
行0列0—3，行1列0—3，行0列4—7，行1列4—7，...
```

对于原逻辑槽位 `i`，新槽位到原A槽位的映射为：

```python
j = i // 8
within = (j // (2*g))*g + (j % g) + 32*((j // g) % 2)
old_slot = within*8 + i % 8
```

只允许 `g∈{1,4,8,16,32}`，其中32严格为identity。设备实现使用等价位运算，避免每个GEMM线程按运行时g做整数除法。核心实现见 [arrival_pair_order.py](../../../../tools/h100/phase6/arrival_pair_order.py)。

| g | 同一权重块在两个M行块之间的逻辑槽位间隔 | 用途 |
| ---: | ---: | --- |
| 1 | 8 | 候选 |
| 4 | 32 | 候选 |
| 8 | 64 | 候选 |
| 16 | 128 | 候选 |
| 32 | 256 | identity及调度代码开销对照 |

### 不变的调度约束

`old_slot % 8 == i % 8`，所以每个来源的目的地序列仍然完全等于A。每八个逻辑槽位依旧覆盖八个目的地，本地/远端节奏不变，交换始终限于同目的地64槽位窗口。所有512个物理tile仍一一覆盖；全远端mask按物理tile绑定，完全不变。

这个机制尝试缩短权重块的重复使用间隔，同时允许更多输入行块处于活跃状态，存在局部性取舍。**不能由逻辑间隔缩短直接推导HBM流量减少或加速。** 权重本身仅约4MiB（Attention）/16MiB（MLP），不应假定A必然发生L2驱逐；A已经让同一权重块被八个目的地复用。

根据历史实际occupancy=2、132SM及当前CUTLASS构造器，两个shape均分派512个单tile DP CTA，`sk_tiles=0`、无cohort raster、无block重映射。这有助于解释局部性候选；逻辑顺序仍不保证CTA完成顺序。新运行继续以实际mapping检查为准。

因此，这是一项**生产顺序的替代机制**，不是新的量化算法，也不是宣称已经实现“到达即消费”。

## 3. 强制同实现：只准两个文件不同

[构建器](../../../../tools/h100/phase6/arrival_op2_build.py)复用原A运行库，只编译一个支持不同g的新S库。对源码哈希逐项核验，唯一允许不同的是：

```text
src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp
phase6_host.hpp
```

所有其他登记源码必须与A逐字节相同，包括：

| 固定项 | A与S |
| --- | --- |
| GEMM主循环、融合epilogue编码 | 同一实现 |
| H128/E4M3、scale计算、FP8转换、packet布局 | 同一实现 |
| 接收端解码器 | 都是原A的CTA解码器，不使用warp替换版 |
| 量化范围 | 每来源448个远端tile全部量化，本地BF16 |
| 来源归约顺序与逐次BF16舍入 | 同一实现 |
| 发布屏障及复用屏障 | 两道均保留 |
| 工作区、逐次配置/复位、输出clone | 同一实现 |
| 编译配置及既定反向路径 | 同一实现 |

新调度仍保留A原始identity查表，再增加生产槽位置换。每个角色的g由主机读取已冻结的环境值，传入swizzle对象；设备kernel不读取环境变量。缺失或非法值明确拒绝，不悄悄选默认g。实际环境、执行计划、mapping及源码登记交叉核验。

两个角色可选择不同g，无需为每个参数重编译运行库。identity32诊断组使用S库、g=32，专门暴露新增调度代码的开销；它不允许作为“新顺序有效”的获胜候选。

## 4. 同量化实测筛选，然后独立确认

### 4.1 筛选阶段：实际完整算子计时

[同codec探测器](../../../../tools/h100/phase6/arrival_op2_probe.py)测六种配置：

```text
A、S(identity32)、S(g1)、S(g4)、S(g8)、S(g16)
```

每个配置使用独立TP8进程，均调用原A API：

```python
op = GemmRSTaco(group, 2048, 4096, K_local, placement='fused')
output = op.forward(input, weight).clone()
```

K_local分别为512和2048。计时包括原发布/复用屏障、解码和输出clone；不运行profiler或到达采样器，不改codec。

固定6个轮换块，再按相反顺序做6块，共12个完整六配置块、72次TP8进程。每个进程对每个角色执行100次预热、200次正式算子调用。每个角色、配置、块取八rank最大墙钟时间，CUDA event时间只作诊断。两角色输入按固定seed生成；候选输入哈希及输出哈希必须与A逐rank一致，并通过既有独立codec参考检查。

对四个候选×两个角色共8个筛选比较，使用10000次完整块bootstrap、seed20261020；采用Bonferroni口径的 **99.375%区间**。只有相对A耗时下降区间下界大于0的候选才准入；在准入候选中选几何平均耗时最小者，同分按预定候选顺序。没有合格候选的角色取g=32。identity32仅诊断，不参与获胜选择。

这降低了有限候选搜索的误选风险，但不是无条件统计保证；窗口独立性、热缓存算子与真实模型之间的差异仍需独立确认。尤其是，两角色分别选出的参数可能在完整模型中发生不同缓存交互。

### 4.2 冻结与回退

筛选全部结束后才生成 `probe/selection.json`，冻结原始rank证据、输入/输出哈希、库身份、硬件、所选g及决策统计。后续完整step不能重新选择g，也不能根据确认结果调参后继续沿用同一确认身份。

若两角色都回退32，批次标记：

```text
status = completed_no_candidate
target_validated = false
formal_optimizer_steps = 0
```

此时保留全部筛选结果，不再对与A同序的候选做长时间训练，也不宣称达到目标。

### 4.3 确认阶段：只比较 A 与冻结的 S

只要至少一个角色选择非identity顺序，就执行两次独立完整模型比较：

| 项目 | 每次比较 | 默认两次合计 |
| --- | ---: | ---: |
| 策略 | A / S | 同一冻结S |
| 完整配对块 | 6块×2轮＝12 | 24 |
| 正式计时窗口 | 24 | 48 |
| 每窗口预热 / 正式step | 10 / 50 | 相同 |
| 正式optimizer step | 1200 | 2400 |
| 正式rank记录 | 192 | 384 |

第二轮反转块及块内次序。每窗口取 `max(八rank elapsed)/50`，包含32层、4微批、前后向、通信、optimizer与清梯度。每次分别做数值预检、两角色实际mapping和profile路径检查；A/S都必须执行原CTA解码器。

校准是独立算子工作量，不计入完整optimizer step。校准时较快不等于完整模型较快；该分离正是为避免把代理目标再次当成最终收益。

## 5. 1.5%验收口径

主比较唯一为S/A，定义耗时下降：

```text
100 × [1 − exp(mean(log(T_S / T_A)))]
```

默认两次独立确认中，**各次combined点估计与配对95%区间下界均须≥1.5%**，同时必须有非identity映射并通过全部执行/数值/来源核验。逐轮结果完整保留，主表不筛掉负收益窗口；慢窗口敏感性另列。

按历史A的534.264 ms/step，1.5%约为8.014 ms；256次目标投影平均需节省31.3 μs/次。这个预算没有证明顺序优化具备这么大的余量。正式验收始终对同场A计时，不能用历史均值代替。

若没有候选，或者冻结S未达标，报告如实给出失败或未验证结果。**本轮没有更换量化实现来补足差距的后门。**

## 6. 本地检查及边界

本轮CPU证据在 [arrival-op2-order-cpu-20261020](../../../../outputs/h100/phase6/arrival-op2-order-cpu-20261020/)。核心排列已核对全部25种双角色跨度组合：每种8192个主机坐标，覆盖完整双射、目的地槽位不变、mask不变、identity、错误shape/参数拒绝。设备swizzle已以CUDA12.8编译为SM90对象；实际CUTLASS的512 DP CTA分派也通过主机构造验证。见 [validation.json](../../../../outputs/h100/phase6/arrival-op2-order-cpu-20261020/validation.json)。

**73项回归全部通过**：50项新调度、同codec探测、独立确认及批次测试，加23项既有消融4/v3门槛回归。见 [tests.log](../../../../outputs/h100/phase6/arrival-op2-order-cpu-20261020/tests.log)。

集成检查进一步验证source-only父构建、同实现S源码生成、探测准备、选择冻结、完整训练准备与报告检查。生成源码中只有前述两个文件不同，**其余349个登记文件完全相同**，见 [same-implementation.json](../../../../outputs/h100/phase6/arrival-op2-order-cpu-20261020/integrated-compile/same-implementation.json)。非identity双角色主机映射及生成的mapping/runtime SM90对象编译通过，见 [集成编译验证](../../../../outputs/h100/phase6/arrival-op2-order-cpu-20261020/integrated-compile/validation.json)。

**source-only只用于CPU排练，不能启动GPU或通过收益验收。** 测试使用的合成耗时只验证统计/失败分支，不是性能证据。

本机没有PyTorch或NVIDIA设备，未运行新GPU算子、完整动态库或训练。原始回传父校准还缺少已登记的 `__pycache__/runtime.cpython-310.pyc`；执行端必须使用原服务器完整父构建和档案，生产核验没有因此放宽。新机制不使用旧BF16到达标签选g，但父构建身份仍须完整。

## 7. H100执行命令

同步更新后的完整 `tools/h100/phase6/`，尤其是新 `arrival_pair_order.py`、`arrival_op2_probe.py` 及改写的op2模块。旧decoder目录已从当前入口移除；不要复用此前的decoder版op2构建。

```bash
cd /workspace/TempName
export H100_OP2_PYTHON=/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX="$CXX"
export DEBUG=0
export MAKEFLAGS=

# 预览，不创建目录、不查询GPU、不启动子进程。
"$H100_OP2_PYTHON" tools/h100/phase6/detach_arrival_op2.py

# 后台：编译一个S库 → 同codec筛选 → 冻结 → 两次独立完整step确认。
"$H100_OP2_PYTHON" tools/h100/phase6/detach_arrival_op2.py --execute
```

默认父构建：

```text
outputs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/build
```

支持 `--build`、`--python`、`--megatron`、`--build-cuda-home`、`--cuda-home`、`--nccl-root`、`--jobs`、`--out`；默认CUDA12.8编译、CUDA12.4训练运行、jobs2、NVLS0。`--trials 4`表示四次独立确认，筛选仍只做一次。`--foreground --execute`可前台运行。

仅可用 `--op2-build` 复用本v2协议的已编译S库；若同时传 `--selection`，会验证它来自同一个构建及完整同codec实测。默认不传这两个选项，新建整个批次。

主要输出：

```text
logs/h100/phase6/arrival-op2-order-2x-50steps-北京时间戳/
  build/arrival-op2-manifest.json
  probe/selection.json                 # 所选g、完整筛选统计、原始证据哈希
  probe/results/                       # 72个进程窗口 × 8rank
  compare1/report.json、report.md       # 有非identity候选才生成
  compare2/report.json、report.md
  batch-state.json
  handoff/container-supervisor.log
```

后台返回只代表已托管。`status=completed`表示完成全部确认，`target_validated=true`才表示两次均通过1.5%要求；`completed_no_candidate`表示筛选后回退A。执行、数值或证据失败立即停止，不覆盖旧结果；完整step未达标仍完成预先指定的重复，不重新挑参数。
