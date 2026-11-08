# OP4：在 OP3 组合组上缩短 AllGather 通信路径

更新：2026-11-02。**只测 OP4 候选，自动先测算子、再测整步；不重新运行组合组对照。** 候选保留 A＋MA89＋MA11＋R＋CR3 消费者融合＋OP3 编码器改进，再将 NCCL 通道数设置为 16。最新批次完成两个整步窗口，平均 **832.249 ms/step**，比历史参考 850.195 ms/step 低 **17.946 ms**；没有同期对照，不能将差值解释为因果收益，也未达到相对历史减少 60 ms 的参考线。详细结果见第 6 节，启动命令见第 4 节。

首个候选窗口的 [supervisor 日志](../../../../Load-5090/logs/5090/phase6/op4-nccl-channels-single-v1-1x-50steps-20261102-103241-089244/handoff/supervisor.log) 显示原历史数值核对失败。八个 rank 的 smoke 三步分别有 59、47、51 个梯度哈希不同，差异均为 `[4096]` 的 FP32 张量；三步参数更新哈希相同，smoke/profile/正式计时记录的 loss、gradient norm 和最终参数哈希相同。历史记录只保存哈希，不能确定梯度差异大小或原因。该批次计划为两个候选窗口，但在第一个窗口核对后失败；子批次报告的 832.244ms/step 不是已完成的 OP4 报告。现在历史 smoke 梯度哈希差异不再阻止验收，差异的 rank、step 和张量名仍记录在 `candidate-evidence.json`；其他数值和执行路径门禁保持不变。

使用新规则对该首窗口的回传证据进行只读审计已通过，包括八 rank 的 profile 检查；这不改变原批次的失败状态，也不补齐原计划中的第二窗口。

## 1. 为什么下一步优化 AllGather

[OP3 v1](op3-improve.md#6-op3-回传结果2026-11-02) 的两组正式均值约 854–855ms：一组是 R＋CR3，另一组是编码器改进。新的 [v2 完整组合组回传](../../../../Load-5090/logs/5090/phase6/op3-combined-v2-1x-50steps-20261102-100813-883313/report.json) 已记录 **850.195ms/step**、9635.44 tokens/s，状态为 `completed`，但只有一个正式窗口。按当前要求，OP4 只测候选；与 850.195ms 的差值是跨批次历史观察，不能声称同期净增量。

本次重新解析 [OP3 v1 原始批次](../../../../Load-5090/logs/5090/phase6/op3-two-arm-v1-2x-50steps-20261102-022324-519378/report.md) 两组各八 rank，并补充新 v2 组合组八 rank，共 **24 份单步 GPU trace**；按 `External id` 关联实际 CPU/autograd owner。每 rank 先求同类 kernel 时长之和，再取八 rank 中位数：

| 每 optimizer step 的通信 | 次数/rank | v1 R＋CR3，ms | v1 编码器改进，ms | v2 完整组合，ms |
| --- | ---: | ---: | ---: | ---: |
| 全部 AllGather | 524 | 408.672 | 428.481 | **403.989** |
| 其中：MA8 QKV/FC1 前向 | 256 | 228.832 | 240.466 | **221.407** |
| 其中：row projection 反向 | 256 | 161.449 | 166.495 | **161.514** |

两个 256 次类别均未与该 rank 的任何非 NCCL GPU kernel 发生时间重叠。它们分别位于下面的计算依赖链：

```text
前向：LayerNorm → BF16 AllGather → QKV/FC1 GEMM → 后续计算
反向：消费者 bias 梯度 → BF16 AllGather → row dgrad GEMM → row wgrad GEMM
```

MA8 已经复用了 QKV/FC1 的前向输入，删除的是这些列投影在反向中重复的输入 AllGather；上述仍存在的前向 gather 和行投影输出梯度 gather 不能用同一份缓存继续删除。每次实际输入为 `1,048,576` 个 BF16（2MiB），聚合输出为 `8,388,608` 个 BF16（16MiB），TP group 为全部八卡。剩余 12 次 AllGather 属于输出 head 和 sequence scatter 的反向，也一起接受通道数调整。

所有 24 份 trace 的 AllGather 入口名都是 `ncclDevKernel_AllGather_RING_LL`，实际 launch 为 **4 个 CTA、每 CTA 544 threads**；新组合组也保留这一特征。这里可靠可观测的是 grid/block，不能由名字推断使用了 LL 协议：NCCL 2.27 的 [kernel 映射代码](https://github.com/NVIDIA/nccl/blob/v2.27.3-1/src/device/generate.py#L128-L139) 将多个 AllGather 算法/协议映射到同一个入口。544 threads 反而与 Simple Ring 的 512 个工作线程加同步 warp 相容。因此“只强制 Simple”没有足够改进依据，本方案不采用它。

相比之下，op3-r 的 rank0 编码器总时长约 11.385ms、解码器约 2.315ms；继续仅微调 codec 缺少支撑 60ms 的时间空间。下一步选择提高通信并行度，并要求实际 AllGather grid 从 4 变为 16，而非仅检查设置了某个变量。

**这些是受 profiler 扰动的 kernel 时长，含 rank 等待，不能相加当成 850ms 的完整关键路径分解。** “未与计算重叠”也不等于整个区间都可消除，更不能据此承诺速度。首个 embedding AllReduce 在部分 rank 有明显启动偏斜，本方案没有把它作为稳态可省时间。

可复算的 [分析工具](../../../tools/5090/phase6/analyze_op4_baseline.py)、[v1 逐 rank 结果](op4-baseline-analysis.json) 和 [v2 组合组逐 rank 结果](op4-combined-analysis.json) 保留全部 trace 相对路径、SHA256、通信 owner、消息尺寸、调用数、时长与计算重叠：

```bash
python3 tools/5090/phase6/analyze_op4_baseline.py \
  --batch ../Load-5090/logs/5090/phase6/op3-two-arm-v1-2x-50steps-20261102-022324-519378 \
  --out /tmp/op4-baseline-analysis.json

python3 tools/5090/phase6/analyze_op4_baseline.py \
  --batch ../Load-5090/logs/5090/phase6/op3-combined-v2-1x-50steps-20261102-100813-883313 \
  --out /tmp/op4-combined-analysis.json
```

## 2. 实际改动：提高 NCCL 通信通道并行度

| 组 | 计算路径 | NCCL 通道配置 |
| --- | --- | --- |
| 历史参考，仅只读 | 已完成的 `op3-combined` 批次 | min/max channels 未设置，大 AllGather 实际 grid 为 4 |
| `candidate`，唯一新运行组 | `op3-combined` 加 OP4 通信配置 | `NCCL_MIN_NCHANNELS=16`、`NCCL_MAX_NCHANNELS=16`，要求 AllGather 实际 grid 为 16 |

这是同一通信量下的通信并行度实验。单次 16MiB 输出由更多通道分摊，有机会增加主机中转路径上的并行搬运，减轻目前四通道的串行等待。固定选择 16，不在同批次扫描多个值再挑最优；四倍通道数不意味着四倍吞吐，PCIe/NUMA 带宽、SHM proxy、额外同步和 SM 竞争都可能抵消收益。

目标环境使用 NCCL 2.27.3；[对应版本通道配置文档](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2273/user-guide/docs/env.html#nccl-min-nchannels) 说明这些变量控制通信通道，并影响计算资源使用。候选同时设置上下界，确保 communicator 有足够容量。仅用 tuner 请求 16 而 communicator 仍只有 4 通道，会在 [NCCL planner](https://github.com/NVIDIA/nccl/blob/v2.27.3-1/src/enqueue.cc#L502-L517) 中被容量上限限制；本实现无需 tuner 插件或重编 NCCL。

AllGather 复制原始 BF16 数据，不做浮点归约。OP3 的量化范围、FP8 packet、decoder 累加顺序、两次消费者 BF16 加法、softmax、激活缓存、梯度统计和 Adam 代码均保持原样。继续使用原来的同步/异步调用及 stream 依赖，保持 `CUDA_DEVICE_MAX_CONNECTIONS=1`、`NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=0`、`NCCL_NVLS_ENABLE=0`；不启用 CUDA Graph。

**通道设置作用于整个 NCCL communicator，不是只改 AllGather。** ReduceScatter、AllReduce 和 SendRecv 的 launch 几何或分块也可能变化，不能将所有端到端变化归给 AllGather。脚本保留这些变化的原始签名，要求调用数、消息量、进程组和入口种类与历史参考相同；不从入口名字声称实际协议不变。历史 smoke 梯度哈希差异仅记录，不作为验收失败；loss、gradient norm、参数更新和最终参数哈希等检查仍要求一致。这里只读旧记录，不启动旧组训练。更多通道还可能增加 NCCL 内部/主机缓冲，Torch allocated 不覆盖这些资源，需关注 OOM 或 SHM 分配失败。

原始 OP3 源码、构建和入口保持原样。新的 [op4.py](../../../tools/5090/phase6/op4.py) 在独立子进程初始化 NCCL 前设定候选环境，调用原有 `op3.py`，从而复用其完整门禁。候选复用已验证 OP3 codec/GemmOnly build；MA9/MA11 兼容库照常准备和校验。若传 `--abc-build`，只在第一个候选子批次新建 OP3 codec，此后复用该 build。

### 2.1. 根据新组合组算子结果选择优化位置

用户指定批次的 [算子报告](../../../../Load-5090/logs/5090/phase6/op3-combined-v2-1x-50steps-20261102-100813-883313/operator/report.md) 中，每角色 20 次预热＋100 次正式 forward，只有一个计时窗口：

| 算子 | 最慢 rank 的 GPU 平均延迟，μs | 同步墙钟平均延迟，μs |
| --- | ---: | ---: |
| Attention 输出投影 | 664.285 | 664.685 |
| MLP 输出投影 | 726.075 | 726.338 |

独立 profile 每角色各 10 次调用，先取每 rank 中位数、再取八 rank 中位数：

| 算子 | GEMM，μs | 编码，μs | SendRecv，μs | 解码含融合，μs | CPU 提交，μs | GPU 跨度，μs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention | 59.816 | 39.040 | 785.738 | 13.808 | 854.721 | 899.763 |
| MLP | 213.367 | 38.640 | 277.702 | 13.432 | 369.166 | 547.148 |

据此，本轮优化选择是：

- **通信及其提交链是优先观察点。** 两种投影的 packet 量相同，但 Attention 的 profile 通信等待明显更大；它可能包含 rank 到达偏斜、CPU 提交和 profiler 干扰，不能直接视为网络带宽差异。16 通道配置也可能影响 SendRecv 的通道资源，候选算子测试会记录实际 grid 和延迟；不预设 SendRecv 一定变成 16 CTA。
- **编码/解码继续融合的量级不足以单独支撑 60ms。** 两角色编码＋解码合计约 104.919μs，按每步各 128 次投影即约 13.430ms 的 profile 工作量。即使理想地全部消除，也只是这个诊断量级；实际收益还更受依赖链限制。保留已有 R、CR3 和 warp 编码器，本轮不再叠加另一套未验证的 codec 算术。
- **MLP GEMM 是次一级候选。** 约 213μs 的 profile 计算比 Attention 大，但改 GEMM 后端/分块需要额外验证 BF16 归约和 producer 输出，也不能从该数值推导 60ms 净收益。本次先隔离通信配置改动，保留原 A 的 GemmOnly。

Attention 的 profile 单项通信时间已经超过其正式算子总延迟，说明两种测量不能直接相减；首调用的大等待也不能直接当作稳态可省时间。纯投影 forward **不包含本轮主要针对的前向列投影和 row-backward AllGather**，因此算子没有提速也不自动判定整步失败，反之算子变快也不代表达成整步目标。

进一步按实际 launch 还原 160 次 profile 投影：Attention/MLP 的阶段间 GPU gap 中位分别约 **4.809/4.896μs**，跨调用 gap 约 **3.616/3.887μs**。不能把表中 854.721/369.166μs 的 CPU scope 全当作可消除空闲。Attention rank6 的阶段间 gap 中位约 225.501μs，部分 kernel 提交偏晚；首调用 NCCL 最长达 42.080ms。这支持检查提交偏斜，但目前仅在 profile 中观察到，不能归因为正式计时的稳定瓶颈。全部调用均保留，没有删除首调用来改善主表。

[算子分析工具](../../../tools/5090/phase6/analyze_op4_operator.py) 和 [分析数据](op4-operator-analysis.json) 核对了 8 份 rank 记录与 16 份 trace 的哈希、真实阶段归属、算子正确性及单窗口计时，可只读重算：

```bash
python3 tools/5090/phase6/analyze_op4_operator.py \
  --batch ../Load-5090/logs/5090/phase6/op3-combined-v2-1x-50steps-20261102-100813-883313 \
  --out /tmp/op4-operator-analysis.json
```

后续若整步 AllGather 已到 16 CTA，而算子 SendRecv 仍高且通道几何未变，再单独研究 P2P 通道和提交调度；本轮不把这些未经验证的额外改动叠入候选。

## 3. 60ms 的预算和判据

512 次主要 gather 若平均每次**净省 117.19μs**，才累计对应 60ms。以新组合组前述两个类别约 383ms 的 profile 量级作粗略筛选，要求约 15.7% 的改善；这个比例只是目标难度，不是预测，因为两个中位数相加、rank 等待变化和 profiler 扰动都不能替代正式步时。

正式指标按原规则：每窗口 10 步预热＋50 个完整 optimizer step，取八 rank 最大墙钟 elapsed，再除以 50。包括前向、反向、通信、optimizer 与清梯度；模型仍为 32 层、hidden 4096、sequence 2048、micro batch 1、global batch 4、TP8/DP1，每步 8192 tokens。

```text
历史差值 = 历史组合组 850.1950773783028 ms − 本次 OP4 平均 ms/step
历史参考线 = 850.1950773783028 − 60 = 790.1950773783028 ms/step
```

- 默认 `--trials 1 --blocks 1`：只测候选的一个正式窗口、50 步，报告算子结果、完整步时、吞吐、显存及历史差值。
- 增加 trials/blocks 也只重复候选；保留全部窗口，报告均值、中位数、最快/最慢，不创建 AB/BA 或配对置信区间。
- 低于约 790.195ms 只能标记为相对历史达到 60ms 的观察。因为没有同期性能对照，`target_validated` 始终为 `false`，不能据此宣称已验证因果增量。

通道配置只在这个固定负载实验中使用；若增加并行度没有缩短整步、导致退化或任何等价性检查失败，就保留失败证据，不将其推广为默认环境配置。attention 局部图重放可作为后续独立备选，但 [MA15 的 H100 结果](../design-h100/New-Mechanism/ma-15.md) 不能用于承诺这里的 60ms，也没有混入本次候选。

## 4. 测试启动

在目标机 `TempName-New` 仓库根目录、已配置的 5090 环境运行：

```bash
bash tools/5090/phase6/run_op4.sh \
  --op3-build /workspace/Load-5090/logs/5090/phase6/op3-two-arm-v1-2x-50steps-20261102-022324-519378/build \
  --trials 1 --blocks 1 --execute
```

**这条命令只启动 OP4 候选：先测算子，再测一个完整训练窗口，共 50 个正式 optimizer step。** 无需先单独运行 OP3 或 OP4 算子；不要同时启动其他 GPU 实验。

默认只读参考为同级 `Load-5090/logs/5090/phase6/op3-combined-v2-1x-50steps-20261102-100813-883313/`。使用其正式报告、原始 rank 记录、算子和 profile 文件进行数值/身份核对，不运行该参考批次；若归档位于别处，可加 `--reference-batch /path/to/completed-op3-combined-batch`。

默认后台执行，加 `--foreground` 可前台观察。不带 `--execute` 只预览，不查询 GPU、不创建目录、不编译。请同步完整新增文件及当前 OP3 脚本，不能只复制入口 shell。启动环境及 `/etc/nccl.conf`、`~/.nccl.conf`、`NCCL_CONF_FILE` 中如有 `NCCL_PROTO`、`NCCL_ALGO`、`NCCL_TUNER_PLUGIN`、`NCCL_MIN/MAX_NCHANNELS`、旧别名 `NCCL_MIN/MAX_NRINGS` 或 `NCCL_MIN/MAX_CTAS` 覆盖，入口会明确报错，避免隐藏配置改变实验；脚本不会修改这些配置文件。

需要先只看候选算子时，可单独运行：

```bash
bash tools/5090/phase6/run_op4.sh \
  --op3-build /workspace/Load-5090/logs/5090/phase6/op3-two-arm-v1-2x-50steps-20261102-022324-519378/build \
  --operator-only --foreground --execute
```

完整入口依次运行：硬件/NCCL → 原生 packet、舍入和输出所有权 → 算子梯度、延迟及 profile → 全模型 smoke → 八 rank 全模型 profile → 正式计时。`--operator-only` 在算子后结束，不验证全模型 AllGather 的 16-CTA grid，也不产生整步达标结论。完整候选结束后核对历史数值轨迹和实际通道变化，失败不产生 OP4 成功报告。

新结果写入 `../Load-5090/logs/5090/phase6/op4-nccl-channels-single-v1-<trials>x-50steps-<时间>/`：

| 位置 | 内容 |
| --- | --- |
| `report.md` / `report.json` | 候选算子、整步结果和明确标注的历史差值 |
| `batch-state.json` / `handoff/supervisor.log` | 总体状态与当前子批次 |
| `candidate-evidence.json` | 候选路径、历史记录哈希及只读数值回归 |
| `compare1/block00-candidate/` | 唯一候选子批：算子、smoke、profile、正式窗口及独立报告 |
| `compare1/block00-candidate/operator/report.md` | 每角色算子梯度验证、延迟和独立 profile |
| `compare1/block00-candidate-supervisor.log` | 候选子批次的执行日志 |

`completed` 表示整步检查和测试完成，不表示一定提速。算子模式使用 `operator_completed`。`--report` 对已完成批次只读复核候选及历史证据，不是续跑命令：

```bash
bash tools/5090/phase6/run_op4.sh \
  --out /path/to/completed-op4-batch --report
```

## 5. 数值、执行路径和本地验证

[op4_evidence.py](../../../tools/5090/phase6/op4_evidence.py) 在原 OP3 验证之外追加：

1. 候选和既有参考的硬件、软件、初始权重、输入、RNG、CPU affinity、模型身份一致；通信环境仅允许 min/max channels 的指定差异。历史数据要求完整文件及 SHA256 相符；不将其解释为本地重新验证了旧二进制或完成新基线训练。
2. 完整模式下，每个 smoke/profile/timing rank 的 loss、gradient norm、最终全部参数指纹与参考相同；smoke 进一步比较全部 trainable 参数的更新轨迹和所有目标投影输出指纹。smoke 梯度哈希差异只记录，不做逐位验收，也不能从哈希评估数值误差。算子模式继续执行候选实际 autograd 与非融合参考的逐位输出、四类梯度和三步更新检查。
3. 整步 profile 每 rank 恰有 524 个 AllGather kernel，全部满足实际 launch、TP group、BF16 消息形状，候选 grid 必须为 `[16,1,1]`，防止设置未生效。入口名不用于证明实际算法/协议；原 OP3 分析器继续审查前后向 owner 和 stream 依赖。
4. 其余 NCCL 入口、payload 和次数与历史参考核对，包括 256 次投影 SendRecv；grid/block/shared-memory 允许变化并留档。算子 SendRecv 的几何只作记录，不强行要求 16 CTA。原 OP3 packet、R 解码、CR3 消费者、MA8/MA9/MA11 检查保留。最终报告绑定原始 rank 文件、trace、历史参考与源码快照的哈希。

本地验证命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tools/5090/phase6 -p 'test_op4*.py' -v
bash -n tools/5090/phase6/run_op4.sh
bash tools/5090/phase6/run_op4.sh --trials 1 --blocks 1
git diff --check
```

原单候选版本的 **40 项 OP4 专项测试、49 项 OP3 回归测试均通过**；完整/仅算子入口预览、shell 语法、文档链接和 `git diff --check` 通过。v1 和 v2 组合组的 24 份整步 trace，以及新组合组的 16 份算子 trace 均已完成只读分析。这些验证不能代替 GPU 正确性门禁；首个实测窗口的梯度哈希差异仍需排查。

## 6. OP4 回传结果（2026-11-02）

新批次的 [报告](../../../../Load-5090/logs/5090/phase6/op4-nccl-channels-single-v1-1x-50steps-20261102-105103-438654/report.md)、[原始结果](../../../../Load-5090/logs/5090/phase6/op4-nccl-channels-single-v1-1x-50steps-20261102-105103-438654/report.json) 和 [候选审计证据](../../../../Load-5090/logs/5090/phase6/op4-nccl-channels-single-v1-1x-50steps-20261102-105103-438654/candidate-evidence.json) 显示状态为 `completed`。实际计划是 **1 trial × 2 blocks**，每个窗口 10 步预热、50 步正式整步计时，共 100 个正式 optimizer step；目录名中的 `1x-50steps` 不表示只有一个窗口。

| 完整训练结果 | block 0 | block 1 | 两窗口汇总 |
| --- | ---: | ---: | ---: |
| ms/step | 833.486 | 831.011 | 832.249（均值及中位数） |
| tokens/s | 9828.60 | 9857.87 | 9843.21（按平均步时计算） |
| 最大 Torch allocated GiB | 19.422869 | 19.422869 | 19.422869 |
| 最大 Torch reserved GiB | 19.472656 | 19.472656 | 19.472656 |

两个窗口采用八 rank 最大 elapsed 计算步时；显存数值不包含 CUDA context、NCCL 等内部占用。历史 OP3 组合组是单窗口 850.195 ms/step，候选两窗口平均低 17.946 ms（约 2.11%），仍高于相对历史减少 60 ms 的参考线 790.195 ms/step。历史与候选不在同期测量，不能计算配对收益或置信区间，也不能把这 17.946 ms 归因于通道设置；报告的 `target_validated=false`。

### 6.1 算子计时与通信 profile

每个 block 独立运行算子测量，每角色 20 次预热和 100 次正式 forward；表中的 GPU 和墙钟延迟分别取八 rank 最慢值。历史组也只有一个窗口，以下仅作跨批次观察。

| 角色 | 历史 GPU / 墙钟 μs | block 0 GPU / 墙钟 μs | block 1 GPU / 墙钟 μs |
| --- | ---: | ---: | ---: |
| Attention 输出投影 | 664.285 / 664.685 | 427.424 / 427.659 | 418.124 / 418.579 |
| MLP 输出投影 | 726.075 / 726.338 | 524.202 / 524.373 | 528.316 / 528.552 |

算子不包含完整训练步或主要 AllGather 路径，不能将延迟差值直接换算为整步收益。详细数据见 [block 0 算子报告](../../../../Load-5090/logs/5090/phase6/op4-nccl-channels-single-v1-1x-50steps-20261102-105103-438654/compare1/block00-candidate/operator/report.md) 和 [block 1 算子报告](../../../../Load-5090/logs/5090/phase6/op4-nccl-channels-single-v1-1x-50steps-20261102-105103-438654/compare1/block01-candidate/operator/report.md)。

两次整步 profile 的每个 rank 均记录 524 次 AllGather，候选实际 grid 均为 `[16,1,1]`，历史参考为 `[4,1,1]`。每 rank 汇总 AllGather kernel 时间后取八 rank 中位数：历史为 **403.989 ms**，候选 block 0 为 **402.005 ms**，block 1 为 **408.304 ms**。这些受 profiler 扰动且包含 rank 等待，不能与正式步时直接相减，也不支持单凭它们认定 AllGather 稳定提速。独立算子 profile 中，两角色 SendRecv 的实际 grid 从历史的 `[4,1,1]` 变为两个候选窗口的 `[15,1,1]`，说明通道设置也影响了其他通信；不能将整步差值全部归给 AllGather。

### 6.2 正确性与验收范围

两个候选窗口的证据审计均通过。历史与候选的 smoke、profile、timing 记录中的 loss、gradient norm 和最终参数指纹一致；smoke 的全部参数更新轨迹和目标投影输出指纹一致。各候选算子的融合与非融合参考通过输出、四类梯度和三步更新的逐位检查。

按当前放宽规则，全模型 smoke 梯度哈希差异只记录，不作为失败条件。两个候选窗口中，八个 rank 的三个 smoke step 分别都有 59、47、51 个梯度张量哈希与历史参考不同；这些哈希无法说明误差幅度，因此 `completed` 不表示全模型梯度逐位一致，也不证明差异在某个数值容差内。原先因该项检查失败的批次仍保留原状态，与本次完成批次分开记录。
