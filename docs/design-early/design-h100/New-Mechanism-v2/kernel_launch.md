# CPU 启动链修复：MA11 为什么从约 450 ms 变成约 340 ms

更新：2026-10-31。分析提交 `d496f53605efa0298d93fa21de3e3d689d8bf2ad`，作者 hjr，提交时间 `2026-10-31 10:39:18 +0800`，标题 `Fix CPU Launch`。

**这次提交删除了 worker 在导入 torch 前“每 rank 强制只用一个逻辑 CPU”的代码，让它继承外部启动环境分配的 CPU 集。MA11 的 CUDA 算术和内核实现没有改变。新服务器上最直接的前后对照是 597.170 → 344.230 ms/step，耗时下降 42.356%；旧服务器的 449.182 → 新服务器的 344.230 ms 是跨机器参照，不能将这 104.952 ms 全部精确归因给这次提交。**

现有证据支持：旧的单核限制严重拖慢了这台新服务器上的主机提交与跨 rank 推进，解除限制后，MA11 能更快地向 GPU 提交原来的工作。新结果已超过旧机器的历史速度，但旧机器 CPU 未知，也没有在旧机器上做解除绑核对照，无法证明旧机器原本也能达到 344 ms。

## 1. 先分清三组 MA11 数据

以下均为无 profiler 的完整 optimizer step 正式均值，不是 kernel 时间，也不是带 profiler 的单步跨度。

| 数据 | CPU 策略 | MA11 均值 | MA11 中位数 |
| --- | --- | ---: | ---: |
| 旧服务器 gxn75，2026-10-27 20:20 批次 | 旧 worker 单逻辑 CPU 强绑 | 449.182 ms | 449.129 ms |
| 新服务器 10-118-122-131，2026-10-31 04:43 MA15 对照臂 | 同一旧强绑策略 | 597.170 ms | 596.787 ms |
| 新服务器 10-118-122-131，2026-10-31 13:21 MA15 对照臂 | inherit，实际允许 CPU 0–191 | 344.230 ms | 333.562 ms |

原始来源为 [旧服务器 MA11 report](../../../../../TempName/logs/h100/phase6/ma11-v1-causal-softmax-1x-50steps-20261027-202038-900556/fullstep/report.json)、[新服务器修复前 report](../../../../../Load/logs/ma15-v1-attention-graph-1x-50steps-20261031-044305-267152/fullstep/report.json)、[新服务器修复后 report](../../../../../Load/logs/ma15-v1-attention-graph-1x-50steps-20261031-132153-299757/fullstep/report.json)。这里 MA15 实验中的 `F4-cache-zcount-csoftmax` 就是完整 MA11，对照臂没有安装 attention graph。

同机前后均值相差 **252.940 ms**，耗时下降 **42.356%**；如果换算固定工作量的吞吐，则提高 **73.480%**。跨机 449.182 → 344.230 ms 的耗时下降为 **23.365%**。这些是均值的描述性比较，不是随机交替 A/B 的置信区间。

## 2. 原来具体有什么问题

### 2.1. “均匀选 CPU 编号”并没有均匀分配物理核

旧的共用 worker 在 `import torch` 前执行：

```python
local_rank = int(os.environ['LOCAL_RANK'])
WORLD = int(os.environ['WORLD_SIZE'])
allowed_cpus = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(
    0, {allowed_cpus[local_rank * len(allowed_cpus) // WORLD]}
)
import torch
```

花括号内只有一个编号，含义是把当前线程允许运行的 CPU 集缩为一个逻辑 CPU。它既不是“给每个 rank 分配一段 CPU”，也没有读取物理核、SMT sibling、CPU socket 或 GPU NUMA 拓扑。

Linux 的 affinity 是线程属性；`sched_setaffinity(0, ...)` 修改调用线程。后续新建的 pthread 通常继承创建者的 CPU affinity，所以在 torch 初始化之前施加这个限制，还可能影响后续创建的主机辅助线程；具体库可以再次修改线程 affinity，不能据此断言所有 CUDA／NCCL 线程最终都在同一核上。语义依据见 [sched_setaffinity(2)](https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html) 和 [pthread_create(3)](https://man7.org/linux/man-pages/man3/pthread_create.3.html)。

即使不增加 CPU 计算线程，训练也需要 CPU 执行 Python、autograd、ATen、CUDA runtime／driver 调用以及同步检查。限制主线程和继承该限制的辅助线程只能在一个逻辑 CPU 上运行，会减少调度余地；选错 GPU 本地 NUMA 又可能增加访问开销。某个 rank 提交变慢，还会让其他 rank 更早进入 collective／IPC 等待。这些影响不会通过 `OMP_NUM_THREADS=1` 自动消除。

### 2.2. 同样的公式在新机器上产生了四个 NUMA 错配

用户提供的 [现场拓扑记录](../cpu-affinity.md) 是双路 **Intel Xeon Platinum 8468**，每路 48 个物理核、每核 2 个硬件线程，共 96 个物理核／192 个逻辑 CPU。容器允许 `0-191`；NUMA 0 为 `0-47,96-143`，NUMA 1 为 `48-95,144-191`。GPU 0–3 属于 NUMA 0，GPU 4–7 属于 NUMA 1。

在 TP8、允许 CPU 为 `0-191` 时，旧公式选择 `[0,24,48,72,96,120,144,168]`：

| rank / GPU | 旧选择 CPU | CPU NUMA | GPU NUMA | 局部性 |
| --- | ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 0 | 同 NUMA |
| 1 | 24 | 0 | 0 | 同 NUMA |
| 2 | 48 | 1 | 0 | 错配 |
| 3 | 72 | 1 | 0 | 错配 |
| 4 | 96 | 0 | 1 | 错配 |
| 5 | 120 | 0 | 1 | 错配 |
| 6 | 144 | 1 | 1 | 同 NUMA |
| 7 | 168 | 1 | 1 | 同 NUMA |

四个 NUMA 错配可由现有拓扑和公式直接推出。**SMT 共核尚未确认**：缺少逐 CPU 的 `core_id`／`thread_siblings_list`。若 CPU `i` 与 `i+96` 恰为 sibling，上述八个 rank 才会落在四个物理核上；不能把这个条件假设写成已经测到的事实。

同一现场快照中 `cpu.max=max 100000`、`nr_throttled=0`、`throttled_usec=0`，没有看到该可见 cgroup 的 CPU quota 节流。这只说明当时的配额记录，不能排除其他主机负载、频率差异或后续调度竞争。原服务器 CPU 信息仍缺失。

## 3.  实际改了什么- inherit 的准确含义

inherit 保留外部启动环境的 CPU 集：外部调度器若已给各 rank 分配合理的 CPU 集，worker 不再覆盖它；若容器给每个 rank 都允许 192 个逻辑 CPU，本次记录就是 0–191。它仍受容器 cpuset／调度器约束。

这不等于每个 rank 新开 192 个线程，也不保证操作系统总把它调度到 GPU 最近的 NUMA。原来的 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1` 没有改变。本次修复撤掉了内部错误的单核选择，尚未实现“为每个 GPU 自动分配一组独占、本地物理核”的策略。

提交没有改变 MA11 softmax、量化算法、FLUX GEMM、CUDA kernel launch geometry、GPU 时钟、驱动、编译参数、通信协议、optimizer 或正式计时规则，也没有给 MA11 加上 CUDA Graph。MA15 的局部 graph 是另一个独立机制。

## 4. 哪些实测支持主机提交问题

### 4.1. 同机前后的任务身份一致

04:43 与 13:21 批次已核对：主机与八张 GPU 的 UUID／顺序一致；PyTorch、CUDA runtime、NCCL、Python 路径、模型、tuning、输入 token、初始参数和 RNG 身份一致。两臂各 358 项 native 源码哈希在归一化构建路径后全部相同，128 份对应正式结果的最终参数哈希也相同。

两批次各有 35 个冻结辅助文件，其中 32 个内容相同。三个变化文件是 `worker.py`（完整训练的 affinity 与报告）、`ma15_bench_base.py`（算子阶段的 affinity 与报告），以及 `ma15_profile.py`（独立于 d496f536 的图节点时间戳容差修正）。最后一项只影响离线 trace 的顺序验证，不改训练、图重放或计时；本页 API／设备时间比较直接从原始 trace 计算。因此不能说两批次“所有 Python 文件都完全相同”，也不能将解析器修正归入 CPU 绑核提交。

这排除了“换成更小模型、少算一步、偷换 kernel 源码”这一类解释。两批次独立构建，不能进一步宣称 native 二进制哈希相同；也没有同步记录每个线程的实际运行 CPU、CPU 频率和整机负载，因而不是把所有外部条件都锁定的因果实验。具体身份检查和冻结 worker diff 保存在 [metrics.json](../../../../../Load/analysis/20261031-ma15-affinity/metrics.json)。

### 4.2. 启动调用明显缩短

以下对三组 MA11 的八 rank trace 分别统计，再取八 rank 中位数。API 列是“每个 rank 内调用时长中位数”的跨 rank 中位数；设备计算列是“每 rank 单步对应 kernel 总时长”的跨 rank 中位数。

| profile 指标 | 旧服务器强绑 | 新服务器强绑 | 新服务器 inherit |
| --- | ---: | ---: | ---: |
| `cudaLaunchKernel` CPU 调用中位数 | 6.096 μs | 7.970 μs | 4.349 μs |
| `cudaLaunchKernelExC` CPU 调用中位数 | 8.780 μs | 11.463 μs | 5.434 μs |
| 256 个 FLUX GEMM 总设备时间 | 22.602 ms | 22.637 ms | 23.025 ms |
| 128 个 MA11 softmax forward 总设备时间 | 14.374 ms | 14.347 ms | 14.511 ms |
| 128 个 MA11 softmax backward 总设备时间 | 5.185 ms | 5.184 ms | 5.198 ms |
| 2,060 个 cuBLAS kernel 总设备时间 | 73.096 ms | 72.803 ms | 74.156 ms |

新服务器修复前的两类启动调用，比旧服务器约慢 31%；解除限制后，它们都明显短于修复前，甚至短于旧服务器的历史值。这里的“启动调用时长”是 CPU 上 CUDA API 的耗时，不是 GPU kernel 的执行时长，也不是完整的排队等待时间。计算列没有表现出与 42% 整步降幅对应的设备计算加速。

MA11 的 rank 0 在修复前后都记录 50,552 个 `cpu_op`、8,990 个 GPU kernel、1,548 个 GPU memset、245 个 GPU memcpy。工作数量保持相同，改变的是工作提交和各 rank 到达的时序。

### 4.3. 主机慢可以表现为 GPU 空隙，也可以表现为 GPU 上的等待

对本进程的 GPU kernel／memcpy／memset 区间求并集，用首个活动开始到最后活动结束的跨度减去并集，得到“本进程 GPU 活动空隙”。它不是整卡利用率；NCCL／IPC 等待内核也算在活动区间里。

同机 MA11 的 rank 0 变化最直观：

| rank 0 单步 profile | 强绑 | inherit |
| --- | ---: | ---: |
| GPU 活动首尾跨度 | 958.466 ms | 586.392 ms |
| GPU 活动并集 | 460.804 ms | 557.238 ms |
| 进程活动空隙 | 497.662 ms | 29.154 ms |
| 大于 10 μs 的空隙数 | 5,717 | 19 |

但是不能把 rank 0 推广到八张卡。强绑批次 rank 4 的空隙也是 299.426 ms，其他六个 rank 则只有约 29–50 ms；那些 rank 的许多时间已经被通信／屏障内核占据。八 rank 中位数如下：

| 八 rank 分别统计后的中位数 | 强绑 | inherit |
| --- | ---: | ---: |
| GPU 活动首尾跨度 | 962.567 ms | 581.898 ms |
| 进程活动空隙 | 37.857 ms | 31.165 ms |
| AllGather kernel 总时长 | 320.922 ms | 174.259 ms |
| IPC barrier kernel 总时长 | 160.483 ms | 76.162 ms |

其他 rank 在等待较慢 rank 到达时，GPU 可以一直有通信／屏障 kernel 活动；所以“GPU 很忙”也不证明有用计算饱和。修复后，各 rank 的推进与等待分布改变。rank 0 活动并集反而增加，但单步跨度显著缩短；这并不矛盾，也不能单凭等待 kernel 变短就宣布 NVLink 带宽提高。

这些 profile 支持的解释是：**主机提交延迟及其放大的跨 rank 等待得到缓解**。API 时长、CPU 算子区间、活动空隙、通信等待有重叠，且 profiler 会扰动执行，不能把它们相加凑成正式 step 节省的 252.940 ms，更不能为 NUMA、SMT、线程竞争各分配一个未经测量的贡献百分比。

## 5. “慢 30%”与“94.6%”应该怎么理解

之前的跨服务器诊断见 [原分析 metrics.json](../../../../../Load/analysis/20261031-cross-host-performance/metrics.json)。其中 MA11 rank 0 的 `cudaLaunchKernel` 中位数为 **5.960 → 7.805 μs**，`cudaLaunchKernelExC` 为 **8.803 → 11.340 μs**；前者约慢 31%，后者约慢 29%。它们来自当时 MA14 对照的数据，不能与本页采用的 04:43 MA15 对照混成同一份 trace。

“94.6%”来自当时**基础量化的 rank 0 profile**：

| 本进程 GPU 活动指标 | 旧服务器 | 新服务器修复前 | 增量 |
| --- | ---: | ---: | ---: |
| 活动首尾跨度 | 984.963 ms | 1,320.665 ms | 335.701 ms |
| 活动空隙 | 75.767 ms | 393.245 ms | 317.478 ms |

以未舍入数据计算，`317.478 / 335.701 ≈ 94.6%`。这说明那份带 profiler 的额外跨度主要表现为该 rank 的进程活动空隙增加。它既不是“GPU 算力下降 94.6%”，也不是“绑核解释了 94.6% 的正式性能损失”，更不是八卡整体空闲比例。

基础量化此前的正式均值为 **582.318 → 765.361 ms**。本次新拉取数据是 MA15／MA11，尚无解除绑核后的独立基础量化复测；虽然共用 worker 修复已覆盖基础量化入口，不能根据 MA11 的 344 ms 推造基础量化的新成绩。

## 6. 可以下什么结论，还不能下什么结论

原先是**旧代码中对机器拓扑敏感的 CPU 绑定策略**，在新机器上产生了明确的 NUMA 错配，并与严重的主机提交延迟同时出现。它是启动代码的可移植性问题；现有证据不支持把它解释为新仓库迁移损坏了 CUDA 数学实现，也没有证据认定新服务器 GPU 硬件故障。

同机前后的源码身份、真实 affinity 记录、正式计时和八 rank profile 共同支持 d496f536 的修复方向。解除内部限制后，MA11 从约 597 ms 降到 344 ms，MA15 从 545.655 ms 降到 315.552 ms，两臂均减少约 42%。**这足以说明修复有实际效果，但不足以精确证明收益中多少来自 NUMA、SMT 共核、辅助线程调度或 CPU 频率。** 如果要继续分解，需要在同一构建与环境中交替复测旧单核、inherit、按 GPU 本地物理核分组等策略，同时采集线程和频率现场。

MA15 自身的收益应在修复后的同一批次内比较：**344.230 → 315.552 ms，配对 +8.109%［95% CI：+4.137%, +13.595%］**。解除 CPU 限制与局部 attention graph 可以同时有收益，不能将旧服务器 449 ms 到当前 MA15 315 ms 的全部下降都算给任何一项。详细结果和显存成本见 [MA15 总结](../New-Mechanism/ma-15.md)。

本页基于已拉取数据做只读分析，没有新跑 GPU 或改动原始结果。新 MA15 批次的训练和 profile 已完成，原监督器仍因 tuple/list 比较错误保留 `failed` 状态；冻结控制器的正式报告及 16 份 profile 已离线复现，详见 [验证记录](../../../../../Load/analysis/20261031-ma15-affinity/validation.json)。这里引用可复核的数据，不把原 batch 状态改写成成功。

复算脚本为 [analyze.py](../../../../../Load/analysis/20261031-ma15-affinity/analyze.py)，输出为 [metrics.json](../../../../../Load/analysis/20261031-ma15-affinity/metrics.json)：包含五组共 40 份 trace 的每 rank 统计、原始 SHA256、八 rank 中位数、正式窗口、身份检查和冻结 worker diff。从包含 `TempName`、`TempName-New`、`Load` 的工作区根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 Load/analysis/20261031-ma15-affinity/analyze.py
```
