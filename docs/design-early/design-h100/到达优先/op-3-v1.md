# H100 到达优先 op3：按 fragment 就绪提前消费

更新：2026-10-20。接续 [op-1.md](op-1.md)、[base-op2.md](base-op2.md) 和 [idea.md](../idea.md)。

**本轮已实现独立 op3 候选：逐 fragment 发布、辅助 stream 提前消费、本地 GEMM 完成后补齐；原来的全 rank 发布屏障已从候选执行路径移除。尚无本轮 H100 正确性、有效重叠或完整 step 的实测结果，因此“比 A 快至少 1%”仍未验证。** CPU 协议测试与 SM90 编译分别检查逻辑和编译可行性，不能替代这一验收。

## 1. 基线和收益预算

A 沿用 op2 的准确含义：双 shape 基础交替调度、全部远端贡献采用 H128/E4M3、本地贡献 BF16、原 scalar CTA 解码器、原 ring 来源次序及逐次 BF16 舍入、单工作区、输出 clone、发布与复用两道屏障。不能拿当前 `src/` 中其他 warp 解码实验当作 A。

op2 合并 A 为 535.492 ms/step；1% 对应约 **5.355 ms/step**。每步有 128 次 Attention 输出投影和 128 次 MLP 输出投影，若节省全部来自这些调用，平均每次需要净省 **20.918 μs**：

```text
128 × (ΔAttention + ΔMLP) ≥ 5354.92 μs/step
```

这是以旧 A 耗时估算的预算。新机、新运行环境应以同批 A 的实测耗时重新计算，不能与旧 535.492 ms 直接配对，也不能把算子百分比当作完整 step 百分比。

## 2. 改动解决哪条依赖

原执行关系为：

```text
主 stream：GEMM + 编码写入 → 全 rank 发布屏障 → 全量解码归约 → 复用屏障
```

op3 的执行关系为：

```text
主 stream：输入就绪事件 → GEMM + 每 fragment 发布
                                      ↓ 本地生产完成
                                等待有限提前扫描结束
                                      ↓
                    只补齐未消费 fragment，各自等待所需来源 → 复用屏障

辅助 stream：等待输入就绪事件 → 有限扫描已就绪 fragment → 直接解码归约
                                             ↓
                              未就绪则退出，本轮不占住 CTA 等待
```

辅助 stream 的事件记录在 GEMM **之前**。它不依赖 GEMM 完成事件或全 rank 发布屏障。每个来源对一个 fragment 的完整载荷发布后，接收端就能独立判断该 fragment 是否收齐。

主 stream 的尾部补齐只等待**本 rank** GEMM 和辅助扫描完成，然后按 fragment 读取来源就绪标记。某个远端 rank 仍在生产其他目的地或其他 tile，不再阻止已经齐备的数据被解码。函数返回的完整输出仍须完成所有 fragment 及复用屏障；未修改 Megatron 后继算子，使其逐 tile 接受输出。

这次保留 A 的生产映射。新机制是“按实际到达就绪提前消费”，没有重新拟合 `tail` 分数，也没有把 op1/op2 的离线生产重排带入候选。只有确认消费路径存在有效重叠后，才适合再做同协议下的生产顺序消融。

## 3. 发布粒度、内存顺序与跨代复用

实现限定单节点 TP8、`M=2048,N=4096`、`K_local∈{512,2048}`、128×128 GEMM tile。每个目标拥有 64 个 tile；每 tile 分为 8 个 epilogue fragment。fragment `f` 覆盖两段各 8 行：

```text
[8f, 8f+7] 和 [64+8f, 64+8f+7]
```

一个 fragment 因而包含 16 个 H128 group。每个接收 rank 共 512 个 fragment、8192 个 H128 group。

生产端的远端编码与本地 BF16 写入都执行：

```text
本 fragment 的所有载荷/scale 写入
→ 所有写线程 __threadfence_system()
→ CTA __syncthreads()
→ 线程 0 st.release.sys 写 ready[source][fragment] = epoch
```

不能只在线程 0 执行 fence；它不足以替代其余载荷写线程的发布。每个回调只发布自己拥有的 fragment，不等待其他 CTA。

消费者由线程 0 使用 `ld.acquire.sys` 读取八个来源的同代标记，再通过 CTA 同步让其余线程进入原解码代码。同步语义参考 [NVIDIA PTX 内存一致性模型](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model)。GPU 实测仍需检查跨设备可见性及长期复用。

每个目的地原本未使用的 `source==destination` FP8 packet slot 用作控制区；该来源的数据始终从本地 BF16 缓冲读取：

| 控制内容 | 大小 |
| --- | ---: |
| 8 来源 × 512 fragment 的 uint32 ready | 16,384 字节 |
| 512 fragment 的 uint32 done | 2,048 字节 |
| 仅诊断使用的 5 个 uint64 统计量 | 40 字节 |
| 合计 | **18,472 字节** |

其他七个来源的 packet 布局、codec 和载荷不变；没有增加 packet 分配尺寸。控制区只在实例初始化时清零并跨 rank 同步。正常调用通过每实例递增的 epoch 隔离旧标记，拒绝 0 和超出 `0x7fffffff` 的代号；不在每轮清 ready/done，避免远端先发布、接收端后清零的竞态。

原复用屏障保留在消费完成之后，因此任何 rank 开始下一代生产之前，所有 rank 已结束上一代读取。所有 rank 仍须按相同次序创建和调用实例。CUDA enqueue 途中异常由实验执行器终止整组，不能重试已部分发出的同一实例。

## 4. 有限提前扫描和尾部补齐

`taco_op3_try_decode` 默认运行两轮，每轮 512 个 128-thread CTA。一个 CTA 负责一个 fragment：

1. 本代已经消费，直接退出。
2. 八来源尚未全部发布，直接退出。
3. 全部就绪，使用 A 的原 scalar CTA 解码计算，依次处理该 fragment 的 16 个 H128 group，再写 `done=epoch`。

两轮扫描在同一辅助 stream 上串行；不同 CTA 拥有不同 fragment。主 stream 等待该 stream 的完成事件后，才读取 done 并启动 `taco_op3_tail_decode`。尾部保持 A 的每 H128 group 一个 CTA 的粒度；已经消费的 fragment 跳过，其余等待各来源标记后计算。

两个阶段直接共用从 A 提取的原解码计算体，保留 `rank+1,…,rank` 来源次序，以及解码后、每次累加后的 BF16 舍入。未替换为 warp 解码器，也未改变量化预算。是否逐位一致需要新的 A/S/O GPU 检查。

**进展条件不依赖“CUDA 一定并发执行两个 kernel”。** 提前扫描没有等待生产者的循环，未获得调度重叠时会有限退出。尾部允许等待远端标记，但它只能在本 rank 的全部生产完成后运行：进入尾部的 rank 已向所有目标发布自己的贡献；仍被等待的 rank 可以继续完成本地生产，不依赖其他 rank 的尾部消费。正常执行下不会形成消费者占住本 GPU、又等待该 GPU 尚未调度生产者的环。

不使用 `use_gemmk=False` 作为“没有 Stream-K”的证明；原调度启发式保留。实际 occupancy、Stream-K 行为及资源竞争仍属于 GPU 预检内容。

## 5. 如何区分协议变化和重叠收益

新增三组同机比较：

| 组别 | 生产/codec | 发布与消费 | 用途 |
| --- | --- | --- | --- |
| A | 原 A | 原发布屏障、原解码、复用屏障 | 性能基线 |
| S | 与 O 相同，生产映射同 A | 新 fragment 发布；本地 GEMM 后按 fragment 消费；无提前扫描 | 协议串行控制 |
| O | 与 S 相同 | 有限提前扫描＋尾部补齐＋复用屏障 | 实际候选 |

`O/A` 回答是否达到实际目标；`O/S` 帮助判断提前消费的增量收益；`S/A` 显示发布标记、移除发布屏障及尾部等待实现的综合影响。不能把整个 `O/A` 差异都归于离线到达排序。

新实验对所有组统一设置 `CUDA_DEVICE_MAX_CONNECTIONS=8`，允许独立 stream 使用不同连接；统一 `FLUX_GEMM_RS_GEMM_ONLY=1`、`NCCL_NVLS_ENABLE=0`。这与旧实验默认连接数不同，必须重测同批 A。配置一致不等于保证实际 kernel 重叠。

固定版本 Megatron 的 TP 参数检查强制连接数为 1。整步准备器从固定 commit 导出独立源码副本，仅把 `megatron/training/arguments.py` 的这一检查改为要求 8；保存完整文件哈希和唯一补丁，A/S/O 共用该副本。原仓库不改，不临时伪造环境变量绕过检查。该版本 `tensor_parallel/layers.py` 对连接数的说明明确区分调度性能和数值正确性；其他 TP 通信与计算的重叠性能仍可能随连接数改变，故新配置结论必须限定在本次同机比较。

诊断开关 `H100_OP3_DIAGNOSTICS=1` 仅用于独立 profile 窗口。记录同一 GPU 上的首次解码开始/完成、最后本地 fragment 发布、提前消费数。有效证据要求：

- trace 中 GEMM 和 `taco_op3_try_decode` 位于不同 stream，时间区间有交集；
- 至少一个完整 fragment 的解码完成早于该 GPU 最后一次生产发布，且提前消费计数为正。

仅看到一个轮询/检查 kernel 与 GEMM 同时运行，不足以证明有用消费。正式计时必须关闭诊断原子操作和时间戳。

## 6. 实现与验证入口

另一台 H100 服务器可使用与 op2 相同形式的一键后台入口。先同步本轮代码到 `/workspace/TempName`，在已能使用完整 8×H100 的容器内执行：

```bash
cd /workspace/TempName
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python \
  tools/h100/phase6/detach_arrival_op3.py --execute
```

新增的 [detach_arrival_op3.py](../../../../tools/h100/phase6/detach_arrival_op3.py) 依次执行：重新生成 A/O 源码并编译 → A/S/O 算子正确性及计时 → 独立 trace/有效提前消费检查 → 两次完整训练比较。没有有效提前消费时记录 `completed_no_overlap` 并停止整步阶段；性能没有达到 1% 时记录完成但目标未通过。默认不复用 op2 构建或本机的 CPU 诊断构建。

默认与原服务器 op2 环境一致：使用启动脚本的 Python，CUDA12.8 编译、CUDA12.4 运行、jobs2、相邻目录 `../Megatron-LM`。未指定 `--nccl-root` 时在新构建目录编译 NCCL。路径不同可覆盖：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python \
  tools/h100/phase6/detach_arrival_op3.py --execute \
  --build-cuda-home /usr/local/cuda-12.8 \
  --cuda-home /usr/local/cuda-12.4 \
  --megatron /workspace/Megatron-LM
```

去掉 `--execute` 仅打印计划，不创建目录、启动进程或查询 GPU；添加 `--foreground` 可前台运行。`--op3-build` 只接受完整、可执行且来源核验通过的 op3 A/O 构建。

启动时打印本批目录，默认位于 `logs/h100/phase6/arrival-op3-2x-50steps-<时间>/`。查看 `batch-state.json`、`handoff/container-supervisor.log`；结果分别在 `operator/report.json`、`operator/profile-summary.json`、`fullstep/report.json`。成功后台启动不表示实验已完成。一键入口的 8 项 CPU 调度回归及无副作用预览已通过，本机未执行 `--execute`。

历史 A、op1/op2 源码和原始实验记录未回写。新增文件：

- [arrival_op3_sources.py](../../../../tools/h100/phase6/arrival_op3_sources.py)：从已核验 A 生成独立候选；修改 runtime、epilogue、C++ 调用关系和 Python 实例配置。
- [arrival_op3_cuda.cuh](../../../../tools/h100/phase6/arrival_op3_cuda.cuh)：控制区布局、system release/acquire 和逐 fragment 发布。
- [arrival_op3_build.py](../../../../tools/h100/phase6/arrival_op3_build.py)：A/O 源码封存、构建与 CPU/SM90 编译检查。
- [arrival_op3_bench.py](../../../../tools/h100/phase6/arrival_op3_bench.py)：独立 fresh-process A/S/O 算子正确性、重复复用、配对计时及 trace 验证。
- [arrival_op3_fullstep.py](../../../../tools/h100/phase6/arrival_op3_fullstep.py)：复用既有完整模型 worker/adapter，生成隔离 Megatron 副本，执行两次独立整步比较并验收 1%。
- [arrival_op3.sbatch](../../../../tools/h100/phase6/arrival_op3.sbatch)：新 H100 节点上的作业入口，第四个参数选择 `micro` 或 `fullstep`。
- [协议模型测试](../../../../tools/h100/phase6/test_arrival_op3_protocol.py)：乱序生产、延迟 rank、旧 epoch、唯一覆盖、复用及进展条件。

从仓库根目录准备新构建，不需要沿用旧 gxn75 的校准身份：

```bash
export OP3_PYTHON=/data/run01/scyb672/conda_envs/flux-megatron-a800/bin/python
export OP3_CUDA=/data/apps/cuda/12.8
export PATH=/data/home/scyb672/run/software/cmake-3.30.0-linux-x86_64/bin:$PATH

"$OP3_PYTHON" tools/h100/phase6/arrival_op3_build.py prepare \
  --out outputs/h100/phase6/arrival-op3-executable

"$OP3_PYTHON" tools/h100/phase6/arrival_op3_build.py build \
  --build outputs/h100/phase6/arrival-op3-executable \
  --cuda-home "$OP3_CUDA" --nccl-root "$PWD/3rdparty/nccl/build" --jobs 2

"$OP3_PYTHON" tools/h100/phase6/arrival_op3_bench.py prepare \
  logs/h100/phase6/arrival-op3-operator \
  --build outputs/h100/phase6/arrival-op3-executable

# 以下两条在完整8×H100分配内执行；默认不加 --execute 只预览。
"$OP3_PYTHON" tools/h100/phase6/arrival_op3_bench.py run \
  logs/h100/phase6/arrival-op3-operator \
  --python "$OP3_PYTHON" --cuda-home "$OP3_CUDA" --execute
"$OP3_PYTHON" tools/h100/phase6/arrival_op3_bench.py run \
  logs/h100/phase6/arrival-op3-operator \
  --python "$OP3_PYTHON" --cuda-home "$OP3_CUDA" --profile --execute

# 算子正确性和提前消费证据核验后，另建完整训练比较。
"$OP3_PYTHON" tools/h100/phase6/arrival_op3_fullstep.py prepare \
  logs/h100/phase6/arrival-op3-fullstep \
  --build outputs/h100/phase6/arrival-op3-executable \
  --megatron ../Megatron-LM --trials 2 --blocks 12
"$OP3_PYTHON" tools/h100/phase6/arrival_op3_fullstep.py run \
  logs/h100/phase6/arrival-op3-fullstep \
  --python "$OP3_PYTHON" --cuda-home "$OP3_CUDA" --execute
```

正式性能验收还需要相同 GPT6.7B 全训练、每窗口 10 步预热＋50 步正式计时、完整 A/S/O 块配对、反序复测。原始八 rank 取完整窗口最大墙钟，不把 rank、窗口内 step 或受 profiler 扰动的时间当作独立样本。算子报告始终保留 `target_validated=false`，不会自动升级为 1% 整步结论。

整步入口默认两次比较，每次 12 个完整 A/S/O 块（前后各 6 块，后半反序），共 72 个正式窗口、3600 个累计 optimizer step。每次比较分别计算配对 log-ratio 的 95% bootstrap 区间，**两次 O/A 的点估计和区间下界都至少为 1%** 才通过性能门槛。所有数值、执行范围、运行库、前端源码和同机身份检查也必须通过；机制证据另由独立算子 profile 验证，不能仅凭整步性能报告推断重叠。

## 7. 当前结果和未完成验收

**45 项 CPU 回归全部通过**：协议模型 12 项、实际源码变换与配置门槛 10 项、构建来源核验 5 项、实验生成/报告/失败清理 18 项。覆盖乱序/延迟生产、连续三代复用、所有提前扫描均未命中、唯一消费、仅线程 0 fence 的错误反例、原解码计算体一致、未完成/被篡改/带诊断的正式证据拒绝。性能报告测试使用临时合成 fixture，仅测试判定逻辑，没有计入 GPU 结果。见 [统一测试日志](../../../../outputs/h100/phase6/arrival-op3-validation-20261020/all-cpu-tests.log) 和 [测试输入哈希](../../../../outputs/h100/phase6/arrival-op3-validation-20261020/cpu-validation.json)。

实际固定 Megatron commit 的导出与校验也已运行通过：1383 个文件中仅调度配置检查文件发生登记变化，原仓库未改；见 [快照核验](../../../../outputs/h100/phase6/arrival-op3-validation-20261020/megatron-snapshot-validation.json)。Python 语法、Slurm shell 语法及本页本地链接检查通过。GPU 正确性入口另外包含 rank0/rank7 的延迟生产压力，但本轮尚未执行。

已完成真实交叉编译，见 [编译结果](../../../../outputs/h100/phase6/arrival-op3-20261020-compile/validation.json) 和 [命令日志](../../../../outputs/h100/phase6/arrival-op3-20261020-compile/compile.log)：

| 检查 | A | O |
| --- | --- | --- |
| 主机双 shape、全 rank 的物理映射 | 通过 | 通过 |
| SM90 mapping/runtime 对象编译 | 通过 | 通过 |
| 真实完整 V2 GEMM 实例编译，包含 epilogue 实例化 | 通过 | 通过 |
| 使用 Torch 2.6.0 头文件编译 C++ THS wrapper | 通过 | 通过 |

合计 18 条真实命令、8 个对象文件；CUDA Toolkit 12.8，Torch 环境 2.6.0+cu124。记录明确为 `completed=true,gpu_used=false,gpu_validated=false,linked_runtime=false`：这不是完整动态库链接或 GPU 执行通过。

新生成 A 与登记 A 的 351 项源码哈希完全一致；O 修改其中 5 个文件并新增一个 helper，其余 346 项保持一致。原 scalar codec 和生产映射均未改变。源码与哈希见 [A/O 封存目录](../../../../outputs/h100/phase6/arrival-op3-20261020/)。该 CPU 编译目录继承诊断父构建属性；正式运行应按第 6 节 fresh prepare，不能把它升级成已验收 GPU 构建。

当前登录节点为 `ln02`，无本地 CUDA 设备。读取 H100 分区后，没有可用的完整八卡节点；`sbatch --test-only` 还返回 `AssocMaxSubmitJobLimit`。这是本次 GPU 实测的实际阻碍；没有取消其他作业、实际提交排队任务或复用旧时间冒充新结果。详见 [环境核验记录](../../../../outputs/h100/phase6/arrival-op3-validation-20261020/environment-audit.json)，其中明确区分了忠实转录的只读结果与自动生成的编译证据。

实现存在清楚的性能风险：每 fragment 的 system fence/标记、有限扫描发起成本、与 GEMM 竞争 SM/内存带宽，以及尾部重复检查，都可能抵消提前消费的收益。两轮扫描也可能全部错过可消费时间窗。**本轮移除了结构上的整阶段消费依赖，但不能据此宣称物理生产尾部、跨 rank 等待或完整 step 已经缩短。**

验收必须同时补齐：新库多输入跨代正确性、有效提前消费 trace、无诊断的完整训练配对结果。若 O/A 没达到 1%，继续以 A 为基线，并报告未达标；不通过降低门槛或只选单轮结果宣称成功。
