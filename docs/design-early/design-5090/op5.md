# OP5 v2：AllGather 接收端 CE＋投影 All-to-All 的 P2P LL 兼容路径

更新：2026-11-03。**SL3060 的 `001026` 批次已通过真实大小的 All-to-All 检查，但在随后的首次 AllGather 上超时，尚未进入训练。** 当前将探针的计时屏障移到独立 Gloo 控制组，并将底层失败详情传回 supervisor；这项兼容性修复尚待目标 GPU 复测，见第 9 节。v2 候选仍保留 OP4 的完整计算组合、16 通道、AllGather 接收端 CE 和 `NCCL_P2P_LL_THRESHOLD=2097152`；此前首次大包 All-to-All 超时的修复经过见第 8 节。

目标仍是将完整 optimizer step 从指定 OP4 批次的 **832.248790ms 降至不高于 782.248790ms**，即至少减少 50ms。只测一个固定候选，不自动扫描参数或挑选最快窗口；目前没有 OP5 整步实测值，不能声称已获得该收益。此前 AllGather 字节统计范围的调整保留，见第 7 节。

## 1. 从指定 OP4 批次出发

参考：[OP4 设计与结果](op4-improve.md)、[5090 baseline](baseline.md)，以及用户指定的 [OP4 批次报告](../../../../Load-5090/logs/5090/phase6/op4-nccl-channels-single-v1-1x-50steps-20261102-105103-438654/report.md)。从原始八 rank elapsed 重新计算，两个完整窗口为：

| 正式结果 | block 0 | block 1 | 参考均值 |
| --- | ---: | ---: | ---: |
| ms/step | 833.486118 | 831.011461 | **832.248790** |
| 正式 optimizer step | 50 | 50 | 共 100 步 |

两窗口均保留；每窗口为 10 步预热＋50 步正式计时，步时取八 rank 最大 elapsed 除以 50。这里使用 OP4 父批次及其两个子批次，不再把 OP3 的 850.195ms 当作 OP5 的性能基准，也不使用先前失败的 OP4 批次。

重新解析两个窗口的 **16 份整步 trace**，按实际 CPU/autograd owner 归属通信，每 rank 先求和，再取八 rank 中位数：

| 每 optimizer step 的 profile 工作 | 次数/rank | block 0，ms | block 1，ms |
| --- | ---: | ---: | ---: |
| 全部 AllGather | 524 | 402.005 | 408.304 |
| MA8 QKV/FC1 前向 AllGather | 256 | 222.228 | 219.860 |
| row projection 反向 AllGather | 256 | 157.118 | 164.418 |
| 上述两个主要类别合计，先逐 rank 相加 | 512 | **381.030** | **385.919** |
| ReduceScatter | 260 | 157.736 | 159.960 |
| 投影 SendRecv | 256 | 164.787 | 174.231 |

512 次主要 AllGather 均未与同 rank 的非 NCCL 计算 kernel 重叠。全部 AllGather 中少量重叠来自其他 owner；不能将“全部 524 次”都描述为零重叠。两个主要类别的中位数之和，与先逐 rank 相加再取中位数不同，预算采用后者。

所有 AllGather 实际 grid 都已经是 `[16,1,1]`。这说明 OP4 的通道设置确实生效，但现有 profile 没显示它稳定缩短 AllGather；继续只从 16 调到 32 缺少依据。两个窗口、全部八 rank 的 AllGather 区间内均没有记录到 HtoD 拷贝。旧批次使用 WARN 日志，没有 SHM 连接的 INFO 证据，所以**这些旧记录不足以直接证明当时具体采用哪一种 host transport**。

这些时长含 rank 等待和 profiler 扰动，不能相加还原正式步时，也不是可直接兑现的收益。编码与解码已较短，单靠进一步消除它们没有足够的 50ms 空间；本轮保留既有 OP3 codec。

[只读分析工具](../../../tools/5090/phase6/analyze_op5_baseline.py)与[逐 rank 分析结果](op5-baseline-analysis.json)包含 trace 路径、SHA256、消息尺寸、调用数、launch 几何、计算重叠和 HtoD 诊断，可复算：

```bash
python3 tools/5090/phase6/analyze_op5_baseline.py \
  --batch ../Load-5090/logs/5090/phase6/op4-nccl-channels-single-v1-1x-50steps-20261102-105103-438654 \
  --out /tmp/op5-baseline-analysis.json
```

## 2. 当前固定候选：接收端 CE＋P2P LL 兼容路径

候选继承 **A＋MA89＋MA11＋R＋CR3 消费者融合＋OP3 warp 编码器＋OP4 16 通道**，额外固定三个 SHM 参数，并加入针对投影 All-to-All 的 P2P LL 阈值：

| 配置 | OP4 历史记录 | OP5 v2 |
| --- | --- | --- |
| `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` | `16` / `16` | `16` / `16` |
| `NCCL_SHM_USE_CUDA_MEMCPY` | 未设置 | **`1`** |
| `NCCL_SHM_MEMCPY_MODE` | 未设置 | **`2`**，接收端 |
| `NCCL_SHM_LOCALITY` | 未设置 | **`2`**，显式固定默认接收端 locality |
| `NCCL_P2P_LL_THRESHOLD` | 未设置 | **`2097152`**，覆盖每 peer 1,114,112 字节的投影交换 |
| `NCCL_P2P_DISABLE` / `NCCL_SHM_DISABLE` / `NCCL_NVLS_ENABLE` | `1` / `0` / `0` | `1` / `0` / `0` |
| 正式训练 `NCCL_DEBUG` | `WARN` | `WARN` |

[NCCL 2.27.3 的 SHM 源码](https://github.com/NVIDIA/nccl/blob/v2.27.3-1/src/transport/shm.cc#L58-L66)定义了这些参数；[接收端 progress 实现](https://github.com/NVIDIA/nccl/blob/v2.27.3-1/src/transport/shm.cc#L378-L410)在 Simple 路径上提交 `cudaMemcpyAsync(..., cudaMemcpyHostToDevice, ...)`，完成后发布就绪状态。上述参数属于本次固定版本的实验控制，探针要求实际 NCCL 为 **2.27.3**，不把它们当作跨版本通用配置。

投影的 P2P LL 选择独立于 AllGather。NCCL 2.27.3 的 [P2P planner](https://github.com/NVIDIA/nccl/blob/v2.27.3-1/src/enqueue.cc#L734-L800)按可用 LL buffer 和消息阈值选择协议，[SendRecv kernel](https://github.com/NVIDIA/nccl/blob/v2.27.3-1/src/device/sendrecv.h#L145-L155)再执行对应分支。v2 用上述阈值避开本次挂起的大包 SendRecv Simple/CE 路径；未设置全局 `NCCL_PROTO`。这是一项有源码依据的兼容修复，实际大包传输是否恢复由新增探针及原 native 检查验证。

采用 Simple/CE 的 AllGather 数据依赖为：

```mermaid
flowchart LR
    A[发送 GPU 的 NCCL kernel] --> B[主机共享内存 FIFO]
    B --> C[接收端 copy engine：HtoD]
    C --> D[接收 GPU 的本地 FIFO]
    D --> E[NCCL kernel 消费本地数据]
    E --> F[原 GEMM / 反向计算]
```

**性能假设：** 让接收 GPU 从本地显存消费数据，有机会减轻 NCCL SM 直接读取主机内存的开销，改善仍在依赖链上的大消息 AllGather。这里只是基于实现和现有 trace 提出的候选，旧数据没有证明直接 host read 已经是瓶颈。通信字节数、ring 转发次数和 PCIe/NUMA 物理带宽并未减少。

选择接收端模式 `2` 是为了只改变接收搬运；发送端仍保留 direct 路径。`SHM_LOCALITY=2` 显式锁定原默认值，不叠加 sender-locality 或 CPU affinity 调整。CE staging 也会增加 proxy 调度、memcpy/event 提交、本地 FIFO 和同步开销；16 通道分出的较小块可能使这些开销抵消带宽收益。这里没有预先把任何开销算成已经省下的时间。

CE 设置仍作用于采用相关 SHM/Simple 路径的通信，可能同时影响 ReduceScatter 和 AllReduce；投影 SendRecv 则新增 P2P LL 兼容设置。因此最终整步变化同时包含这些配置的影响，不能全部归给 AllGather。LL 路径本身也可能增加传输耗时，50ms 目标需重新验证。

不从 AllGather 入口名的 `_RING_LL` 或通用 SendRecv kernel 名字断言实际协议。AllGather 保留实际 HtoD 活动审计；All-to-All 独立 profile 检查 SendRecv 调用及其区间内没有 HtoD 拷贝。后者是未观察到接收 CE 拷贝的证据，不能单独证明 LL 协议。

独立 packet 探针和投影算子 profile 的零 HtoD 检查作为门禁；完整模型同时含其他合法 CE 通信，因此对完整模型 SendRecv 只记录 HtoD 时间相交次数与字节数，不把没有 owner 关联的时间重叠归因为投影传输错误。原有完整模型 SendRecv 次数、消息、owner 和 stream 依赖检查仍执行。

模型、TP8/DP1、32 层、hidden 4096、sequence 2048、micro batch 1、global batch 4、8192 tokens/step、FP8 packet 内容、原 source 归约顺序、BF16 舍入、消费者融合、MA8 缓存、梯度统计、Adam 和 stream 依赖均沿用 OP4。不缩短正式窗口、不关闭正确性检查、不启用 CUDA Graph、不引入新的近似量化。

## 3. 50ms 预算与结果解释

```text
历史均值 = (833.4861183352768 + 831.0114610381424) / 2
         = 832.2487896867096 ms/step
目标线   = 历史均值 − 50
         = 782.2487896867096 ms/step

如果仅由 512 次主要 AllGather 提供收益：
每次所需平均净减少 = 50,000 / 512 = 97.65625 μs
```

50ms 相当于约 **6.01% 的正式整步耗时**，或上述 381–386ms profile 量级的约 **13%**。后一个比例仅用于筛选目标难度：profile 与正式计时不同，既不能预测 CE 会节省多少，也不能按通信改善比例外推整步收益。其他通信的变化可能增加或抵消收益。

报告同时保留：

- 所有候选窗口的均值、中位数、最快/最慢值、tokens/s 和原始显存记录。
- `historical_delta_ms`：相对全部 OP4 历史窗口均值的观察差值。
- `historical_target_observed`：全部候选窗口平均步时是否不高于 782.248790ms；算子模式始终为 false。
- `target_validated=false`：当前入口只测候选，没有同期 OP4 对照，不能据此声称已经验证因果性的 50ms 增量。

若完成但未达线，仍报告全部结果；不会挑最快窗口、自动换设置重跑或把未达标写成达标。若探针或数值/路径审计失败，则父批次标记 `failed`，保留失败记录，不生成成功报告。后续要确认因果收益，需要另做同期 OP4/OP5 对照；本入口不偷偷增加旧组训练。

## 4. 测试启动命令

在目标机的 `TempName-New` 仓库根目录执行：

```bash
bash tools/5090/phase6/run_op5.sh \
  --place 60 --trials 1 --blocks 2 --execute \
  --build /workspace/Load-5090/SL3060/<OP3或ABC批次>/build
```

`--place 60` 和 `--place 61` 分别将新批次写入 `Load-5090/SL3060`、`Load-5090/SL3061`；不传时仍写入 `Load-5090/logs/5090/phase6`，显式 `--out` 优先。`--trials` 可为 1 或 2，`--blocks` 为每个 trial 的正整数窗口数，每窗口 50 个正式训练步骤。

`--build` 识别 OP3 或 OP1 ABC 构建清单：OP3 直接复用，ABC 则用于编译新的 OP3；原有 `--op3-build`、`--abc-build` 参数仍可使用。open-compress 的 build 不属于这条构建链，不能传给 OP5。若不传构建参数，脚本会在新批次下依次构建 baseline、ABC 和 OP3。首次候选窗口运行成功后，批次根目录的 `build` 是指向实际 OP3 构建的符号链接，可供后续运行复用；完整构建及其依赖必须保留。需要编译时默认并行数为 `nproc`，可用 `--jobs N` 覆盖。

OP4 参考目录的自动定位不等于 OP3 构建可任意搬迁：OP3/ABC 清单保存构建及其依赖的绝对路径，仍须通过各自校验。如果构建目录已从原位置搬走，不能仅修改 `--build` 路径就假定可复用；可使用原位置的完整构建及其依赖，或不传 `--build` 重新构建。

**一条命令自动完成：硬件/NCCL 预检 → 真实大小 All-to-All 探针 → 同尺寸 AllGather 探针 → 原 native 与算子检查/计时 → 全模型 smoke/profile → 10 步预热＋50 步正式训练 → OP5 审计和报告。** 默认后台运行；加 `--foreground` 可前台观察。不带 `--execute` 只预览，不访问 GPU、不创建结果目录、不编译。

这条命令自动创建新的 v2 批次；重新测试时不传此前失败批次的 `--out`。默认参考是以下 OP4 父批次：

```text
../Load-5090/SL3060/op4-nccl-channels-single-v1-1x-50steps-20261102-105103-438654/   # --place 60
../Load-5090/SL3061/op4-nccl-channels-single-v1-1x-50steps-20261102-105103-438654/   # --place 61
```

指定 `--place` 时优先查找对应目录；若对应目录不存在且旧的 `../Load-5090/logs/5090/phase6/` 下该批次仍存在，则回退旧位置。不传 `--place` 时默认使用旧位置。若它存放在其他位置，加 `--reference-batch /path/to/completed-op4-parent-batch`，显式路径优先，应传父目录，不能传其中 `block00-candidate`。入口校验该批次所有正式窗口和文件哈希，从原始 rank elapsed 重算均值；数值轨迹及 trace 对照固定使用第一个历史子窗口。没有重新运行历史 GPU 工作，也不声称重新验证了归档中缺失的历史二进制。

本次修复后的首次测试建议保持上面的 `--trials 1 --blocks 1`；需要观察重复性时可使用 `--blocks 2`，它表示两个候选正式窗口，共 100 步。`--operator-only --foreground --execute` 会运行同尺寸通信探针和投影算子后结束，不运行完整模型，也不产生整步达标结论。

需要同步下列文件的当前版本，以及已有的 `op4_evidence.py`；其余依赖使用当前完整仓库，不能只复制 shell：

```text
tools/5090/phase6/run_op5.sh
tools/5090/phase6/op5.py
tools/5090/phase6/op5_evidence.py
tools/5090/phase6/op5_probe.py
tools/5090/phase6/op4_evidence.py
```

OP4 审计 helper 只新增可选的通信检查器、参考通道数和干预说明参数；不传参数时沿用原行为。OP3 worker、codec、GEMM 构建和训练入口未修改，可以复用命令中已有的 OP3 build。MA9/MA11 兼容库继续按原流程准备。

启动环境、`/etc/nccl.conf`、`~/.nccl.conf` 和 `NCCL_CONF_FILE` 中若有本轮通道、SHM CE、P2P LL 阈值、算法、全局协议或 tuner 覆盖，入口会明确报错，避免隐藏变量污染实验；脚本不会编辑配置文件。只在八卡空闲时运行，不与其他 GPU 实验并行。

## 5. 新增验证与证据位置

[控制器](../../../tools/5090/phase6/op5.py)和[启动脚本](../../../tools/5090/phase6/run_op5.sh)复用 OP3 的完整门禁，再通过 [OP5 审计](../../../tools/5090/phase6/op5_evidence.py)与历史 OP4 核对。

### 5.1 同尺寸通信探针

[op5_probe.py](../../../tools/5090/phase6/op5_probe.py)先执行原有硬件/Apex/NCCL 预检，再启动独立的八 rank 进程：

1. 使用与训练相同的 GPU、NCCL 2.27.3、通信配置和 `CUDA_DEVICE_MAX_CONNECTIONS=1`。先测投影真实 wire 形状：每个远端 peer `8192×136=1,114,112` 字节，每 rank 合计 `7,798,784` 字节，类型 `uint8`，self split 为 0。
2. All-to-All 使用随 source、destination 和轮次变化的非均匀字节模式，三轮检查完整输出位串；在非默认 stream 上复用缓冲、调用异步 Work 并等待，其中一轮延迟 rank7。验证覆盖全部 56 条有向远端链路。之后完成独立预热、延迟计时及 10 次 profile，检查 SendRecv 调用完整且其区间内没有 HtoD 拷贝；任何失败都会阻止后续训练。
3. 再测 AllGather：每 rank 输入 `1,048,576` 个 BF16（2MiB），输出 `8,388,608` 个 BF16（16MiB）。继续执行三轮位串校验和缓冲复用检查，20 次预热＋100 次正式调用，并另做 10 次独立 profile。通信延迟报告采用 CUDA event 和同步墙钟的八 rank 最大平均值；它们均不是 optimizer step，不能直接换算为正式训练收益。
4. 仅探针使用 `NCCL_DEBUG=INFO`、`NCCL_DEBUG_SUBSYS=INIT,SHM`。要求实际连接日志为 `SHM/direct/CE`，八 rank 均覆盖 16 通道；它说明 communicator 配置的 SHM 连接，不能据此断言其中每一种通信都实际经过 CE。正式训练仍使用原来的 WARN。
5. 每次 profile AllGather 必须实际启动 16 CTA，并包含同 device、完整落在该 kernel 时间区间内的 HtoD 拷贝。七份远端贡献的理论量为 **14 MiB/次**；保留已采用的 **10.5–17.5 MiB/次**统计区间（含边界）。实际逐次字节数和跨边界排除的拷贝数量均留档。时间归属无法证明每个拷贝的实际归属或完整 payload 覆盖，独立逐位正确性检查和 transport 日志检查继续保留。

探针通过后才启动完整候选。它与训练使用独立 communicator，因此完整候选结束后的审计还会对八 rank 的 **524 次模型 AllGather 逐次检查同样的 HtoD 时间归属统计范围**。若当前 profiler 无法提供这些证据，批次失败并要求检查原 trace，不降级成“设置变量即成功”。HtoD 时长包含在等待它的 NCCL kernel 区间中，不能再加一次。

### 5.2 原有数值与路径门禁

继续执行 native packet/舍入/输出所有权、投影算子输出与四类梯度及三步更新逐位检查、模型 smoke、八 rank 全模型 profile、完整正式训练与原始文件哈希核验。

与 OP4 历史记录相比，只允许三个 CE 配置及 `NCCL_P2P_LL_THRESHOLD=2097152` 的指定差异；保留硬件、软件、初始参数、输入、RNG、CPU affinity 和模型身份检查。loss、gradient norm、全部最终参数指纹、smoke 参数更新轨迹及投影输出仍要求一致。沿用 OP4 的既有规则：历史 smoke 梯度哈希差异记录但不阻断，哈希不能说明误差大小；没有把这一项改成新的放宽条件。

AllGather 调用数、BF16 消息尺寸、TP group、真实 launch、16-CTA grid 继续严格审查；其余 NCCL 入口、payload、进程组和调用数与 OP4 核对。完整候选必须同时通过原 OP3 replay 和 OP5 的 CE 审计。

### 5.3 输出与只读复核

新批次位于：

```text
../Load-5090/logs/5090/phase6/op5-shm-recv-ce-p2p-ll-single-v2-<trials>x-50steps-<时间>/
```

| 位置 | 内容 |
| --- | --- |
| `report.md` / `report.json` | 正式步时、历史差值、50ms 目标线与达线观察 |
| `batch-state.json` / `handoff/supervisor.log` | 总体状态、失败原因与当前阶段 |
| `transport-probe/report.json` | 真实大小 All-to-All 与 AllGather 的正确性、延迟、SHM 连接及 profile 审计 |
| `transport-probe/process.log` / `rank*-profile.json.gz` / `rank*-alltoall-profile.json.gz` | 探针原始 INFO 日志、八 rank 的两类 trace；均在 `transport-probe/` 下 |
| `candidate-evidence.json` | 历史数值回归、全部模型 AllGather CE 覆盖、原始文件哈希 |
| `compare1/block00-candidate/` | 原 OP3 子批：构建校验、算子、smoke、profile、50 步正式窗口 |

`completed` 表示检查完成，是否达到历史 50ms 目标另看 `historical_target_observed`。完整运行的每窗口显存记录位于子批 `report.json` 和逐 rank JSON；Torch allocated/reserved 不覆盖 NCCL 的新增 GPU FIFO、主机共享内存和 CUDA context，不能把它们当作整卡总占用。

对已完成结果只读复核：

```bash
bash tools/5090/phase6/run_op5.sh \
  --out /path/to/completed-op5-batch --report
```

`--report` 重新核对探针、子批、历史记录、trace 和源码快照，不续跑训练。结果及依赖 build 应保留在目标机；本地回传副本缺少二进制时，不能把只读历史分析当成完整 replay。

## 6. 本地验证与当前边界

本地已核对 OP4 两个正式窗口原始 elapsed、历史证据哈希，并解析全部 16 份整步 trace。原版的 **13 项 OP5 测试、41 项 OP4 回归、49 项 OP3 回归，共 103 项全部通过**；shell 语法、完整/算子模式预览、文档本地链接及文件空白检查通过。CPU 测试覆盖候选独占配置、只预览无副作用、50ms 判据、历史双窗口均值、数值门禁、CE 时间归属统计、真实 transport 日志、失败中止、探针先于训练，以及报告与探针证据的绑定。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tools/5090/phase6 -p 'test_op5*.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tools/5090/phase6 -p 'test_op4*.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tools/5090/phase6 -p 'test_op3*.py' -v
bash -n tools/5090/phase6/run_op5.sh
bash tools/5090/phase6/run_op5.sh --trials 1 --blocks 1
git diff --check
```

**当前没有通过完整 OP5 验收的 GPU 运行，也没有 OP5 算子或整步实测值。** 是否净省至少 50ms，以目标机完整运行的结果为准。

## 7. 探针失败与审计范围调整（2026-11-02）

[失败批次](../../../../Load-5090/logs/5090/phase6/op5-shm-recv-ce-single-v1-1x-50steps-20261102-122453-878129/batch-state.json) 通过硬件预检及 8 rank 通信探针逐位正确性检查，在 trace 审计时因为某次统计到 15,794,176 字节而未满足原先恰好 14,680,064 字节的门禁。检查全部 80 次探针 AllGather 后，按完整时间包含统计的字节范围为 11,599,872–17,760,256 字节；另有 3 次拷贝跨越相应 AllGather 的结束边界，已排除并记录。按用户要求，当前门禁调整为 11,010,048–18,350,080 字节，统计仍按同设备和时间归属，不声称证明完整传输字节。使用新规则对该批次探针原始记录只读重审通过，含 `SHM/direct/CE` 连接日志及 8 rank 的逐位检查；原批次状态不变，算子、完整模型及正式计时尚未运行。


## 8. 大包 All-to-All 超时与 v2 兼容修复（2026-11-02）

用户指定的 [144159 批次状态](../../../../Load-5090/logs/5090/phase6/op5-shm-recv-ce-single-v1-1x-50steps-20261102-144159-279295/batch-state.json)为 `failed`。其 [AllGather 探针报告](../../../../Load-5090/logs/5090/phase6/op5-shm-recv-ce-single-v1-1x-50steps-20261102-144159-279295/transport-probe/report.json)已通过，记录 GPU 延迟 **3372.067μs**、同步墙钟延迟 **3373.441μs**，并保留八 rank 逐位校验及 `SHM/direct/CE` 连接证据。这只是通信微基准结果，不能当作完整训练性能。

随后子批在 `native-reference-wire-and-mapping-probes` 阶段失败。[native 原始日志](../../../../Load-5090/logs/5090/phase6/op5-shm-recv-ce-single-v1-1x-50steps-20261102-144159-279295/compare1/block00-candidate/native-probe/op3-combined/process.log)第 11 行起显示八个 rank 均在第一次大包交换上超时：

```text
SeqNum=1, OpType=ALLTOALL_BASE
NumelIn=7798784, NumelOut=7798784
Timeout(ms)=300000
```

这里每 rank 向七个远端 peer 各发送 `1,114,112` 字节的量化 packet，self split 为 0；NCCL watchdog 超时后终止进程。日志没有显示此前的 HtoD 字节区间门禁失败，也没有 OOM 证据。原有小消息 NCCL 预检和大消息 AllGather 通过，未覆盖这笔实际投影 wire 大小的 All-to-All，这是此次补充前置探针的原因。

**修复采用有限范围的兼容路径：** 新增 `NCCL_P2P_LL_THRESHOLD=2097152`，保留 16 通道与 AllGather 的 SHM 接收端 CE。NCCL 2.27.3 的 P2P planner 在具备 LL buffer 时按该阈值选择 LL，消息大小覆盖每 peer 的投影 packet；[SHM proxy](https://github.com/NVIDIA/nccl/blob/v2.27.3-1/src/transport/shm.cc#L378-L410)只对 Simple 使用 CE，因而这一设置避开已挂起的 P2P Simple/CE 组合。具体内部为何未能推进，现有日志不足以确定；不能据此把某个 CUDA 队列、驱动或 NCCL 内部缺陷写成已经证实的根因。

v2 将真实大小、self split 为 0 的 All-to-All 放到同尺寸 AllGather 之前，完成多轮位串检查、延迟及独立 profile 后，才继续原 native/算子/模型检查。完整 native packet、消费者舍入、输出所有权和数值门禁均继续执行。全局 `NCCL_PROTO`、`CUDA_DEVICE_MAX_CONNECTIONS=1`、codec、模型和 OP3 build 保持原配置，原有 300 秒 native collective 超时也不延长。之前按用户要求采用的 AllGather ±25% 统计区间保持不变。

重新同步当前脚本后，使用第 4 节命令启动新批次；默认目录和协议变为 `op5-shm-recv-ce-p2p-ll-single-v2`，探针协议也升级为 v2。旧 `144159` 批次及其失败状态保留，不重写为成功、不用 `--report` 代替复测。**当前修复尚无目标机 GPU 通过记录；All-to-All 是否恢复、LL 兼容路径的额外开销及 50ms 目标，都以新批次完整结果为准。**

本次修复的 **26 项 OP5 CPU 测试与 41 项 OP4 回归，共 67 项全部通过**；shell 语法、完整/算子模式预览、文档本地链接和 `git diff --check` 通过。测试覆盖实际 packet 调用流程中的异步等待、全部 rank 路由、旧数据/部分拷贝拒绝、56 个有向连接、trace 消息尺寸和 splits、记录哈希绑定，以及完整模型时间重叠仅作诊断的边界。CPU 测试不能替代目标 GPU 的兼容性验证。

## 9. All-to-All 后首次 AllGather 超时（2026-11-03）

[001026 批次](../../../../Load-5090/SL3060/op5-shm-recv-ce-p2p-ll-single-v2-1x-50steps-20261103-001026-767905/batch-state.json)的直接故障在 [transport-probe/process.log](../../../../Load-5090/SL3060/op5-shm-recv-ce-p2p-ll-single-v2-1x-50steps-20261103-001026-767905/transport-probe/process.log)：八个 rank 的首次 AllGather 均出现 `SeqNum=136, OpType=_ALLGATHER_BASE, Timeout(ms)=120000`，最后完成的序号是 135。此前各 rank 已完成 3 次 packet 逐位检查、20 次预热、100 次计时和 10 次 profile，共 133 次 All-to-All，记录均为 `bitwise_correct=true`。rank JSON 中随后保存的 BF16 校验失败可能发生在 watchdog 中止通信之后，不能据此判定为独立的数据损坏根因。

原探针的两次计时屏障调用默认 NCCL 组。[PyTorch 的 barrier 实现](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L4725-L4784)会插入单元素 AllReduce，因此首次 AllGather 之前实际执行的是 133 次 packet 交换加 2 次额外 AllReduce。这与失败序号一致，但现有日志不足以证明具体的 NCCL 内部挂起机制。

当前兼容性处理将 `measure()` 的屏障改为独立 Gloo 控制组，屏障仍位于计时区间外，GPU 同步继续保留。All-to-All 与 AllGather 仍按原顺序共用同一 NCCL 组，真实 packet 大小、异步等待、逐位检查、CE profile 门禁、120 秒 collective 超时及正式训练配置均保留。**这移除了探针附带的 AllReduce，是否消除本次 GPU 挂起仍需复测；探针通过也不能替代后续 native、算子和完整模型检查。**

探针协议升级为 `rtx5090-op5-shm-ce-probe-v3`，`settings.coordination_backend=gloo` 纳入命令、rank 记录和报告核验；父批次仍为相同的 OP5 v2 候选。失败时，`handoff/supervisor.log` 和 `batch-state.json.error` 会直接包含底层 watchdog 信息、rank 阶段与原始日志路径。旧失败批次保留，不能用修改后的脚本继续其冻结的 `--inside` 命令。

按该批次 `handoff/launch.json` 恢复的参数，在目标机仓库根目录启动**新批次**：

```bash
bash tools/5090/phase6/run_op5.sh \
  --place 60 --trials 1 --blocks 1 --jobs 128 --worker-timeout 3600 \
  --python /workspace/venvs/tempname-5090/bin/python \
  --megatron /workspace/Megatron-LM \
  --cuda-home /usr/local/cuda-12.9 --build-cuda-home /usr/local/cuda-12.9 \
  --reference-batch /workspace/Load-5090/SL3060/op4-nccl-channels-single-v1-1x-50steps-20261102-105103-438654 \
  --execute
```

本次 OP5 CPU 回归共 37 项，35 项通过，2 项因本地缺少对应历史归档而跳过。新增检查覆盖屏障组隔离及计时边界、旧探针证据拒绝、残缺 rank 文件下的错误摘要，以及子进程失败传回 supervisor。当前环境没有目标机 CUDA/GPU，尚无此次修改后的 GPU 通过记录。
