# RTX 5090 Flux 四组量化对比（回传快照，阶段性报告）

**批次尚未完成：** 回传的 `batch-state.json` 为 `running`，停留在 `compare1/block04-flux-basic`。这是拉回时的文件快照，不代表目标机此刻一定仍在运行；目录尚无完整批次的最终报告。

协议 `rtx5090-phase6-flux-quant-four-arm-v1`；原计划 1 trial × 8 blocks × 4 组，共 32 个正式窗口、1600 个正式 optimizer step。主表只使用 **block00–block03：4 个完整块、16 个正式窗口、128 份正式 rank 记录、800 个正式 optimizer step**。

`samples.json` 收录了 **18 个正式窗口**，其中 block04 的两个窗口已汇总。另一个 block04 窗口有完整 rank 数据但尚未收录，见下文。四组第 5 个窗口尚不齐，未纳入主表。

**本报告是阶段性观察，不是原协议的完整报告。** 原计划的八块平衡排序尚未完成，已执行的四块也未构成完整平衡周期；未对剩余数据进行推断或填补。

## compare1：四个完整块

| 组别 | 平均 ms/step | 中位数 | 最慢 | tokens/s | 相对无量化均值降幅 | 最大 allocated GiB | allocated＋native GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flux＋改进量化 | 985.177 | 986.486 | 987.003 | 8315.26 | -0.040% | 17.678090 | 20.334431 |
| Flux＋量化融合 | 986.811 | 986.777 | 987.109 | 8301.48 | -0.206% | 17.008412 | 18.336583 |
| Flux＋基础量化 | 968.480 | 968.436 | 969.041 | 8458.62 | +1.655% | 17.473256 | 17.473256 |
| Flux（无量化） | 984.779 | 985.747 | 985.949 | 8318.62 | +0.000% | 17.490834 | 17.490834 |

每窗口 10 步预热＋50 步正式计时，步时为八 rank 最大 elapsed 除以 50；每组均值使用已完成的四个配对窗口，吞吐按每步 8192 tokens 计算。正降幅表示相对无量化组更快；均值比与下表配对估计的统计口径不同。

| 配对（前者 / 后者） | 配对降幅点估计 | 前者更快块数 |
| --- | ---: | ---: |
| Flux＋改进量化 / Flux＋量化融合 | +0.166% | 3/4 |
| Flux＋改进量化 / Flux＋基础量化 | -1.724% | 0/4 |
| Flux＋改进量化 / Flux（无量化） | -0.040% | 1/4 |
| Flux＋量化融合 / Flux＋基础量化 | -1.893% | 0/4 |
| Flux＋量化融合 / Flux（无量化） | -0.207% | 0/4 |
| Flux＋基础量化 / Flux（无量化） | +1.655% | 4/4 |

配对点估计采用四个完整块的对数耗时比几何均值；**原实现要求至少六个配对块且完整计划为八块，因此这里不计算 95% CI，也不作显著性或最终排名结论**。负值表示前者更慢。

## 已完成的配对窗口

| block | Flux＋改进量化 | Flux＋量化融合 | Flux＋基础量化 | Flux（无量化） |
| --- | ---: | ---: | ---: | ---: |
| block00 | 986.990 | 987.109 | 968.413 | 981.672 |
| block01 | 980.733 | 986.638 | 969.041 | 985.936 |
| block02 | 987.003 | 986.916 | 968.006 | 985.559 |
| block03 | 985.981 | 986.582 | 968.458 | 985.949 |

各单元为 ms/step；没有剔除窗口。

## block04 的不完整数据

| 组别 | ms/step | 回传记录状态 |
| --- | ---: | --- |
| Flux（无量化） | 985.630 | 已写入 samples.json |
| Flux＋改进量化 | 986.005 | 已写入 samples.json |
| Flux＋基础量化 | 968.413 | 8 份 rank 均完成且日志有 SCALE_PROCESS_COMPLETE；尚未写入 samples.json，本地按八 rank 最大 elapsed 独立重算 |

本快照没有 `block04-flux-fused`；由于四组不齐，上述三行均未混入主表、配对计算和数值差异统计。尚未写入 `samples.json` 的窗口只作为独立复核观察，不代表控制器已经将其确认为正式样本。

## 检查与数值观测

- 回传记录中的硬件/NCCL 预检、两组 native 探针、mapping 和四组完整模型 smoke 已通过核对。32 个逐 rank profile 分析记录均为 `execution_validated=true`，相应 trace 哈希均匹配。
- `profile-summary.json` 顶层 `completed` 和 `execution_validated` 仍为 `false`；控制器只有完成后续全部计时任务才将它们改为 `true`。因此不能将本批写成完整通过。
- 使用批次冻结 controller 核对 8 个 smoke/profile 窗口及 19 个已有八 rank 的正式窗口，共 216 份 rank 记录。前 18 个正式窗口的重算样本与 `samples.json` 一致；最后一个仅有 rank 记录和完成日志。native 二进制和完整 build 未随日志回传，未独立重验二进制或运行完整 `--report`；本次核对 profile trace 哈希，未重新解析全部 trace。
- 下表使用 1 个 smoke、1 个 profile 和 4 个完整计时块，共 6 个配对窗口中观测到的最大差值；不同量化/归约路径未要求训练轨迹逐位相同。这些观察不是收敛或模型质量等价证明。

| 数值观测（前者 / 后者） | loss 最大绝对差 | gradient norm 最大绝对差 |
| --- | ---: | ---: |
| Flux＋改进量化 / Flux＋量化融合 | 0 | 0 |
| Flux＋改进量化 / Flux＋基础量化 | 3.45332 | 56.3494 |
| Flux＋改进量化 / Flux（无量化） | 2.00606 | 40.4213 |
| Flux＋量化融合 / Flux＋基础量化 | 3.45332 | 56.3494 |
| Flux＋量化融合 / Flux（无量化） | 2.00606 | 40.4213 |
| Flux＋基础量化 / Flux（无量化） | 3.62291 | 96.7708 |

allocated＋native 先逐 rank 相加后取最大，包含独立 TACO workspace，但不含 CUDA context、NCCL 等内部显存；逻辑贡献字节数不等于 NCCL 实测链路流量。

环境：单机 `SL3060`，8×RTX 5090（SM120，170 SM）、TP8/DP1；PyTorch `2.8.0a0+5228986c39.nv25.06`、CUDA `12.9`、NCCL `2.27.3`、驱动 `595.84`；56 条跨卡有向 peer-access 链路不可用。无量化组采用 BF16 Reduce-Scatter，量化组的 GEMM、decoder、传输与舍入不同；组间差值为端到端组合比较。

原始证据：[批次状态](../../batch-state.json)、[正式样本](../../samples.json)、[profile 汇总](../../profile-summary.json)、[native 探针](../../native-probe/report.json)、[硬件预检](../../environment-preflight.json)、[控制日志](../../handoff/supervisor.log)、[未汇总的 block04 基础量化窗口](block04-flux-basic/process.log)。

## 构建目录迁移后的复用

`--flux-build` 的清单会记录九组 baseline 父构建。若将两个批次从 `logs/5090/phase6/` 搬到同一个 `SL3060/` 或 `SL3061/` 目录，校验器会先从当前 Flux 批次的同级目录查找父构建，再尝试清单记录的原位置；父构建清单仍须与原 SHA256 完全一致。NCCL 头文件、静态库以及父构建的 codec 也从当前构建位置检查原哈希，历史清单不被改写。

例如，将 `flux-quant-four-arm-v1-1x-50steps-20261101-201716-772182/build` 放入 `/workspace/Load-5090/SL3060/` 时，其父目录也应存在 `/workspace/Load-5090/SL3060/baseline-nine-arm-nccl-staged-v2-1x-50steps-20261101-013331-414102/build/`，并保留完整二进制、源码快照和依赖。若父构建或文件未随批次保留，需要先恢复原构建；路径重定位不会生成缺失的文件。其余源码身份、依赖、GPU 环境和 Python/CUDA ABI 校验继续执行。
