# MA15 profile 离线修复

`ma15-v1-attention-graph-1x-50steps-20261031-044305-267152` 已完成 16 个正式窗口、800 个 optimizer step，以及两臂各 8 份 rank profile。`fullstep/report.json` 和 `report.md` 已有统计结果；失败在最后的 profile 验收汇总。

| 同期比较 | MA11 对照 | MA15 attention graph |
| --- | ---: | ---: |
| 平均 ms/step | 597.170 | 545.655 |

配对耗时下降为 8.622%，95% CI 为 [8.070%, 9.252%]。这是原实验的单轮结果，使用原来的 CPU 绑核策略；离线修复不会把它变成新的 affinity 策略下的测量。

失败原因是候选 rank2 的一次 backward graph replay：同 stream、同 context、同设备、同一次 `cudaGraphLaunch` 关联的 GEMM 与 scale kernel，在 trace 中有约 0.094727 μs（95 ns）的区间重叠。旧分析器要求所有相邻事件区间最多重叠 0.01 μs，因而报 `GPU dependency order changed`。

[NVIDIA 对 CUPTI 时间戳的说明](https://forums.developer.nvidia.com/t/concurrent-execution-of-cuda-kernels-from-different-contexts/73333/2) 指出，结束时间采集方式可能使同一 stream 出现很小的表观重叠，不等于实际工作并发。修复只对同一次 graph launch 内的相邻计算 kernel 使用有限容差，要求 device/context/stream/correlation 均一致，且重叠同时不超过 0.5 μs 和较短 kernel 时长的 5%。本例实际允许上限为 0.1296 μs。所有接受的重叠会逐项记录在 `timestamp_order_check` 中。

计算节点的类型、数量、顺序、launch geometry、独立输入/输出拷贝、RNG prologue、跨 replay 关联和 stream 检查均保留。输入、输出、RNG 以及不同 replay 的边界仍采用原严格检查；更大的重叠、错流、错设备、错 correlation 会失败。没有删除事件或修改原始 trace。

修复当前仓库的 `ma15_profile.py` 会让新实验使用正确的检查。对于已经冻结并完成采集的旧实验，用专门的离线入口，避免修改 frozen scripts 或 manifest：

```bash
python3 tools/h100/phase6/reanalyze_ma15_profile.py \
  logs/h100/phase6/ma15-v1-attention-graph-1x-50steps-20261031-044305-267152 \
  --apply
```

从 `TempName-New` 仓库根目录执行，实验路径按实际位置填写。省略 `--apply` 只验证和打印摘要。此工具不导入 torch、不启动 GPU、不编译、不重新 prepare；原构建和旧测量不变。

它先核验冻结源码哈希，使用原冻结 controller 重算 operator 报告和 profile、正式训练报告，再仅在内存中应用记录了哈希的 MA15 graph 时间戳分析修正，重新验证全部 16 份 fullstep profile。任何验证失败均不会把 batch 状态改为完成。

成功应用后生成：

- `fullstep/profile-summary.json`：补齐的 profile 结果及分析器来源。
- `profile-reanalysis/receipt.json`：原/新分析器哈希、输入文件哈希、修改范围和逐 rank 容差记录。
- `profile-reanalysis/batch-state.before.json`：原失败状态的完整备份。
- `batch-state.json`：明确标记 `completion_mode="offline_profile_reanalysis"`，恢复本次实验完成状态，保留先前错误及备份位置。

原始 `fullstep-profile.log` 的报错、rank JSON、trace、worker、manifest、正式计时报告都保留。此入口不会自动恢复已经退出的 MA14–MA18 串行 supervisor，也不会自动启动后续实验。

`prepare` 与编译是不同步骤：`fullstep prepare` 生成新 worker、配置和哈希清单，本身不调用编译器；高层启动器未显式复用已验证构建时，会先执行 build-prepare 和 compile。普通的串行 `--execute` 默认会编译 MA14–MA18，不能把它当成仅重新 prepare。本次旧数据汇总修复不需要上述任一步骤。

开发验证：29 项 MA15 profile 测试、15 项 MA15 harness/launcher 测试和 3 项离线修复测试通过；包含真实量级的 95 ns 表观重叠、大幅重叠拒绝、跨 launch/device/context/stream 拒绝、冻结源码篡改拒绝、失败时不发布完成状态及原数据保留。
