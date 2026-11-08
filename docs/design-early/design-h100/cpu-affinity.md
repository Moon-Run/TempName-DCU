# 启动 CPU affinity 修复与复测

2026-10-31 的新服务器输出确认：`10-118-122-131` 使用双路 Intel Xeon Platinum 8468，共 96 个物理核、192 个逻辑 CPU。GPU 0–3 的 CPU affinity 为 `0-47,96-143`（NUMA 0），GPU 4–7 为 `48-95,144-191`（NUMA 1）。实验容器允许 `0-191`，可见 cgroup 的 `cpu.max` 为 `max 100000`，`nr_throttled` 和 `throttled_usec` 都为 0；这一快照没有显示 CPU quota 节流。

旧 worker 在导入 torch 前按 `allowed_cpus[local_rank * len(allowed_cpus) // WORLD]` 选一个逻辑 CPU，并把当前线程的 affinity 缩成单 CPU。对于这台机器的 TP8，结果如下：

| rank / GPU | 旧选择 CPU | CPU NUMA | GPU NUMA | 结果 |
| --- | ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 0 | 同 NUMA |
| 1 | 24 | 0 | 0 | 同 NUMA |
| 2 | 48 | 1 | 0 | 错配 |
| 3 | 72 | 1 | 0 | 错配 |
| 4 | 96 | 0 | 1 | 错配 |
| 5 | 120 | 0 | 1 | 错配 |
| 6 | 144 | 1 | 1 | 同 NUMA |
| 7 | 168 | 1 | 1 | 同 NUMA |

因此四个 rank 的 NUMA 错配已经有直接证据。现有输出没有逐 CPU 的 `core_id` / `thread_siblings_list`，不能进一步认定哪些 rank 共用同一个物理核。驱动版本为 `595.71.05`；旧服务器 CPU 信息仍未知，所以绑核问题不能单独证明全部跨机性能差距都由它造成。

修复后的默认策略为 **inherit**：worker 保留调用环境已有的 CPU affinity，不再按逻辑编号强制缩成单核。容器 cpuset、Slurm 或外部启动器的 CPU 分配仍然生效；已正确设置的每 rank CPU 集也不会被覆盖。`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1` 等已有线程数设置保持原值。继承 CPU 集不等于自动建立 NUMA 最优绑定，也不会突破容器允许的 CPU 集合。

修复位于实际运行的 worker、算子入口及生成模板中。MA11、MA14–MA19、MAV2、基础量化等 H100 shell 入口，以及复用相同模板的 Phase7 和 RTX 5090 路径，在**新 prepare** 时获得该策略；无需给每个 shell 入口添加重复的 `taskset`。新 worker 的报告记录 `cpu_affinity_policy: "inherit"` 和实际 `cpu_affinity`，用于确认运行时拿到的集合。

必须同步修改后的源码并使用新输出目录、重新 prepare。已经冻结的 `logs/`、`outputs/`、source package 或 worker 保留原哈希与原策略；更新仓库不会自动改变这些产物。少数 A800 prepare 从历史 worker 创建新副本，会在新副本中迁移已识别的旧启动绑核块，并记录迁移前后哈希；未知绑核形状会拒绝 prepare。不要直接编辑旧 `scripts/worker.py` 或改 manifest 绕过校验。需要保留旧策略作 A/B 时，使用独立保存的旧源码和新目录，并在结果中明确 CPU 策略不同。

在原实验容器、原 Python/CUDA 环境中，先预览两组命令。以下命令从 `TempName-New` 仓库根目录执行，默认的恢复环境仍由原启动脚本加载：

```bash
H100_AFFINITY_MA11="$PWD/logs/h100/phase6/ma11-affinity-inherit-$(date +%Y%m%d-%H%M%S)"
H100_AFFINITY_BASIC="$PWD/logs/h100/phase6/basic-affinity-inherit-$(date +%Y%m%d-%H%M%S)"
bash tools/h100/phase6/run_ma11.sh --trials 1 --out "$H100_AFFINITY_MA11"
bash tools/h100/phase6/run_basic_quant.sh --trials 1 --out "$H100_AFFINITY_BASIC"
```

正式复测在同一 shell 中顺序执行，`--foreground` 保证第一组结束后才启动第二组。保持相同模型、输入、10 步预热与 50 步正式计时，不同时运行其他 GPU 实验。两条命令会按现有流程创建独立构建与新冻结 worker：

```bash
set -euo pipefail
bash tools/h100/phase6/run_ma11.sh --trials 1 --out "$H100_AFFINITY_MA11" --foreground --execute
bash tools/h100/phase6/run_basic_quant.sh --trials 1 --out "$H100_AFFINITY_BASIC" --foreground --execute
```

这两组分别产生完整 MA11 对照与基础量化对照，不是仅运行一个 GPU kernel。核对 MA11 的 `fullstep/report.md`、基础量化的 `compare1/report.md`，并查看 rank JSON 的 affinity 字段；两臂应采用相同 CPU 策略。若外部启动环境允许 `0-191` 且未给每 rank 单独分配，继承结果应为完整集合，而不是旧规则的单个 CPU。

可在正式计时之外采集现场信息。以下输出目录必须尚不存在；正在运行的 worker 可以逐个追加 `--pid 实际PID`：

```bash
python3 tools/h100/diagnostics/collect_host_performance.py \
  --output /tmp/h100-affinity-evidence-20261031-01
```

采集器只读取 CPU、NUMA、cgroup、线程 affinity 和 GPU 状态，不启动 GPU 负载。不要把未指定 worker PID 时得到的启动器 CPU 集，当成所有工作线程已经具有该集合的证明。

本次修复消除了内部单核强绑及其确定的 NUMA 错配来源；它不代表已在该服务器验收吞吐。应以修复后的同机正式计时为准，不能承诺恢复旧服务器的 449.182 ms/step（MA11）或 582.318 ms/step（基础量化）。若仍慢，再比较实际 CPU 频率、线程调度、驱动提交开销和跨 rank 等待。

开发端验证：新增 affinity 回归覆盖 20 个直接 worker、8 个生成 worker，以及 192 CPU、非连续受限 CPU 集、外部单 CPU 三种情况，共 672 个启动情境；全 tools AST 检查没有内部 affinity setter 残留。A800 的 28 项 CPU 检查、H100 重点回归中的 166 项与补充 29 项、RTX 5090 的 12 项、Phase7 worker 生成门禁均通过。历史模板转换器的 5 项专项测试和独立复审通过，`git diff --check` 通过。

未完成的环境依赖检查如实保留：MAV2 的 worker dependency-closure 测试需要本机未安装的 PyTorch；Phase7 部分完整集成测试缺少登记的历史 `calibration-complete.json` 输入。没有降低这些检查的标准、伪造历史校准文件或启动 GPU 训练；服务器复测仍须按上面的新会话命令执行。
