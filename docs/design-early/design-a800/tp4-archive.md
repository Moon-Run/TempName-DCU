# 单节点TP4冻结存档（2026-10-06）

为后续单节点TP8工作保留三种可复跑配置：**S1024/mb8、S2048/mb4、S256/mb32**。此前提到的“S512”已确认指S256/mb32，不新增不存在的S512结果。

## 配置与历史结果

三种均为单节点4×A800、TP4/DP1/PP1/CP1；12层、H2048、FFN8192、32 heads、BF16，8192 token/optimizer step，实际padded vocab8704。micro batch与global batch相同，每step一个微批；MLP GEMM均为`[M,N,K_local]=[8192,2048,2048]`。序列长度不同会改变attention工作量，不能跨形状相除当作优化收益。

下列下降百分比是所存正式实验中，远端＋到达优先＋选择性量化相对冻结Flux融合v2的两轮配对点估计；完整区间、另一节点复现和所有对照见[TP4记录](base-phase6-tp4.md)。这是历史结果的存档，不是2026-10-06重新测出的收益。

| 配置 | micro / global batch | 实际候选版本 | 正式窗口 | 对融合v2下降，第1/2轮 | 冻结4%验收 |
| --- | --- | --- | ---: | --- | --- |
| S1024/mb8 | 8 / 8 | opt2 final，`compact-v1`，六组 | 72 | 4.101% / 4.114% | 通过 |
| S2048/mb4 | 4 / 4 | opt2 final，`compact-v1`，六组 | 72 | 3.162% / 3.131% | 未通过 |
| S256/mb32 | 32 / 32 | opt final，旧解码分派，四基础八组 | 128 | 5.060% / 5.090% | 通过 |

S256保留当时实际运行的旧库、八组配置及4%门槛；不改成compact-v1，也不把约5%的实测收益改写成当时的验收门槛。S1024/S2048的候选来自`outputs/a800/phase6/tp4-opt2-final-build-20261004`；S256来自`outputs/a800/phase6/tp4-opt-final-build-20261004`。

## 归档入口

| 配置 | 独立存档 | 原正式实验 |
| --- | --- | --- |
| S1024/mb8 | [tp4-s1024-mb8-20261006](../../../outputs/a800/archives/tp4-s1024-mb8-20261006/archive-manifest.json) | [tp4-opt2-s1024-confirm-nodea-20261004](../../../logs/a800/phase6/tp4-opt2-s1024-confirm-nodea-20261004/acceptance.md) |
| S2048/mb4 | [tp4-s2048-mb4-20261006](../../../outputs/a800/archives/tp4-s2048-mb4-20261006/archive-manifest.json) | [tp4-opt2-s2048-confirm-nodea-20261004](../../../logs/a800/phase6/tp4-opt2-s2048-confirm-nodea-20261004/acceptance.md) |
| S256/mb32 | [tp4-s256-mb32-20261006](../../../outputs/a800/archives/tp4-s256-mb32-20261006/archive-manifest.json) | [tp4-opt-s256-confirm-nodeb-20261004](../../../logs/a800/phase6/tp4-opt-s256-confirm-nodeb-20261004/acceptance.md) |

每份存档包含：

- `template/`：当次冻结的config、worker、adapter、量化对照、到达表、mask、控制器及核验脚本。
- `artifacts/build/`：原生Flux、必须融合v2基线、各候选的动态库/Python封装/overlay和GPU映射检查程序的实体副本。加载器链接仅指向存档内部，不依赖候选构建目录的可变链接。
- `source-submission.json`、`source-verification.json`和`source-acceptance.*`：配置与已完成测量的来源；原始逐rank测量仍在原日志目录。
- `archive-manifest.json`：每个文件的完整SHA256及受保护原始代码/库清单；`archive.py`为冻结的核验与复跑准备工具。

原生Flux库SHA256为`b7a87ecd1855d85d668447fbb97d05b9574bf7fbcd8e8a45e08c52677a002714`；必须融合v2库为`fc29708ce409c0d47570403d53ea53328eb761e4f556b508bf74e582d36942c3`。各候选完整哈希见各存档的`artifacts/build/manifest.json`，不可混用不同版本的runtime与封装。

存档仍使用项目既有环境：`/data/run01/scyb672/conda_envs/flux-megatron-a800/bin/python`，Megatron提交`3ea68ad6042cc1204386ae9364358f7c4de1bc37`，CUDA 12.4运行环境。它不是整个系统环境的容器镜像；原必须baseline文件也保留供验收器对照哈希。

## 核验与复跑

以下示例采用S1024；换用另外两份存档时，不手工改batch、decoder、mask或策略数。作业号必须先按[instruction](instruction.md)核验，输出目录必须是新目录。

```bash
cd /data/run01/scyb672/hjr/TempName
tp4_archive="$PWD/outputs/a800/archives/tp4-s1024-mb8-20261006"
tp4_run="$PWD/logs/a800/phase6/NEW-TP4-REPLAY"
tp4_job=189341
python3 "$tp4_archive/archive.py" audit "$tp4_archive" --originals
python3 "$tp4_archive/archive.py" prepare "$tp4_archive" "$tp4_run" --job-id "$tp4_job"
squeue --steps -j "$tp4_job"
srun --jobid="$tp4_job" --overlap --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --gpus=4 --kill-on-bad-exit=1 \
  /data/run01/scyb672/conda_envs/flux-megatron-a800/bin/python \
  "$tp4_run/campaign.py" > "$tp4_run/driver.log" 2>&1
PYTHONPATH="$tp4_run/scripts" python3 "$tp4_run/assess.py" "$tp4_run"
python3 "$tp4_archive/archive.py" audit "$tp4_archive" --originals
```

`prepare`只生成新运行目录并更新当前作业及存档路径，保持冻结实验脚本和模型配置；不会重编译。正式复跑包括映射、codec、模型smoke/profile及两轮平衡计时。S1024/S2048为六块×六组×两轮，S256为八块×八组×两轮。

后续TP8实现使用独立目录；每次结束后以三份归档的`audit --originals`检查原代码及库。数值/加载预检可验证可运行性，但不替代正式性能回归或保证新节点上百分比绝对不变。

本次TP8适配后的[保护核验](../../../logs/a800/phase6/tp8-quant-measure-20261006/tp4-preservation-checks.json)确认：三份归档、受保护源码以及各原实验的运行库均保持原哈希。

2026-10-06三份归档均通过文件与受保护原件的哈希核验、新复跑目录的配置/路径检查，见[归档核验](../../../logs/a800/phase6/tp4-archive-check-20261006/archive-audit.json)。在189341用S1024归档完成六组smoke＋六组profile，共12窗口/48份rank记录，均通过；动态库从归档内部加载，见[复跑核验](../../../logs/a800/phase6/tp4-archive-check-20261006/replay-verification.json)。首次直接启动launcher缺少训练环境PATH，修正启动环境后在新预检子目录重试，旧失败日志保留，未改归档代码或库。本次只验证归档可运行性，不新增TP4正式性能结论。
