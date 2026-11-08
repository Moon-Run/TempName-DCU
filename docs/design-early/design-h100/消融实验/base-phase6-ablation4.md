# H100 GPT 6.7B TP8 消融实验4：60步完整实测总结

更新：2026-10-19。依据回传提交 `fbaf8b5d`（Add Ablation-5 data），汇总批次 `ablation5-then4-60steps-20261018-230611-029519` 下 `ablation4/compare1`、`ablation4/compare2`。**两次均已完成：每次50个正式窗口、每窗口预热10步＋正式60步，合计100个窗口、6000个正式optimizer step。** 两套严格报告及运行状态均完成，CPU离线复算一致。

本页第1—6节只使用本批60步结果。第8节保留原2026-10-18的20步历史记录，两个口径分别统计。消融5预算结果见 [消融5实测总结](base-phase6-ablation5.md)。

## 1. 实测结论

- **C完整版在两次实验、四轮中均快于B及融合v2。** C相对B的两次配对耗时下降为 **3.344% [3.157%, 3.524%]**、**3.524% [3.187%, 3.876%]**；C相对融合v2为 **3.460% [3.182%, 3.765%]**、**3.665% [3.296%, 4.059%]**（方括号为95% CI）。这些比较的四个逐轮区间也均为正。
- C两次合并平均 **516.960 ms/step**，A为534.490、B为535.344、融合v2为536.057、Megatron＋量化为1009.491。C相对Megatron的两次配对下降为48.666%、48.914%。**C/B同时改变选择性量化覆盖率和候选实现，不能把约3.3%—3.5%的收益单独归因于mask。**
- **调度的单独增益仍缺少跨两次一致的区间支持。** A/Flux的compare1为+0.305% [0.026%, 0.603%]，compare2为+0.280% [−0.055%, 0.630%]；B/A两次点估计为−0.185%、−0.135%，区间均跨零。不能宣称到达优先稳定提速，也不能据此证明严格等效或稳定退化。
- 四轮全部完整块保留，无慢窗口排除；主表包含所有正式样本。训练loss为固定合成token上的短程轨迹，解释见第3、5节。

## 2. 数据完整性与固定条件

| 项目 | compare1 | compare2 | 合计 |
| --- | ---: | ---: | ---: |
| 轮数 / 完整五组块 | 2 / 10 | 2 / 10 | 4 / 20 |
| 正式窗口 / rank记录 | 50 / 400 | 50 / 400 | 100 / 800 |
| 正式optimizer step（不重复计rank） | 3000 | 3000 | 6000 |
| smoke/profile窗口 / rank记录 | 10 / 80 | 10 / 80 | 20 / 160 |
| profile trace / 策略角色mapping | 40 / 8 | 40 / 8 | 80 / 16 |

两次run_id分别为 `54e3d5c9eb1e4e9cb83029af9cd0dfba`、`295e0809e892438fbceb07b1a4a950df`。每次均为 `--blocks 5 --rounds 2 --nvls-enable 0`，各自重新预检；第二轮反转块顺序和块内策略顺序。顶层批次按“消融5两次→消融4两次”执行，四次总计200个正式窗口、1600份正式rank、12,000个正式step，全部完成。

单节点 `gxn75`、8×H100 80GB；GPT6.7B，32层、H4096/FFN16384、32 heads、S2048、micro batch1/global batch4、TP8/DP1/PP1/CP1、BF16，seed1234、token_seed4321、padded vocab33792。PyTorch2.6.0+cu124、运行CUDA12.4、NCCL2.21.5；Megatron提交固定为 `3ea68ad6042cc1204386ae9364358f7c4de1bc37`，复用原消融4构建。冻结通信设置包含 `NCCL_NVLS_ENABLE=0`、`NCCL_DEBUG=WARN`、`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`。

五组均覆盖32个Attention输出投影和32个MLP输出投影。Megatron使用原生GEMM＋Tensor TACO；Flux/A/B共用固定融合v2、单工作区和输出clone，分别使用原始顺序、交替顺序、交替＋到达表；C沿用优化codec、双工作区和实例持有配置/输出。详细定义见 [实验方案](消融实验4.md)。

**C的预算仍为远端来源贡献的1/64，最多每来源7/448个远端tile。** Attention实际选择 `[7,7,7,7,4,7,7,7]`，合计53/3584＝1.479%；MLP为 `[7,7,7,7,1,7,7,7]`，合计50/3584＝1.395%。两角色合计103/7168＝1.437%。这是远端通信贡献比例，不是模型参数比例；预算上限和实际覆盖量不同。

新会话的 `execution_timing` 与 `scripts/config.json` 均冻结10步预热、60步正式计时。`submission.config` 中保留的20步字段属于原构建身份，本次执行和统计使用独立冻结的60步配置，未将旧20步会话直接改名复用。

## 3. 全样本完整step耗时与loss

窗口耗时为 `max(8 rank elapsed_seconds) × 1000 / 60`。每步4个微批、8192 token，计入全模型前向/反向、通信、optimizer更新、调度及清梯度。主表展示C、A、B、原本的Flux v2、Megatron＋量化五组。**Flux v2（原始顺序＋完整量化）**对应本批实际执行的 `flux_fused_v2_attn_mlp`，使用本次60步数据；它采用原始Flux顺序、固定融合v2及全远端量化，A/B在同一融合实现上分别增加交替排序、交替＋到达排序。[消融6](base-phase6-ablation6.md) 的“Flux＋朴素量化”则使用独立Tensor TACO通信实现，两者是不同的实验组。

| **指标** | **C：完整版（选择性量化）** | **A：交替＋完整量化** | **B：交替＋到达＋完整量化** | **Flux v2（原始顺序＋完整量化）** | **Megatron＋量化** |
| --- | ---: | ---: | ---: | ---: | ---: |
| 平均耗时（ms/step） | 516.960 | 534.490 | 535.344 | 536.057 | 1009.491 |
| C完整版相对耗时下降 | 0.000% | +3.280% | +3.434% | +3.562% | +48.790% |
| 平均训练loss | 3.161155 | 3.184181 | 3.184181 | 3.184181 | 2.362192 |
| 最终训练loss的平均 | 3.011703e-05 | 6.098071e-06 | 6.098071e-06 | 6.098071e-06 | 9.502610e-06 |

每组包含两次实验的20个窗口、1200个正式optimizer step。平均耗时对未舍入窗口取算术平均；“C完整版相对耗时下降”按 `100 × (1 − C平均耗时 / 该列平均耗时)` 计算，正值表示C更快。这是合并均值的描述性比值，区别于第4节的同轮同块配对统计。

两项loss均从原始正式计时rank记录复算，仅计rank0，排除预热、smoke和profile。每窗口八rank的loss序列一致；每组取 `W=20`：

```text
平均训练loss = sum_w(sum_s(losses_rank0[w][s])) / (W × 60)
最终训练loss的平均 = sum_w(losses_rank0[w][-1]) / W
```

最终loss指第60个正式step的值（之前另有10个预热step），不是整个批次最后一个窗口。由于末步值很小，表中采用科学计数法；两次各策略的60步loss轨迹逐项一致，A/B/Flux三组轨迹也一致。这里没有将重复的TP记录当成独立样本。

| 指标 | C：完整版（选择性量化） | A：交替＋完整量化 | B：交替＋到达＋完整量化 | Flux v2（原始顺序＋完整量化） | Megatron＋量化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| compare1 平均耗时（ms/step） | 518.244 | 535.183 | 536.170 | 536.820 | 1009.557 |
| compare1 中位耗时（ms/step） | 518.516 | 534.939 | 535.963 | 537.282 | 1009.629 |
| compare2 平均耗时（ms/step） | 515.677 | 533.796 | 534.517 | 535.295 | 1009.424 |
| compare2 中位耗时（ms/step） | 515.612 | 534.649 | 535.099 | 534.827 | 1008.041 |
| 两次合并吞吐（tokens/s） | 15846.47 | 15326.77 | 15302.32 | 15281.95 | 8114.98 |

逐轮平均耗时（ms/step，每格5个窗口）：

| 实验 / 轮次 | C：完整版（选择性量化） | A：交替＋完整量化 | B：交替＋到达＋完整量化 | Flux v2（原始顺序＋完整量化） | Megatron＋量化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| compare1 第1轮 | 519.143 | 536.201 | 536.190 | 537.193 | 1010.006 |
| compare1 第2轮 | 517.345 | 534.166 | 536.151 | 536.447 | 1009.108 |
| compare2 第1轮 | 516.665 | 534.254 | 534.056 | 535.288 | 1010.378 |
| compare2 第2轮 | 514.689 | 533.338 | 534.979 | 535.301 | 1008.471 |

吞吐为 `8192000 / 平均ms`，即总token数除以总耗时。每次中位数基于该次10个窗口，不用两次中位数的平均代替合并中位数。

## 4. 同场完整块配对比较

每次各有10个完整五组块。下降率为 `100 × [1 − exp(mean(log(T_X/T_Y)))]`，正值表示X更快；95%区间按完整块bootstrap10000次、seed20261009计算。60个step和8个TP rank均不视作独立样本。两次分别报告，不额外拼接跨实验置信区间；未做多重比较校正，无4%验收门槛。

| X / 参考Y | compare1 配对耗时下降% [95% CI] | compare2 配对耗时下降% [95% CI] |
| --- | ---: | ---: |
| Flux v2 / Megatron | +46.826 [+46.744, +46.904] | +46.970 [+46.769, +47.151] |
| A / Megatron | +46.989 [+46.824, +47.162] | +47.119 [+46.931, +47.330] |
| A / Flux v2 | +0.305 [+0.026, +0.603] | +0.280 [-0.055, +0.630] |
| B / Megatron | +46.890 [+46.763, +47.002] | +47.048 [+46.812, +47.320] |
| B / Flux v2 | +0.120 [-0.156, +0.395] | +0.146 [-0.212, +0.580] |
| B / A | -0.185 [-0.443, +0.142] | -0.135 [-0.493, +0.206] |
| C / Megatron | +48.666 [+48.533, +48.799] | +48.914 [+48.736, +49.077] |
| C / Flux v2 | +3.460 [+3.182, +3.765] | +3.665 [+3.296, +4.059] |
| C / A | +3.165 [+2.801, +3.570] | +3.394 [+3.099, +3.693] |
| C / B | +3.344 [+3.157, +3.524] | +3.524 [+3.187, +3.876] |

C/B四个逐轮下降为3.180%、3.508%、3.255%、3.793%，对应区间全部为正。A/Flux仅一次合并区间完全为正；B/A两次合并区间均跨零。C/B涉及codec、工作区和配置/输出生命周期变化，因此选择性预算本身的影响应看 [消融5](base-phase6-ablation5.md) 中固定C实现的比较。

## 5. 慢窗口与数值解释

| 实验 / 轮次 | 最大窗口 / 本轮同组中位数 | 保留完整块 | 排除块 |
| --- | ---: | ---: | ---: |
| compare1 / 第1轮 | 1.007474 | 5/5 | 无 |
| compare1 / 第2轮 | 1.005198 | 5/5 | 无 |
| compare2 / 第1轮 | 1.009609 | 5/5 | 无 |
| compare2 / 第2轮 | 1.008727 | 5/5 | 无 |

阈值固定为本轮同组窗口中位数的1.10倍，任一策略触发即同步排除整块；本批20个块全部保留，敏感性结果与全样本一致。正式窗口均通过八rank、60步有限loss/grad_norm、实际投影路由/调用、运行身份、通信快照及零skip/fallback检查。

两次smoke预检的最大相对BF16 L2误差：C为0.439100%，A/B/Flux为2.545994%，Megatron＋量化为2.546476%；各组均低于既定5%门槛，相对独立codec参考也低于1%门槛。这是预检投影输出误差，不能直接当作全模型任务准确率。

冻结的 [worker](../../../../logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/ablation4/compare1/scripts/worker.py) 在每个窗口初始化模型，并一次生成固定的4个合成微批；每个optimizer step都重新遍历同一组 `batches`。60步末尾接近零的loss反映这一固定训练输入上的拟合，不能据此判断真实语料泛化、验证困惑度或长期收敛。两次重复计时也不等于不同训练种子的重复质量评估；20步历史均值/末步loss不与本批混合。

## 6. 核验与原始证据

在临时目录建立回传输入的只读链接，仅将重算报告写入临时目录，调用维护代码的 `report(..., offline=True)`。两套报告经JSON序列化后，除 `offline` 标志外与回传报告逐项一致，包括逐窗口耗时、完整块配对区间、敏感性分析和证据哈希；顶层批次登记的报告SHA256也吻合。

另从全部800份正式rank记录核验60步loss/grad_norm、完成身份与有限值，确认每窗口八rank的loss序列完全一致，并用rank0复算两项loss均值。两次输入、模型/输入/运行环境指纹、运行库登记、冻结worker、角色计划与通信配置一致；run_id和预检各自独立。未修改原始报告、rank、日志或状态，未启动新的GPU实验。离线核验依据回传证据，不声称重新加载执行端动态库。

- compare1：[报告JSON](../../../../logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/ablation4/compare1/report.json)、[报告Markdown](../../../../logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/ablation4/compare1/report.md)、[submission](../../../../logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/ablation4/compare1/submission.json)、[预检](../../../../logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/ablation4/compare1/preflight.json)、[完成状态](../../../../logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/ablation4/compare1/all-state.json)、[原始窗口](../../../../logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/ablation4/compare1/results/)。
- compare2：[报告JSON](../../../../logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/ablation4/compare2/report.json)、[报告Markdown](../../../../logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/ablation4/compare2/report.md)、[submission](../../../../logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/ablation4/compare2/submission.json)、[预检](../../../../logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/ablation4/compare2/preflight.json)、[完成状态](../../../../logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/ablation4/compare2/all-state.json)、[原始窗口](../../../../logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/ablation4/compare2/results/)。
- [本实验批次状态](../../../../logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/ablation4/batch-state.json)、[四次实验总状态](../../../../logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/batch-state.json)、[启动参数](../../../../logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/handoff/launch.json)、[监督进程身份](../../../../logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/handoff/supervisor.json)。

## 7. 复测入口

使用 [`ablation4_repeat.py`](../../../../tools/h100/phase6/ablation4_repeat.py)，在原75容器、仓库根目录执行：

```bash
cd /workspace/TempName
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python \
  tools/h100/phase6/ablation4_repeat.py --execute
```

默认复用本次已完成的四套构建 `outputs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/build`，在 `logs/h100/phase6/ablation4-repeat-时间戳/` 下创建全新的 `compare1`、`compare2`，不覆盖上面这批结果。默认Python为启动脚本的解释器，运行CUDA为 `/usr/local/cuda-12.4`。换路径时显式传参：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python \
  tools/h100/phase6/ablation4_repeat.py \
  --build /workspace/TempName/outputs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/build \
  --out /workspace/TempName/logs/h100/phase6/ablation4-new-batch \
  --megatron /workspace/Megatron-LM --cuda-home /usr/local/cuda-12.4 --execute
```

脚本固定每次 `--blocks 5 --rounds 2 --nvls-enable 0`，新窗口预热10步、正式60步，每次3000个正式step，两次共6000个。两套先分别prepare和dry-run，核对输入/步数一致、run_id独立；然后顺序执行各自的 `run --stage all --execute`，各自重做8项mapping和10个smoke/profile窗口，再执行50个正式窗口并调用严格 `report`。第一套报告与阶段状态通过后才开始第二套；任一步失败即停止，保留现场，不自动重试。每次启动沿用原控制器的GPU空闲检查，运行中不周期查询占用。

输出包括 `compare1/report.{json,md}`、`compare2/report.{json,md}`、各阶段日志和 `batch-state.json`。只有两套严格报告均通过且运行身份一致，批次状态才为 `completed`，登记100个正式窗口/800份正式rank。指定的 `--out` 必须不存在；省略 `--execute` 只打印完整命令计划，不创建目录、不启动子进程或查询GPU。脚本前台执行，不自动建立后台服务；需要独立托管时可由既有容器监督进程调用。

需要按本批顺序再执行“消融5两次→消融4两次”时，使用 [`detach_ablation5_then4.py`](../../../../tools/h100/phase6/detach_ablation5_then4.py)；参数和输出目录说明见 [消融5方案第9节](消融实验5-选择性量化.md#9-先消融5两次再消融4两次统一60步)。本页总结的是已经回传并核验完成的批次。

## 8. 历史20步记录（2026-10-18）

以下保留原批次 `ablation4-gpt67-tp8-both-20261018-hostservice` 的独立结果，来源提交 `62e07733`。每窗口正式20步，两次合计100窗口/2000个正式step；以下“本次”均指该历史批次，不与上面的60步结果合并。

### 历史1. 实测结论

- **C 在两次实验、四轮中均快于融合 v2。** 两次合并各自两轮后的配对耗时下降分别为 **3.539% [3.312%, 3.766%]**、**3.199% [2.882%, 3.486%]**；四个逐轮点估计为3.171%–3.578%，对应区间均为正。C 的整步平均为516.361、517.240 ms，融合 v2 为535.309、534.331 ms。
- **交替排序与到达优先的单独增益没有得到本次数据支持。** A/Flux 两次合并的区间都跨零；B/A 分别为−0.245%、−0.267%，区间也跨零。compare1第2轮、compare2第1轮的 B/A 区间完全为负；compare2合并的 B/Flux 为−0.354% [−0.634%, −0.045%]。不能用“差距小”宣称严格等效，也不能把个别负区间推广成长期稳定退化。
- **固定融合实现已经占据相对朴素量化的大部分收益。** Flux v2 相对 Megatron＋量化的配对耗时下降为46.942%、47.031%；C 相对 Megatron 为48.820%、48.725%。A/Flux 与 B/A 是本实验可以分离的调度比较，C/B 的3.768%、3.540%同时包含选择性量化和候选实现变化，不能解释为纯 mask 收益。
- 四轮均未触发“本轮同组中位数×1.10”的慢窗口阈值，主结果没有删除窗口，敏感性结果与主表一致。

### 历史2. 数据完整性与固定条件

| 项目 | compare1 | compare2 | 合计 |
| --- | ---: | ---: | ---: |
| 轮数 / 五组完整配对块 | 2 / 10 | 2 / 10 | 4 / 20 |
| 正式窗口 / rank记录 | 50 / 400 | 50 / 400 | 100 / 800 |
| 正式 optimizer step（不重复计rank） | 1000 | 1000 | 2000 |
| smoke/profile 窗口 / rank记录 | 10 / 80 | 10 / 80 | 20 / 160 |
| profile trace / 策略角色mapping | 40 / 8 | 40 / 8 | 80 / 16 |

两次 run_id 分别为 `36f5fe380921468ca16a217009f607f3`、`4587cb5d32a446638726ee3c4ef6aa1f`。两次独立新建模型、窗口和预检；运行库、冻结 worker、校准计划、模型/输入/环境指纹及通信配置的登记一致，未复用计时窗口。

单节点 `gxn75`，8×H100 80GB HBM3、每卡132 SM。GPT6.7B、32层、H4096/FFN16384、32 heads、S2048、micro batch1/global batch4、TP8/DP1/PP1/CP1、BF16，seed1234、token_seed4321，padded vocab33792。Python3.10.20、PyTorch2.6.0+cu124、运行CUDA12.4、NCCL2.21.5，编译CUDA12.8/GCC11。Megatron提交固定为 `3ea68ad6042cc1204386ae9364358f7c4de1bc37`；本次构建/准备登记的维护提交为 `92cd94bc58e41aae060959782f65181e89ca404b`，C计算来源为 `ff38f5b1ff1c80cb831f58c64b95f42b0379879a`。

五组均作用于32个 `mlp.linear_fc2` 和32个 `self_attention.linear_proj`，共64个输出投影：

| 简称 | 调度与执行实现 | 远端量化 |
| --- | --- | --- |
| Megatron | 原生GEMM＋Tensor TACO通信；不导入Flux | 全远端 |
| Flux v2 | 原始Flux顺序；固定融合v2、CTA解码、单工作区、逐次配置/复位、输出clone | 全远端 |
| A | 与Flux v2同实现，改为基础交替顺序 | 全远端 |
| B | 与Flux/A同实现，在交替基础上增加冻结到达表 | 全远端 |
| C | 与B同到达表；保留候选优化codec、warp解码、双工作区及实例持有配置/输出 | 定向选择性mask |

Attention shape为 `[2048,4096,4096]`、本地K512；MLP为 `[2048,4096,16384]`、本地K2048。全量组每来源量化448个远端tile；本次C的Attention每来源选择数为 `[7,7,7,7,4,7,7,7]`，MLP为 `[7,7,7,7,1,7,7,7]`。本地贡献、GEMM及既定反向路径保持BF16/STE；“完整量化”不表示整网FP8或权重量化。

**C的选择性量化预算上限是远端贡献的 `1/64 = 1.5625%`，不是 `1/16 = 6.25%`。** [冻结配置](../../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/submission.json)中的 `fraction=0.015625`；每来源448个远端tile最多选择 `448/64=7` 个，并不要求用满预算。选择单位是 `[source rank, physical tile]` 的来源贡献：本次 [实际mask](../../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/scripts/selection-plan.json) 中，Attention合计53/3584（约1.479%），MLP合计50/3584（约1.395%）；两角色按相同调用次数合计103/7168（约1.437%）。这是远端通信贡献的量化比例，不是模型权重的量化比例；其余远端贡献和本地贡献仍为BF16。

两次均冻结 `NCCL_NVLS_ENABLE=0`、`NCCL_DEBUG=WARN`、`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`；通信快照中的 `/etc/nccl.conf` 和 `/root/.nccl.conf` 均不存在。完整五组定义、构建隔离及通信快照规则见 [消融实验4](消融实验4.md)。

### 历史3. 全样本完整step耗时

每窗口先预热10步、计时20步，每步含4个微批、8192 token，覆盖全模型前向/反向、通信、optimizer更新、调度及清梯度。窗口耗时为 `max(8 rank elapsed_seconds) × 1000 / 20`，不是某层、某个GEMM或八rank均值。每组每轮5个窗口，每次合并10个窗口，两次合计每组20个窗口。

| **指标** | **C：完整版（选择性量化）** | **A：交替＋完整量化** | **B：交替＋到达＋完整量化** | **Flux v2（原始顺序＋完整量化）** | **Megatron＋量化** |
| --- | ---: | ---: | ---: | ---: | ---: |
| 平均耗时（ms/step） | 516.800 | 535.028 | 536.399 | 534.820 | 1008.837 |
| C完整版相对耗时下降 | 0.000% | +3.407% | +3.654% | +3.369% | +48.773% |
| 平均训练loss | 8.432632 | 8.344758 | 8.344758 | 8.344758 | 6.720901 |
| 最终训练loss的平均 | 5.509021 | 6.573645 | 6.573645 | 6.573645 | 1.926777 |

上表合并compare1、compare2的全部正式窗口，每组20个窗口、400个正式optimizer step。平均耗时直接对20个未舍入窗口取算术平均；由于两次各10个窗口，也等于两次未舍入均值的等权平均。Flux v2列由本历史批次的 `flux_fused_v2_attn_mlp` 正式窗口及rank0 loss记录复算，使用20步口径。

“C完整版相对耗时下降”按 `100 × (1 − C平均耗时 / 该列平均耗时)` 计算，正值表示C更快，C自身为0。此行是合并均值的描述性对照，不是第4节同轮同块的配对几何平均下降，也不附带配对置信区间；计算使用未舍入值，最后显示三位小数。

**两项loss都从本次原始正式计时记录计算，不包含预热、smoke或profile。** 已核验100个窗口的800份rank文件哈希、完成/运行身份、20步loss长度与有限值；每窗口八rank的 `losses` 序列完全一致，只计rank0一次，不能把同一TP组的八份记录当成八个独立样本。每组取窗口数 `W=20`：

```text
平均训练loss = sum_w(sum_s(losses_rank0[w][s])) / (W × 20)
最终训练loss的平均 = sum_w(losses_rank0[w][-1]) / W
```

“最终训练loss”是各窗口第20个正式step的 `losses[-1]`，先取每窗口末步值，再跨20个窗口求算术平均，不是只取整套实验最后一个窗口。loss均使用未舍入值计算，显示六位小数。两次实验各组的这两项loss均值分别一致；Flux/A/B的loss均值相同来自本次原始记录。每窗口重新初始化模型，这些短程合成token记录不代表长期收敛。

分实验耗时与合并吞吐保留如下。吞吐按总token数除以总时间计算，即 `8192000 / 平均ms`；中位耗时分别基于各次10个窗口，不将两次中位数取平均。

| **指标** | **C：完整版（选择性量化）** | **A：交替＋完整量化** | **B：交替＋到达＋完整量化** | **Flux v2（原始顺序＋完整量化）** | **Megatron＋量化** |
| --- | ---: | ---: | ---: | ---: | ---: |
| compare1 平均耗时（ms/step） | 516.361 | 535.266 | 536.579 | 535.309 | 1008.917 |
| compare1 中位耗时（ms/step） | 516.218 | 535.903 | 536.741 | 535.304 | 1007.344 |
| compare2 平均耗时（ms/step） | 517.240 | 534.790 | 536.220 | 534.331 | 1008.758 |
| compare2 中位耗时（ms/step） | 517.046 | 534.563 | 536.204 | 533.637 | 1007.207 |
| 两次合并吞吐（tokens/s） | 15851.38 | 15311.35 | 15272.21 | 15317.30 | 8120.24 |

逐轮平均耗时（完整step；每格5个窗口）：

| **指标** | **C：完整版（选择性量化）** | **A：交替＋完整量化** | **B：交替＋到达＋完整量化** | **Flux v2（原始顺序＋完整量化）** | **Megatron＋量化** |
| --- | ---: | ---: | ---: | ---: | ---: |
| compare1 第1轮平均耗时（ms/step） | 516.670 | 536.480 | 535.988 | 535.842 | 1007.047 |
| compare1 第2轮平均耗时（ms/step） | 516.051 | 534.051 | 537.169 | 534.776 | 1010.786 |
| compare2 第1轮平均耗时（ms/step） | 516.585 | 534.517 | 536.435 | 533.808 | 1008.357 |
| compare2 第2轮平均耗时（ms/step） | 517.896 | 535.063 | 536.005 | 534.854 | 1009.160 |

### 历史4. 同场配对比较

以下每列使用对应实验的10个完整五组块，将两轮合并；保持同一轮、同一块内的X/Y配对。配对下降为 `100 × [1 − exp(mean(log(T_X/T_Y)))]`，正值表示X更快。95%区间按完整五组块bootstrap10000次、seed20261009计算；20个step和8个rank不视作独立样本。未给两次实验额外拼接一个跨批次置信区间。

| X / 参考 Y | compare1 配对耗时下降% [95% CI] | compare2 配对耗时下降% [95% CI] |
| --- | ---: | ---: |
| Flux v2 / Megatron | +46.942 [+46.769, +47.116] | +47.031 [+46.887, +47.181] |
| A / Megatron | +46.946 [+46.789, +47.125] | +46.985 [+46.847, +47.132] |
| A / Flux v2 | +0.008 [-0.274, +0.337] | -0.086 [-0.295, +0.104] |
| B / Megatron | +46.816 [+46.657, +46.977] | +46.843 [+46.676, +47.029] |
| B / Flux v2 | -0.237 [-0.566, +0.055] | -0.354 [-0.634, -0.045] |
| B / A | -0.245 [-0.583, +0.086] | -0.267 [-0.521, +0.013] |
| C / Megatron | +48.820 [+48.684, +48.972] | +48.725 [+48.554, +48.899] |
| C / Flux v2 | +3.539 [+3.312, +3.766] | +3.199 [+2.882, +3.486] |
| C / A | +3.532 [+3.336, +3.726] | +3.282 [+2.913, +3.594] |
| C / B | +3.768 [+3.494, +4.030] | +3.540 [+3.226, +3.893] |

逐轮负例保留：compare1第2轮 B/A 为 **−0.584% [−0.997%, −0.052%]**；compare2第1轮 B/A 为 **−0.359% [−0.634%, −0.041%]**。其余轮次与最好单块、配对中位加速比可直接查两套原始报告。C/B改变codec、工作区、配置/输出生命周期与量化覆盖率，不能用该比较单独证明选择性量化有效。所有区间未做多重比较校正，协议无4%验收门槛。

### 历史5. 慢窗口、数值检查与结论边界

| 实验 / 轮次 | 最大窗口耗时 / 本轮同组中位数 | 保留完整块 | 排除块 |
| --- | ---: | ---: | --- |
| compare1 / 第1轮 | 1.005442 | 5/5 | 无 |
| compare1 / 第2轮 | 1.006074 | 5/5 | 无 |
| compare2 / 第1轮 | 1.008560 | 5/5 | 无 |
| compare2 / 第2轮 | 1.010074 | 5/5 | 无 |

固定阈值为本轮同组原始窗口中位数的1.10倍，任一组触发则同步排除对应整块五组；不迭代阈值。本次全部20个完整块保留，不能把相对朴素量化的大幅差距解释为剔除慢窗口造成。

严格报告逐窗口核验八rank齐全、实际库登记、64个投影的路由/调用、通信快照、初始化/RNG/token/运行环境、有限loss/梯度及零skip/意外fallback；数值预检采用相对BF16 L2≤5%、相对独立codec参考L2≤1%的既定门槛。每个窗口重新初始化训练，因此这些合成token上的短程记录不构成真实语料准确率、验证困惑度或长期收敛结论。

### 历史6. 核验与原始证据

本次在临时目录建立只读输入链接，调用维护代码 `ablation4_report.report(..., offline=True)`，重新核验冻结文件、预检/计时证据哈希及全部800份正式rank，并复算两套完整报告。JSON序列化后仅 `offline` 标志与回传报告不同，其他字段（含逐窗口耗时、配对区间、敏感性与证据哈希）全部一致；两次 `measurement_identity` 和角色计划哈希也相同。未写入或覆盖原始会话的报告、日志、状态及rank记录，未在本机重新读取H100动态库或启动GPU。

60步改动后的兼容性复核也已完成：再次在临时目录复算这两套20步历史报告，除新增的 `execution_timing={warmup_steps:10,timed_steps:20}`、每次 `formal_optimizer_steps=1000` 以及离线标志外，全部原有JSON字段仍与回传报告一致。

- compare1：[报告JSON](../../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/report.json)、[报告Markdown](../../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/report.md)、[submission](../../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/submission.json)、[预检](../../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/preflight.json)、[完成状态](../../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/all-state.json)。
- compare2：[报告JSON](../../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare2/report.json)、[报告Markdown](../../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare2/report.md)、[submission](../../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare2/submission.json)、[预检](../../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare2/preflight.json)、[完成状态](../../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare2/all-state.json)。
- [原始compare1窗口](../../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/results/)、[原始compare2窗口](../../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare2/results/)、[执行端路径/环境登记](../../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/handoff/session-paths.json)、[构建登记快照](../../../../logs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/compare1/provenance/build-manifest.json)。
