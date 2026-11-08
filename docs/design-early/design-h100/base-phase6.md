# H100 单节点 Phase6 TP8 实测

模型范围：默认模型为[32层GPT 6.7B](../model-gpt-6.7b.md)。融合基线对齐后的新实验与历史 `phase6-warp-v1` 四轮结果分别记录，旧0.64B结果另列；不同模型和基线版本的收益不混用。

## GPT 6.7B TP8 融合 v2 基线对齐实验

更新：2026-10-13。数据提交 `7483993`、`24a553c` 对应第1–4轮，`a9c723e` 对应新监控体系a–d轮；最新提交 `944b9df` 又带回两套完整测试，顺延为 **e、f、g、h**；a–d与e–h按两批测试分成两张表，轮次标签保持不变。每套各72窗口、576份计时rank记录；新监控体系a–h合计288窗口、2304份计时rank记录，连同原1–4轮共432窗口、3456份计时rank记录。原始报告各自仍编号为round1/round2。

首次base构建的源码提交为 `d786da51e0aeafec1bb2b5d071c5c59e93e6de43`。本节各次实验 `taco_fused` 的实际标识为 **`a800-fused-v2-cta-v1`**，使用 A800 冻结融合 v2 的非模板 CTA 解码及旧配置流程；全部八 rank 的预检记录均确认执行了 `taco_decode_ring_kernel(FluxTacoConfig, ...)`。历史结果中的融合列对应 `phase6-warp-v1`，与本节分开解读。

在 `gxn74` 单节点8×H100上使用32层、H4096/FFN16384、32 heads、配置词表32768（实际 padded vocab 为33792）、S2048/mb1/global4、BF16、TP8/DP1配置，不调参。每 step 累积4个微批、8192 token，MLP全局 shape 为 `[2048,4096,16384]`、本地 K=2048；window64、量化预算上限1/64。新 base 完整构建，编译 CUDA 12.8、GPU运行 CUDA 12.4；候选库和选择计划复用 `gpt67-tp8-20261010-112249` 会话的已核验产物，内容不变。六种策略均在本会话重新计时，复用关系见 [reuse-verification.json](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/handoff/reuse-verification.json)。

第3–4轮复用第1–2轮的base、候选及选择计划；两次submission的模型配置、计划SHA、各变体路径、构建文件哈希清单、源码manifest、校准设备及控制脚本哈希均一致。复测新建compare并重做六组smoke/profile与完整计时，没有复用前次计时窗口，见[复测输入与复用关系](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-005925-repeat/handoff/repeat-inputs.json)。

**a–h均为新监控体系下的测试结果。** a/b对应 `gpt67-tp8-fusedv2-20261012-211935-repeat1` 的round1/round2，c/d对应同前缀 `repeat2` 的round1/round2；e/f对应 `gpt67-tp8-fusedv2-20261013-001822-repeat1` 的round1/round2，g/h对应同前缀 `repeat2` 的round1/round2。每套启动前忙时每10分钟检查；空闲后每10秒确认，连续3次满足条件才启动；运行中不再周期查询GPU显存、利用率和计算进程，第一套完成并核验后再检查并启动第二套。必要的GPU正确性与profile预检照常执行，见 [watcher.md](watcher.md)。

a–d对应的两套会话的 `execution.json` 和 `handoff/repeat-inputs.json` 均记录 `gpu_monitoring=before_each_run_only`，执行器哈希匹配新入口，且没有生成运行中的 `handoff/gpu-occupancy.jsonl`。实际队列为 `gpt67-tp8-fusedv2-20261012-211935-queue`，回传状态为两套均完成。与第3–4轮相比，模型配置、选择计划、基线/候选构建及库哈希、校准设备与模型控制脚本一致；监控方式和执行时段不同，结果差异不能直接解释为关闭监控的因果收益。

新增e–h的两套会话同样记录 `gpu_monitoring=before_each_run_only`，没有运行中的GPU占用采样文件；实际队列为 `gpt67-tp8-fusedv2-20261013-001822-queue`，两套均已完成。与a–d相比，配置、计划、变体路径/库哈希、校准设备和模型控制脚本保持一致，新建了各自compare与run_id并重新执行预检和计时。

**本次冻结对象是a–h实际复用的同一套候选实现**，包含远端组合 `remote_arrival_selective` 和交替组合 `interleaved_arrival_selective` 两个变体；对应构建目录为 `outputs/h100/phase6/gpt67-tp8-20261010-112249/candidates-20261010-153517/`。两批测试并非两版候选，实际构建、库哈希和选择计划一致。优化前版本登记为 `h100-phase6-candidate-v1-20261013`，以[冻结manifest](../../../outputs/h100/phase6/frozen-candidate-v1-20261013/manifest.json)的 `candidate` 字段为准；后续仅优化候选，四个对照 `native`、`original`、`native_taco`、`taco_fused` 的实现保持固定。

### 完整 step 对照

每窗口10步预热＋20步完整 optimizer step，窗口耗时取八rank最大elapsed除以20；计时包含前向、反向、通信/同步、optimizer更新与清梯度，独立profile不参与统计。

**round1和round2是两轮完整训练计时，都包含前向和反向。** 默认六策略中，round1用循环移位安排六个配对块，每块各跑六个策略；round2把块顺序及每块内的策略顺序同时反转，以减轻先后执行和时间漂移的影响。这里的“反转”指测试调度顺序，不是神经网络反向传播；每轮都是36个窗口，每个窗口均执行上述完整step。调度定义见 [common.py的orders](../../../tools/h100/phase6/common.py#L80)，实际计时worker的 `forward_only=False`，随后执行 `optimizer.step()` 和清梯度，见 [g/h轮冻结worker](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat2/compare/scripts/worker.py#L126)。

a/b、c/d、e/f、g/h分别是四套完整测试的round1/round2：a/c/e/g对应第一轮，b/d/f/h对应第二轮。候选当前替换的是MLP `linear_fc2` 前向路径，最终表格仍测完整训练step，不能将第二轮耗时理解为单独反向耗时。


**远端组合第4轮按用户要求剔除两个明显慢窗口，保留4个配对块；第3轮及交替组合第4轮仍使用全部6块。** 每行候选与四个对照均取同一组保留配对块内的中位数。筛选统计不作为全样本总体收益，原实验验收仍以各自全样本report为准。**a–h均使用各轮全部6个配对块，未剔除慢窗口。** 单位为ms/step。

远端组合第4轮对应复测目录的 `round2`。剔除 `window-026-remote_arrival_selective`（block 4，1521.523 ms/step）和 `window-031-remote_arrival_selective`（block 5，989.273 ms/step），四个对照同步排除block 4/5，仅保留block 0–3（block编号从0起）。保留的远端窗口为535.482、532.863、550.437、542.329 ms/step。原始窗口文件和report不删除、不改写；其他候选/轮次的统计不受影响。

**第1–4轮**

| 候选 | 轮次 | 配对块数 | 候选 ms/step | 新融合 v2 ms/step | Megatron＋量化 ms/step | 原生 Megatron ms/step | 原生 Flux ms/step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 远端组合 | 1 | 6 | 541.272 | 553.170 | 829.511 | 583.145 | 548.262 |
| 远端组合 | 2 | 5 | 540.468 | 548.839 | 828.641 | 579.180 | 545.241 |
| 远端组合 | 3 | 6 | 538.959 | 553.811 | 827.683 | 580.558 | 545.284 |
| 远端组合 | 4 | 4 | 538.906 | 658.759 | 828.409 | 578.528 | 549.664 |
| 交替组合 | 1 | 5 | 542.748 | 551.076 | 829.547 | 582.275 | 547.656 |
| 交替组合 | 2 | 6 | 540.493 | 549.213 | 825.354 | 577.261 | 544.148 |
| 交替组合 | 3 | 6 | 542.186 | 553.811 | 827.683 | 580.558 | 545.284 |
| 交替组合 | 4 | 6 | 541.868 | 553.222 | 828.409 | 579.234 | 554.080 |

**a–d轮（新监控体系，2026-10-12批次）**

| 候选 | 轮次 | 配对块数 | 候选 ms/step | 新融合 v2 ms/step | Megatron＋量化 ms/step | 原生 Megatron ms/step | 原生 Flux ms/step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 远端组合 | a | 6 | 536.570 | 549.161 | 823.840 | 578.916 | 542.286 |
| 远端组合 | b | 6 | 538.877 | 547.494 | 823.649 | 575.296 | 542.045 |
| 远端组合 | c | 6 | 538.186 | 547.839 | 821.867 | 576.683 | 544.092 |
| 远端组合 | d | 6 | 537.781 | 550.921 | 824.445 | 574.857 | 543.966 |
| 交替组合 | a | 6 | 539.979 | 549.161 | 823.840 | 578.916 | 542.286 |
| 交替组合 | b | 6 | 538.331 | 547.494 | 823.649 | 575.296 | 542.045 |
| 交替组合 | c | 6 | 537.689 | 547.839 | 821.867 | 576.683 | 544.092 |
| 交替组合 | d | 6 | 538.288 | 550.921 | 824.445 | 574.857 | 543.966 |

**e–h轮（新监控体系，2026-10-13批次）**

| 候选 | 轮次 | 配对块数 | 候选 ms/step | 新融合 v2 ms/step | Megatron＋量化 ms/step | 原生 Megatron ms/step | 原生 Flux ms/step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 远端组合 | e | 6 | 536.234 | 548.523 | 825.594 | 575.820 | 543.516 |
| 远端组合 | f | 6 | 537.758 | 545.486 | 823.403 | 574.514 | 539.808 |
| 远端组合 | g | 6 | 539.750 | 551.103 | 824.155 | 576.218 | 542.504 |
| 远端组合 | h | 6 | 538.590 | 547.133 | 818.670 | 575.587 | 540.055 |
| 交替组合 | e | 6 | 537.680 | 548.523 | 825.594 | 575.820 | 543.516 |
| 交替组合 | f | 6 | 536.422 | 545.486 | 823.403 | 574.514 | 539.808 |
| 交替组合 | g | 6 | 536.086 | 551.103 | 824.155 | 576.218 | 542.504 |
| 交替组合 | h | 6 | 536.662 | 547.133 | 818.670 | 575.587 | 540.055 |

候选相对两个必须 baseline 及两个原生参考的对照如下。下降百分比采用对应统计口径下配对块的几何平均 `100×(1−exp(mean(log(T候选/T基线))))`；95%区间按对应配对块bootstrap 10000次，随机种子20261009。第1–2轮及远端第4轮的区间只描述保留样本，未计入事后筛选的不确定性；第3轮、交替第4轮及a–h使用全部六块，与各自report一致。参考组“耗时增加”为相应配对下降的相反数，负值表示候选更快；不能用上表中位数相除替代配对收益。

**第1–4轮**

| 候选 | 轮次 | ms/step | 对 Megatron＋量化下降% [95% CI] | 对新融合 v2 下降% [95% CI] | 比原生 Megatron 耗时增加% | 比原生 Flux 耗时增加% |
| --- | --- | ---: | --- | --- | ---: | ---: |
| 远端组合 | 1 | 541.272 | +34.543 [+34.292, +34.787] | +2.938 [+0.832, +5.925] | -7.066 | -1.480 |
| 远端组合 | 2 | 540.468 | +34.855 [+34.425, +35.283] | +1.771 [+1.035, +2.316] | -6.716 | -0.955 |
| 远端组合 | 3 | 538.959 | +34.928 [+34.246, +35.596] | +5.750 [+1.939, +9.856] | -7.259 | -12.138 |
| 远端组合 | 4 | 538.906 | +34.730 [+33.820, +35.555] | +18.052 [+2.872, +30.860] | -7.656 | -1.682 |
| 交替组合 | 1 | 542.748 | +34.305 [+33.924, +34.656] | +2.728 [+0.486, +6.197] | -6.328 | -0.488 |
| 交替组合 | 2 | 540.493 | +34.394 [+33.661, +35.058] | +2.458 [+0.370, +5.421] | -6.154 | -0.381 |
| 交替组合 | 3 | 542.186 | +34.667 [+34.313, +35.111] | +5.373 [+1.555, +9.133] | -6.888 | -11.787 |
| 交替组合 | 4 | 541.868 | +34.394 [+33.365, +35.281] | +12.301 [+0.439, +23.280] | -14.556 | -2.582 |

**a–d轮（新监控体系，2026-10-12批次）**

| 候选 | 轮次 | ms/step | 对 Megatron＋量化下降% [95% CI] | 对新融合 v2 下降% [95% CI] | 比原生 Megatron 耗时增加% | 比原生 Flux 耗时增加% |
| --- | --- | ---: | --- | --- | ---: | ---: |
| 远端组合 | a | 536.570 | +32.886 [+28.800, +35.004] | +10.293 [-7.234, +31.316] | -14.876 | +1.898 |
| 远端组合 | b | 538.877 | +34.195 [+33.631, +34.697] | +6.148 [+0.525, +15.177] | -5.946 | -0.315 |
| 远端组合 | c | 538.186 | +27.432 [+10.835, +34.684] | +5.315 [+1.383, +12.051] | +3.253 | +9.683 |
| 远端组合 | d | 537.781 | +34.736 [+34.496, +35.003] | +2.387 [+2.072, +2.652] | -6.989 | -1.246 |
| 交替组合 | a | 539.979 | +24.370 [+0.076, +34.745] | -1.091 [-49.759, +30.847] | -4.074 | +14.829 |
| 交替组合 | b | 538.331 | +34.604 [+34.225, +35.098] | +6.731 [+1.249, +16.203] | -6.530 | -0.934 |
| 交替组合 | c | 537.689 | +34.568 [+34.334, +34.836] | +14.625 [+1.536, +35.575] | -6.900 | -1.103 |
| 交替组合 | d | 538.288 | +34.632 [+34.342, +34.973] | +2.232 [+1.843, +2.601] | -6.841 | -1.090 |

**e–h轮（新监控体系，2026-10-13批次）**

| 候选 | 轮次 | ms/step | 对 Megatron＋量化下降% [95% CI] | 对新融合 v2 下降% [95% CI] | 比原生 Megatron 耗时增加% | 比原生 Flux 耗时增加% |
| --- | --- | ---: | --- | --- | ---: | ---: |
| 远端组合 | e | 536.234 | +34.853 [+34.592, +35.099] | +8.874 [+1.641, +21.398] | -6.889 | -13.023 |
| 远端组合 | f | 537.758 | +34.955 [+34.458, +35.615] | +1.676 [+1.222, +2.205] | -6.727 | -0.995 |
| 远端组合 | g | 539.750 | +36.313 [-15.055, +65.728] | +10.445 [-54.895, +46.653] | -11.289 | +6.245 |
| 远端组合 | h | 538.590 | +12.147 [-25.110, +34.533] | -1.832 [-17.450, +8.716] | +4.032 | +5.522 |
| 交替组合 | e | 537.680 | +34.884 [+34.539, +35.274] | +8.918 [+1.757, +21.224] | -6.934 | -13.065 |
| 交替组合 | f | 536.422 | +34.876 [+34.441, +35.284] | +1.556 [+1.007, +1.989] | -6.614 | -0.875 |
| 交替组合 | g | 536.086 | +34.071 [-29.093, +65.947] | +7.293 [-73.436, +47.071] | -8.167 | +9.985 |
| 交替组合 | h | 536.662 | +34.566 [+34.265, +34.884] | +24.154 [+1.681, +49.362] | -22.515 | -21.405 |

远端第4轮筛选后为 **538.906 ms/step**，相对新融合v2的配对下降为 **18.052% [2.872,30.860]**。这一数值只对应保留的4块：新融合v2在保留块中仍包含761.488、820.066 ms/step两个慢窗口，中位数为658.759 ms/step，因此筛选后的收益明显受候选侧剔除和baseline慢窗口影响，不能视为稳定的全样本加速比，也不改写原report中的 `remote_arrival_selective=false`。

第1–2轮既有筛选统计中，远端对新融合v2下降2.938%/1.771%，交替下降2.728%/2.458%，点估计均低于4%；这些数字不改写首次实验的全样本验收结果（两候选均未通过）。

**原始第3–4轮的全样本报告中，交替组合通过本次两轮验收，远端组合未通过。** 交替对新融合v2下降5.373%/12.301%，对Megatron＋量化下降34.667%/34.394%，均满足点估计至少4%且95%区间下界大于0；远端对新融合v2为+5.750%/−14.421%，第4轮区间很宽并跨零，因而不通过。这里的`interleaved_arrival_selective=true`仅对应本次复测，不能表述为四轮全部达标。

第4轮存在明显慢窗口：远端两个窗口约1521.523、989.273ms/step，新融合v2两个窗口约761.488、820.066ms/step；第3轮原生Flux也有约1106.581ms/step的窗口。这些记录均保留在原始全样本报告中；本页仅对远端第4轮按上面的窗口清单做筛选。慢窗口解释了中位数差距与配对收益可能不一致、部分置信区间较宽。通过当前协议门槛不等于证明运行环境无干扰或后续复现必然达到同等收益。

**新监控体系a–d：两套测试的两候选均未通过完整两轮验收。** 远端组合相对新融合v2的配对下降依次为10.293%、6.148%、5.315%、2.387%；交替组合为−1.091%、6.731%、14.625%、2.232%。a轮两候选相对融合v2的区间均跨零，导致a/b这套不通过；d轮两候选相对融合v2的下降均不足4%，导致c/d这套不通过。不能因b或c单轮点估计较高就将整套标为达标。

新模式下仍能看到慢窗口：a轮融合v2约1117.288、交替约1249.385ms/step；b轮融合v2约752.118ms/step；c轮融合v2约1263.628、远端约997.395ms/step。这些窗口均按原协议保留，部分收益及置信区间因此波动较大。运行中未采集占用状态，无法据此判断慢窗口是否由其他任务造成，也不能从启动前空闲确认推断全过程独占。

**新增e–h：两套测试的两候选也均未通过完整两轮验收。** 远端组合相对新融合v2的配对下降依次为8.874%、1.676%、10.445%、−1.832%；交替组合为8.918%、1.556%、7.293%、24.154%。e/f这套在f轮均不足4%；g/h这套在g轮两候选的区间均跨零，远端h轮也未确认正收益。h轮交替的高点估计不能替代整套验收。

新增轮次仍按全部六块计算，较大点估计须结合区间和原始窗口解读，不因结果正负或波动自动剔除数据。关闭运行中GPU状态采样只改变监控条件，不证明这些窗口没有其他干扰。

本节各次均未在同一配对实验中同时测量旧 `phase6-warp-v1` 基线，因此不能将跨会话绝对耗时之差直接归因于基线改动。固定合成 token 的数值与短程训练检查不构成真实语料收敛验证。

### 核验与证据

| 本节轮次 | 来源会话 | 原报告轮次 | run_id | 全样本验收 |
| --- | --- | --- | --- | --- |
| 1–2 | `gpt67-tp8-fusedv2-20261011-233427` | round1/round2 | `a7cac271fa2f407db093f20d5a584612` | 远端false，交替false；上表沿用已有筛选统计 |
| 3–4 | `gpt67-tp8-fusedv2-20261012-005925-repeat` | round1/round2 | `fd640c7d6fcc49d2ae203e979cb378e5` | 远端false，交替true；本页远端第4轮另做筛选，其余为全样本 |
| a、b | `gpt67-tp8-fusedv2-20261012-211935-repeat1` | round1/round2 | `9d3cd8a3d15f40d78ba0f2a00ec8b87c` | 新监控体系；远端false，交替false；全样本 |
| c、d | `gpt67-tp8-fusedv2-20261012-211935-repeat2` | round1/round2 | `3aa54b8428aa45878c93d745c019918c` | 新监控体系；远端false，交替false；全样本 |
| e、f | `gpt67-tp8-fusedv2-20261013-001822-repeat1` | round1/round2 | `51f29917a2304634a6c9b6eec82fc90e` | 新监控体系；远端false，交替false；全样本 |
| g、h | `gpt67-tp8-fusedv2-20261013-001822-repeat2` | round1/round2 | `e5c5e72add794ee494e403096c71ec78` | 新监控体系；远端false，交替false；全样本 |

原1–4轮对应的两次执行端状态均为`completed`，各自重新完成六组smoke/profile（96份rank记录）与72个计时窗口。新增第3–4轮在开发端CPU核对了576份计时rank及96份预检rank的身份、模型/设备、完成状态、零skip/fallback、10/20步协议、预检指纹和记录中的运行库路径/哈希；submission、冻结scripts、构建manifest副本及预检文件哈希匹配。全部八rank的预检内核列表均包含非模板CTA解码，筛选前的全样本中位数、配对收益、95%区间及验收布尔值复算与最新report一致；本次再用保留的4块重算远端第4轮表格。

新增a–d对应两套均为`completed`。2026-10-13在开发端CPU核对新增1152份计时rank、192份smoke/profile rank：模型/设备/运行身份、10/20步协议、完成与零skip/fallback、预检指纹及记录的库哈希均通过；submission、冻结scripts、构建manifest及preflight产物哈希匹配，八rank CTA内核列表完整。四轮全部中位数、配对收益、95%区间及验收状态复算与报告一致；监控模式字段和驱动哈希也匹配。未运行新GPU实验或在本机核验原路径动态库实体。

新增e–h在开发端CPU核对了1152份计时rank和192份smoke/profile rank：完成状态、身份、模型/设备、10/20步协议、预检指纹与记录中的库哈希通过，冻结输入和preflight文件哈希匹配；各轮中位数、配对收益、95%区间及验收布尔值复算与报告一致。监控模式、执行驱动哈希和八rank CTA内核证据也已核对。本次仅更新文档，未运行GPU或修改原始报告；原1–4轮及a–d表格数值保持不变。

第1–2轮、第3轮及交替第4轮数值保持不变；远端第4轮仅按本节明确列出的两个窗口及对应配对块进行剔除。原始日志、report和运行文件均未修改；本机未重新运行GPU或核验执行端原路径动态库实体。

- [计时来源窗口](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/compare/results/) · [来源报告](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/compare/report.json)
- [冻结输入与运行库来源](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/compare/submission.json) · [预检](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/compare/preflight.json) · [CTA 内核核验](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/handoff/cta-trace-verification.json)
- [计时状态](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/compare/timing-state.json) · [会话完成状态](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/execution.json) · [构建核验](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/handoff/build-verification.json)
- [环境记录](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/handoff/environment.md) · [GPU 占用采样](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261011-233427/handoff/gpu-occupancy.jsonl)
- 第3–4轮：[计时来源窗口](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-005925-repeat/compare/results/) · [来源报告JSON](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-005925-repeat/compare/report.json) · [Markdown](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-005925-repeat/compare/report.md)
- 第3–4轮：[冻结输入](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-005925-repeat/compare/submission.json) · [预检](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-005925-repeat/compare/preflight.json) · [CTA核验](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-005925-repeat/handoff/cta-trace-verification.json) · [复用记录](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-005925-repeat/handoff/repeat-inputs.json)
- 第3–4轮：[计时状态](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-005925-repeat/compare/timing-state.json) · [执行状态](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-005925-repeat/execution.json) · [交接摘要](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-005925-repeat/handoff/summary.md) · [GPU占用采样](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-005925-repeat/handoff/gpu-occupancy.jsonl)
- a、b轮：[报告JSON](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat1/compare/report.json) · [报告Markdown](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat1/compare/report.md) · [原始窗口](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat1/compare/results/) · [冻结输入](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat1/compare/submission.json) · [预检](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat1/compare/preflight.json) · [独立核验](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat1/compare/verification.json) · [执行/监控状态](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat1/execution.json) · [复用与驱动记录](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat1/handoff/repeat-inputs.json)
- c、d轮：[报告JSON](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat2/compare/report.json) · [报告Markdown](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat2/compare/report.md) · [原始窗口](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat2/compare/results/) · [冻结输入](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat2/compare/submission.json) · [预检](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat2/compare/preflight.json) · [独立核验](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat2/compare/verification.json) · [执行/监控状态](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat2/execution.json) · [复用与驱动记录](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-repeat2/handoff/repeat-inputs.json)
- 新监控队列：[完成状态](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-queue/watchdog-state.json) · [启动前GPU检查](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-queue/gpu-checks.jsonl) · [队列日志](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261012-211935-queue/watchdog.log)
- e、f轮：[报告JSON](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat1/compare/report.json) · [报告Markdown](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat1/compare/report.md) · [原始窗口](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat1/compare/results/) · [冻结输入](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat1/compare/submission.json) · [预检](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat1/compare/preflight.json) · [独立核验](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat1/compare/verification.json) · [执行/监控状态](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat1/execution.json) · [复用与驱动记录](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat1/handoff/repeat-inputs.json)
- g、h轮：[报告JSON](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat2/compare/report.json) · [报告Markdown](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat2/compare/report.md) · [原始窗口](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat2/compare/results/) · [冻结输入](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat2/compare/submission.json) · [预检](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat2/compare/preflight.json) · [独立核验](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat2/compare/verification.json) · [执行/监控状态](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat2/execution.json) · [复用与驱动记录](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-repeat2/handoff/repeat-inputs.json)
- e–h队列：[完成状态](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-queue/watchdog-state.json) · [启动前GPU检查](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-queue/gpu-checks.jsonl) · [队列日志](../../../logs/h100/phase6/gpt67-tp8-fusedv2-20261013-001822-queue/watchdog.log)

## GPT 6.7B TP8 历史完整四轮结果（phase6-warp-v1）

更新：2026-10-11，依据已回传的两份完整报告。原实验的两轮记为**第1–2轮**，复测的两轮顺延为**第3–4轮**，在同一组表中展示。共144个正式窗口、1152份计时rank记录、2880个正式optimizer step。各轮沿用原报告的独立配对统计，原始文件中的轮次编号不修改。

模型为32层GPT，H4096、FFN16384、32 heads、词表32768，约6.729B；单节点8×H100，S2048/mb1/global4、BF16、TP8/DP1/PP1/CP1。每step累积4个微批、8192 token；MLP全局shape为 `[2048,4096,16384]`，TP8本地K为2048。window64、量化预算上限1/64。

四轮的模型配置、计划SHA、构建文件清单/哈希、源码manifest、控制脚本哈希及校准设备信息一致。第3–4轮复用既有校准与构建，在新的compare目录重新执行预检和测量；不是算法改版。两份报告均为 `completed=true`，两候选均为 **`accepted=false`**。

### 完整step对照结果

每轮6个位置平衡块，每个策略测6个窗口；每窗口先预热10步，再连续计时20个完整step，包含前向、反向、量化/反量化、通信同步、optimizer更新和清梯度。窗口耗时取8rank最大elapsed除以20，表内ms/step为6个窗口的中位数。

下降百分比采用配对块耗时比的几何平均，95%区间由10000次块bootstrap得到；参考组“耗时增加”为对应配对下降的相反数。不能用表内中位数相除替代，也不把四轮窗口合并重算为一个收益数。

**远端组合 `remote_arrival_selective`**

| 轮次 | ms/step | 对Megatron＋量化下降% [95% CI] | 对融合v2下降% [95% CI] | 比原生Megatron耗时增加% | 比原生Flux耗时增加% |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 554.391 | +33.998 [+32.704, +35.245] | -0.882 [-2.611, +0.545] | -7.839 | -0.192 |
| 2 | 534.545 | +34.579 [+34.230, +34.898] | +3.003 [-0.876, +9.568] | -7.095 | -0.910 |
| 3 | 537.350 | +32.395 [+28.849, +35.225] | -0.521 [-5.065, +2.413] | -7.570 | +2.949 |
| 4 | 534.255 | +34.822 [+34.393, +35.232] | +0.724 [-0.904, +2.149] | -8.579 | -4.259 |

**交替组合 `interleaved_arrival_selective`**

| 轮次 | ms/step | 对Megatron＋量化下降% [95% CI] | 对融合v2下降% [95% CI] | 比原生Megatron耗时增加% | 比原生Flux耗时增加% |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 548.140 | +34.482 [+33.529, +35.535] | -0.143 [-0.916, +0.760] | -8.515 | -0.924 |
| 2 | 536.271 | +32.704 [+29.201, +34.499] | +0.223 [-8.957, +9.567] | -4.432 | +1.930 |
| 3 | 553.728 | +29.876 [+24.340, +34.296] | -4.265 [-10.074, -0.348] | -4.128 | +6.784 |
| 4 | 537.893 | +33.869 [+32.343, +35.126] | -0.727 [-2.223, +0.346] | -7.242 | -2.860 |

六组绝对耗时如下，单位ms/step：

| 策略 | 第1轮 | 第2轮 | 第3轮 | 第4轮 |
| --- | ---: | ---: | ---: | ---: |
| 原生Megatron `native` | 589.601 | 574.219 | 580.638 | 570.325 |
| 原生Hopper Flux `original` | 553.859 | 540.876 | 544.512 | 546.045 |
| Megatron＋量化 `native_taco` | 836.608 | 817.482 | 825.564 | 822.847 |
| 冻结Flux＋融合V2 `taco_fused` | 549.006 | 534.061 | 542.422 | 536.254 |
| 远端组合 `remote_arrival_selective` | 554.391 | 534.545 | 537.350 | 534.255 |
| 交替组合 `interleaved_arrival_selective` | 548.140 | 536.271 | 553.728 | 537.893 |

四轮中，两候选相对Megatron＋朴素量化均有明确下降，但相对融合v2尚无稳定优势。远端组合四轮的区间均跨零；交替组合第3轮为−4.265% [−10.074,−0.348]，该轮显著变慢，其余三轮区间跨零。原协议按每次两轮分别要求相对两个必须baseline下降至少4%、且95%区间下界大于0；两次报告都未通过。统一四轮展示不改变原验收结果。

### 轮次来源与核验

下表将本文轮次映射到原始报告；原文件仍各自使用round1/round2。

| 本文轮次 | 原始对比目录 | 原报告轮次 | 运行标识与状态 |
| --- | --- | --- | --- |
| 第1–2轮 | `compare-20261010-153517` | round1 / round2 | run_id=`e1cc608329cc4bb2ae8e2506f8352ab5`；completed，resumed=true |
| 第3–4轮 | `compare-20261011-114343-repeat` | round1 / round2 | run_id=`b113bd6141374034aa3e050cbf3aab95`；completed，未设置resumed=true |

第1–2轮曾在 `native_taco` 解码时OOM，随后在原会话恢复；失败备份不参与计时统计。第2轮融合v2的一个窗口为659.164ms/step，明显高于该轮其他窗口，按既定协议保留，可影响配对点估计和置信区间。第3–4轮状态未标续跑，但不等于已有全过程独占GPU的证据；运行条件仍需结合环境/占用记录判断。

2026-10-11已在开发端CPU核对两套对比共1152份计时rank记录：身份、完成/通过、零skip/零fallback、预热/计时步数和preflight指纹均通过；冻结scripts/preflight哈希、各轮中位数、全部配对收益及95%区间与原报告一致。没有重新执行GPU或在本机核验原路径动态库。本次仅合并文档表格和轮次编号，不修改任何原始数据或统计。

实现范围仍限MLP `linear_fc2` 前向：量化MLP为SM90编译的GemmV2，attention及原生Flux参考走Hopper V3。固定随机token的短程step与数值检查不构成真实语料收敛证据。旧小模型也未显示相对融合v2的稳定优势，不能仅归因于模型扩大或本次OOM。

### 证据入口

- 第1–2轮：[JSON](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/report.json) · [Markdown](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/report.md) · [状态](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/timing-state.json) · [冻结输入](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/submission.json) · [预检](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/preflight.json) · [原始结果](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/results/)
- 第3–4轮：[JSON](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/report.json) · [Markdown](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/report.md) · [状态](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/timing-state.json) · [冻结输入](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/submission.json) · [预检](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/preflight.json) · [原始结果](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261011-114343-repeat/results/)
- [原OOM日志](../../../logs/h100/phase6/gpt67-tp8-20261010-112249/compare-20261010-153517/results/round2/window-007-native_taco.failed-20261011-112116/process.log)

源码与数据索引见 [instruction.md](instruction.md)，测试及数据交付见 [instruction-h100.md](instruction-h100.md)。

## 2026-10-10 完整量化六组两轮首测

北京时间 2026-10-10 00:22:34–00:59:39（UTC 2026-10-09 16:22:34–16:59:39），在 `gxn74` 单节点 8×H100 80GB HBM3 上，按 [A800 TP8 完整量化首测](../design-a800/base-phase6-tp8.md#2026-10-06-完整量化组合适配与首测不优化) 的六组、两轮完整 optimizer-step 口径完成测量，耗时约 37 分钟。**全部 72 个窗口及独立复算通过，两候选均未通过 4% 性能门槛。** 会话目录沿用部署时的 UTC 日期 `tp8-20261009-deployment`。

本轮使用冻结的默认 window64、1/64 预算和本机实测到达计划，不扫描参数。A800 文档顶部后续 window16 优化结果不作为本轮配置或对照。H100 GPU 正确性和性能证据在本记录独立维护，部署指南及接续说明中的旧“尚未 GPU 验证”描述属于此前状态。

### 环境与模型

| 项目 | 本次配置 |
| --- | --- |
| 节点 / GPU | `gxn74`；8×NVIDIA H100 80GB HBM3，每卡 132 SM，SM90；两两 NV18 / peer access |
| 驱动 | 580.178.04 |
| Python / PyTorch | `/opt/conda/envs/flux-h100`；Python 3.10.20，PyTorch 2.6.0+cu124，NumPy 1.26.4 |
| CUDA / 编译器 | CUDA 12.4 用于运行与 Apex，CUDA 12.8.93 用于 Flux 编译；GCC/G++ 11.4.0，CMake 3.30.0 |
| NCCL | 运行时 2.21.5；独立 Release 静态库 2.19.3，SM90 |
| Megatron | `core_v0.12.3`，提交 `3ea68ad6042cc1204386ae9364358f7c4de1bc37` |
| Apex | 提交 `e13873debc4699d39c6861074b9a3b2a02327f92`；`apex_C`、`amp_C`、`fused_layer_norm_cuda` |
| 并行 | TP8 / DP1 / PP1 / CP1，sequence parallel |
| 模型 | 12 层 GPT，H2048、FFN8192、32 heads，BF16 |
| 输入 / batch | S2048 / micro batch 1 / global batch 4；每 step 累积 4 个微批、8192 token；实际 padded vocab 9216 |
| 初始化 | 模型 seed 1234，token seed 4321；六组使用同一参数、RNG 与输入指纹 |

### 六组定义与实现边界

| 策略 | 角色与实现 |
| --- | --- |
| `native` | 原生 Megatron，参考组；不加载 Flux |
| `original` | 原生 Hopper Flux，参考组；attention `linear_proj` 与 MLP `linear_fc2` 使用 Hopper GemmV3 |
| `native_taco` | 必须 baseline：原生 Megatron＋朴素 TACO，保留原生 linear forward，不加载 Flux |
| `taco_fused` | 必须 baseline：原始顺序 Flux＋全远端融合 TACO，使用本次冻结的 H100 库 |
| `remote_arrival_selective` | 远端优先基础排序＋到达优先＋选择性量化 |
| `interleaved_arrival_selective` | 交替基础排序＋到达优先＋选择性量化 |

量化只作用于 MLP `linear_fc2` 前向的远端贡献，使用 FP8 E4M3/H128；本地贡献、GEMM 与 STE 反向保持 BF16。Flux 量化 MLP 是重新编译为 SM90 的 Phase6 GemmV2 路径，attention 仍用 Hopper GemmV3；不是基于 WGMMA/TMA 的 Hopper 原生融合 TACO epilogue。基线及候选库均为本次 H100 独立构建，不复用 A800 二进制。本轮计时不启用采样、profiler 或 CUDA Graph。

### 本机校准与预检

同一节点、同一组 GPU UUID 和设备顺序重新采集两种 BF16 基础排序的到达数据，以 pass 0/1 拟合、pass 2 留出诊断。采用 window64 和 1/64 上限；模型对应 shape `[2048,2048,8192]`，每来源 256 个 tile 中有 224 个远端 tile。

| 计划 | 每来源选中 tile，按 rank 0–7 | 总选中 / 总远端 tile | 实际选择率 | 有效训练 tile | 留出近临界命中率 | 采样中位扰动 |
| --- | --- | --- | ---: | --- | ---: | ---: |
| 远端优先 | `[0,3,0,3,0,3,0,3]` | 12 / 1792 | 0.6696% | 254 / 256 | 33.33% | 21.52% |
| 交替 | `[3,3,3,3,3,3,3,3]` | 24 / 1792 | 1.3393% | 125 / 256 | 41.67% | 10.21% |

掩码是本机校准的实际产物，不强行补齐未选中的来源。交替计划有效训练 tile 较少，且两种计划留出命中率均有限；这些指标仅为校准诊断，不代表重排量化后的收益。没有进行重排或量化后的联合重新校准。

部署阶段依赖检查、15 项 CPU 协议测试和八 rank NCCL all-reduce / all-gather / reduce-scatter / all-to-all 已通过。四组基线/采样库及两组真实计划候选构建完成；六组完整模型 smoke 和六组独立 profile 共 96 份 rank 记录全部完成并通过，零 skip、零 fallback。48 份 GPU trace 核验了对应 Hopper V3、SM90 编译的 V2 及 TACO decode 内核；两候选实际 GPU swizzle 均匹配冻结计划。每层量化 mask、逻辑通信字节、初始参数/RNG/token 指纹、实际加载库路径和哈希均通过预检。

上述是 H100 整模型前向与执行预检，不将 A800 的独立 codec、连续偏斜调用或 STE 专项报告算成本次 H100 验证。共 864 条模块前向检查，其中 384 条为量化前向双参考预算检查。整模型量化前向最大相对 BF16 L2 误差为 2.5124%，相对 codec 参考最大为 0.0198%，均低于既定 5% / 1% 预算。

| 策略 | 最大相对 BF16 L2 误差 | 最大相对 codec L2 误差 |
| --- | ---: | ---: |
| `native_taco` | 2.5118% | 0.0000% |
| `taco_fused` | 2.5124% | 0.0198% |
| `remote_arrival_selective` | 0.3103% | 0.0013% |
| `interleaved_arrival_selective` | 0.7703% | 0.0107% |

### 统计口径

每轮 6 个位置平衡块，每块覆盖六组，第二轮反转块与组顺序。两轮共 72 个独立八 rank 窗口；每窗口 10 步预热＋20 步完整 optimizer step，总计 720 步预热、1440 步正式 step、576 份计时 rank 记录。窗口耗时取 8 rank 最大 elapsed 除以 20；完整 step 包含前向、反向、梯度处理和 optimizer 更新。

ms/step 为每轮同策略 6 个窗口的中位数。配对下降百分比为 `100 × (1 − exp(mean(log(T候选/T基线))))`，正值表示更快；95% 区间以配对块 bootstrap 10000 次得到，不能用表中中位数相除替代。参考组“耗时增加%”为上述下降值的相反数，负值表示候选更快。预检/profile 不混入计时统计。

验收沿用既定 4% 标准：每个候选在两轮中，分别对 `native_taco` 和 `taco_fused` 的点估计均至少下降 4%，且各 95% 区间下界大于零。原生 Megatron 与原生 Flux 只作参考。

### 单节点TP8完整量化六组对照表

下表按 A800 文档格式汇总“远端组合”`remote_arrival_selective`。本次直接在节点 `gxn74` 执行，“配置 / 作业”列以节点名标识实验。所有百分比从同块窗口计算，不能用中位数相除。

| 配置 / 作业 | 轮次 | 远端组合 ms/step | 对Megatron＋量化下降% [95% CI] | 对融合v2下降% [95% CI] | 比原生Megatron耗时增加% | 比原生Flux耗时增加% |
| :------ | :- | -----------: | :----------------------- | :---------------- | ---------------: | -----------: |
| S2048/mb1/gb4 / gxn74 | 1 | 205.601 | +33.636 [+33.255, +34.004] | +0.532 [-0.271, +1.583] | -6.781 | -1.449 |
| S2048/mb1/gb4 / gxn74 | 2 | 207.051 | +33.441 [+33.142, +33.914] | -0.368 [-0.934, +0.084] | -6.119 | -1.235 |

交替组合 `interleaved_arrival_selective` 使用相同口径，补充如下：

| 配置 / 作业 | 轮次 | 交替组合 ms/step | 对Megatron＋量化下降% [95% CI] | 对融合v2下降% [95% CI] | 比原生Megatron耗时增加% | 比原生Flux耗时增加% |
| :------ | :- | -----------: | :----------------------- | :---------------- | ---------------: | -----------: |
| S2048/mb1/gb4 / gxn74 | 1 | 205.333 | +33.605 [+33.494, +33.720] | +0.485 [-0.237, +1.495] | -6.737 | -1.402 |
| S2048/mb1/gb4 / gxn74 | 2 | 206.494 | +33.554 [+33.096, +34.097] | -0.197 [-0.761, +0.299] | -6.279 | -1.403 |

六组每轮的绝对耗时如下，单位 ms/step：

| 策略 | 第 1 轮 | 第 2 轮 |
| --- | ---: | ---: |
| 原生 Megatron `native` | 220.847 | 219.944 |
| 原生 Hopper Flux `original` | 208.521 | 208.876 |
| Megatron＋量化 `native_taco` | 309.753 | 310.567 |
| 冻结 Flux＋融合 V2 `taco_fused` | 206.101 | 206.031 |
| 远端组合 `remote_arrival_selective` | 205.601 | 207.051 |
| 交替组合 `interleaved_arrival_selective` | 205.333 | 206.494 |

**H100 TP8 环境可以执行本轮实验，完整测量与数值检查已完成；两候选尚未建立相对冻结融合 V2 的稳定整步收益。** 对 Megatron＋量化的两轮下降约 33.4%–33.6%，区间均为正；对冻结融合 V2 则第一轮略快、第二轮略慢，四个区间均跨零，点估计也均不足 4%。因此两候选 `accepted=false`。本轮止于默认配置首测，没有继续优化，也不据此计算 H100 相对 A800 的硬件加速比。

计时阶段 576 份 rank 记录全部完成并通过，零 skip、零 fallback；模型、TP 拓扑、GPU UUID、运行库哈希和参数/RNG/token 指纹与预检一致。每策略的 12 个窗口中，同 rank 的初始 loss、20 步正式 loss 和梯度范数记录完全一致，最大差异均为零。跨全部窗口及 rank 的峰值 allocated / reserved 显存为 2.185 / 2.301 GiB；实验结束后八卡均为 0 MiB、0% 利用率。

本次输入是固定 seed 的随机 token，测量的是短程完整训练 step。前向误差、有限 loss/梯度和同策略轨迹重复通过不等于真实数据训练或收敛验证。量化 MLP 的 SM90 编译 V2 实现边界仍如上所述。

### 原始产物与复算

构建与计划：`outputs/h100/phase6/tp8-20261009-deployment/`；校准与本轮实验：`logs/h100/phase6/tp8-20261009-deployment/`。

- [冻结选择计划](../../../outputs/h100/phase6/tp8-20261009-deployment/selection-plan.json)
- [冻结提交与输入清单](../../../logs/h100/phase6/tp8-20261009-deployment/compare/submission.json)
- [完整预检](../../../logs/h100/phase6/tp8-20261009-deployment/compare/preflight.json)
- [完整配对报告](../../../logs/h100/phase6/tp8-20261009-deployment/compare/report.md) / [未舍入统计](../../../logs/h100/phase6/tp8-20261009-deployment/compare/report.json)
- [独立复算与数值核验](../../../logs/h100/phase6/tp8-20261009-deployment/compare/verification.json)
- [全部窗口及八 rank 原始耗时汇总](../../../logs/h100/phase6/tp8-20261009-deployment/compare/window-statistics.json)
- [计时原始文件哈希](../../../logs/h100/phase6/tp8-20261009-deployment/compare/timing-artifact-hashes.json)
- [计时完成状态](../../../logs/h100/phase6/tp8-20261009-deployment/compare/timing-state.json)
- [各窗口原始记录](../../../logs/h100/phase6/tp8-20261009-deployment/compare/results/)

首次执行命令如下；控制器拒绝覆盖已开始的阶段，不能在原目录重跑计时。

```bash
cd /workspace/TempName
source outputs/h100/phase6/tp8-20261009-deployment/env.sh
python tools/h100/phase6/run.py "$H100_LOG/compare" \
  --stage timing --python /opt/conda/envs/flux-h100/bin/python \
  --cuda-home "$CUDA124" --execute
```

完成后的 CPU 复算命令如下，不会重新启动 GPU 测量：

```bash
source outputs/h100/phase6/tp8-20261009-deployment/env.sh
python tools/h100/phase6/report.py "$H100_LOG/compare"
python "$H100_BUILD/verify-timing.py"
```
