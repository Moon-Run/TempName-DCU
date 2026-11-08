# H100新baseline：原版TACO CUDA codec的50步实测

更新：2026-10-21。最新回传批次为 **`baseline-taco-cuda-3x-50steps-20261021-162929-989672`**，对应 [本页实验设计](#8-实验设计与复测)。A/B/C保持原消融6实现；两组baseline改用作者独立TACO CUDA编解码kernel，通信替换为NCCL All-to-All。三次比较均完成，合计 **150个正式窗口、7500个正式optimizer step、1200份正式rank记录**。

2026-10-19的PyTorch朴素codec结果完整保留在 [旧版消融6结果](消融实验/base-phase6-ablation6.md)。下文统计只使用本轮CUDA codec批次，旧版约48%的收益不作为本轮结论。

## 1. 实测结论

- **C仍是五组中最快：517.734 ms/step。** 三次合并均值相对新Flux基线下降 **15.213%**、相对新Megatron基线下降 **18.197%**；三次各自的配对95%置信区间均完全为正。
- **基线加强后，收益幅度明显小于旧结果。** 新Flux、Megatron基线分别为 **610.632、632.905 ms/step**，旧朴素版本为989.377、1008.970 ms/step。旧“约48%”主要反映与更慢朴素实现的差距，不能沿用于新基线；新旧批次的变化仍是跨批次描述性比较。
- **C相对A/B仍有稳定收益，但不是单因素消融。** 三次C/A配对下降为 **3.740%、3.490%、3.365%**；C/B为 **3.241%、3.261%、3.741%**，区间均为正。C同时改变量化覆盖、codec、工作区与配置/输出生命周期。
- **旧到达调度B/A没有稳定复现。** 三次配对下降分别为 **+0.516%、+0.236%、−0.391%**；第三次区间跨零，不能只保留前两次正收益。
- **共享同一CUDA codec后，Flux基线稳定快于Megatron基线。** 三次配对下降为 **3.435%、3.588%、3.534%**，区间均为正，仍包含GEMM与实现管理开销差异。
- **数值门禁通过不代表训练轨迹等价。** 两个新基线的50步loss轨迹完全相同，末步为 **0.01916869**，且最后两步回升；A/B末步为3.261156e-05，C为3.057346e-04。应同时保留性能和数值结果，不能据此宣布五组长期训练质量相同。

本轮比较对象是“作者原版TACO codec＋NCCL通信”的移植实现，**不包含COCCL的集合通信调度与重叠，不能将上述15%/18%收益表述为相对论文完整TACO系统的收益。**

## 2. 五组与固定条件

| 组别 | 前向实现 | 量化及生命周期 |
| --- | --- | --- |
| C | 原candidate，交替＋原到达表 | 原1/64预算上限及冻结mask；双工作区、实例持有配置/输出 |
| A | 融合v2＋基础交替 | 全远端量化；单工作区、逐次配置/复位、输出clone |
| B | 与A相同的融合v2，使用原到达表 | 与A相同的量化范围和生命周期 |
| Flux＋CUDA量化 | 原生SM90 GemmOnly后，调用独立TACO CUDA codec | 全远端量化；独立编码、All-to-All、独立解码、逐源BF16归约 |
| Megatron＋CUDA量化 | 原生Megatron GEMM后，调用同一CUDA codec | 与Flux基线相同的codec、通信与归约 |

两组baseline使用固定提交`ca717e05b6dd6d21fff4d2617d209c95be63f3ef`中的`fused_ash_compress_kernel`与`fused_ash_decompress_kernel`，保持H128/E4M3、两个FP32 scale和作者的设备端计算体。没有将编码融合进GEMM，也没有把解码与归约融合；不使用旧PyTorch Hadamard路径或融合v2替代这两个kernel。

A/B/C的variant定义、动态库登记和选择计划与旧消融6逐项一致。C仍为1/64上限：Attention **53/3584（1.479%）**、MLP **50/3584（1.395%）**个远端贡献被量化，合计103/7168（1.437%）；没有改成消融7的1/16。所有组本地贡献保持BF16。

环境为`gxn75`的8×H100 80GB HBM3，每卡132 SM；PyTorch2.6.0+cu124、CUDA运行时12.4、NCCL2.21.5，`NCCL_NVLS_ENABLE=0`。模型为GPT6.7B：32层、H4096/FFN16384、32 heads、S2048、TP8/DP1/PP1/CP1、BF16、micro batch1/global batch4，seed1234、token_seed4321、padded vocab33792。

五组均覆盖32个Attention输出投影和32个MLP输出投影的前向RS，反向仍沿用原实验路径。每个optimizer step包含4个微批、8192 token。每窗口独立初始化，10步预热＋50步正式计时；窗口时间取八rank最大墙钟除以50，覆盖前后向、通信、optimizer和清梯度。累计7500步不是同一个模型连续训练7500步。

## 3. 完成范围与全样本结果

| 项目 | compare1 | compare2 | compare3 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 完整五组块 | 10 | 10 | 10 | 30 |
| 正式窗口 / rank记录 | 50 / 400 | 50 / 400 | 50 / 400 | 150 / 1200 |
| 正式optimizer step | 2500 | 2500 | 2500 | 7500 |
| smoke/profile窗口 / rank | 10 / 80 | 10 / 80 | 10 / 80 | 30 / 240 |
| profile trace / 角色mapping | 40 / 6 | 40 / 6 | 40 / 6 | 120 / 18 |

每次5块×2轮，第二轮反转块顺序和块内顺序；三次有不同run ID，分别执行预检和严格报告。各组保留30个正式窗口、1500个正式step。

| 指标 | C | A | B | Flux＋CUDA量化 | Megatron＋CUDA量化 |
| --- | --- | --- | --- | --- | --- |
| 平均耗时（ms/step） | 517.734 | 536.692 | 536.041 | 610.632 | 632.905 |
| 全部窗口中位耗时（ms/step） | 517.446 | 536.618 | 535.755 | 610.428 | 632.682 |
| C相对该列耗时下降 | +0.000% | +3.532% | +3.415% | +15.213% | +18.197% |
| 平均训练loss | 3.793364 | 3.821014 | 3.821014 | 3.821911 | 3.821911 |
| 最终训练loss的平均 | 3.057346e-04 | 3.261156e-05 | 3.261156e-05 | 1.916869e-02 | 1.916869e-02 |
| 吞吐（tokens/s） | 15822.79 | 15263.87 | 15282.41 | 13415.61 | 12943.49 |

“C相对该列耗时下降”由三次合并算术均值计算，是描述性对照。吞吐为`8192000 / 平均ms`。平均loss先取每窗口rank0的50个正式值平均，再跨30窗口平均；末步loss取第50个正式step，不包含预热，不重复计八个TP rank。

| 平均ms/step | C | A | B | Flux＋CUDA量化 | Megatron＋CUDA量化 |
| --- | --- | --- | --- | --- | --- |
| compare1 | 517.821 | 537.947 | 535.164 | 610.509 | 632.226 |
| compare2 | 517.739 | 536.458 | 535.192 | 609.959 | 632.660 |
| compare3 | 517.643 | 535.672 | 537.767 | 611.428 | 633.829 |

## 4. 三次独立配对比较

下降率为`100×[1−exp(mean(log(T_X/T_Y)))]`，正数表示X更快。每次以10个完整五组块bootstrap10000次、seed20261009；不将50个step或八rank当作独立样本，无多重比较校正，无4%收益门槛。

| X / 参考Y | compare1下降% [95% CI] | compare2下降% [95% CI] | compare3下降% [95% CI] |
| --- | --- | --- | --- |
| C / A | +3.740 [+3.299, +4.107] | +3.490 [+3.197, +3.791] | +3.365 [+3.129, +3.595] |
| C / B | +3.241 [+2.918, +3.547] | +3.261 [+2.968, +3.563] | +3.741 [+3.464, +3.968] |
| C / Flux＋CUDA量化 | +15.182 [+15.000, +15.370] | +15.119 [+14.852, +15.384] | +15.339 [+15.140, +15.478] |
| C / Megatron＋CUDA量化 | +18.096 [+17.777, +18.397] | +18.165 [+17.948, +18.360] | +18.330 [+18.112, +18.549] |
| B / A | +0.516 [+0.140, +0.887] | +0.236 [+0.031, +0.437] | -0.391 [-0.768, +0.017] |
| Flux＋CUDA量化 / Megatron＋CUDA量化 | +3.435 [+3.135, +3.742] | +3.588 [+3.189, +3.986] | +3.534 [+3.271, +3.776] |
| A / Flux＋CUDA量化 | +11.887 [+11.524, +12.289] | +12.050 [+11.871, +12.227] | +12.391 [+12.120, +12.670] |
| A / Megatron＋CUDA量化 | +14.913 [+14.625, +15.194] | +15.206 [+14.900, +15.532] | +15.486 [+15.210, +15.824] |
| B / Flux＋CUDA量化 | +12.341 [+12.128, +12.564] | +12.257 [+12.001, +12.524] | +12.048 [+11.804, +12.339] |
| B / Megatron＋CUDA量化 | +15.352 [+15.189, +15.568] | +15.406 [+15.153, +15.641] | +15.156 [+14.896, +15.415] |

C对各组的收益方向在三次中一致；B/A只有前两次区间完全为正，第三次为负点估计且跨零。A对新Flux基线约12%的收益包含GEMM、融合和调度变化，不能单独归因于交替顺序；C对A/B约3%—4%的增量也不能只归因于减少量化。

## 5. 精度、执行路径与loss

三次smoke共 **7680项模型投影forward检查**，相对BF16门槛5%、相对独立codec参考门槛1%，全部通过；下表为各组跨三次、八rank、64模块的最大值：

| 组别 | 最大相对BF16 L2 | 最大协议参考误差（按BF16参考范数归一化） |
| --- | ---: | ---: |
| C | 0.439100% | 0.008438% |
| A | 2.545994% | 0.017685% |
| B | 2.545994% | 0.017685% |
| Flux＋CUDA量化 | 2.545994% | 0.017685% |
| Megatron＋CUDA量化 | 2.545994% | 0.017685% |

两个新基线另有 **192个独立codec预检case**，涵盖零值、小值、离群值与实际七packet完整shape。最大相对BF16误差为 **2.726618%**，最大独立codec误差为 **0.008076%**，实际packet解释检查误差为 **0**；固定5%/1%/0.5%门槛均通过，已返回输出也未被后续调用覆盖。

48份新基线profile trace核验了 **12,288次压缩kernel、12,288次解压kernel**及All-to-All证据；每份trace各256次，均为rank-local调用计数。Flux组还核验每份trace256次GemmOnly标记。两个基线加载同一CUDA库，A/B/C未加载它；本轮不存在用旧张量codec或融合v2冒充新基线的情况。

每组30条正式loss轨迹逐项一致；新Flux/Megatron两组之间也逐项一致，A/B之间一致。A/B/C轨迹与旧消融6相同，新CUDA基线与旧朴素基线的轨迹则不同。代码和数值预检通过，不意味着两种执行方式对训练逐位等价。

尤其要保留新基线末段的回升：正式第48、49、50步loss分别为 **3.216704e-05、1.038385e-03、1.916869e-02**。该现象在重复窗口中一致出现。它不属于缺失或非有限记录，因而没有违反本轮预注册门禁；但也不能由`correctness_passed=true`推导为长期收敛正常或与A/B/C质量等价。当前资料尚不能定位回升原因。

模型始终重复固定四个合成微批，没有真实语料验证集。这些短程轨迹既不足以证明长期发散，也不足以代替论文训练精度复现；后续评价应将数值轨迹与性能分别报告。

## 6. 慢窗口及与旧朴素baseline的关系

| 比较 | 第1轮最大窗口/同组中位数 | 第2轮最大窗口/同组中位数 | 保留完整块 |
| --- | --- | --- | --- |
| compare1 | 1.003696 | 1.009442 | 10/10 |
| compare2 | 1.007807 | 1.008306 | 10/10 |
| compare3 | 1.008036 | 1.006872 | 10/10 |

敏感性阈值为本轮同组中位数×1.10；三次六轮均未触发，30个完整块全部保留，主表没有剔除慢样本。

| 对象 | 旧PyTorch朴素codec，ms/step | 新CUDA codec，ms/step | 跨批次描述性下降 |
| --- | --- | --- | --- |
| Flux基线 | 989.377 | 610.632 | 38.281% |
| Megatron基线 | 1008.970 | 632.905 | 37.272% |

以上新旧对照没有跨批次配对置信区间。CUDA codec替换后，两个baseline的整步耗时分别下降约38.28%和37.27%，同时短程数值轨迹也发生变化；不能将其解释为严格控制其他条件后的单一kernel加速量。A/B/C构建保持不变，但本批均值较旧批慢约0.28%—0.36%，说明也存在批次间运行差异。

**后续与较接近原版TACO量化的baseline比较，应引用本轮C对Flux约15.21%、对Megatron约18.20%的合并均值下降，并同时给出第4节的逐次配对区间及第5节的数值边界。** 旧约48%结论保留为朴素实现对照，不与本轮样本合并。

## 7. 离线核验与原始证据

| 比较 | run_id | 报告 | 运行证据 |
| --- | --- | --- | --- |
| compare1 | `e8d15a26bc4140b28656ef07ea958e70` | [报告JSON](../../../logs/h100/phase6/baseline-taco-cuda-3x-50steps-20261021-162929-989672/compare1/report.json) / [Markdown](../../../logs/h100/phase6/baseline-taco-cuda-3x-50steps-20261021-162929-989672/compare1/report.md) | [预检](../../../logs/h100/phase6/baseline-taco-cuda-3x-50steps-20261021-162929-989672/compare1/preflight.json) / [submission](../../../logs/h100/phase6/baseline-taco-cuda-3x-50steps-20261021-162929-989672/compare1/submission.json) |
| compare2 | `ea234267860343ba9edea4e1f0645c7f` | [报告JSON](../../../logs/h100/phase6/baseline-taco-cuda-3x-50steps-20261021-162929-989672/compare2/report.json) / [Markdown](../../../logs/h100/phase6/baseline-taco-cuda-3x-50steps-20261021-162929-989672/compare2/report.md) | [预检](../../../logs/h100/phase6/baseline-taco-cuda-3x-50steps-20261021-162929-989672/compare2/preflight.json) / [submission](../../../logs/h100/phase6/baseline-taco-cuda-3x-50steps-20261021-162929-989672/compare2/submission.json) |
| compare3 | `d703d4346c284a7789a25b53cf6aed34` | [报告JSON](../../../logs/h100/phase6/baseline-taco-cuda-3x-50steps-20261021-162929-989672/compare3/report.json) / [Markdown](../../../logs/h100/phase6/baseline-taco-cuda-3x-50steps-20261021-162929-989672/compare3/report.md) | [预检](../../../logs/h100/phase6/baseline-taco-cuda-3x-50steps-20261021-162929-989672/compare3/preflight.json) / [submission](../../../logs/h100/phase6/baseline-taco-cuda-3x-50steps-20261021-162929-989672/compare3/submission.json) |

[批次总状态](../../../logs/h100/phase6/baseline-taco-cuda-3x-50steps-20261021-162929-989672/batch-state.json)为`completed`。三次预检均离线重算一致；三份完整报告除`offline`标志外逐项一致，批次登记的报告SHA256也一致。报告复算保留原预检文件的序列化身份，避免回传后文件遍历顺序不同影响证据哈希。统计和来源复核见 [审计JSON](../../../outputs/h100/phase6/baseline-taco-cuda-results-audit-20261021/audit.json)、[复核脚本](../../../outputs/h100/phase6/baseline-taco-cuda-results-audit-20261021/audit.py) 与 [运行日志](../../../outputs/h100/phase6/baseline-taco-cuda-results-audit-20261021/audit.log)。

审计还从全部原始rank记录重新计算窗口最大墙钟、平均/末步loss和误差最大值，核对三次输入/初始化/软件身份、实际调用次数、零skip/fallback、50步分母，以及A/B/C相对旧批次的构建与计划一致性。

原始报告、rank JSON、trace和状态均未修改，本次未运行新的GPU实验或调整数值门槛。回传保留了codec构建源码、manifest和运行端库哈希，但没有`libbaseline_taco.so`二进制；离线复核没有重新加载或哈希验证缺失的库。动态库实际执行由运行端身份记录及kernel trace交叉核验。

实验定义及一行启动命令见 [本页实验设计及启动命令](#8-实验设计与复测)。旧朴素实现的原始统计见 [历史结果页](消融实验/base-phase6-ablation6.md)，不覆盖其原始证据。

## 8. 实验设计与复测

以下保留本轮的实现范围、预注册检查及启动说明。

### 8.1. 五组的固定范围

| 组别 | 实现 | 量化范围及生命周期 |
| --- | --- | --- |
| A | 消融6原A：融合v2＋基础交替 | 全远端量化；单工作区、原配置/复位与输出clone |
| B | 消融6原B：融合v2＋原到达表 | 全远端量化；与A相同生命周期 |
| C | 消融6原C：原candidate、交替＋原到达表 | **原1/64预算上限与冻结mask**；双工作区、实例持有配置/输出 |
| Flux＋量化 | 原生Hopper SM90 GemmOnly产生BF16 partial，再独立编码、通信、解码、归约 | 全远端量化；不使用GemmRS或融合v2 |
| Megatron＋量化 | 保留原生Megatron GEMM，将前向RS替换为同一独立编码、通信、解码、归约 | 全远端量化；不加载Flux |

A/B/C直接复用原消融6使用的ablation4父构建，包含原动态库、角色计划、到达表和C的mask；不重做校准，不改成消融7的1/16或op4的闭包mask。本轮新代码没有修改旧A/B/C源文件。默认父构建为：

```text
outputs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/build
```

五组作用范围均为32层的Attention输出投影和MLP输出投影，共64个RowParallelLinear前向RS；本地贡献保持BF16。反向继续沿用原实验的BF16 AllGather及dgrad/wgrad流程，不另外扩大到其他通信位置。完整训练每步仍包含前向、反向、通信、optimizer更新和清梯度。

### 8.2. “接近原本TACO”的具体含义

参考作者 [COCCL_TACO仓库](https://github.com/liuman1125-ops/COCCL_TACO)，固定提交 **`ca717e05b6dd6d21fff4d2617d209c95be63f3ef`**。本地已随工具保存 [原始taco.cu](../../../tools/h100/phase6/baseline_upstream/taco.cu)、[RS配置](../../../tools/h100/phase6/baseline_upstream/taco_RS.config)、[许可证](../../../tools/h100/phase6/baseline_upstream/LICENSE.txt) 和 [来源及哈希](../../../tools/h100/phase6/baseline_upstream/provenance.json)，远端无需下载另一份TACO仓库。

构建器从固定源码中原文提取设备辅助函数及以下两个kernel，**不重写其设备端计算体**：

- `fused_ash_compress_kernel<__nv_bfloat16,128>`：统计、自适应缩放、Hadamard、FP8量化在独立kernel中完成。
- `fused_ash_decompress_kernel<__nv_bfloat16,128>`：独立解压、逆Hadamard及尺度恢复，输出BF16。

固定E4M3、SATFINITE、H128、`target_range=448`、`lambda=1e-6`，每组两个FP32 scale。保持作者的128线程CTA及shared-memory变换，不采用融合v2的单warp四元素重写。必要的移植仅包括头文件、C ABI启动包装及当前PyTorch CUDA stream接入；不调用COCCL的配置解析、内存管理或集合通信API。

每个目标rank的packet保持作者的布局：`[FP8 payload][FP32 quant scales][FP32 adaptive scales]`。每128个输入元素对应136字节，七个远端目标的每rank每投影逻辑发送量为 **7,798,784字节**，BF16对应14,680,064字节，减少46.875%。这是逻辑字节数，不是实测链路流量。

只支持本实验完整H128分组：M2048、N4096、TP8；Attention/MLP的K_local分别为512/2048。拒绝不整除的尾组，避免将作者尾组处理与本项目补零布局混为同一协议。

两组执行路径为：

```text
本地BF16 GEMM（Flux GemmOnly或原生Megatron）
  → 拼接七个目标的远端partial
  → 作者独立融合压缩kernel，一次launch覆盖七个packet
  → NCCL All-to-All（本地split为0）
  → 作者独立融合解压kernel，一次launch覆盖七个packet
  → 原来源顺序rank+1,…,rank逐次BF16相加，本地partial直接加入
  → 返回独立输出
```

这里保留独立codec之外的张量拼接、临时分配和逐来源归约开销；没有增加GEMM-量化融合、解码-归约融合、双缓冲或自定义通信流水。作者kernel内部已有的融合是要保留的原实现部分。

**因此本轮是“原版TACO codec＋NCCL通信”的移植基线，仍不等同于论文完整TACO系统。** COCCL的集合通信调度、分块重叠及完整系统集成不在本轮范围内。比较结果应按此名称报告。

### 8.3. 实验规模及统计口径

沿用消融6的GPT6.7B、32层、H4096/FFN16384、32 heads、S2048、TP8/DP1/PP1/CP1、BF16、micro batch1/global batch4；seed1234、token_seed4321，每optimizer step为8192 token。固定`NCCL_NVLS_ENABLE=0`，其余通信环境沿用消融6的捕获与核验机制。

默认串行运行 **compare1、compare2、compare3三次独立比较**。每次5块×2轮×5组＝50个正式窗口；第二轮反转块顺序和块内顺序，每窗口独立初始化、10步预热＋50步正式计时。总计：

| 项目 | 每次比较 | 三次合计 |
| --- | ---: | ---: |
| 正式计时窗口 | 50 | 150 |
| 正式rank记录 | 400 | 1200 |
| 正式optimizer step | 2500 | 7500 |
| smoke/profile窗口 | 10 | 30 |
| profile trace | 40 | 120 |

7500是独立窗口累计工作量，不是同一个模型连续训练7500步。每次都有独立run ID、预检和严格报告，上一比较完成后才准备下一比较。

窗口时间为八rank最大墙钟除以50。配对下降率使用`100×[1−exp(mean(log(T_X/T_Y)))]`，按完整五组块bootstrap10000次、seed20261009。主表保留全部样本及负收益；慢窗口只进入独立敏感性附表，无4%或其他预设性能通过门槛。报告同时保存平均/中位step耗时、吞吐、rank0正式50步平均loss及末步loss。

Flux/Megatron对照共用同一codec和通信，但仍包含GEMM及实现管理开销。C/A、C/B及A/新Flux基线也各自包含多项实现差异，不能将全部收益归因于单一调度或量化策略。训练数据仍为固定四个合成微批的重复训练，50步loss不代表真实语料长期收敛。

### 8.4. 计时前必须通过的检查

1. **A/B/C来源不变。** 核验父构建、动态库、角色计划、实际mapping与C的mask；三次比较的输入、构建和软件身份必须一致。
2. **两基线实际加载同一CUDA库。** 固定源码、设备端提取片段、编译命令和动态库SHA256；禁止用旧PyTorch codec或融合v2库替代。A/B/C不得加载新codec库。
3. **独立codec检查。** 每个新基线的smoke进程在模型初始化前验证零值、小值、离群值及实际七packet完整shape，比较独立tensor参考及实际packet解释结果，并检查后续调用未覆盖已返回输出。BF16相对L2上限5%；与codec参考之差按BF16输入范数归一化后上限1%；实际packet检查上限0.5%。诊断使用独立随机数生成器，不进入模型计时或profile。
4. **模型检查。** 保留消融6全部64个目标投影的forward精度检查、初始化/RNG/token身份、有限loss/grad norm、零skip/fallback和调用次数核验。正式每投影200次，另有40次预热；不根据新结果放宽阈值。
5. **实际GPU路径检查。** 新基线每份完整step profile必须有256次作者压缩kernel、256次作者解压kernel及对应通信标记；Flux组还须有256次GemmOnly标记。若执行融合v2的解码归约kernel或缺少实际CUDA codec调用则拒绝验收。

任何编译、精度、来源、执行路径或记录完整性检查失败都保留现场并停止，不自动重试或跳过。通过这些短程检查不等于证明与原系统逐位等价或长期收敛。

### 8.5. 另一台服务器的一行启动

同步本次新增`baseline*.py`、`baseline_codec.cu`、整个`baseline_upstream/`目录及 [detach_baseline.py](../../../tools/h100/phase6/detach_baseline.py)，连同已有实验工具。在原服务器/容器的`/workspace/TempName`目录执行：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/detach_baseline.py --execute
```

默认复用上述A/B/C构建，fresh编译一个原生Flux GemmOnly构建和一个独立TACO CUDA库。沿用CUDA12.8编译、CUDA12.4运行、jobs2、相邻`../Megatron-LM`及原NCCL开发库记录。双fork后台托管，终端关闭后继续；需保持容器和主机运行。

去掉`--execute`只打印计划，不创建目录、不编译、不fork或查询GPU。加`--foreground`可前台诊断。路径变更时使用`--build`、`--megatron`、`--build-cuda-home`、`--cuda-home`、`--python`；若原NCCL开发库已移动，传入`--nccl-root`，需包含`include/nccl.h`与`lib/libnccl_static.a`。

已有消融6的GemmOnly构建可用`--flux-build /path/to/flux-build`复用，仍会严格核验。`--codec-build`仅接受本次固定协议构建的独立库。原A/B/C构建缺失时启动失败，不用新建mask或其他实验的库替代。

默认输出：

```text
logs/h100/phase6/baseline-taco-cuda-3x-50steps-<北京时间戳>/
  batch-state.json                    批次状态与三次比较完成数
  handoff/container-supervisor.log   后台总日志
  flux-build/                       原生GemmOnly构建
  codec-build/manifest.json          上游来源、编译命令、动态库哈希
  compare1/                         独立预检、profile、正式窗口与报告
  compare2/
  compare3/
```

每个compare含`submission.json`、`scripts/`、`provenance/`、`results/`、`preflight.json`及`report.json/report.md`。只有三份完整报告均通过后，批次才标记`status=completed`；它表示实验完成，不保证性能为正。

回传整个批次。若省略大型二进制，保留所有manifest、冻结脚本、rank JSON、trace和报告；离线报告入口为：

```bash
python3 tools/h100/phase6/baseline.py report /path/to/batch/compare1 --offline
```

### 8.6. 运行前本地验证

已完成独立codec的真实SM90编译与动态库链接，见 [构建登记](../../../outputs/h100/phase6/baseline-validation-20261021/codec-build/manifest.json)。另在屏蔽GPU的进程中验证了动态库加载、C ABI、非法参数及替换库拒绝，见 [ABI检查](../../../outputs/h100/phase6/baseline-validation-20261021/abi-validation.json)。编译和这些检查均未运行GPU kernel。

**10项CPU测试全部通过，无跳过**，见 [CPU验证登记](../../../outputs/h100/phase6/baseline-validation-20261021/cpu-validation.json) 与 [测试代码](../../../tools/h100/phase6/test_baseline.py)。覆盖A/B/C定义和调度保持、真实生成adapter的五组路由与调用计数、上游kernel原文保持、源码变化拒绝、缺失CUDA profile拒绝、数值门槛/动态库替换拒绝、source-only不可执行、完整50窗口模拟报告及截断记录拒绝、三次启动顺序及失败停止。模拟rank和trace仅用于测试报告器，不作为实验结果。另已通过 [启动预览](../../../outputs/h100/phase6/baseline-validation-20261021/launch-preview.txt)，确认无副作用。

上述为运行前的本地验证；本轮八卡实测、逐次区间及数值边界见本页第1–7节。
