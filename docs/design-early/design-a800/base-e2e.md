# Megatron-LM 与 Flux 端到端对比：6.7B配置与历史结果

## 当前模型：GPT 6.7B（2026-10-10，未测试）

后续主模型改为 **32层、hidden=4096、FFN=16384、32 heads、vocab=32768**，保持独立输入/输出嵌入、BF16、learned absolute位置编码与固定合成token。默认S2048、micro batch1/global batch4、8192 token/step；按TP词表补齐，约6.725B（TP4）/6.729B（TP8）。本次不编译、不测试、不提交GPU作业。

A800的通用端到端入口默认值、Phase6 TP4/TP8模型准备及新形状校验，以及H100默认配置均已更新。完整配置、校准与入口说明见 [GPT 6.7B统一说明](../model-gpt-6.7b.md)，H100产物存放规则见 [H100 instruction](../design-h100/instruction.md#生成数据与文件放置规范)。原始六组对照仍只优化MLP `linear_fc2`；新到达表与量化mask必须基于 `[M,N,K_global]=[2048,4096,16384]` 重新校准。

**下文所有已完成耗时、loss、显存和正确性记录属于原12层约0.64B或更早4层模型。** 保留它们用于历史追溯，不将旧结果改标为6.7B；旧shape专用回放入口和旧固定作业命令也不作为6.7B启动入口。

## 历史0.64B实验更新

2026-10-03最新融合v2复测已完成：融合v2相对旧融合的完整step耗时下降8.622% / 8.715%，相对独立编码下降0.282% / 0.318%；相对原始Flux仍慢4.177% / 4.089%。仅MLP量化、attention BF16原始映射、无重排；五组两轮100窗口，见[base-phase4.md页首](base-phase4.md)。下文约232 ms的融合结果属于初版。

仅MLP量化、无tile重排的四组两轮端到端已完成。原始顺序Flux＋独立TACO相对原始Flux（BF16）为-4.447% / -4.395%，两轮区间均为负，重复退化。原始顺序Flux＋融合TACO相对原始Flux（BF16）为-13.853% / -13.947%，两轮区间均为负，重复退化。原始顺序Flux＋融合TACO相对原始顺序Flux＋独立TACO为-9.005% / -9.149%，两轮区间均为负，重复退化。 数值、控制组与误差边界见[base-phase4.md](base-phase4.md)。

当前另行进行仅MLP的TACO量化端到端对照，attention保持BF16原始Flux、所有组固定原始tile映射；包含stock Flux、同构建关闭量化控制、分离TACO、融合TACO。方案和结果入口见[base-phase4.md](base-phase4.md)，不与tile重排收益混算。

四组MLP-only复测已完成：MLP远端相对原始Flux为+0.113% / +0.051%，未达到两轮区间下界均为正的重复增益标准。MLP到达优先相对MLP远端基础为-0.134% / -0.179%，两轮区间上界均为负，重复退化。 64个窗口/256份rank记录通过，详细时长、吞吐与配对区间见[only-tile.md](only-tile.md)和[完整报告](../../../logs/a800/e2e/mlp-arrival-tp4-20261003-r2/report.md)。

本次补测扩为四组MLP-only：新增“MLP远端优先＋到达优先、attention原始Flux”，与三组基线一起重测。每轮8个配对块、两轮反序；运行结果统一在`logs/a800`，构建在`outputs/a800`，均不纳入Git。两种窗口及点估计/95%区间的说明见[only-tile.md](only-tile.md)。

## 当前研究范围：仅MLP重排（2026-10-03用户确认）

研究主线面向MLP层；attention暂不加入远端优先、交替或到达优先重排。attention保留原始Flux或原始Megatron，作为明确固定的后端条件，不能在同一组对照中混用后再归因于MLP重排。本轮用户指定 **MLP远端优先＋attention原始Flux**，优化对象具体为12层模型的`mlp.linear_fc2`前向GEMM-RS；`attention.linear_proj`保持原始Flux映射，其他算子和反向口径沿用现有基线。此处“MLP重排”不表示MLP所有GEMM均已改写。

本轮在现有179147内进行三组完整optimizer step对照：原始Megatron、原始Flux、MLP远端优先＋attention原始Flux。基线一起重跑；沿用12层、H=2048、FFN=8192、S=2048、TP4、global batch=4、BF16，每窗口10步预热+20步计时，每轮6个位置平衡块，第二轮反序。共完成36个计时窗口，属于同一分配内两轮独立进程，不能标为独立Slurm作业。

实验目录：[mlp-remote-tp4-e2e-20261003](../../../outputs/a800/mlp-remote-tp4-e2e-20261003/)；[进度](../../../outputs/a800/mlp-remote-tp4-e2e-20261003/campaign-state.json)；[入口与构建](../../../tools/a800/e2e/mlp/)。先检查attention逐tile映射与原始Flux一致、MLP逐tile映射与原远端策略一致，再做模型输出/kernel预检和正式计时。模块选择按本冻结模型不同shape在host构造时确定，无新增GPU查表；入口逐层核对名称与shape，不支持任意模型套用。原179147驻留step及外层循环保留。

后续MLP交替、MLP基础＋到达优先也遵守上述模块范围；此次只验证MLP远端基础，不同时更改到达标签或引入压缩。下文历史实验曾同时重排attention与MLP，其数值保留，但不能标成MLP-only结果。

本轮三组两轮端到端已完成，36个窗口/144份rank记录通过；MLP-only远端组合相对原始Flux两轮配对点估计均更快，但两轮95%区间均跨零，尚不能确立可重复的端到端正增量。 详细数值见[only-tile.md的MLP-only结果](only-tile.md)及[完整报告](../../../logs/a800/e2e/mlp-remote-tp4-e2e-20261003/report.md)。

<!-- distributed-results:start -->
## 双节点实测更新（2026-10-02）

本轮复用`outputs/a800/phase3/three-way-build`：94项文件及`outputs/a800/megatron-e2e/scale-mapping`的6项文件哈希全部匹配，未修改或重编译`tools/a800/phase3`及其Flux二进制。复用依据是节点内仍调用相同IPC/GEMM-RS kernel，跨节点交换由新的Python适配器处理；这不代表C++直接跨节点路径已经验证。审计见[build-reuse-audit.json](../../../logs/a800/e2e/distributed/build-reuse-audit.json)。

集群提交插件要求多节点增加`--qos=gpugpu`，已补入`distributed/submit.py`；仍使用普通`gpu_a800`分区，查询到该QoS优先级为0。首次缺少QoS的提交被Slurm拒绝，未产生GPU作业。

固定12层、H=2048、S=2048、FFN=8192、32 heads、BF16、micro batch=1/global batch=4，均为8192 token/step；两节点各4×A800。TP4/DP2每副本累积2步，TP8/DP1累积4步。每轮四个位置平衡块，每候选4个窗口，每窗口10步预热+20步完整optimizer计时。各窗口取8rank最慢墙钟；profile在独立进程中运行。

| 作业 | 布局 | 阶段 | 结果 | 产物 |
| --- | --- | --- | --- | --- |
| 178774 | TP4/DP2 | preflight | 完成 | [记录](../../../logs/a800/e2e/distributed/0627962ded3e/submission.json) |
| 178775 | TP8/DP1 | preflight | 失败（退出码1） | [记录](../../../logs/a800/e2e/distributed/93b45d69af69/submission.json) |
| 178776 | TP4/DP2 | timing | 完成 | [记录](../../../logs/a800/e2e/distributed/0d7697b0e98d/submission.json) |
| 178777 | TP4/DP2 | timing | 完成 | [记录](../../../logs/a800/e2e/distributed/8543423361d9/submission.json) |
| 178781 | TP8/DP1 | preflight | 完成 | [记录](../../../logs/a800/e2e/distributed/tp8-sync-preflight/submission.json) |
| 178785 | TP8/DP1 | timing | 完成 | [记录](../../../logs/a800/e2e/distributed/b58dd8138140/submission.json) |
| 178786 | TP8/DP1 | timing | 完成 | [记录](../../../logs/a800/e2e/distributed/e54091370074/submission.json) |

### 预检与数值检查

| 作业/布局 | 原始Flux max-abs / relative-L2 | 远端 max-abs / relative-L2 | 交替 max-abs / relative-L2 |
| --- | ---: | ---: | ---: |
| 178774 TP4/DP2 | 0.0078125 / 8.29e-05 | 0.0078125 / 8.86e-05 | 0.0078125 / 8.84e-05 |
| 178781 TP8/DP1 | 0.0156250 / 0.00483 | 0.0117188 / 0.00483 | 0.0117188 / 0.00483 |

每格为全部rank/目标模块首次前向检查的最大值，逐元素按`atol=rtol=0.02`与本地GEMM+NCCL RS组合容差比较。通过预检的作业还完成各节点映射检查、实际kernel及调用次数验证、四实现短训练、有限loss/梯度和DP副本参数一致性检查。以上不替代完整配对梯度误差验收或真实数据收敛。

### 正式计时

| 作业/布局 | 实现 | ms/step | tokens/s | 初始loss | 末步loss | allocated峰值GiB | 相对原始Flux下降%（配对95%区间） |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 178776 TP4/DP2 | 原生Megatron | 304.842 | 26,874 | 9.475574 | 0.059440 | 4.14 | -1.38 [-4.83, +1.87] |
| 178776 TP4/DP2 | 原始Flux | 301.656 | 27,159 | 9.475458 | 0.065248 | 4.46 | +0.00 [+0.00, +0.00] |
| 178776 TP4/DP2 | 远端重排 | 303.186 | 27,020 | 9.475638 | 0.059730 | 4.46 | +0.25 [-1.46, +1.61] |
| 178776 TP4/DP2 | 交替重排 | 301.849 | 27,139 | 9.475571 | 0.060676 | 4.46 | -0.19 [-1.86, +1.45] |
| 178777 TP4/DP2 | 原生Megatron | 295.292 | 27,742 | 9.475574 | 0.059440 | 4.13 | +1.29 [-2.16, +3.50] |
| 178777 TP4/DP2 | 原始Flux | 301.326 | 27,187 | 9.475458 | 0.065248 | 4.46 | +0.00 [+0.00, +0.00] |
| 178777 TP4/DP2 | 远端重排 | 302.168 | 27,111 | 9.475638 | 0.059730 | 4.46 | -0.25 [-1.18, +0.72] |
| 178777 TP4/DP2 | 交替重排 | 299.923 | 27,314 | 9.475571 | 0.060676 | 4.46 | +0.91 [-0.38, +2.05] |
| 178785 TP8/DP1 | 原生Megatron | 1089.572 | 7,519 | 9.543221 | 0.069668 | 2.13 | -8.12 [-14.42, -2.14] |
| 178785 TP8/DP1 | 原始Flux | 990.456 | 8,271 | 9.543336 | 0.066672 | 2.25 | +0.00 [+0.00, +0.00] |
| 178785 TP8/DP1 | 远端重排 | 965.002 | 8,489 | 9.543344 | 0.067097 | 2.25 | +3.19 [-0.25, +6.51] |
| 178785 TP8/DP1 | 交替重排 | 1017.515 | 8,054 | 9.543452 | 0.067946 | 2.25 | -2.29 [-7.78, +4.47] |
| 178786 TP8/DP1 | 原生Megatron | 1065.206 | 7,693 | 9.543221 | 0.069668 | 2.13 | -8.56 [-14.57, -0.31] |
| 178786 TP8/DP1 | 原始Flux | 962.536 | 8,515 | 9.543336 | 0.066672 | 2.25 | +0.00 [+0.00, +0.00] |
| 178786 TP8/DP1 | 远端重排 | 964.983 | 8,499 | 9.543344 | 0.067097 | 2.25 | +1.65 [-3.03, +5.21] |
| 178786 TP8/DP1 | 交替重排 | 1019.037 | 8,039 | 9.543452 | 0.067946 | 2.25 | -1.67 [-9.50, +5.60] |

- [178776汇总](../../../logs/a800/e2e/distributed/0d7697b0e98d/summary.csv)、[窗口](../../../logs/a800/e2e/distributed/0d7697b0e98d/windows.csv)、[配对块](../../../logs/a800/e2e/distributed/0d7697b0e98d/paired_blocks.csv)。
- [178777汇总](../../../logs/a800/e2e/distributed/8543423361d9/summary.csv)、[窗口](../../../logs/a800/e2e/distributed/8543423361d9/windows.csv)、[配对块](../../../logs/a800/e2e/distributed/8543423361d9/paired_blocks.csv)。
- [178785汇总](../../../logs/a800/e2e/distributed/b58dd8138140/summary.csv)、[窗口](../../../logs/a800/e2e/distributed/b58dd8138140/windows.csv)、[配对块](../../../logs/a800/e2e/distributed/b58dd8138140/paired_blocks.csv)。
- [178786汇总](../../../logs/a800/e2e/distributed/e54091370074/summary.csv)、[窗口](../../../logs/a800/e2e/distributed/e54091370074/windows.csv)、[配对块](../../../logs/a800/e2e/distributed/e54091370074/paired_blocks.csv)。

ms/step与tokens/s分别取窗口中位数；配对百分比用各块耗时比的几何均值计算，不能直接用两个中位数相除重算。正值表示相对原始Flux更快；区间来自同一作业内4个配对块，不等同于跨节点/跨作业总体置信区间。两种布局的累积步数、跨节点通信内容不同，分别报告；不将单节点结果和双节点结果合并统计。


### TP8工作区同步修正

首次预检178775的远端候选在第8层attention输出rank0超出容差（max-abs=0.3251953125，relative-L2=0.297066），该作业失败且未计入性能结果。源码核对发现现有`GemmRS.forward()`在最终本地归约前执行barrier；分层适配器连续复用同一个工作区时，还需保证所有rank完成归约/读取。修正版在克隆每段输出后调用现有`forward_barrier()`，再进入下一段GEMM；未修改误差阈值或重编译Flux，新增同步成本计入正式性能。

修正版预检178781已在原容差下通过全部四实现的数值、映射、实际kernel与短训练检查。修正仅涉及Python分层路径；节点内TP4路径未改变。

同步改动见[修正补丁](../../../logs/a800/e2e/distributed/hierarchical-workspace-sync.patch)和[适配器](../../../tools/a800/e2e/distributed/adapter.py)；另有[CPU工作区回归](../../../tools/a800/e2e/test_hierarchical_workspace.py)。

实际padded vocabulary随TP变化：TP4=8704，TP8=9216；每rank原生参数量分别为164,296,704和84,583,424。各布局内部的四候选配置一致，但跨布局的词表补齐和分片初始化不同，不能将两者时间/loss作为严格同模型strong-scaling或训练轨迹等价对照。

TP4/DP2正式作业178776/178777及TP8/DP1正式作业178785/178786均已完成，Slurm退出码均为`0:0`，共64个计时窗口、512份rank结果。状态与节点见[Slurm验收记录](../../../logs/a800/e2e/distributed/slurm-accounting.txt)。两个通过的预检为178774、178781；首次TP8失败作业178775保留追溯，不计入性能结果。

### 结果判断与边界

TP8/DP1的178785/178786中，原始分层Flux相对原生的配对耗时下降为7.51%/7.89%，两轮作业内95%区间分别为[2.10%,12.60%]与[0.31%,12.72%]；这是分层Flux替换原生路径的收益，不能归因于重排。远端相对原始Flux为+3.19%/+1.65%，交替为−2.29%/−1.67%，四个区间均跨零，重排未显示稳定整步收益。第一轮原生/交替CV为6.56%/5.32%，第二轮原始Flux/远端CV为6.33%/5.69%，共享环境波动较大，结论限于本轮配置和节点。

TP4/DP2的178776/178777中，远端重排相对原始Flux的配对耗时下降为+0.25%/−0.25%，交替为−0.19%/+0.91%，四个95%区间均跨零，未显示稳定整步加速；原始Flux相对原生的区间也均跨零。当前固定global batch=4且关闭通信重叠的双节点配置，吞吐低于历史单节点4卡配置；不能据此推广到其他batch、网络或重叠设置。

已完成作业保存全部rank有限loss/梯度、零跳步及参数更新证据。运行于普通队列共享节点，节点内拓扑与网络信息均随窗口记录；已检查日志存在跨节点`NET/IB/.../GDRDMA`通道。两轮作业仍不足以宣称任意拓扑的稳定收益。更大节点数、模型、PP/CP、压缩路径和真实数据收敛不属于本轮已验证范围。

<!-- distributed-results:end -->

<!-- scale-results:start -->
## 当前实验：12层 GPT，扩大 hidden/sequence

本轮统一采用**12层GPT**。主对照为原生Megatron-LM、原始Flux、远端重排、交替重排；记录loss、完整optimizer step时间、tokens/s、梯度范数及显存峰值。

| 层数 | hidden | sequence | FFN | heads | TP/GPU | 当前状态 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 12 | 2048 | 2048 | 8192 | 32 | 4 | 预检178746；计时178759/178760，已完成2轮 |
| 12 | 2048 | 2048 | 8192 | 32 | 8 | 已按用户要求取消全部8卡作业 |
| 12 | 4096 | 4096 | 16384 | 64 | 8 | 已按用户要求取消全部8卡作业 |

两节点各4卡需要跨节点通信。Megatron可使用这种部署，但本次三种Flux接入设置`nnodes=1`、CUDA IPC/NVLink和单节点启动；跨节点需另行接入与验证，不能直接沿用当前对照。按用户要求，本轮8卡任务全部取消，继续12层4卡实验。8卡配置仅保留为未运行模板。

### 12层4卡的实测结果

| 实现 | 178759 ms/step | 178759 tokens/s | 178760 ms/step | 178760 tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 原生 Megatron-LM | 209.389 | 39,123 | 209.195 | 39,160 |
| 原始 Flux | 203.839 | 40,189 | 203.749 | 40,206 |
| 远端重排 | 203.369 | 40,282 | 203.667 | 40,223 |
| 交替重排 | 203.685 | 40,219 | 203.748 | 40,207 |

| 实现 | 178759 初始loss | 178759 step 11 loss | 178759 step 30 loss | 178759 峰值显存GiB | 178760 初始loss | 178760 step 11 loss | 178760 step 30 loss | 178760 峰值显存GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原生 Megatron-LM | 9.475574 | 5.981937 | 0.059440 | 4.14 | 9.475574 | 5.981937 | 0.059440 | 4.14 |
| 原始 Flux | 9.475458 | 6.062554 | 0.065248 | 4.46 | 9.475458 | 6.062554 | 0.065248 | 4.46 |
| 远端重排 | 9.475637 | 6.031703 | 0.059717 | 4.46 | 9.475637 | 6.031703 | 0.059717 | 4.46 |
| 交替重排 | 9.475571 | 6.011055 | 0.060676 | 4.46 | 9.475571 | 6.011055 | 0.060676 | 4.46 |

表中所有完成窗口的loss和梯度范数均有限，跳步为0；显存取各rank/窗口中最大的PyTorch allocated峰值。

| 实现 | 178759 相对原始Flux耗时下降% | 178759 末步梯度范数 | 178760 相对原始Flux耗时下降% | 178760 末步梯度范数 |
| --- | ---: | ---: | ---: | ---: |
| 原生 Megatron-LM | -2.82 | 2.424112 | -2.67 | 2.424112 |
| 原始 Flux | +0.00 | 2.819257 | +0.00 | 2.819257 |
| 远端重排 | +0.20 | 2.445806 | +0.06 | 2.445806 |
| 交替重排 | +0.05 | 2.506007 | +0.03 | 2.506007 |

正值表示配对耗时下降，负值表示增加。绝对ms/step取窗口中位数，配对百分比按相同块内比值汇总。

- [178759 完整汇总](../../../logs/a800/e2e/scale/178759/summary.csv)、[窗口](../../../logs/a800/e2e/scale/178759/windows.csv)、[配对块](../../../logs/a800/e2e/scale/178759/paired_blocks.csv)。
- [178760 完整汇总](../../../logs/a800/e2e/scale/178760/summary.csv)、[窗口](../../../logs/a800/e2e/scale/178760/windows.csv)、[配对块](../../../logs/a800/e2e/scale/178760/paired_blocks.csv)。

### 扩展测试口径

保持BF16、sequence parallel、DP=PP=CP=1、micro batch=1/global batch=4、Apex Adam和原有融合开关。hidden=2048、sequence=2048时每步8192有效token；attention与MLP目标shape分别为(2048,2048,2048)、(2048,2048,8192)，局部K为512和2048。12层共24个目标模块，每step实际目标调用数为96。

每轮4个位置平衡块、四种实现各4个独立窗口；每窗口10步预热后计时20步，计划两个独立作业重复。ms/step取全部rank中最慢墙钟，包含完整前向/反向、梯度同步、Adam和清梯度；数据预置GPU，排除初始化、预热、profile与checkpoint。与历史4层测试的窗口长度不同，结果分表报告。

每个窗口重建相同模型/optimizer/RNG和输入；本轮初始loss直接取该窗口第一步预热，不再借用另一作业。固定合成token重复使用，loss下降不代表真实数据泛化。

入口为`tools/a800/e2e/scale/`，脚本支持TP=4/8的全rank统计和映射检查。当前只验证4卡运行；被取消的8卡作业不计为已验证。

```bash
# 12层4卡预检（普通gpu_a800队列）
sbatch --gpus=4 --cpus-per-task=8 --export=ALL,E2E_CASE=l12-h2048-s2048-tp4,E2E_SCALE_STAGE=preflight tools/a800/e2e/scale/run.sbatch

# 预检完成后指定结果目录，运行两轮独立计时
sbatch --gpus=4 --cpus-per-task=8 --export=ALL,E2E_CASE=l12-h2048-s2048-tp4,E2E_SCALE_STAGE=timing,E2E_SCALE_PREFLIGHT=/绝对路径/预检目录 tools/a800/e2e/scale/run.sbatch
python3 tools/a800/e2e/scale/report.py
```
<!-- scale-results:end -->




## 双节点每节点4卡与通用多节点入口

可以采用两节点各4×A800，但必须先确定并行布局。历史表中的8卡作业取消记录继续保留；下面是双节点起步方案和通用多节点入口。历史取消作业未恢复；本轮新提交的双节点实测结果见本页顶部。

| 路线 | 并行布局 | 跨节点通信 | 对当前研究的意义 |
| --- | --- | --- | --- |
| A：先验证分布式训练 | TP=4、DP=2、PP=CP=1 | 两个DP副本之间同步梯度；TP组各位于一个节点 | 保留已验证的节点内Flux三策略，检查加入跨节点梯度同步后的整步收益 |
| B：验证跨节点GEMM-RS | TP=8、DP=1、PP=CP=1 | 一个TP组跨两个节点，包含前向RS和反向AllGather | 直接检验跨节点算子路径；需要新的Flux适配、数值与映射验收 |

建议先完成A，再开展B。若研究问题是“重排能否改善跨节点GEMM-RS”，必须完成B；A只能回答双节点训练中的模型整步效果。`GemmRS`的`nnodes`表示其TP组跨越的节点数：A中仍为1，B中才为2。节点总数不能直接填入每个算子的`nnodes`。

### 固定模型与统计口径

首先采用已完成单节点测量的12层、hidden=2048、sequence=2048、FFN=8192、heads=32、BF16和sequence parallel，保持micro batch=1、global batch=4。每步有效token均为8192，吞吐为`8192 / 完整step秒数`，不再乘TP、DP或节点数。

- A：每个DP副本每step累积`4 / (1 × 2) = 2`个microsteps；两个副本分配同一全局batch中的不同样本，TP组内数据保持一致。每rank每step目标模块调用数为`24 × 2 = 48`，局部K仍为512/2048。
- B：累积4个microsteps；每rank每step有96次逻辑目标模块调用，局部K为256/1024；全局目标shape仍为`(2048,2048,2048)`和`(2048,2048,8192)`。分层实现可能将一次逻辑调用拆成多个底层GEMM-RS，需分别计数。
- 如果A另测global batch=8以保持每副本4个microsteps，应单独列为吞吐扩展实验，每步16384 token；不能与global batch=4的loss/step直接混报。

计时继续采用四组位置平衡、每组4个窗口、10步预热+20步计时、两个独立作业重复。每个窗口统一初始化、固定全局样本顺序，包含前向、反向、DP梯度同步（A）、Adam及清梯度。窗口开始前全rank同步，结束时等待GPU完成，再对8个rank的本地持续时间取最大值；不用跨主机绝对时间戳相减。profile和checkpoint独立运行。

### 历史入口的限制与通用入口改造范围

以下根据本仓库源码核对；当前`scale/`入口仍是单节点版本，不能直接用`sbatch --nodes=2`启动本方案。

| 文件 | 当前限制 | 双节点适配要求 |
| --- | --- | --- |
| `tools/a800/e2e/scale/run.sbatch` | `nodes=1`、一个task、禁用IB，只采集本机拓扑 | 申请2节点、每节点4卡、每节点一个launcher task；逐节点采集GPU/NIC/拓扑和环境 |
| `tools/a800/e2e/scale/launch.py` | `--standalone`，每节点进程数直接取TP | 两个launcher共用master地址/端口，每节点4进程，分别设置node rank；四策略和窗口执行顺序一致 |
| `tools/a800/e2e/scale/worker.py` | 可见GPU数等于WORLD，全部GPU互访，断言DP=1且TP=WORLD | 区分LOCAL_WORLD_SIZE=4与WORLD_SIZE=8；只检查节点内P2P；CPU绑定按本地进程数划分；A加入DP数据分片和初始化一致性检查 |
| `tools/a800/e2e/scale/adapter.py` | `GemmRS(tp_group, 1, ...)`，现有IPC初始化 | A审计每个TP子组的共享内存初始化与rank索引；B选择并验证跨节点实现，禁止跨主机使用CUDA IPC指针 |
| `config.json`、`summarize.py`及报告 | TP与全局rank数等同，按TP收集结果 | 独立记录nodes、gpus_per_node、TP、DP、world_size；A同样必须收齐8个rank，校验batch及调用数 |

启动契约为Slurm `--nodes=2 --ntasks-per-node=1 --gpus-per-node=4`，由`srun`在两节点各运行一个launcher；每个launcher调用`torch.distributed.run --nnodes=2 --nproc_per_node=4 --node_rank=<0或1> --master_addr=<首节点可达地址> --master_port=<本作业端口>`。移除`--standalone`，不要让8个Slurm task各自再生成4个GPU进程。上述契约已在新增`tools/a800/e2e/distributed/`中实现；原有`scale/`保留历史复跑用途。通用入口的提交命令见下文。

网络预检先确认两节点接口、路由、IB/RoCE设备及NCCL实际选用的传输。现有`NCCL_IB_DISABLE=1`会禁用IB路径，双节点性能实验不应无条件继承；接口/HCA按实际分配节点选择并记录。Socket可用作功能排查，但其结果应单列，不能当作IB性能。`FLUX_FORCE_NVLINK=1`也不能作为节点间存在NVLink的证据。

### 路线B的Flux实现边界

仓库已有`python/flux/gemm_rs_sm80.py`中的`GemmRS_multinode`：先在节点内执行GEMM-RS，再通过`batch_isend_irecv`交换结果并相加，可作为SM80双节点原型参考。C++ `src/gemm_rs/ths_op/gemm_reduce_scatter.cc`也有`nnodes>1`分支。本轮分层适配器已通过顶部所列双节点测试；C++直接跨节点分支仍未经本轮验证。

本轮以分层原型为参考实现了独立适配器，适配接口、ring/reduce选项、进程组和输出生命周期，并补齐工作区复用后的同步；没有直接把该原型替换到原有`op.forward(..., reduce_scatter_option=...)`。两节点时该原型将M按节点拆分，底层节点内算子M=1024、局部TP=4，必须重新验证这些实际shape的映射与误差。三个Flux候选使用相同跨节点交换和归约，仅改变节点内映射，结果应标为“分层GEMM-RS中的节点内重排”；不能宣称已优化跨节点调度。若改用C++直接跨节点分支，还需单独确认构建依赖、实际kernel/调度器和候选映射是否生效。

### 分阶段验收

1. **资源与通信：** 保存两节点GPU UUID、节点内拓扑、NIC、软件/构建哈希、rank→host/local rank/TP/DP映射；8个rank均能完成NCCL集合通信。A检查两组TP和四组DP通信；B检查八rank的RS/AllGather，并对照确定性参考结果。
2. **算子与模型预检：** 先跑原生Megatron，再跑原始Flux与两个重排候选；检查输出、输入/权重梯度、参数更新、有限loss/梯度及零跳步。A检查DP同步后同一TP分片在两个副本中一致；B重新校验实际shape和全部rank，无静默回退。正式计时前固定并记录数值误差阈值，不能仅凭loss下降通过精度验收。
3. **短训练与正式计时：** 预检通过后执行同一拓扑内的四组配对实验，保存8个rank原始记录、完整退出码、窗口中位数、波动与配对区间。单节点与双节点、TP4/DP2与TP8/DP1分别出表；同一作业缺任一rank或节点失败都不能计为完成。

### 通用测试入口与复跑方法

新增`tools/a800/e2e/distributed/`，不将双节点4卡写死为唯一配置。`submit.py`支持命令行或JSON配置，默认只解析并打印计划，添加`--submit`才调用普通`gpu_a800`队列；每次提交生成独立结果目录；本中心多节点自动添加`--qos=gpugpu`（仍为普通`gpu_a800`分区）。多节点需所有节点可见相同代码、Conda环境、冻结Flux构建和结果目录。

| 可配置项 | 参数/行为 |
| --- | --- |
| 拓扑 | `--nodes`、`--gpus-per-node`、`--tp`；DP自动取`nodes × gpus_per_node / TP`，当前PP=CP=1 |
| 模型 | `--layers`、`--hidden`、`--ffn`、`--heads`、`--sequence`、`--vocab` |
| 数据 | `--micro-batch`、`--global-batch`、`--seed`、`--token-seed`；统一生成全局token bank后按DP分片 |
| 对照 | `--policies native original remote_first interleaved`，可选子集；原生组不依赖Flux构建 |
| Flux路径 | 默认`--flux-transport local`；跨节点TP显式指定`hierarchical`，组内跨节点数自动推导 |
| 测量 | `--warmup-steps`、`--timed-steps`、`--blocks`；blocks必须是候选数量的倍数以保证位置平衡 |
| 前向误差 | `--atol`、`--rtol`默认0.02；预检逐目标模块首次前向对照本地GEMM+NCCL RS，记录max-abs/relative-L2 |
| 作业 | `--time`、`--cpus-per-task`、`--python`；NCCL接口/HCA等通过环境变量传入并记录 |

从TempName根目录执行：

```bash
# 先查看可选参数与解析结果，不申请GPU
python3 tools/a800/e2e/distributed/submit.py --help
python3 tools/a800/e2e/distributed/submit.py --nodes 2 --gpus-per-node 4 --tp 4

# 两节点各4卡，节点内TP4、跨节点DP2：四种实现预检
python3 tools/a800/e2e/distributed/submit.py --nodes 2 --gpus-per-node 4 --tp 4 --submit

# 两节点各4卡，TP8跨节点：四种实现使用同一分层通信路径
python3 tools/a800/e2e/distributed/submit.py --nodes 2 --gpus-per-node 4 --tp 8 \
  --flux-transport hierarchical --submit

# 扩展到四节点各4卡、TP8/DP2；同时演示模型与batch参数
python3 tools/a800/e2e/distributed/submit.py --nodes 4 --gpus-per-node 4 --tp 8 \
  --flux-transport hierarchical --layers 12 --hidden 4096 --ffn 16384 \
  --heads 64 --sequence 4096 --global-batch 8 --submit

# JSON可复用（也可使用历史结果目录的config.json）；CLI显式参数覆盖JSON中的同名设置
python3 tools/a800/e2e/distributed/submit.py \
  --config tools/a800/e2e/distributed/examples/tp8-dp2.json --submit

# 使用与预检完全相同的配置正式计时；指定上一步输出的结果目录
# 独立重复两次此命令，产生两个不同作业/结果目录
python3 tools/a800/e2e/distributed/submit.py \
  --config tools/a800/e2e/distributed/examples/tp8-dp2.json \
  --stage timing --preflight /绝对路径/已完成的预检目录 --submit
```

结果位于`logs/a800/e2e/distributed/<run_id>/`：保存解析配置、脚本快照/哈希、作业号、逐节点网络/GPU信息、逐rank训练记录、独立profile、映射日志及退出码。预检包括两步预热+三步短训练、独立profile；正式计时自动写`windows.csv`、`summary.csv`、`paired_blocks.csv`和`analysis.json`。统计收齐`world_size`个rank，检查DP副本初始/末态参数一致及全部参数指纹发生更新，取全rank最大耗时。tokens/s按全局batch计算，不按卡数重复计数。性能区间来自单作业内配对块，独立作业需另行重复。

**支持边界：** 当前运行环境仍限定A800、BF16、sequence parallel和既有冻结构建。TP至少为2，并须完整落在节点内或跨完整节点。Flux暂只接受TP不小于每节点卡数、节点内TP=2/4/8以及`sequence × micro_batch`可被`128 × TP`整除；原生组允许节点内多个TP组。新shape仍须通过映射与实际kernel校验；若调度到不同tile/hparams或出现不支持的cohort，预检失败，不静默回退。分层实现仅改变节点内映射，跨节点交换/归约三策略共用，不代表跨节点调度优化。

**状态：** 通用启动、训练适配、分层Flux、全rank汇总、配置校验和CPU回归检查已实现；已完成TP4/DP2和TP8/DP1各两轮双节点正式计时，完整结果、同步修正与首次失败记录见本页顶部实测更新；更大拓扑仍须单独预检。前向容差检查、有限梯度和DP参数一致性不等于完整配对梯度精度验收，也不代表真实数据收敛；后续按上面的分阶段验收继续补充。已有单节点历史实测与新入口结果分别报告。

## 历史基线：4层 GPT

本页按同一小型GPT配置，对比**原生Megatron-LM、原始Flux、远端重排、交替重排**的loss、完整optimizer step时间、有效tokens/s和梯度范数。本页汇总实测运行指标。

### 4层历史结果

已完成178711、178712共2轮独立Slurm作业，均正常退出。每轮32个计时窗口，四种实现各8个窗口；各窗口20步预热后计时30个完整optimizer steps。

| 实现 | 178711：ms/step | 178711：tokens/s | 178712：ms/step | 178712：tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 原生 Megatron-LM | 67.795 | 60,418 | 67.523 | 60,661 |
| 原始 Flux | 64.555 | 63,450 | 64.385 | 63,617 |
| 远端重排 | 64.520 | 63,485 | 64.451 | 63,553 |
| 交替重排 | 64.666 | 63,341 | 64.432 | 63,571 |

上表为各实现8个窗口的中位数。每个窗口先取四个rank中最慢的耗时，再除以30步；`tokens/s = 4096 / 每步秒数`。TP=4处理同一批token，不再额外乘4。

### 同一训练步位置的 loss

每个窗口都从相同模型初始化和空optimizer状态重新开始，使用相同token、RNG和训练设置。初始step 1 loss取独立短训练作业178710；step 21和step 50分别取各计时窗口预热后的首步与最后一步。三列的采样位置分别标明。

| 实现 | 初始loss（独立短训练step 1） | 178711：step 21 | 178711：step 50 | 178712：step 21 | 178712：step 50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原生 Megatron-LM | 9.278128 | 0.822296 | 0.00386960 | 0.822296 | 0.00386960 |
| 原始 Flux | 9.278234 | 0.822967 | 0.00387328 | 0.822967 | 0.00387328 |
| 远端重排 | 9.278172 | 0.822328 | 0.00387237 | 0.822328 | 0.00387237 |
| 交替重排 | 9.278246 | 0.821097 | 0.00386978 | 0.821097 | 0.00386978 |

loss表中的计时段数值同样取8个窗口的中位数。四组都在重复使用的固定合成数据上训练，loss下降只说明本次短训练行为，不能代表真实语料的收敛或泛化能力。

### 梯度与运行状态

| 实现 | 178711：末步梯度范数 | 178712：末步梯度范数 | 跳过更新 | loss/梯度NaN或Inf |
| --- | ---: | ---: | ---: | --- |
| 原生 Megatron-LM | 0.022569 | 0.022569 | 0 | 无 |
| 原始 Flux | 0.022508 | 0.022508 | 0 | 无 |
| 远端重排 | 0.022549 | 0.022549 | 0 | 无 |
| 交替重排 | 0.022616 | 0.022616 | 0 | 无 |

梯度范数来自Megatron原生optimizer返回的全局范数；保留全部rank、全部step的原始值。



### 4层历史配置

| 项目 | 设置 |
| --- | --- |
| GPU/并行 | 单节点4×A800，NV8；TP=4，DP=PP=CP=1 |
| 模型 | 4层GPT；hidden=1024，FFN=4096，16个attention heads |
| 序列/batch | sequence=1024；micro batch=1，global batch=4；每步累积4个microsteps |
| 词表 | token范围0–8191；实际padded vocabulary=8704 |
| 精度/后端 | BF16；local Transformer、Apex LayerNorm；sequence parallel开启 |
| 优化器 | 原生Megatron FP32主权重 + Apex Adam；lr=1e-4，weight decay=0.01，clip-grad=1.0 |
| 数据 | 固定GPU驻留synthetic token bank；token seed=4321，模型seed=1234；4096有效tokens/step |
| 其他 | dropout=0；不使用activation checkpointing或CUDA Graph；关闭梯度累积融合及所列bias/softmax融合 |
| 软件 | Megatron-LM core_v0.12.3（3ea68ad6042cc1204386ae9364358f7c4de1bc37）；PyTorch 2.6.0+cu124；独立Conda环境flux-megatron-a800 |

原生组保留Megatron的RowParallelLinear GEMM+NCCL路径。Flux三组共用已有IPC修复、BF16精度、归约选项及hparams，仅tile映射不同；本次模型接入采用`ring_reduction=True`。替换4层中的attention输出投影和MLP第二线性层，共8个模块。反向保持sequence AllGather和本地dgrad/wgrad计算，optimizer及其余模型结构相同。

真实目标shape为`(M,N,K_global)=(1024,1024,1024)`及`(1024,1024,4096)`，局部K分别为256和1024。独立诊断确认三种Flux策略每rank每step实际执行32次目标GEMM-RS，无回退；tile=128×128×32、StreamK-SK、3 stages。


### 4层历史复跑入口

脚本位于`tools/a800/e2e/`；需要现有独立Conda训练环境及`outputs/a800/phase3/three-way-build`。所有GPU作业通过普通`gpu_a800`队列提交。

```bash
# 从TempName根目录运行四组准备与诊断
sbatch --export=ALL,E2E_STAGE=validate,E2E_ALLOW_EXPLORATORY=1 tools/a800/e2e/run.sbatch

# 准备作业完成后，指定其结果目录，再执行下面的命令两次进行独立复测
sbatch --export=ALL,E2E_STAGE=exploratory-timing,E2E_ALLOW_EXPLORATORY=1,E2E_VALIDATION_DIR=/绝对路径/准备作业目录 tools/a800/e2e/run.sbatch

# 用已完成的两轮结果生成本页及CSV
python3 tools/a800/e2e/report.py logs/a800/e2e/comparison/178710 \
  logs/a800/e2e/comparison/178711 \
  logs/a800/e2e/comparison/178712
```

