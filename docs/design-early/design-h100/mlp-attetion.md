# H100 单节点 TP8：Attention与MLP作用范围对比

更新：2026-10-14。**状态：四组范围对比代码已实现，71项CPU回归、校准及32窗口dry-run通过；采样库与共同候选库完成无GPU编译和链接（含Python binding及mapping程序）。尚未执行真实Attention校准或本方案GPU验证。** 独立入口、验证摘要与实际校准/测试命令见 [README-mlp-attention.md](../../../tools/h100/phase6/README-mlp-attention.md)。已有 [交替五组入口](../../../tools/h100/phase6/README-ablation2.md) 的CPU/编译检查只覆盖其MLP范围，不能作为本方案的Attention验证。本文独立于 [消融实验2.md](消融实验/消融实验2.md)，不改写旧分组、旧会话或旧结果。

## 1. 测什么：只保留用户指定的四组

本轮固定单节点8×H100、GPT 6.7B、TP8，比较朴素Megatron＋量化和交替版完整候选，在“仅Attention”和“Attention＋MLP”两个作用范围下的完整训练step性能。

这里的范围精确定义为：

- **Attention**：每层 `self_attention.linear_proj`，即注意力输出投影的前向GEMM后、ReduceScatter对应的远端输出贡献；不包含Q/K/V投影、QK矩阵乘法、Softmax、AV矩阵乘法或KV cache。
- **MLP**：每层 `mlp.linear_fc2`，即FFN第二个线性层的前向输出贡献；不包含 `linear_fc1` 或激活函数。
- “完整候选”指 **交替基础顺序＋到达优先＋定向选择性量化**；基础顺序固定 `interleaved`。它不表示整个模型使用FP8计算，GEMM及本地贡献仍为BF16，反向沿用当前BF16/STE实现。

| 本轮策略ID | 对应需求 | Attention输出投影 | MLP输出投影 |
| --- | --- | --- | --- |
| `native_taco_attn_mlp` | native_taco－MLP－Attention | 原生Megatron GEMM＋朴素TACO远端通信 | 原生Megatron GEMM＋朴素TACO远端通信 |
| `native_taco_attn` | native_taco－Attention | 原生Megatron GEMM＋朴素TACO远端通信 | 原生Megatron BF16路径，无量化 |
| `interleaved_full_attn` | 交替版完整代码－Attention | 交替＋到达优先＋选择性量化 | 原生Megatron BF16路径，无候选替换/量化 |
| `interleaved_full_attn_mlp` | 交替版完整代码－Attention－MLP | 交替＋到达优先＋选择性量化 | 交替＋到达优先＋选择性量化 |

策略ID统一使用 `attn`，文档文件名保留用户指定的 `mlp-attetion.md`。四组均为新ID，不覆盖旧 `native_taco`、G1–G4的含义；不增加远端顺序、MLP-only、随机mask、全远端候选或算子性能组。

**旧 `native_taco` 确实只量化MLP通信。** 现有 [adapter.py](../../../tools/a800/phase5/e2e/adapter.py) 在 `native_taco` 分支中直接返回Attention的原生forward，且 `stop_audit()` 也只保留MLP的量化forward。新 `native_taco_attn` 和 `native_taco_attn_mlp` 必须显式扩展作用范围，不能通过重命名旧策略获得。

候选两组统一从当前冻结实现派生：选中的模块采用SM90编译的GemmV2和原候选混合传输/解码路径。两个Attention-only组的未选中MLP均保留原生Megatron BF16 forward，使该部分实现一致；这是本范围实验的显式约定，不能直接照搬旧候选“未量化模块也走Flux”的路由。除作用范围、对应shape的计划和必要路由外，不同时更改GEMM参数、解码算法、工作区或其他模型计算。

## 2. 固定配置与两种独立shape

| 项目 | 固定值 |
| --- | --- |
| 模型 | GPT 6.7B，32层、hidden4096、FFN16384、32 heads |
| 并行与设备 | 单节点8×H100，TP8/DP1/PP1/CP1；默认沿用已验证的132 SM设备配置，记录UUID与顺序 |
| 输入与batch | S2048、micro batch1、global batch4；每optimizer step累计4个微批、8192 token |
| 初始化 | seed1234、token_seed4321；配置词表32768、实际padded vocab33792；四组固定相同初始参数、RNG和合成token规则 |
| 量化 | H128/E4M3，远端贡献；本地仍为BF16。native_taco按原朴素实现量化所选模块的全部远端贡献，候选按物理tile选择 |
| 候选参数 | interleaved，合法到达窗口64；每种shape各自使用每来源1/64的选择预算上限，不在两种shape之间重新分配预算 |
| 计时 | 每窗口10步预热＋20步完整optimizer step，包含前向、反向、通信、参数更新和清梯度 |

两类输出尺寸相同，但K不同：

| 模块类型 | 全局 `[M,N,K]` | TP8本地K | 每个模型实例的模块数 |
| --- | --- | --- | ---: |
| Attention `linear_proj` | `[2048,4096,4096]` | 512 | 32 |
| MLP `linear_fc2` | `[2048,4096,16384]` | 2048 | 32 |

这两个shape必须使用独立的到达置换、物理mask及哈希。不得因为M/N相同，就按输出尺寸命中同一张表；分派至少核对模块角色与完整shape，CUDA查表也必须区分本地K。不能通过把模型FFN改成4096来伪造Attention测试，模型本身始终保持6.7B配置。

默认候选起点是 `h100-phase6-candidate-v1-20261013` 的已测交替实现，见 [冻结manifest](../../../outputs/h100/phase6/frozen-candidate-v1-20261013/manifest.json)，其构建源码提交为 `ff38f5b1ff1c80cb831f58c64b95f42b0379879a`。旧MLP计划、库及报告保持原样。原生基线的数学实现沿用现有朴素TACO，只扩展模块范围，不优化baseline的打包、编解码或通信算法。

## 3. 校准、计划与构建要求

1. **Attention单独做H100 BF16校准。** 采样/映射对应 `[2048,4096,4096]`、interleaved及候选将采用的GemmV2算法。沿用训练pass 0/1、留出pass 2的隔离规则；采样库不进入正式计时。
2. MLP可在设备、顺序、配置、映射与来源身份核验一致时复用冻结H100计划；否则在新会话重新校准。Attention缺少实测计划时，先标记“待H100校准”，不将MLP计划或CPU合成计划当作可执行Attention计划。
3. 两个候选使用**相同的Attention计划与物理mask**，不能分别重新拟合；双模块候选的MLP计划单独冻结。开启另一模块不应改变Attention的已选对象、参数或通信协议。
4. 计划按模块角色及完整shape组织，记录校准输入、训练/留出遍次、设备身份、置换、各来源实际选择数和哈希。本地mask为零，预算只是上限；不得为了制造收益强行补满选择量。
5. 新建范围对比构建，支持Attention本地K512和MLP本地K2048。当前 [common.py](../../../tools/h100/phase6/common.py)、[calibrate.py](../../../tools/h100/phase6/calibrate.py)、[plan.py](../../../tools/h100/phase6/plan.py) 和 [build.py](../../../tools/h100/phase6/build.py) 以MLP单shape为主，不能只删除 `MLP-only` 检查而不补齐多shape生成、查表和验证。
6. 两个候选应从同一套源码派生，优先用显式scope选择路径；基线两组同样共享一套朴素TACO实现，仅模块集合不同。保持每模块/每实例的计划和工作区状态隔离，不能让相邻Attention与MLP调用覆盖彼此配置或提前复用输出缓冲。

## 4. 必须修改和核验的执行路径

独立入口 `mlp_attention.py` 不改写旧五组实验的分组与默认行为；采用协议ID `h100-gpt67-tp8-mlp-attention-v1`，在manifest、worker记录和report中统一记录协议、策略及作用范围。

- **原生基线路由**：按明确模块集合调用原始 `RowParallelLinear.forward` 的模块级通信替换，不修改进程全局通信函数，不替换原生GEMM/bias/线性反向。`native_taco_attn` 只有32个Attention模块启用tensor TACO；`native_taco_attn_mlp` 有32个Attention＋32个MLP模块启用。两组都不得导入Flux候选库。
- **候选路由**：`interleaved_full_attn` 的Attention采用完整候选，MLP直接保留原生Megatron BF16 forward；`interleaved_full_attn_mlp` 两类均采用完整候选。不能把未选中模块误判为fallback，未选中路径必须预先登记；真正的未预期回退仍为失败。
- **预热与正式阶段一致**：现有adapter的安装、`make_forward()`、`stop_audit()`，以及worker中的native分流、模块计数、字节核验、库加载和shape登记都要按scope适配，防止smoke/profile正确但正式计时又恢复成旧MLP-only行为。
- **报告按角色核验**：每组记录全部64个目标投影模块的实际路径、shape和调用次数，另记录启用量化/完整候选的32或64个模块。原生组验证原forward保留；候选组验证加载库、到达表、物理mask及预期wire bytes。两个投影输出尺寸相同不代表计划可以通用。
- **kernel证据**：现有 [ablation_routing.py](../../../tools/h100/phase6/ablation_routing.py) 与 [ablation2_report.py](../../../tools/h100/phase6/ablation2_report.py) 默认量化发生在MLP且Attention为native/V3，必须增加scope感知，不能硬套旧“必须出现Hopper V3 attention”的条件。双模块候选可在两个目标投影均使用V2，其他Attention核心算子仍保持模型原有实现。
- **正确性**：每个启用模块按现有BF16参考与codec参考预算检查，BF16输出路径沿用原容差；loss/梯度必须有限，初始化及输入指纹一致，零skip和零意外fallback。预检应覆盖Attention单独启用、两类交错调用和两个shape配置不会串用；短程检查通过不等于长期收敛验证。

需要保存必要映射、数值和profile证据，但本轮只报告完整训练step性能，不新增独立算子性能实验。

## 5. 默认调度与测量规模

四组同一配对块内全部执行。每轮4个块，第一轮固定随机排列后循环移位，使每组在四个执行位置各出现一次；第二轮反转块和块内策略顺序。调度seed20261013，prepare时冻结。

```text
4组 × 4个配对块 × 2轮 = 32个正式窗口
32窗口 × 8rank = 256份正式计时rank记录
每窗口10步预热＋20步完整optimizer step，共640个正式计时step
```

默认只运行一套，不自动增加其他组或独立复测。可支持在启动前将每轮块数设为4的正整数倍；四块是简化首测，区间精度有限，按实际保留块数解释结果。

正式计时前四组各一次smoke及profile，共8个预检窗口、64份rank记录、32份压缩profile trace。候选映射覆盖三个“策略×启用shape”组合：Attention-only的Attention，以及Attention＋MLP的两个shape；未选中MLP的原生BF16路由也要核对。校准及映射检查不计作正式训练窗口，新增Attention校准证据另行回传。

四组必须在新会话同期重新计时，不能拿历史MLP-only窗口或旧native_taco耗时充当本轮对照。GPU检查沿用 [watcher.md](watcher.md)：整套开始前检查，预检/计时期间不周期查询资源占用。默认不创建等待队列；若后续明确使用排队，仍按忙600秒、空闲10秒、连续3次规则，不在组间重复检查。

## 6. 比较对象与报告

**主要加速比必须匹配作用范围，不能用同一个baseline替代两个基线。**

| 比较 | 主要回答的问题 |
| --- | --- |
| `interleaved_full_attn` 对 `native_taco_attn` | 仅Attention范围内，完整候选相对朴素量化的整体收益 |
| `interleaved_full_attn_mlp` 对 `native_taco_attn_mlp` | Attention＋MLP范围内，完整候选相对朴素量化的整体收益 |
| `interleaved_full_attn_mlp` 对 `interleaved_full_attn` | 已启用Attention候选后，将MLP也纳入候选实现的净效果 |
| `native_taco_attn_mlp` 对 `native_taco_attn` | 原生基线中额外量化MLP通信的净效果 |

两套候选与原生基线的计算/通信实现不同，不能把整体差距全归于排序或量化选择。候选从Attention-only扩展到双模块时，MLP由原生Megatron BF16转为GemmV2完整候选；该比较回答“扩展整套实现是否值得”，**不是仅增加MLP量化的纯算法消融**。本轮按用户要求保持四组，不额外加入工程对照。

当前没有同期MLP-only组，因此不能用本轮四组单独断言“优化Attention比仅优化MLP更好”；历史MLP-only结果只能作背景。也不把原生BF16 Megatron或固定Flux融合v2作为未测的第五个分母。

报告要求：

1. 全部四组逐轮ms/step中位数与tokens/s；每窗口取八rank最大elapsed除以20，吞吐按8192 token/step计算。
2. 两张主要加速表：同范围配对比值 `T_baseline,b / T_candidate,b` 的**中位数**、**最大值**；最好值附轮次、block及双方耗时，不能分别取两组极值相除。
3. 上表四个比较均报告配对耗时下降 `100 × (1 − exp(mean(log(T_X/T_Y))))` 和按完整配对块bootstrap的95%区间（10000次、seed20261009）。两轮分别报告，可附同一会话合并8块的结果；不足两块时不估计区间。
4. 主表保留全样本。可附10%慢窗口敏感性表：按每轮每策略原始中位数的1.10倍标记，只要四组任一组触发，就剔除整块四组；不迭代阈值，四个比较共用保留块，列明触发策略与剔除块。没有剩余块时写“无可用配对块”。
5. 不强求双模块候选比Attention-only快；负收益照实记录。正确性、完整性与性能结论分开，不套用旧“两种必须baseline均达到4%”验收。

## 7. 开发、验证与交付

优先复用现有H100准备/运行流程和 [ablation2.py](../../../tools/h100/phase6/ablation2.py) 的证据、调度及统计机制，但新增独立scope协议和入口；保留旧 `native_taco` 的MLP-only语义与五组模式。维护代码在 `tools/`、`src/`、`include/`、`python/` 等维护目录修改，不直接编辑历史 `outputs/` 快照。

必要CPU检查包括：四组scope及32窗口调度；两种shape/K的精确分派；两个候选同一Attention mask；本地不量化和预算；原生forward保留及无Flux导入；预热/正式路径一致；逐角色模块计数；错误shape、哈希、scope或缺rank时拒绝；同范围配对统计、统一整块筛选；dry-run不查询GPU、不启动worker。可进行环境允许的无GPU编译，不将其写为GPU验证。

开发端本次只实现代码、CPU检查、dry-run及无GPU编译；不启动GPU实验、Slurm或自动等待队列。新的Attention真实校准、完整映射/数值/profile和四组计时留给有H100的执行会话。CPU合成计划只能用于测试，不得被标记为正式可执行输入。

新会话路径从仓库根目录起算，例如：

```text
outputs/h100/phase6/mlp-attention-gpt67-tp8-<时间>/
  calibration/            # 校准构建与冻结输入
  plans/                  # Attention、MLP分别登记计划及哈希
  candidates/             # 新范围构建/源码、manifest及编译日志
logs/h100/phase6/mlp-attention-gpt67-tp8-<时间>/
  calibration/            # 新Attention实测记录；MLP如复用，保留来源登记
  compare/                # submission、scripts、schedule、预检、rank与报告
  handoff/                # 环境、版本、范围、复用关系、完成状态及缺项
```

执行端交付256份正式rank、64份预检rank、32份压缩profile trace、各候选/shape映射、校准原始记录、计划、冻结输入、库/源码/控制器哈希、命令和进程日志、完整及筛选报告（如启用）。实际规模调整时按冻结调度计算，失败记录保留并注明缺项。

Git规则沿用 [instruction-h100.md](instruction-h100.md)：维护代码、文档和正式H100实验的文本/trace证据需要回传；`.so/.o/.a`、构建缓存、Python缓存、环境及checkpoint保持忽略。CPU演练/编译检查产物可以只留本机，验证摘要写入使用说明。最终交付真实可执行的准备、Attention校准、计划拟合、构建、dry-run、预检、计时及报告命令；功能未实现前不把拟定参数当作已有CLI。
