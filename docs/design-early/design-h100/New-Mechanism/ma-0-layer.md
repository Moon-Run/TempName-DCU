# MA0：MA11 与 Flux＋基础量化的完整子层计时

更新：2026-10-28。目标是在 **gxn75 单节点、8×H100、TP8 DP1** 上，直接比较 MA11 与 Flux＋基础量化接收端融合 v2 的完整 Attention／MLP 子层。当前是测试实现与协议，**尚无本轮 H100 性能结果**。

## 1. 比较对象与范围

| 组别 | 实现 |
| --- | --- |
| MA11 | `F4-cache-zcount-csoftmax`：CR3 消费链融合、MA8 激活复用、MA9 保序零统计、MA11 注意力中间态消除 |
| Flux＋基础量化 | `basic`：原生 Hopper Flux GemmOnly、独立 CUDA 编码、NCCL All-to-All、接收端解码／BF16 归约融合 v2 |

沿用 [baseline.md](baseline.md) 对应实现和固定模型：32 层，hidden=4096、FFN=16384、32 heads、sequence=2048、BF16、microbatch=1、TP8。每次子层调用处理一个 microbatch；序列并行输入为每 rank `[256, 1, 4096]`。

两组的量化策略不同：MA11 使用原选择性量化预算，basic 使用全远端量化。因此这里是完整方案对比，不是单独 MA11 softmax 的消融，也不要求跨组输出或梯度逐位一致。两组各自运行已有原始训练 smoke 和算子数值检查。

**这次测量完整子层的独立执行时间，不是单个输出投影，也不是训练 step 除以层数。** 固定输入、上游梯度和初始权重，逐层调用真实 Megatron 方法；32 层全部保留，分别记录前向、反向和合计耗时。

## 2. 子层边界

| 区域 | 前向覆盖 | 反向覆盖 |
| --- | --- | --- |
| Attention | 输入 LayerNorm → QKV 投影与通信 → QK、mask、softmax、PV → 输出投影与通信 → bias／residual | 从 Attention 残差输出梯度回传至子层输入，包含该子层参数梯度及原 TP 通信 |
| MLP | 前置 LayerNorm → FC1 与通信 → GELU → FC2 与通信 → bias／residual | 从 MLP 残差输出梯度回传至子层输入，包含该子层参数梯度及原 TP 通信 |

Megatron 原 `_forward_attention` 最后还计算 `pre_mlp_layernorm`。独立 Attention 测量时将这一模块临时替换为 Identity，调用结束后恢复；MLP 测量先调用原 `pre_mlp_layernorm`，再调用原 `_forward_mlp`。这使两个区域各包含自己的前置 LayerNorm，避免重复计算或漏计。两组使用完全相同的边界，并在计时前对照原方法验证输出逐位一致。临时替换通过 `try/finally` 恢复，不改 native 算子、通信顺序、原 residual 绑定或 autograd 公式。

反向使用 `torch.autograd.backward`，同时计算输入和参数的局部梯度，包含原 autograd 路径中的 TP 通信。完整训练步末尾由 `finalize_model_grads` 统一处理的跨模块梯度汇总／通信不归入单个子层。没有模块 backward hook、额外 autograd Function 或 profiler。MA9 属于 optimizer 步尾统计，**不在任何 Attention／MLP 计时区域中**；不能把 MA9 收益归入子层加速。

## 3. 计时与统计协议

1. 准备并冻结原 native 源码和依赖，编译或校验复用的原生库。
2. 两组分别执行原完整模型 smoke：2 步预热、3 步训练，检查数值、机制安装和独立输出。两组都通过后才进入子层计时。
3. 每个计时窗口启动新的 8-rank 进程。构建相同初始模型，做一次不更新权重的完整前反向以初始化工作区，关闭原审计路径。
4. 对 32 层的 Attention／MLP 各执行 **10 次预热＋50 次正式调用**。输入和上游梯度由独立固定种子生成；跨组核对逐 rank、逐区域的输入、梯度种子数据及权重哈希。调用之间清理梯度，整个计时窗口不执行 optimizer 更新。
5. 每次调用前做 barrier 和设备同步，均在计时区间外。CUDA Event 分别位于前向开始、前向结束／反向开始、反向结束。前反向之间不插入同步；结束后同步再读取事件。计时包含区间内实际 TP 通信和依赖等待。
6. 默认 **8 个完整配对块、16 个计时窗口**，AB／BA 各四块；奇偶块反向遍历层号与角色，减轻顺序漂移。每次匹配调用取八 rank 耗时最大值，再对 50 次调用求均值；每个区域都保留全部配对块。
7. 报告逐层结果以及 32 层等权平均结果。分别列出基础量化耗时、MA11 耗时、减少的 ms、配对降幅和按完整块 bootstrap 的 95% CI。合计使用每个 rank 自己的完整前反向区间，不把不同 rank 的前向／反向最大值相加。

有收益或没有收益都正常完成全部窗口。数值失败、缺失 rank／层、输入或权重身份不一致、进程失败会停止并保留日志。每个窗口启动前检查 GPU 上是否已有计算进程，不终止其他用户任务；实验期间仍需安排独占八卡。

这些结果是固定输入和权重下的独立子层基准，**不代表完整训练中逐层的重叠时间，也不能将节省乘以 32 层、4 个 microbatch 作为整步收益**。CUDA Event 时间也可能包含主机提交不及时造成的设备空闲；它不是单纯 kernel 执行时长之和。全层平均 CI 是主汇总；逐层 CI 未做多重比较校正。

## 4. 在 75 上运行

在 `/workspace/TempName` 使用原 75 Python 环境，默认 CUDA 编译工具链为 `/usr/local/cuda-12.8`：

```bash
cd /workspace/TempName
bash tools/h100/phase6/run_ma0_layer.sh --execute
```

不加 `--execute` 只预览，不创建目录、不查询 GPU、不编译。默认后台启动，关闭 VS Code 不依赖原终端继续运行；需要前台查看时加 `--foreground`。

首次运行会导出固定源码快照并编译。若 75 的 `/workspace/tp8-dp2-bench` 已完成原生编译，可以显式复用：

```bash
bash tools/h100/phase6/run_ma0_layer.sh --execute \
  --package /workspace/tp8-dp2-bench \
  --build-root /workspace/tp8-dp2-bench/build/node1-e9f5dbfbabdc
```

复用前校验包哈希、原始构建身份、native 输入、库文件、Python／Torch／编译器身份，以及包内 adapter／机制与当前代码一致。复用的是源码和原生库，**实际 worker 仍为单节点 TP8 DP1，不启动双节点通信预检或 DP2 训练**。不修改原包和原构建。默认复用的编译配方也会产生一个未参与本次测量的 Megatron CUDA codec；实际只启动表中两组。

可用 `MA0_LAYER_PYTHON` 覆盖 Python，`MA0_LAYER_CUDA_HOME` 覆盖 CUDA。`--out` 必须是新目录，避免覆盖失败记录；不要在已有测试未退出时重复启动。

## 5. 输出与验证

新结果写入 `logs/h100/phase6/ma0-layer-时间戳/`：

| 文件 | 用途 |
| --- | --- |
| `batch-state.json` | 当前阶段、完成／失败状态与错误 |
| `handoff/container-supervisor.log` | 后台控制器日志 |
| `plan.json`、`workers.json` | 模型、测量协议和冻结 worker 身份 |
| `preflight.json` | 两组完整模型 smoke 是否均通过 |
| `results/blockXX-组别/rankN.json` | 每层、每次调用的原始前向／反向／合计 CUDA Event 时间 |
| `results/*/process.log` | 该窗口的初始化及测量日志 |
| `report.json`、`report.md` | 32 层平均和每层的 ms、减少 ms、配对降幅及 CI |

本地 CPU 回归命令：

```bash
python3 -B -m unittest discover -s tools/h100/phase6 -p 'test_ma0_layer.py' -v
```

回归覆盖子层边界及异常恢复、事件与同步顺序、八 rank 聚合、样本拒绝、真实生成 worker 的 GPU 导入前启动、完整 18-worker 控制流程及 smoke 失败提前退出。控制器模拟不加载 CUDA，不能替代 75 上的真实数值、编译和性能验收。

2026-10-28 本地验证：新增层级测试 **9 项通过**，随提交 TP8-DP2 相关回归 **55 项通过**，启动脚本语法及预览通过。另以已有 MA11／basic 各八份真实 smoke 记录核对新验收字段，并确认历史两组的初始参数、RNG 和 token 指纹一致；这只验证记录兼容性，不是本轮层级计时结果。
