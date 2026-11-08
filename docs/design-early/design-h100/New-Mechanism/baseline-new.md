# 新服务器五组 baseline：MA11、MA89、CR3 与两组量化基线

更新：2026-10-29。此入口在**同一台单节点八卡 H100 服务器、同一批次**中比较五组完整 optimizer step，重新生成本机 baseline。当前为已实现的测试协议与启动说明，尚未产生这套五组同场实验的 GPU 结果。

```bash
cd /workspace/TempName-New
source env/activate.sh
bash tools/h100/phase6/run_baseline_new.sh --execute
```

入口为 [run_baseline_new.sh](../../../../tools/h100/phase6/run_baseline_new.sh)，默认后台运行并打印批次目录。所有五组自动按配对块轮换执行，不需要依次手工启动旧的 `run_ma11.sh`、基础量化或历史 baseline 脚本。

## 1. 五组实现与对照范围

| 展示名称 | 本次组 ID | 使用的原实现 | 量化与机制 |
| --- | --- | --- | --- |
| **MA11** | `ma11` | `F4-cache-zcount-csoftmax` | CR3＋MA8 前后向激活复用＋MA9 保序零统计＋MA11 注意力中间态消除 |
| **MA89** | `ma89` | `F4-cache-zcount` | CR3＋MA8＋MA9，保留原 attention mask／softmax 路径 |
| **CR3** | `cr3` | 原 F4 | 交替顺序、到达优先、选择性量化、接收端预取与 bias／residual 消费融合；不安装 MA8／MA9／MA11 |
| **Flux＋基础量化** | `flux-basic` | 接收端融合 v2，`basic` | 原生 Hopper Flux GemmOnly＋独立 TACO CUDA 编码＋NCCL All-to-All＋融合接收端解码／BF16 归约 |
| **Megatron＋CUDA量化** | `megatron-cuda` | `native_cuda_taco_attn_mlp` | 原 Megatron GEMM＋独立 TACO CUDA 编解码＋NCCL All-to-All＋原 BF16 求和 |

五组均覆盖 Attention 输出投影和 MLP FC2。前三组使用同一份历史冻结到达置换及 1/64 预算上限的选择性量化 mask；后两组使用全远端量化，所有组的本地贡献均保留 BF16。Flux＋基础量化是 [a-baseline.md](a-baseline.md) 页首的接收端融合 v2；Megatron＋CUDA量化保留原 `RowParallelLinear.forward`，不加载 Flux 动态库。

本次不重新标定到达计划，不叠加 MA14，也不更换原数值 kernel。MA11、MA89、CR3 分别编译各自的原生库：MA89 不含 MA11 softmax helper，CR3 不含 MA9／MA11 helper；三个组共用原 CR3 row adapter，只有对应候选安装自己的机制。

MA11／MA89 检验注意力中间态消除的增量，MA89／CR3 比较激活复用与零统计组合，MA11 与两组量化基线比较完整方案。不同量化策略之间不要求逐位相同的训练轨迹。

## 2. 固定模型、环境与计时

| 项目 | 固定设置 |
| --- | --- |
| 硬件／拓扑 | 单节点 8×H100 80GB、每卡 132 SM；TP8、DP1、PP1、CP1 |
| 模型 | GPT6.7B 配置：32 层、hidden=4096、FFN=16384、32 heads、sequence=2048 |
| 精度／数据 | BF16，固定合成 token；相同初始化、RNG 与数据指纹 |
| batch | microbatch=1、global batch=4；每步四个 microbatch、8192 token |
| Python／Torch | `/workspace/conda-envs/tempname-h100/bin/python`；PyTorch 2.6.0+cu124，ABI=0 |
| 工具链 | CUDA 12.8 编译原生库，CUDA 12.4 运行，GCC/G++ 11；固定 Apex 与 Megatron 源码 |
| 运行设置 | `CUDA_DEVICE_MAX_CONNECTIONS=1`、`NCCL_NVLS_ENABLE=0`；冻结实际 NCCL 环境及配置文件身份 |
| 正式窗口 | 新建 8-rank 进程，10 步预热＋50 步完整 optimizer step |
| 计时边界 | 原前向、反向、TP 通信、梯度处理、optimizer 更新、调度器和清梯度；每窗取八 rank 最大墙钟／50 |

采用原 `full-step-one-clear-v2` 计时循环。训练核心函数和计时循环有 AST 身份检查；参数梯度／更新指纹只在 smoke 阶段计算，profile 与正式计时分开。这里测完整训练步，不使用 MA0 的独立子层 CUDA Event 计时。

## 3. 同场轮换与统计口径

默认 **1 个独立 trial、10 个五组完整配对块、50 个正式窗口、400 份正式 rank 记录、2500 个正式 optimizer step**。每组十个窗口、500 个正式步骤；预热另计。

每个配对块包含五组各一次。顺序采用五组的 Williams 平衡设计：五个旋转序列及其逆序构成十块，再固定随机种子打乱块顺序。十块内每组在五个运行位置各出现两次，每一对组别的两种先后关系各出现五次。`--blocks` 仅接受 10 的正整数倍；`--trials 2` 独立报告两轮，不将其拼成一次更大的 trial。

报告同时提供：

- 五组全部正式窗口的算术平均步时、中位数、最慢窗口及吞吐；主表的 MA11 相对降幅统一为 `100 × (1 − MA11平均步时 / 对照平均步时)`。
- 五组之间全部十对同期配对结果：以块内对数耗时比计算点估计，按完整块进行 10,000 次 bootstrap，给出双侧 95% CI。
- allocated／reserved 峰值：每窗口先取八 rank 最大值，再跨窗口求平均。
- 全部原始窗口及各 rank 记录，不挑最快窗口、不删除慢样本，也不混入历史机器的测量值。

平均耗时比和配对对数比是不同统计量，分别保留。无论性能提升、持平或退化，均完成全部窗口及最终 profile。`completed/done` 表示正确性、完整执行和路径检查通过，不要求某个固定百分比门槛。

## 4. 执行流程与正确性

```text
八卡空闲与环境检查
  → 准备原始源码并编译四套 Flux 库、两套 CUDA codec 与 Megatron helper
  → 五组完整模型 smoke
  → 同场五组轮换计时
  → 五组各一次独立完整 step profile
  → 核对原始记录、生成 report.json／report.md
```

四套 Flux 库分别服务 MA11、MA89、CR3 和 Flux GemmOnly；Megatron＋CUDA量化只使用独立 codec。NCCL 开发库在本批编译一次并共享，运行时使用原 PyTorch NCCL。Megatron 按固定提交归档到本批 build 内并预编译 dataset helper，不依赖旧服务器的编译产物。

所有组先完成各自原有的投影数值检查；两个独立 codec 保留零值、小值、离群值、完整 shape 等测试，基础量化保留输出独立所有权检查。统一 smoke 在两步预热后记录三步、每 rank 全部 389 个可训练参数的梯度与更新指纹。

MA11、MA89、CR3 的 smoke 梯度／更新指纹必须一致；正式计时中，每个完整配对块还核对三组 loss、grad norm 与最终参数指纹一致。五组共同核对硬件、软件、初始参数、RNG、token 与通信设置。不同量化方案保留各自数值门槛，不强行要求后两组与前三组逐位一致。

独立 profile 使用原有的 launch correlation 检查器：

- 三个 CR3 系列组：验证 GEMM→发布屏障→融合 decoder→LayerNorm，以及原 row backward。
- CR3：保留原列并行反向输入 AllGather、逐参数零统计和原 softmax；MA89：验证 MA8 缓存与 MA9 统计，并保留原 softmax；MA11：再验证融合 softmax 及原位 view 回放消除。
- 两组量化基线：验证原编码、NCCL All-to-All、对应解码／求和路径；基础量化使用 Hopper GemmOnly，Megatron 基线保持原 GEMM。

每个 worker 启动前检查 GPU 计算进程；发现占用即停止，不终止其他任务。此检查不是 GPU 预约机制，运行期间仍需独占八卡。数值、源码／库身份、工作量、rank 完整性或实际路径检查失败时停止并保留证据；失败目录不覆盖、不续接正式窗口。

## 5. 启动与日志

已通过 `env/setup_h100.sh` 的新容器直接使用页首命令；启动器自动读取 `outputs/h100/container-recovery/env.sh`。默认 Megatron 路径为同级 `/workspace/Megatron-LM`。

```bash
# 只预览：不创建目录、不查询 GPU、不编译。
bash tools/h100/phase6/run_baseline_new.sh

# 一轮，默认后台运行。
bash tools/h100/phase6/run_baseline_new.sh --execute

# 两轮独立 trial：100 个正式窗口、5000 个正式 optimizer step。
bash tools/h100/phase6/run_baseline_new.sh --trials 2 --execute

# 前台运行，便于与其他测试串行安排。
bash tools/h100/phase6/run_baseline_new.sh --foreground --execute
```

首次运行会编译所需原生库。只有这套五组入口产生且身份校验通过的 build 才可用 `--baseline-build /absolute/path/to/build` 复用；不能将 MA0 包或单独 MA11 的 build 当作五组 build。可通过 `--megatron` 指定源码位置，`BASELINE_NEW_PYTHON` 指定 Python，`BASELINE_NEW_CUDA_HOME`／`BASELINE_NEW_BUILD_CUDA` 指定运行／编译 CUDA。

新批次位于 `logs/h100/phase6/baseline-new-1x-50steps-时间戳/`：

| 文件 | 用途 |
| --- | --- |
| `batch-state.json` | 当前阶段、错误或最终 `completed/done` |
| `handoff/container-supervisor.log` | 后台监督器输出 |
| `handoff/source/`、`handoff/controllers.json` | 本次使用的控制器／模板及哈希 |
| `build/selective/compile.log`、`build/flux/compile.log` | 三套选择性量化库和 GemmOnly 的编译日志 |
| `plan.json`、`workers.json` | 运行顺序、模型、源码配置、库身份 |
| `preflight.json` | 五组 smoke 是否全部通过 |
| `results/compare1/blockXX-组ID/process.log` | 每个正式窗口的训练日志 |
| `results/*/rankN.json` | 预检／profile 的 rank 记录；正式记录在对应 compare/block 目录 |
| `profile-summary.json` | 五组共 40 份真实 trace 的路径检查与文件哈希 |
| `samples.json`、`report.json`、`report.md` | 全部窗口、统计数据与可读汇总 |

跟踪打印出的批次目录，例如：

```bash
tail -f logs/h100/phase6/baseline-new-实际时间戳/handoff/container-supervisor.log
```

在源码版本和冻结输入未变的情况下，可对拉回的批次做 CPU 只读报告复核；无需服务器上的 build 动态库：

```bash
python3 -B tools/h100/phase6/baseline_new.py \
  --report --out logs/h100/phase6/baseline-new-实际批次目录
```

该命令从 smoke／正式／profile rank 记录重算计时和数值身份，验证保存的 profile 检查结果及原 trace 哈希，不重新执行 GPU，也不重新解析全部 trace。正式结果只有在五组 profile 全部通过后才生成。

## 6. 本地验证与上机边界

CPU 回归入口：

```bash
python3 -B -m unittest discover -s tools/h100/phase6 -p 'test_baseline_new.py' -v
```

检查覆盖五组位置／先后关系平衡、真实源码准备和变更拒绝、原训练与计时 AST、worker 启动保护、五组各阶段记录校验、负收益仍完成全部窗口、smoke 失败提前停止和 profile 失败禁止完成报告。模拟 worker 使用明确的 CPU fixture，不代表 H100 性能结果。

2026-10-29 本地检查：**14 项 CPU 回归全部通过**，单／双 trial 无副作用预览、Python／Bash 语法及文档链接检查通过；启动前冻结源码失败也会写入明确的 `failed` 状态。

另以五组已有的真实 rank0 trace 检查新入口对原 profile 解析器的接入；这是历史数据兼容性复核，不是新五组同场实验。本机未执行四套原生库的 CUDA 编译或八卡训练，实际新 baseline 以服务器运行后完整生成的本批报告为准。历史 [baseline.md](baseline.md) 及其结果保持不变。
