# TP8＋DP2 独立测试包

本目录用于 gxn74＋gxn75 的 16 张 H100：每节点 TP8、跨节点 DP2，只比较 **MA11、Flux＋基础量化 v2、Megatron＋CUDA量化**。两节点都放在 **`/workspace/tp8-dp2-bench`**，不覆盖 `/workspace/TempName`，无需拉取其 Git 历史或实验数据。

包内包含通信检测、正式训练入口、三组冻结的 worker、MA11 与 Flux GemmOnly 编译源码、两个 CUDA codec、量化选择计划，以及固定版本的 Megatron／CUTLASS／NCCL 源码。不包含旧日志、profile、模型 checkpoint、语料或 Conda 环境。使用现有 Python 环境和 CUDA 12.8，在各节点首次运行时编译；构建不联网下载依赖。

Megatron 固定为 `3ea68ad6042cc1204386ae9364358f7c4de1bc37`。包中 `package.json` 记录每个文件的 SHA256，两节点必须使用同一包。无需依赖两边旧仓库的 HEAD。`provenance.json` 中原工作区路径仅供追溯，不作为运行依赖。

## 放置与校验

把 `tp8-dp2-bench.tar.gz` 和 `tp8-dp2-bench.tar.gz.sha256` 传到 **两个容器的 `/workspace/`**，分别执行：

```bash
cd /workspace
sha256sum -c tp8-dp2-bench.tar.gz.sha256
tar -xzf tp8-dp2-bench.tar.gz
cd /workspace/tp8-dp2-bench
sha256sum -c SHA256SUMS > package-check.log
```

目标目录应为新目录；若已经运行过旧包，先将旧目录改名保留，再解压本次包。只移动本包即可，74 不需要更新旧的 TempName 仓库。75 也使用同一个包执行，避免代码版本不一致。

## 当前修订：完整流程复核与 profile 前置

在修复计时阶段 `forward_checks` 误报后，又发现 profile 的 DP 归约识别过窄：本包 Megatron 的 `param_and_grad_buffer.py` 使用 `_coalescing_manager`，PyTorch 2.6 会调用 `allreduce_coalesced`，对应 trace 的 `Collective name` 也是 `allreduce_coalesced`。旧解析器只接受 `allreduce`，会在后置 profile 中误报 `No DP2 gradient AllReduce metadata in profile`。这个命名由 [PyTorch 2.6 NCCL 实现](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L3636) 和既有单节点 GPU trace 交叉确认。

当前解析器接受普通及合并 AllReduce，并继续要求：DP group size=2、正确的 DP group description、FP32 大消息、输入输出元素数一致，以及该 scope 内 CUDA launch 与实际 NCCL AllReduce kernel 的关联。仅有元数据、无实际 kernel、错误归约或错误归属都不能通过。

**当前顺序为：通信预检 → 编译/缓存校验 → 三组 smoke → 三组独立 profile → 全部正式计时 → 汇总。** 进入第一个计时窗口前，强制检查三组 smoke 和每组 16 rank 的 profile 都已通过。每个窗口仍是新进程、10 步预热＋50 步计时；profile 不计入性能样本。默认仍为 42 个模型进程窗口，其中正式计时 36 个，负收益不会中断后续对照。

本地通过 35 项 CPU 测试，包含两控制器并发运行完整流程的模拟、默认全部 36 个计时窗口、两轮独立报告、失败通知、profile 解析失败前置阻断及负收益继续执行。模拟中的 worker、编译器和 GPU profile 是明确的 CPU 替身，**不能当作双节点 GPU 实测**。另回放了三组 48 份真实 smoke、75 上 8 份已有计时记录，并以历史单节点 GPU trace 验证两组基线的投影事件解析；实际 DP2 profile 仍需两台执行新前置阶段确认。

使用仓库传送时，请确认克隆的是本轮修订。假设仓库位于 `/workspace/Load`，旧任务已退出，在两台容器分别同步以下 **10 个文件**；这会保留已有 `build/` 和 `runs/`：

```bash
cd /workspace/Load
for file in README.md SHA256SUMS package.json \
  app/build-package.json app/tp8_dp2_protocol.py app/tp8_dp2_train.py \
  app/tp8_dp2_profile.py app/test_tp8_dp2_validation.py \
  app/test_tp8_dp2_profile.py app/test_tp8_dp2_campaign.py
do
  cp "tp8-dp2-bench/$file" "/workspace/tp8-dp2-bench/$file"
done
cd /workspace/tp8-dp2-bench
sha256sum -c SHA256SUMS > package-check.log
python3 -B -m unittest discover -s app -p 'test_tp8_dp2_*.py' -v
```

包内有 18 项便携 CPU 测试，剩余 17 项位于 TempName 的源码回归中。也可用累计补丁 `tp8-dp2-preflight-fix.tar.gz` 和对应 `.sha256` 文件覆盖到 `/workspace/`，其内容与上述仓库文件相同。此轮仍复用经验证的 `node0-e9f5dbfbabdc`／`node1-e9f5dbfbabdc` 原生库缓存。`tp8-dp2-03` 尚未启动时可继续使用该编号，先同步本轮全部文件再按下方两端命令启动。

## 已有部署：2026-10-28 计时校验修复

`tp8-dp2-02` 已通过 IB 通信预检、两端编译及三组 smoke，随后在 `compare1/block00-ma11` 完成 50 步后，控制器报 `Incomplete original projection numeric checks`。原因是控制器对所有模式都要求 64 份 `forward_checks`，而三个冻结 adapter 只在 `E2E_MODE=smoke` 时执行这些参考计算。计时与 profile 模式正确输出空字典，旧校验器却将其误判为失败。

本次修复按模式校验：smoke 必须覆盖全部 64 个目标模块且全部通过；timing／profile 必须为空，防止参考计算污染计时或执行路径。训练算子、数据、native 构建配方和计时口径未改变。

下列 `tp8-dp2-audit-fix.tar.gz` 是前一次仅修复计时检查的补丁，**尚不包含上方 profile 前置修复**；当前部署应采用上方仓库同步步骤或新的 `tp8-dp2-preflight-fix.tar.gz`。保留旧补丁操作记录如下：

```bash
cd /workspace
sha256sum -c tp8-dp2-audit-fix.tar.gz.sha256
tar -xzf tp8-dp2-audit-fix.tar.gz -C /workspace
cd /workspace/tp8-dp2-bench
sha256sum -c SHA256SUMS > package-check.log
python3 -B app/test_tp8_dp2_validation.py
```

补丁只包含静态修复文件，保留 `build/`、`runs/`。新包使用新的完整包 ID 校验两端实现，同时通过 `app/build-package.json` 保存的原始清单核对 **native／依赖源码、依赖符号链接及构建脚本**完全相同，然后复用原来的 `build/node0-e9f5dbfbabdc`、`build/node1-e9f5dbfbabdc`。原有 Python／Torch／nvcc 身份及全部已记录构建产物的 SHA256 仍会核验；不接受不完整、损坏或构建输入不同的缓存。

更新后用新编号 **`tp8-dp2-03`** 从三组 smoke 重新开始，不能混入上一轮未完成对比的样本。74：

```bash
cd /workspace/tp8-dp2-bench
nohup bash run_training.sh --execute --node-rank 0 \
  --python /opt/conda/envs/flux-h100/bin/python \
  --run-id tp8-dp2-03 > tp8-dp2-03-node0.log 2>&1 &
```

75：

```bash
cd /workspace/tp8-dp2-bench
nohup bash run_training.sh --execute --node-rank 1 \
  --python /workspace/conda-envs/flux-h100-ablation3-20261017/bin/python \
  --run-id tp8-dp2-03 > tp8-dp2-03-node1.log 2>&1 &
```

缓存存在时日志会出现 `VERIFY cached native libraries`。本地已通过 24 项 CPU 回归，回放三组共 48 份 smoke rank 记录，并检查 75 拉取的 8 份 MA11 首窗口计时记录与初始化身份；当前缺少该计时窗口的 74 rank 文件，尚未取得三组完整性能比较或 profile 结果。

## 容器 IB 用户态依赖

若出现 `Failed to open libibverbs.so[.1]`／`Error: network IB not found`，设备文件可见并不足以证明用户态库可用。Ubuntu 22.04 容器内需具备 `libibverbs1`、`ibverbs-providers`、`ibverbs-utils`，可在缺少依赖的容器中补装：

```bash
apt-get update
apt-get install -y --no-install-recommends libibverbs1 ibverbs-providers ibverbs-utils
ldconfig
```

使用各节点指定的训练 Python 验证 `ctypes.CDLL('libibverbs.so.1')` 能加载，并通过 `ibv_devinfo -d mlx5_1 -i 1` 等检查四个选定端口；通过后仍需执行本包的真实 collective 预检。74 第一轮因缺少 verbs 库失败，第二轮已通过通信预检。原生 IB 后端不要求额外的 `libnccl-net.so` 插件。

## 单独检测通信

以下两条命令分别在对应容器运行，启动间隔不超过 3 分钟。默认控制网络 `bond0`、地址 `10.18.133.74`，数据网络使用两节点都 ACTIVE 的 `mlx5_1:1,mlx5_4:1,mlx5_6:1,mlx5_8:1`，关闭 NVLS。会在本进程提高 memlock，并检查 GPU 空闲；不终止其他用户的任务。

74：

```bash
cd /workspace/tp8-dp2-bench
bash run_link.sh --execute --node-rank 0 \
  --python /opt/conda/envs/flux-h100/bin/python \
  --run-id tp8-dp2-link-check
```

75：

```bash
cd /workspace/tp8-dp2-bench
bash run_link.sh --execute --node-rank 1 \
  --python /workspace/conda-envs/flux-h100-ablation3-20261017/bin/python \
  --run-id tp8-dp2-link-check
```

检查输出目录的 `state.json`：两端都必须为 `completed`。实际执行 TP AllGather／ReduceScatter、8 组跨节点 DP AllReduce，并从 NCCL 数据通道日志验证 IB，拒绝 Socket 回退。该阶段不训练。

## 正式三组训练

下面入口会自动再次执行通信检测，再编译或验证缓存、执行三组 smoke 和独立 profile，全部通过后才运行完整计时。无需先手动执行上面的通信命令。两节点必须使用**同一个 `--run-id`**，且输出目录必须不存在；重试请两边一起换新 ID。

74：

```bash
cd /workspace/tp8-dp2-bench
nohup bash run_training.sh --execute --node-rank 0 \
  --python /opt/conda/envs/flux-h100/bin/python \
  --run-id tp8-dp2-01 > tp8-dp2-01-node0.log 2>&1 &
```

75：

```bash
cd /workspace/tp8-dp2-bench
nohup bash run_training.sh --execute --node-rank 1 \
  --python /workspace/conda-envs/flux-h100-ablation3-20261017/bin/python \
  --run-id tp8-dp2-01 > tp8-dp2-01-node1.log 2>&1 &
```

两条命令应尽快依次启动。默认协调端口 **29580**、torchrun／自动通信检测端口 **29581**；单独 `run_link.sh` 使用 **29574**。不需要 SSH 免密或共享文件系统。控制器等待两端同阶段就绪，任何构建、正确性或执行路径错误均记录失败并通知另一端；**性能无收益不会中止剩余对照**。不带 `--execute` 仅校验文件并打印计划，不启动 GPU 或网络通信。

查看进度：

```bash
tail -f tp8-dp2-01-node0.log  # 74；75 改为 node1
```

首次编译进度还可查看 `build/node0-*/compile.log`（75 为 `node1-*`）。成功构建按包 ID 缓存；失败构建不会当作可用缓存。若需重新编译，使用 `--build-root /workspace/tp8-dp2-build-retry-node0`（75 对应 node1）指定新目录。

## 口径与结果

- 固定 32 层、hidden 4096、FFN 16384、32 heads、sequence 2048、BF16、MBS=1、**GBS=4**。每节点 **2 个 micro-batch**，全局每 optimizer step 8192 tokens。两份 DP 数据不同，同一 TP 组数据相同。
- rank 0–7 是 74 的 TP 组，rank 8–15 是 75 的 TP 组；DP 组为 `[0,8]` 至 `[7,15]`。量化只用于原有 TP 通信，跨节点 DP 梯度同步保留 Megatron 原实现。
- Flux 基础组保留 **receiver fusion v2**：Flux GemmOnly＋独立 CUDA 编码＋NCCL All-to-All＋接收端解码/BF16 求和融合。Megatron CUDA 组保留原生 GEMM＋独立 CUDA 编解码；MA11 保留 CR3＋MA8-v2＋MA9＋MA11。
- 每组 smoke 为 2 步预热＋3 步检查：核对全部可训练参数的 DP 梯度与更新指纹、量化数值，以及 MA11 原生算术自检。三个对照使用相同初始化和全局输入；量化策略不同，不要求三组 loss／梯度轨迹逐位相同。
- 默认 **12 个配对块×3 组＝36 个计时窗口**，每窗口新进程、10 步预热＋50 步完整 optimizer step。六种组内顺序各出现两次，共 1800 个正式 optimizer step、576 份 rank 计时记录。所有窗口保留；跨 16 rank 的最慢耗时定义窗口 ms/step。
- 每组在正式计时前另开独立 profile，检查实际 launch 归属、MA8/MA9/MA11 路径及真实 DP2 AllReduce（含 `allreduce_coalesced`）。profile 不计入性能样本。MA8／MA11 调用数按 2 个本地 micro-batch 校验；DP 新增的梯度缩放不套用旧单节点全局 `aten::mul` 总数。
- 报告给出三组算术平均／中位 ms/step，以及 MA11 相对两基线的逐块 log 耗时比 bootstrap 95% CI。默认一轮属探索性结果；可两端都指定 `--trials 2`，两轮分别报告，不混用历史单节点耗时。

两端结果各位于 `runs/tp8-dp2-01/node0/`、`runs/tp8-dp2-01/node1/`。完成需 **两端 `state.json.status=completed`**，且有 `report.json`。`timing-progress.json` 只是未完成的计时进度，不能代替最终通过标记。`report.json` 汇总全部 16 rank；原始日志、trace 留在各自节点，分析时请拉取两端结果目录。

本地生成与 CPU 验证不能代替双节点实测。第二轮日志已证明通信预检及三组 smoke 通过，但尚无完整性能结论；修复后的完整流程仍需在 74／75 实跑。此测试沿用固定合成全局 batch，用于端到端性能比较，不代表真实语料收敛验证。
