# 双节点 TP8＋DP2：MA11 与两组量化基线

更新：2026-10-28。目标为使用 gxn74、gxn75 各八张 H100，仅比较 **MA11、Flux＋基础量化（接收端融合 v2）、Megatron＋CUDA量化**。独立包第二轮 `tp8-dp2-02` 已完成双节点 IB 通信预检、编译和三组 smoke；MA11 首个正式窗口完成后，控制器因误用 smoke 数值检查条件而中止。**该校验错误已修复，完整性能比较仍待重跑。** 安装、补丁及两端命令见 [独立包使用说明](tp8-dp2-package.md)。原单节点 `run_ma11.sh` 等入口不能直接用于本实验。

## 1. 已确认的两节点环境

以下来自用户回传的环境摘要和随后执行命令的反馈，GPU 占用为探测时快照。

| 项目 | gxn74 | gxn75 |
| --- | --- | --- |
| GPU | 8×H100 80GB HBM3，132 SM／卡 | 相同 |
| 本地 GPU peer access | 全部通过 | 全部通过 |
| 探测时计算进程 | 未列出 | 未列出 |
| 驱动 | 580.178.04 | 580.65.06 |
| PyTorch／CUDA runtime／NCCL | 2.6.0+cu124／12.4／2.21.5 | 相同 |
| CUDA 编译器 | `/usr/local/cuda-12.8/bin/nvcc` | 相同；另有 12.4 |
| 仓库 | `/workspace/TempName` | 相同 |
| 训练 Python | `/opt/conda/envs/flux-h100/bin/python` | `/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python` |
| Megatron | `/workspace/Megatron-LM` | 相同 |
| Megatron commit | `3ea68ad6042cc1204386ae9364358f7c4de1bc37`，后续命令确认 | 同一提交 |
| 探测时仓库 commit | `ac6740b17f3fbda1b26e7d91fe357772312edb3e` | `ec0b9b159ef465e372a420e499a7ec90f056b1df` |
| 引导网卡与 IP | `bond0`／`10.18.133.74` | `bond0`／`10.18.133.75` |
| memlock | 起初 65536 bytes；`ulimit -l unlimited` 已成功 | 相同 |

74 的原探测摘要未打印 Megatron commit；后续直接查询已确认与 75 一致，不能据空摘要认定缺少 Megatron。两个仓库提交仍不同，正式训练前需同步相同实现和依赖。驱动版本差异予以记录，当前环境信息尚不能代替真实 collective 验证。

`ulimit` 设置由当前 shell 的子进程继承，不会自动应用到其他 shell 或新容器。通信启动器在 `--execute` 时为本次进程设置并核对 unlimited memlock；修改仅作用于该进程及子进程。NCCL 对容器共享内存和锁页内存限制有专门说明，见 [NCCL 2.21.5 Docker 排障](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/troubleshooting.html#docker)。

## 2. 本次预检的网络选择

74 有八个活动 IB 端口，75 的活动端口为下列四个；其余 DOWN 端口不用于本次预检。以下四组均在摘要中报告 ACTIVE、InfiniBand、400 Gb/sec 链路速率。

| HCA／端口，两节点相同名称 | 74 的对应网卡／IP | 75 的对应网卡／IP |
| --- | --- | --- |
| `mlx5_1:1` | `ibp78s0`／`192.168.20.74` | `ibp24s0`／`192.168.20.75` |
| `mlx5_4:1` | `ibp96s0`／`192.168.30.74` | `ibp92s0`／`192.168.30.75` |
| `mlx5_6:1` | `ibp201s0`／`192.168.50.74` | `ibp140s0`／`192.168.50.75` |
| `mlx5_8:1` | `ibp221s0`／`192.168.70.74` | `ibp164s0`／`192.168.70.75` |

预检使用 `10.18.133.74:29574` 进行 torchrun rendezvous，Gloo／NCCL 引导接口为两边同名的 `bond0`。GPU 跨节点传输显式指定 `NCCL_NET=IB`、`NCCL_IB_DISABLE=0`、`NCCL_IB_HCA==mlx5_1:1,mlx5_4:1,mlx5_6:1,mlx5_8:1`；前导 `=` 表示精确匹配 HCA 名称。`NCCL_NVLS_ENABLE=0` 沿用现有实验设置。参数含义见 [NCCL 2.21.5 环境变量](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/env.html)。

端口可见与 ACTIVE 仅是本地状态。400 Gb/sec 是报告的链路速率，不代表实测带宽，也不证明四条链路同时饱和。预检除检查通信数值外，还核对每个进程日志中的实际 `via NET/IB` channel；仅出现 `Using network IB` 不算传输已验证。GDRDMA 标记单独记录，不把 IB 传输等同于所有路径均已使用 GPUDirect RDMA。

## 3. 两节点执行通信预检

将 [check_tp8_dp2_link.sh](../../../../tools/h100/phase6/check_tp8_dp2_link.sh) 与同目录的 [check_tp8_dp2_link.py](../../../../tools/h100/phase6/check_tp8_dp2_link.py) 同步到两台容器。此预检不依赖 Megatron 或已有 Flux build，但两边 worker 文件必须完全相同。启动时两台八张 GPU 都应空闲；脚本发现计算进程会停止，不终止其他任务。

在 **74 容器**的仓库根目录执行：

```bash
bash tools/h100/phase6/check_tp8_dp2_link.sh \
  --node-rank 0 \
  --python /opt/conda/envs/flux-h100/bin/python \
  --execute
```

然后在三分钟内于 **75 容器**的仓库根目录执行：

```bash
bash tools/h100/phase6/check_tp8_dp2_link.sh \
  --node-rank 1 \
  --python /workspace/conda-envs/flux-h100-ablation3-20261017/bin/python \
  --execute
```

去掉 `--execute` 只打印命令与环境，不创建目录、不查询 GPU、不调整资源限制。测试为前台执行，两边终端需保留到结束。每次自动新建各自节点的日志目录，无需共享文件系统；重跑使用新目录。若修改 `--run-id`、`--master-addr`、`--master-port`、`--hca`，两边必须一致。默认等待第二节点 180 秒，collective 超时 90 秒，单节点监督进程总时限 420 秒。

预检固定全局 16 个 rank：74 为 0–7，75 为 8–15。

- TP 组为 `[0,1,…,7]` 与 `[8,9,…,15]`，检查节点内 AllGather、ReduceScatter。
- DP 组为 `[0,8]`、`[1,9]`、…、`[7,15]`，检查跨节点小消息和 16 MiB 消息 AllReduce，后者重复五次并逐元素核对。
- 全部 rank 通过 Gloo 交换 worker 哈希、run ID、软件、GPU UUID 与 TP／DP 映射，拒绝两台误启动成独立单机组或不同脚本。
- 每个节点核对自己的八份 NCCL 进程日志，保留实际 IB／Socket／GDRDMA channel 证据。

脚本打印输出目录，主要文件为：

| 文件 | 用途 |
| --- | --- |
| `state.json` | `status`、`collectives_passed`、`ib_transport_validated` |
| `process.log` | torchrun 与 worker 输出；失败时首先查看 |
| `local-preflight.json`、`launch.json` | 本地 IB／memlock、启动参数与进程环境 |
| `all-ranks.json` | 交换并核验过的全部 16 份结果，两节点各有一份 |
| `rank*.json`、`nccl.*.log` | 本节点八进程的结果与原始 NCCL 日志 |
| `transport.json` | 每进程实际传输路径的日志证据 |

**两台 `state.json` 都为 `completed`，且 collective 和 IB 传输验证均为 true，才算完成本次双节点通信预检。** 这只说明指定通信路径及测试消息正确，不是三组训练已运行，也不是训练性能结论。

本地已验证 CPU 元数据拒绝、启动参数、memlock 分支、预览无副作用、channel 判读和成功／失败状态；未在本机运行 H100／IB 通信。测试入口为 [test_check_tp8_dp2_link.py](../../../../tools/h100/phase6/test_check_tp8_dp2_link.py)。

## 4. 通信通过后的三组训练适配范围

保持 global batch=4、micro batch=1、TP8、DP2、PP=CP=1，因此每个 DP 副本每 optimizer step 执行 `4 / (1 × 2) = 2` 个 microbatch；两副本处理不同样本分片，同一 TP 组内输入一致，每步全局 token 数仍为 8192。

| 对比组 | 需要保留的实现 |
| --- | --- |
| MA11 | CR3 选择性量化与消费链＋MA8-v2 激活复用＋MA9 保序零统计＋MA11 注意力中间态消除 |
| Flux＋基础量化 | `a-baseline.md` 页首接收端融合 v2：GemmOnly＋原 CUDA 编码＋TP 组内 All-to-All＋融合解码归约 |
| Megatron＋CUDA量化 | 原 Megatron GEMM＋独立 CUDA 编解码＋TP 组内 All-to-All |

独立包的正式入口同步两节点的统一窗口顺序，使用原生 DP 梯度同步，并核对同 TP 分片的跨副本梯度与参数更新；耗时为全局 16 rank 的最大窗口墙钟／正式步数。TP 内 IPC 与 native 参数仍为本节点八卡，每个 TP 算子的 `nnodes` 保持 1；分别记录全局 rank、TP rank、DP rank 与 local rank。调用次数按两 microbatch 验证，统计参数筛选按 TP rank 判断，因此 global rank 8 与 rank 0 均为完整统计参数的 TP rank 0。

MA11 使用选择性量化，两组历史量化基线使用全远端量化，不能要求三个方案的 loss／参数轨迹互相逐位一致。正确性应分别针对原算术／codec 参考、DP 副本同步和对应路径验证；性能在这次双节点配置内同场比较，完整保留慢窗口。既有单节点 450–630 ms 等数据不作为本次实测窗口。

## 5. 无需拉取大仓库的独立包

初版归档 `tp8-dp2-bench.tar.gz` 为 **11,518,538 bytes（约 11 MiB）**，含 3215 个受哈希保护的文件及包清单；SHA256 为 `805567d900fc63289ec102e3930122050bed69f908e885d37b727e966799b03f`。用户已将工作目录移到 `Load/tp8-dp2-bench`，本次修复同步更新该目录，并另提供 `tp8-dp2-audit-fix.tar.gz`；以上初版归档哈希不代表修复后的包。**74 和 75 都更新到同一修订，目标路径均为 `/workspace/tp8-dp2-bench`。** 安装与两端命令见 [使用说明](tp8-dp2-package.md)。

导出入口为 `tools/h100/phase6/export_tp8_dp2.py`，只读取所需源码、已固定的选择计划和依赖提交，不遍历全仓库日志／输出目录。正式入口 `run_training.sh` 自动执行通信预检、各节点源码编译/缓存校验、三组 smoke、三组独立 profile，全部通过后才运行 36 个正式计时窗口。各节点预先编译自己的 Megatron dataset helper，不依赖共享文件系统；成功 native build 按构建身份缓存。

本地验证：15 项独立包 CPU 回归与 10 项通信预检 CPU 回归通过；归档在另一全新目录解压后，3216 条 SHA256 校验通过，74／75 的通信与训练共四个入口预览通过且未创建运行目录。冻结 worker 与控制器的本地导入依赖检查通过。**上述结果不包含 H100 编译、GPU 通信或训练实测；这些仍由两节点执行确认。**

后续真实日志：第二轮三组各 16 rank 的 smoke 已通过，控制器误将计时阶段空的 `forward_checks` 视为缺失证据。本次修复要求 smoke 完整检查、timing/profile 无参考检查，并用原包清单验证 native 输入未变后复用编译缓存。新增回归后共 24 项 CPU 测试通过；真实数据回放覆盖 48 份 smoke 和 75 上 8 份首窗口计时记录。后者约 648.643–648.647 ms/step，仅有 MA11 一个窗口的半数 rank，不作为三组对比或性能收益结论。

随后完整逻辑复核又修正了 DP profile 对 `allreduce_coalesced` 的漏识别，并将 profile 全部前置。35 项 CPU 回归通过，覆盖两端并发控制、全部窗口、两轮报告、异常通知及负收益继续执行；其中 18 项随独立包提供。历史 GPU trace 用于确认事件格式和基线路径，不作为本次 DP2 的新增性能证据。当前累计补丁为 `tp8-dp2-preflight-fix.tar.gz`，同步文件与启动步骤见 [独立包说明](tp8-dp2-package.md)。
