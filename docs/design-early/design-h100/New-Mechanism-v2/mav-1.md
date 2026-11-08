# MAV1：MA11 上的固定线性层 C++ autograd 执行计划

更新：2026-10-30。本文承接 [分析.md](分析.md) 的 N1 方案，将它实现为独立候选 **MAV1**。目标是相对同期完整 MA11 降低 **5% 完整 optimizer step 时间**。

MAV1 将 QKV、FC1、attention output、FC2 的自定义线性层前后向迁入 C++ autograd。固定模型每步四个 microbatch，每类前向／反向各 256 次，合计 **1024 个执行区间**。它减少这些区间内部的 Python 解释执行和跨语言提交；外层 Module、TransformerLayer、DDP 和训练循环继续运行原路径。

**实现不等于性能验证。当前尚无 MAV1 的 H100 八卡数值或性能结果，不能宣称已经低于 450 ms，也不能宣称达到 5%。** 本机没有可用 CUDA 设备；本地验证范围与上机验收分开记录。

## 1. 启动 MA11／MAV1 对比

在已配置好原实验环境的 **TempName-New 仓库根目录**执行：

```bash
# 只预览任务、输出位置和命令；不构建、不启动训练
bash tools/h100/phase6/run_mav1.sh

# 默认两个独立 trial，自动构建并运行完整对比
bash tools/h100/phase6/run_mav1.sh --execute

# 前台执行，适合直接观察构建或首轮数值检查
bash tools/h100/phase6/run_mav1.sh --foreground --execute

# 只做一个探索性 trial
bash tools/h100/phase6/run_mav1.sh --trials 1 --execute

# 复用已完成构建、且源码和二进制身份检查通过的 MAV1 双臂 build
bash tools/h100/phase6/run_mav1.sh \
  --mav1-build /absolute/path/to/verified-mav1-build --execute
```

可用 `MAV1_PYTHON=/absolute/path/to/python` 指定原 H100 环境的解释器；Megatron 路径、CUDA 构建／运行路径、构建并发数等使用启动器的 `--help` 参数。不要为了使用本候选更换 PyTorch、NCCL 或原模型配置；native 扩展依赖原 PyTorch 2.6 的 ABI。

两臂为：

| 组别 | 策略 ID | 实际路径 |
| --- | --- | --- |
| MA11 对照 | `F4-cache-zcount-csoftmax` | 原 MA8-v2＋MA9＋MA11，以及原 MA4-F row 消费融合 |
| MAV1 候选 | `F4-cache-zcount-csoftmax-native-linear` | 完整 MA11＋固定 native 线性层计划 |

对照不安装 MAV1 adapter。双方保留同一量化 mask、到达顺序、row producer／decoder、MA8 输入缓存、MA9 零统计和 MA11 softmax。候选没有混入 MA14–20。

## 2. 实际实现边界

### 2.1. 固定计划和独立 autograd context

每个目标模块在初始化时建立自己的 native plan，绑定角色、形状规则、原 ProcessGroup 和通信选项。row plan 持有原两个 native GemmRS 槽，负责交替使用、失败封锁和保活。

实现分别位于 [mav1_native.cc](../../../../tools/h100/phase6/mav1_native.cc)、[mav1_pybind.cc](../../../../tools/h100/phase6/mav1_pybind.cc) 和 [mav1_adapter.py](../../../../tools/h100/phase6/mav1_adapter.py)。源码生成与构建入口为 [mav1_sources.py](../../../../tools/h100/phase6/mav1_sources.py)、[mav1_build.py](../../../../tools/h100/phase6/mav1_build.py)；主实验和路径检查为 [mav1_fullstep.py](../../../../tools/h100/phase6/mav1_fullstep.py)、[mav1_profile.py](../../../../tools/h100/phase6/mav1_profile.py)。

输入、权重、bias、residual 仍作为显式 Tensor 参数进入 C++ Function，保留原 autograd 边。每次调用的输入和 MA8 完整激活由该次 context 独立保存，保留版本检查；不以共享输出池覆盖尚未反向的激活。

context 通过注册的 custom class／IValue 强持有 plan；row plan 还保存两个槽的 IPC packet、选择 mask 和量化索引 Tensor 引用。仅持有 GemmRS 对象不能保证这些原本由 Python codec 管理的存储仍然有效，因此候选同时改变了 GemmRS 的 pybind holder，并显式保活底层存储。该改动只存在于候选。

```text
原 MA11：Python Function.forward/backward → 多次 Python→ATen／c10d／Flux
MAV1：   C++ Function.forward/backward    → 同一套 ATen／c10d／Flux
```

native backward 由 autograd engine 直接进入。它不回调 Python 完成线性计算，不直接写 FP32 `main_grad`；仍返回原 BF16 梯度，供原 AccumulateGrad／DDP hook 按原 microbatch 顺序累加。C++ 节点在 forward 阶段建立，并非事后替换已经生成的 Python backward 节点。

### 2.2. column 前后向

QKV／FC1 前向仍先得到独立完整输入，执行原完整矩阵乘法，并由 MA8 缓存供反向使用。反向顺序保留：

```text
读取该次 retained_input
→ dgrad GEMM
→ 原 wgrad 输入布局处理
→ 原 ProcessGroup 异步 ReduceScatter
→ 原 wgrad GEMM
→ 原 bias 梯度归约（适用时）
→ 原 Work.wait
→ 返回原布局的梯度
```

没有恢复已被 MA8 删除的 256 次反向输入 AllGather；没有将 RS 挪到 wgrad 后面，也没有把异步通信改成同步执行。前向的 `skip_bias_add` 返回接口继续匹配原模块。

### 2.3. row 前后向

row 前向直接调用已有类型化 native GemmRS 消费接口，保持原 producer、发布屏障及 decoder＋bias＋residual 两次 BF16 舍入。双槽状态进入 native plan，避免再从 C++ 回调 Python `forward_consumer()`。

反向仍先求本地 bias 梯度，原样传递 residual 梯度，再进行原 AllGather、dgrad GEMM 和 wgrad GEMM。原 TransformerLayer 的残差绑定和 BDA 所有权检查保留。

### 2.4. 没有改变的数值与通信契约

- GEMM 使用原 ATen 操作和布局，不能改为 FP32 梯度直接累加或跨 microbatch 合并。
- 复用原 ProcessGroup；不建立新的 NCCL communicator，不采用 MA17 的同流直接 NCCL 路径。
- 原 Work、stream 依赖、必要 wait、allocator 生命周期和双槽保护保留。
- MA11 的有限 mask sentinel、softmax 归约树与 BF16 舍入保留。
- 原 optimizer、梯度范数、裁剪、Adam、参数回写和清梯度保留。

同一 API 不自动保证同一数值或设备执行。候选仍必须通过完整梯度／参数指纹和独立 profile；失败时停止，不能静默回退为原路径后宣称 MAV1 已运行。

## 3. 5% 的判定口径

旧 [MA11 首轮报告](../../../../../TempName/logs/h100/phase6/ma11-v1-causal-softmax-1x-50steps-20261027-121342-060643/fullstep/report.json) 中完整步时为 **450.6624015 ms**。以这个历史量级换算，5% 需要净省 **22.5331201 ms**，到约 **428.1292814 ms**；平均每个目标执行区间需要净兑现约 **22.005 μs**。

这些是目标难度，不是 MAV1 的实测或预测。旧 trace 中 CPU scope 的未归属区间包含必要 native 工作和 profiler 成本；不能当作可直接删除的 Python 时间。主机工作缩短也可能被 GPU 计算覆盖。

每个 trial 使用 8 个完整 AB/BA 平衡块、16 个独立训练窗口；每窗口 10 步预热＋50 步正式。步时取八 rank 最大窗口墙钟／50，覆盖前后向、通信、optimizer 更新与清梯度。两个 trial 合计 1600 个正式 optimizer step、256 份正式 rank 记录。

配对统计按完整块的对数耗时比做 10,000 次 bootstrap，逐 trial 报告，保留所有慢窗口。采用新实验同期 MA11 作为分母：

| 结论 | 条件 |
| --- | --- |
| 正增量得到验证 | 每个 trial 的点估计和 95% CI 下界均大于 0，且正确性、路径检查通过 |
| 观察到 5% 点估计 | 点估计达到 5%；区间下界不足时仍不能称达到完整目标 |
| 5% 目标得到验证 | 每个 trial 的点估计和 95% CI 下界均至少为 5%，且全部验证通过 |

`completed` 表示实验流程完成，不表示性能达标。若只有 1–2% 增量、区间跨零或出现退化，应完整保留结果。

## 4. 产物与验证边界

启动器打印本批唯一输出目录，重点查看：

- `batch-state.json`：当前阶段、失败原因、最终验收状态。
- `fullstep/preflight.json`：完整模型正确性及候选安装证据。
- `fullstep/report.json`：所有正式窗口、逐 trial 配对收益及 5% 判定。
- `fullstep/profile-summary.json`：native 计划与共同 MA11 设备路径的独立验证。
- `handoff/` 和各阶段日志：冻结控制器、启动参数、构建及运行失败信息。

原 row projection 算子阶段只验证共同投影实现；它不能替代 MAV1 训练前后向的正确性或性能证据。正式计时不运行逐次 oracle 或额外 profiler；必要的 native 检查、状态管理、分配及全部实际提交成本留在计时中。

完整模型继续核对每 rank 全部 **389 个参数**的梯度和更新指纹、正式 loss／grad_norm 轨迹及最终参数指纹。独立 profile 需要同时确认 native 路径执行、原 Python Function 没有回退，以及原 MA8／MA9／MA11 和通信依赖保留。

候选 smoke 在正式模型运行前，使用独立计划和私有叶子完成 **8 类 GPU 字节对照**：四种线性角色各测试连续与非连续输出梯度，覆盖前向、输入／权重／bias／residual 梯度。它不触发模型的 DDP hook，不改变训练 RNG，也不计入模型计划的调用次数。真实候选 GPU 自测仍须上机执行。

profile 使用 C++ context 保存的 `plan_id/call_id` 配对前后向，要求 128 个计划覆盖四个 microbatch、共 1024 个 native scope。旧 Python 线性 Function 回退、缺少 kernel、错误 stream／依赖和重复归属都应失败；全步仍要求原来的 524 次 AG、260 次 RS 和 256 次发布屏障。这里的全步计数含 LM head 等非目标路径，不能当作 MAV1 删除的通信数量。

## 5. 本地验证记录

本机解释器为 `/data/run01/scyb672/conda_envs/flux-megatron-a800/bin/python`，PyTorch **2.6.0+cu124**，CUDA 设备数 **0**；可用 CUDA 12.8 编译器。因此本地检查不能代替 H100 八卡执行，尤其不能证明 GPU 数值、NCCL 时序或 5% 性能目标。

本地完成以下检查：

| 检查 | 结果与范围 |
| --- | --- |
| MAV1 单元／集成协议测试 | **43 项通过**，含 adapter、build、bench、harness、launcher、profile |
| 原 MA11 回归测试 | **53 项通过** |
| 实际源码 prepare＋verify | 通过；对照 358 个源码文件，候选 361 个；对照保持注册 MA11 字节身份 |
| 候选源码差异 | 仅新增 native 头文件、C++ 实现、pybind 注册文件，并修改 GemmRS holder；CUDA 算术源码未改 |
| 实际 C++／pybind 编译、链接、导出与导入 | 通过；使用真实 PyTorch 2.6／CUDA 头文件及 ABI；链接检查用明确的 GemmRS 方法／registry fixtures，不是完整 Flux 链接 |
| 原 GemmRS 候选 pybind 源码语法编译 | 通过，覆盖 shared holder 与原接口兼容 |
| 启动器检查 | shell 语法、默认双 trial 预览通过；预览没有创建训练批次 |

汇总见 [validation.json](../../../../outputs/h100/phase6/mav1-validation-20261030/validation.json)，测试输出见 [MAV1 tests.log](../../../../outputs/h100/phase6/mav1-validation-20261030/tests.log) 与 [MA11 regression.log](../../../../outputs/h100/phase6/mav1-validation-20261030/ma11-regression.log)。native 检查的具体边界、命令与源码哈希见 [native validation.json](../../../../outputs/h100/phase6/mav1-validation-20261030/native-check/validation.json)。

可复跑本地检查：

```bash
python -m unittest discover -s tools/h100/phase6 -p 'test_mav1_*.py' -v
python -m unittest discover -s tools/h100/phase6 -p 'test_ma11_*.py' -v

# 输出目录必须不存在；编译／链接测试不运行 GPU
python tools/h100/phase6/check_mav1_native.py \
  --cuda-home /absolute/path/to/cuda \
  --out /absolute/path/to/fresh-native-check
```

**尚未完成的设备验证：** 完整双臂 Flux 编译、真实候选 GPU 字节自测、八卡完整训练与候选真实 trace。启动器会依次执行这些阶段；只有全部通过并满足第 3 节的同期配对门槛，才可报告 MAV1 达到 5%。
