# MA9：保持原 CR3，对优化器零梯度统计做精确多张量归约

**更新：2026-10-27，MA9 首轮完整八卡实测及离线复核通过，本批次达到 2% 验收门槛。** 同期 CR3／MA9 平均为 **503.310／489.878 ms/step**，平均减少 **13.431 ms/step**；完整块配对收益 **+2.669%［95% CI：+2.306%, +3.062%］**，8 个配对块全部更快。完整梯度／参数更新、零统计值、原 CR3 对照身份和实际 GPU 路径均通过，四份算子／训练报告及 profile 汇总由原始记录重算一致。**仍只有一个独立 trial，尚未独立复现。** 详见[第 9 节](#9-ma9-首轮八卡实测与离线复核2026-10-27)。

首次上机曾因动态库接口未导出退出，修复后新批次已完整完成；失败记录保留于[第 8 节](#8-动态库接口导出修复2026-10-27)。本轮对照为原始 CR3（脚本名称仍为 `F4`），候选为 `F4-zcount`。只替换候选优化器的零梯度统计，不改变模型前后向、梯度范数、裁剪、Adam 更新及统计结果。

在 H100 容器的 `TempName` 仓库根目录执行：

```bash
bash tools/h100/phase6/run_ma9.sh --execute

# 两个独立 trial，验证复现
bash tools/h100/phase6/run_ma9.sh --trials 2 --execute
```

## 1. 先解决 MA8 对照不再是原 CR3 的问题

用户指出 MA8-v1 改动了 CR3 对照；源码核对确认这一点。[当时冻结的 ma8_adapter.py](../../../../logs/h100/phase6/ma8-v1-cache-2x-50steps-20261026-221355-516887/fullstep/scripts/ma8_adapter.py) 的 `install_activation_cache` 对 `F4` 和 `F4-cache` 都安装新的 column 入口。`F4` 的全部 64 个 QKV／FC1 模块会经过 `MA8ReferenceLinear`，额外执行状态检查、字典计数及前后向 `record_function`，正式计时也会经过这些代码。此处描述历史 v1；后续 MA8-v2 已修改对照接入，本节不将两版混为一谈。

因此，**MA8-v1 回传报告中的约 5% 不能视为相对原 CR3 的有效增量**。保留 native row 投影源码相同，不足以证明整个 CR3 对照未变；也不能拿其约 541 ms 对照和历史 501.433 ms 跨批次相减，认定包装的确切开销。本轮不引用 MA8 的正数作为成功依据，不叠加激活缓存。

MA9 将对照身份落实为代码检查，而不只写在说明里：

- `F4` native 来源逐文件等于注册的 MA4-F；候选在 runtime 尾部增加新统计 kernel 的 include、对应 header 和三个接口的显式导出项，原有 GEMM、codec、barrier、decoder 内容保持不变。
- **两组生成的 `adapter.py`、`taco_support.py` 均逐字节等于原 MA4-F 批次。** 固定 SHA256 分别为 `a66c0124…`、`d994d174…`，完整值在 [ma9_fullstep.py](../../../../tools/h100/phase6/ma9_fullstep.py)。
- 原 worker 的全部八个函数，包括 `forward_backward_step`、`update`、`step`、`clear_gradients`，以及 profile／正式计时循环，与原批次的 AST 指纹一致；更改任一调用或语句即拒绝。
- 新安装语句只在初始化的 `F4-zcount` 条件分支执行。`F4` 不导入 MA9 优化器适配器，不安装新 autograd Function，不增加逐层计数或标记。实验名称、构建核对及结果记录位于训练循环之外。
- 冻结原 Megatron commit 和完整源码；完整模型验证与 trace 另行确认运行路径。禁止从 MA6／MA7／MA8 build 改名复用。

原始依据为 `ma4-f-1x-50steps-20261025-115124-977713`，即 [baseline.md](baseline.md) 中 501.433 ms 的 CR3 批次。历史时间只用于目标量级，MA9 的收益仍由本轮同期 F4 计算。

## 2. 新瓶颈在 optimizer step 尾部

[原 ChainedOptimizer.step](../../../../logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/megatron/megatron/core/optimizer/optimizer.py) 实际执行：

```text
prepare_grads
  → 原 gradient norm
  → 原 gradient clipping
  → count_zeros
      → 逐参数 count_nonzero／减法／累加
      → TP8 标量 AllReduce
      → .item()
  → 原 Adam 更新
  → 原 clear_gradients
```

虽然配置的 `log_num_zeros_in_grad=False`，`ChainedOptimizer.count_zeros` 在 statistics group 共享的分支仍无条件调用 `count_zeros_fp32`。训练 worker 丢弃第三个返回值，也不等于该计算不存在。**MA9 保留这个统计，不通过关闭统计、删掉返回值或改配置取得加速。**

原 [count_zeros_fp32](../../../../logs/h100/phase6/ma4-f-1x-50steps-20261025-115124-977713/fullstep/megatron/megatron/core/optimizer/clip_grads.py) 按参数顺序执行：

```python
num_zeros = grad.numel() - torch.count_nonzero(grad)
total_num_zeros = num_zeros + total_num_zeros
```

本次冻结版本中，一次 `count_nonzero` 对应非零比较、bool→int64 转换、int64 reduction；之后还有整数减法和 FP32 标量累加。每个被选中的参数共五个计算 kernel，部分大 reduction 另有原有的 4 字节清零。

对**原始 CR3 全部八份 trace**的复核如下。通过 CPU 调用范围和 CUDA launch correlation 关联，同一 rank 内测量，未跨 rank 相减绝对时间：

| 项目，每 optimizer step | rank0 | rank1–7 |
| --- | ---: | ---: |
| 筛选后参与统计的参数张量 | 389 | 每 rank 194 |
| 零统计计算 kernel | 1,945 | 每 rank 970 |
| 这些计算 kernel 的工作量之和 | 10.531 ms | 8.687–8.765 ms |
| 原标量 AllReduce kernel | 0.010 ms | 11.521–14.606 ms |
| CPU 首个 count→AllReduce→item 完成区间 | 34.876 ms | 34.832–34.933 ms |

rank0 多统计一组 TP 复制参数，是原 `param_is_not_tensor_parallel_duplicate` 筛选语义，应保持；不能为了平衡负载而重复计数或漏计。rank0 更晚完成逐参数工作、其他 rank 在最终集合通信等待，说明这个统计尾部值得独立优化。所有 rank 的 Adam 更新都在统计读回之后。

上述 CPU 区间、GPU 工作之和及 AllReduce 时长相互重叠，**不能相加，更不能直接说能省 35 ms**。这组 profile 的整步 GPU span 为 818.808–836.798 ms，明显扰动了正式约 501 ms 的训练。它证明执行了什么、依赖在哪里；实际净省多少仍需无 profiler 配对计时。

可复算证据：[analyze.py](../../../../outputs/h100/phase6/ma9-validation-20261026/baseline-analysis/analyze.py)、[analysis.json](../../../../outputs/h100/phase6/ma9-validation-20261026/baseline-analysis/analysis.json)，含原源码、八份 trace 的 SHA256。没有使用 MA8 包装后的 F4 trace 估算本轮预算。

## 3. 机制：一次多张量扫描，一次精确按序归并

MA9 改变统计的计算粒度与中间结果表示：把每参数五个独立 kernel，替换为全体被选梯度共两个 kernel。

```mermaid
flowchart LR
    G[原裁剪后的 FP32 梯度] --> C[Kernel 1：各 tensor 分块计零]
    C --> P[int64 分块计数]
    P --> T[Kernel 2：逐 tensor 归并；按原参数顺序 FP32 累加]
    T --> A[原 TP8 标量 AllReduce]
    A --> I[原 item 返回相同统计]
    I --> O[原 Adam 更新]
```

实现为 [ma9_zero_count.cuh](../../../../tools/h100/phase6/ma9_zero_count.cuh) 和 [ma9_adapter.py](../../../../tools/h100/phase6/ma9_adapter.py)：

1. Python 沿用原参数遍历及 shared／TP duplicate／缺失梯度过滤。只在第一次预热时上传 `[梯度地址, 元素数, 起始 chunk]` 描述表，分配少量持久 scratch；以后逐次核对地址、顺序、形状、dtype、连续性及 stream。发生变化立即失败，不使用过期指针或静默回退。
2. 第一个 kernel 每 CTA 256 threads，扫描最多 32,768 个 FP32 元素。直接计数，输出一个 int64 partial；不生成逐元素 bool 或 int64 临时张量。
3. 第二个 kernel 的一个 CTA 先归并每个参数的 int64 partial，再由一个线程按原参数顺序做原 FP32 累加。每次完全覆盖 scratch 和结果，无逐步 memset，无 float atomic，无新的 stream。
4. 使用原 group、原 FP32 标量、原 SUM AllReduce 和原 `.item()`。原 `step`、梯度范数、裁剪、Adam、清梯度函数不改，更新依赖不变。

这与 MA6 的 decoder tile／mask／布局、bias 与 AllGather 重排，MA7 的延迟 wgrad、MA8 的列激活缓存均不同。它不依赖通信与 GEMM 的新重叠，也不靠改变对照模型增加时间；减少的是**原步尾统计真实执行的扫描、中间张量和提交次数**。

### 数学等价不能只看最终整数

原统计虽然来自整数，累加器却是 FP32。设按原顺序得到每参数整数零数 `z_i`，候选必须保持：

```text
t_0 = FP32(0)
t_i = round_fp32(round_fp32(z_i) + t_(i-1))
```

不能替换成 `FP32(sum_int64(z_i))`。例如 `[2^24, 1, 1]` 的原顺序结果为 `2^24`，整数先求和再转 FP32 得 `2^24+2`。CUDA 实现显式使用 `__ll2float_rn` 和 `__fadd_rn`；round-to-nearest-even 语义见 [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html)。

零判定检查 FP32 位模式：`+0/-0` 都计零，NaN、无穷和非零 subnormal 都不计零，避免外层构建的 fast-math／FTZ 将 subnormal 错当成零。整数分块归约不改变计数。保存的梯度只读，统计不会修改任何梯度位。

当前只支持原 TP8、PP=CP=DP=1、非 DTensor/FSDP、普通 FP32 main gradients 的配置。precision-aware optimizer、存储地址或计算 stream 变化会拒绝；不将额外功能混入首轮实验。

## 4. 为什么值得以 2% 为目标

本节保留上机前的预算依据；实际是否达到目标以第 9 节同期完整训练结果为准。

历史 501.433 ms 的 2% 为 **10.029 ms/step**。这次优化对象具有三个同时存在的成本：

- **970–1945 次计算 launch 收敛为两次**，尤其减少 rank0 多出近千次 launch 的步尾提交工作。
- 一个 FP32 元素在原链中的主要逻辑流量约为 `4+1+1+8+8=22` 字节：读梯度、写／读 bool、写／读 int64。新扫描主要只读 4 字节梯度，再写少量分块计数；不再为整份梯度生成中间数组。缓存、实际事务与 reduction 工作区仍影响实测，不能把逻辑字节直接按 HBM 峰值换算成收益。
- 保留的最终标量 collective 位于 Adam 之前。提前完成本地统计有机会使这个全局同步点和更新一起提前，目标不是把早到 rank 的等待另换位置。

设计时这些证据尚未证明能省 10 ms。只看 10.531 ms 的原 rank0 计算量，扣除新扫描成本，未必能达到 2%；方案的额外机会在于减少逐参数提交与中间分配，并缩短 rank 到达不均衡。**如果无 profiler 实测这些部分已被隐藏，或统计尾部净省不足 10 ms，应按不足 2% 报告。** 本轮不自动叠加旧机制或筛选最快窗口来补足目标。

## 5. 可执行验收流程

| 项目 | 固定协议 |
| --- | --- |
| 对照／候选 | 原 `F4`／`F4-zcount` |
| 模型 | TP8、32 层、hidden 4096、sequence 2048、global batch 4、四个 microbatch、BF16、零 dropout |
| 环境 | 沿用 CUDA 12.4 runtime／12.8 build、`CUDA_DEVICE_MAX_CONNECTIONS=1`、`NCCL_NVLS_ENABLE=0` |
| 窗口 | 10 步预热＋50 步完整 optimizer step，包含前后向、通信、统计、更新及清梯度 |
| 配对 | 每 trial 八个平衡 AB／BA 块、16 窗口、800 个正式步骤；两 trial 为 1600 步 |
| 步时 | 八 rank 最大窗口墙钟／50 |
| 统计 | 完整块配对对数比，10,000 次 bootstrap 95% CI；保留全部负收益与慢窗口 |

启动器顺序为构建 → 原 row 投影算子正确性／计时／profile → 完整模型 smoke → 正式训练配对 → 独立完整 step profile。算子阶段验证原 CR3 投影未变，**不触发 optimizer 融合，不能拿其耗时作为 MA9 机制收益**。

完整模型 smoke 执行两步预热及三步检查，逐 rank 比较全部 **389 个训练参数**的完整梯度和参数更新指纹。候选每个 smoke step 还逐参数比较 native int64 count，并额外运行原 `count_zeros_fp32`，要求包括分布式归约后的返回统计完全一致。该额外核对只出现在 smoke，正式计时／profile 均不执行。

候选 smoke 还运行真实 CUDA 边界检查：空列表、空 tensor、signed zero、NaN／Inf／subnormal、chunk 尾部、超过 `2^24` 的 FP32 累加和 scratch 重用。失败即停止，不放宽容差。正式窗口核对同块两组的全部 loss、grad norm、末态参数指纹和完整执行次数。

[ma9_profile.py](../../../../tools/h100/phase6/ma9_profile.py) 对实际 GPU 路径进行核对：

- F4 的零统计仍有原 389／194 次 `count_nonzero` 和五倍计算 kernel，不存在 MA8 column 包装、新统计 marker 或 kernel。
- 候选有且仅有两个零统计 kernel，顺序、stream 和 launch 关联正确；没有原 `count_nonzero` 回退、逐步 scratch 清零或描述表复制。
- 两组保留一次原 FP32 单标量 TP8 AllReduce、一次 4 字节 DtoH 读回，Adam 在统计完成之后开始。
- 两组原 CR3 的 256 次前向 producer→barrier→decoder＋bias/residual→LayerNorm 路径继续检查。

候选的统计 marker 仅在独立 profile 模式启用；F4 无需新增 marker，直接通过原 ATen 调用关联。

报告分别列出点估计是否达到 2%、CI 是否确认正收益、CI 下界是否达到 2%。只有**正确性、对照身份、实际路径全部通过，且每个请求的 trial 点估计及 CI 下界均 ≥2%**，才标记 `target_validated=true`。单 trial 仍是探索；两 trial 均通过才有复现证据。中途 timing 报告的判断须结合最终 `batch-state.json`，profile 失败不能算批次完成。

## 6. 启动和结果位置

```bash
# 仅预览；不查询 GPU、不建立批次目录
bash tools/h100/phase6/run_ma9.sh

# 后台完整执行，与 run_ma6.sh --execute 同一使用方式
bash tools/h100/phase6/run_ma9.sh --execute

# 两轮独立 trial
bash tools/h100/phase6/run_ma9.sh --trials 2 --execute

# 前台查看全过程
bash tools/h100/phase6/run_ma9.sh --foreground --execute
```

入口为 [run_ma9.sh](../../../../tools/h100/phase6/run_ma9.sh)，监督器为 [detach_ma9.py](../../../../tools/h100/phase6/detach_ma9.py)。支持 `MA9_PYTHON=/path/to/python`、`--out`、`--ma9-build`；复用 build 必须通过当前 MA9 完整源码及二进制身份核对。环境发现沿用已有 container-recovery 文件。

查看启动器打印的新批次目录：

- `batch-state.json`：当前阶段、是否完整完成、最终 2% 判定。
- `handoff/container-supervisor.log`：后台监督日志。
- `fullstep/preflight.json`：三步完整梯度／更新一致性。
- `fullstep/report.json`、`report.md`：全部窗口、同期 CR3 步时、配对收益／CI、原对照身份与显存峰值。
- `fullstep/profile-summary.json`：原路径、两个统计 kernel 与同步依赖证据。
- `fullstep/results/`：各 rank 原始记录、smoke 统计核对状态、独立 trace。

冻结控制器支持只读离线重算，不修改旧实验的报告、脚本或日志。

## 7. 首次上机前的本地验证及边界

首次上机前：**59 项本地测试通过、零跳过**；见 [validation.json](../../../../outputs/h100/phase6/ma9-validation-20261026/validation.json)。

- 10 项 PyTorch `2.6.0+cu124` CPU adapter 测试：真实 `inspect.getsource`、实例安装边界、原参数过滤、顺序 FP32 舍入反例、数值边界、地址／stream 变化拒绝、统计返回值比较。CUDA ABI 和 collective 在这些 CPU 测试中为模拟，明确不算 GPU 数值验证。
- 49 项控制器测试：原 F4 来源、候选唯一 native 增量、篡改拒绝、真实冻结 worker 的在线检查、平衡配对、负结果保留、完整参数门槛、profile、离线重算和启动器。
- 新 CUDA header 用 nvcc 12.8 针对 SM90 普通及 fast-math 编译均通过，无 spill；这是独立 kernel 编译，尚非完整 Flux 动态库链接或 H100 执行证据。
- 八份原始 CR3 trace 通过新的零统计校验器；生成代码、shell 语法和两 trial 预览通过。

上述记录属于首次上机前的本地验证，不包含后续 GPU 结果，也未被回写为 GPU 验证成功。修复后的实际 H100 数值、执行路径与完整训练结果见第 9 节；原 CR3 对照及数值门槛保持不变。


## 8. 动态库接口导出修复（2026-10-27）

首次批次 [`ma9-v1-zcount-1x-50steps-20261027-005917-206111`](../../../../logs/h100/phase6/ma9-v1-zcount-1x-50steps-20261027-005917-206111/batch-state.json) 已完成编译与原 F4 算子预检，但 `preflight-F4-zcount` 的八个 rank 均在读取 `ma9_zero_count_protocol` 时失败，见 [process.log](../../../../logs/h100/phase6/ma9-v1-zcount-1x-50steps-20261027-005917-206111/operator/results/preflight-F4-zcount/process.log)。错误为 `libflux_cuda.so: undefined symbol: ma9_zero_count_protocol`，没有开始正式训练；这不是 Bash 参数或 GPU 抢占错误。串行启动器收到非零退出码后停止，因此当时没有启动 MA10。

根因是继承的 `src/cuda/version.ld` 仅允许原有的 `taco_*` 等接口导出，其 `local: *;` 隐藏了新 `ma9_zero_count_*` 函数。此前的独立 CUDA 对象编译没有经过完整导出表链接，因此不能发现该问题。

本次修复：

- [ma9_sources.py](../../../../tools/h100/phase6/ma9_sources.py) 仅给候选导出表添加 `ma9_zero_count_protocol`、`ma9_zero_count_chunk_size`、`ma9_zero_count_cuda` 三个明确名称。原 F4 导出表、既有投影和新统计 kernel 算术均不变。
- [ma9_build.py](../../../../tools/h100/phase6/ma9_build.py) 在 `built=True` 验证中调用 [native_exports.py](../../../../tools/h100/phase6/native_exports.py)，使用 `nm -D --defined-only` 检查真实动态函数导出。候选必须含全部接口，F4 不得暴露候选接口；文件哈希正确也不能替代这个检查。该检查同时覆盖构建结束、build 复用和在线 worker 的 GPU 初始化前验证。
- 新的导出变换和验证器都纳入构建身份；不能修改旧 manifest 哈希或失败快照绕过验证。

MA9／MA10 联合修复共 **71 项 CPU 回归检查通过，零跳过**，见 [validation.json](../../../../outputs/h100/phase6/ma9-ma10-export-fix-20261027/validation.json)。测试包含实际 ELF 链接／加载、控制组接口隔离、导出表篡改拒绝，以及旧导出表生成的库即使重新记录哈希仍在 GPU 初始化前被拒绝。

另用实际 CUDA helper 编译并链接动态库：旧表复现缺失符号；修复表的普通及 fast-math 版本均可由 `ctypes` 解析全部接口，host 查询返回 protocol=9、chunk=32768。见 [native-validation.json](../../../../outputs/h100/phase6/ma9-ma10-export-fix-20261027/native-validation.json)。**这份修复时记录只证明 CUDA helper 动态库验证，不包含完整 Flux 链接、GPU kernel 执行或性能结果。** 后续完整上机结果另记于第 9 节。

同步修复后使用新批次：

```bash
bash tools/h100/phase6/run_ma9.sh --execute
```

启动命令及串行运行顺序保持原样。旧 `--ma9-build` 的源码／构建身份已过期，不可直接复用；失败批次及旧测试记录保留不改。

## 9. MA9 首轮八卡实测与离线复核（2026-10-27）

### 9.1. 完成范围与结论

批次 [`ma9-v1-zcount-1x-50steps-20261027-081044-826367`](../../../../logs/h100/phase6/ma9-v1-zcount-1x-50steps-20261027-081044-826367/batch-state.json) 已完成修复后的构建、两组算子检查、完整模型预检、全部正式窗口及独立 profile。最终状态为 `completed / done`，`target_validated=true`，不是只有计时阶段结束。

本轮保留 **16 个正式窗口、128 份 rank 记录、800 个正式 optimizer step**，两组各八个窗口、400 步。每窗口 10 步预热＋50 步正式训练，按八 rank 最大墙钟／50 计算步时；未剔除慢窗口或负向波动。八个完整块的配对对数比及 10,000 次 bootstrap 使用原协议，点估计和 CI 下界均超过 2%。

**结论：这一次固定配置的 MA9 相对同期原 CR3 有已通过数值与路径检查的整步收益，并达到本批次 2% 门槛。** `trials=1`、`exploratory=true`；同一 trial 的前后半轮不构成两次独立复现，暂不据此把默认 CR3 基线替换为“稳定 MA9”。

### 9.2. 同期完整训练：8/8 个配对块更快

| 范围 | 配对块 | CR3/F4 平均 ms/step | F4-zcount 平均 ms/step | 配对收益 %［95% CI］ |
| --- | ---: | ---: | ---: | --- |
| 前半轮 | 4 | 504.122 | 491.491 | +2.505［+2.032, +3.228］ |
| 后半轮 | 4 | 502.497 | 488.265 | +2.832［+2.513, +3.150］ |
| 全部 | 8 | **503.310** | **489.878** | **+2.669［+2.306, +3.062］** |

平均减少 **13.431 ms/step**；各块减少 **10.169–18.373 ms/step**，块内收益为 **+2.031% 至 +3.623%**。两组全部窗口范围分别为 **500.086–507.133 ms**、**487.567–493.440 ms**；窗口中位数分别为 **503.374／488.754 ms**。按每步 8192 token 计算，吞吐从 **16,276.26** 增至 **16,722.53 tokens/s**。

数据来源：[完整训练 report.json](../../../../logs/h100/phase6/ma9-v1-zcount-1x-50steps-20261027-081044-826367/fullstep/report.json)、[report.md](../../../../logs/h100/phase6/ma9-v1-zcount-1x-50steps-20261027-081044-826367/fullstep/report.md)。上述收益仅比较本批次的两个组，不将历史 501.433 ms 或 MA8-v1 包装后的对照代入计算。两组 row 投影算子相同，其独立算子计时只作一致性检查，不用来解释 optimizer 融合收益。

### 9.3. 原 CR3 身份与完整数值检查

本次回传 `adapter.py` 和 `taco_support.py` 已与原 MA4-F 批次实际文件逐字节核对，SHA256 与第 1 节固定值一致。冻结控制器同时通过全部八个原训练函数、原计时循环及候选专属安装分支的检查；返回的 **1,383 个 Megatron 文件**也与计划中记录的哈希一致。所有正式 F4 rank 记录的 `ma9_zero_count_state` 均为 `{"installed": false}`，实际 F4 trace 中保留原 `count_nonzero` 链，没有 MA8 column 包装或 MA9 统计 kernel。

完整模型 [preflight.json](../../../../logs/h100/phase6/ma9-v1-zcount-1x-50steps-20261027-081044-826367/fullstep/preflight.json) 通过：两组八个 rank 的三步检查中，每步全部 **389 个参数的完整梯度及更新后参数指纹一致**。候选每 rank 的两步预热＋三步 smoke 共五次原统计对照均通过，六类 native 边界自测也全部通过。

正式窗口每 rank 的候选状态均为 `calls=60`、`kernel_calls=120`、`initializations=1`、`audits=0`；rank0 选中 389 个张量，其他 rank 各 194 个，无失败，smoke 额外统计核对未进入正式计时。两组同块同 rank 的全部 loss、grad norm 轨迹及末态参数指纹一致。两组 mean loss 均为 `3.793363792030723`，final loss 均为 `0.00030573460389859974`，final grad norm 均为 `0.01800245891878419`。这是固定合成输入的短程等价验证，不是长程收敛实验。

### 9.4. 实际路径：统计由 970–1945 个计算 kernel 降至两个

两组共 **16 份完整训练 trace**通过复算。每 rank 的 Attention／MLP 各 128 次投影仍走原 CR3 的 producer→barrier→融合 decoder→LayerNorm，launch geometry 保持原配置；两组每组覆盖八 rank 共 2,048 次投影。

| 零统计路径，每 rank 每 profile step | CR3/F4 | F4-zcount |
| --- | ---: | ---: |
| 原 `count_nonzero` 次数，rank0／其他 rank | 389／194 | 0／0 |
| 统计计算 kernel，rank0／其他 rank | 1,945／970 | 2／2 |
| 原 reduction workspace 清零，rank0／其他 rank | 131／130 | 0／0 |
| 原 FP32 单标量 TP8 AllReduce | 1 | 1 |
| 原 4 字节 DtoH 统计读回 | 1 | 1 |
| Adam 在统计完成之后启动 | 是 | 是 |

两个候选 kernel 均通过 CUDA launch 关联及同 stream 顺序检查，无原统计回退、描述表复制或额外 memset。零统计计算 kernel 工作量的八 rank 中位数从 **8.714 ms** 降至 **1.095 ms**；GPU 统计首个 kernel 到读回结束的区间中位数从 **34.349 ms** 降至 **3.692 ms**；CPU 统计调用区间中位数从 **34.395 ms** 降至 **6.611 ms**。这些独立 profile 数据支持减少扫描、提交及步尾等待的机制解释，**不能将这些差值相加，或用约 30.7 ms 的 profile 区间差替代正式训练实测的 13.431 ms**。

见 [profile-summary.json](../../../../logs/h100/phase6/ma9-v1-zcount-1x-50steps-20261027-081044-826367/fullstep/profile-summary.json)。

### 9.5. 显存、离线复算与下一步

以“每窗口八 rank 峰值的最大值，再跨窗口平均”为口径：

| 峰值指标，GiB | CR3/F4 | F4-zcount |
| --- | ---: | ---: |
| allocated | 16.428822 | 16.429080 |
| reserved | 16.638672 | 16.509766 |

allocated 的该均值增加约 **0.264 MiB**，没有出现激活缓存级别的显存增长；reserved 较低仅作为本轮 allocator 观测，不推广为稳定节省。

离线审计使用回传的冻结控制器，隔离进程、禁用字节码写入、未导入 Torch 或启动 GPU。算子 `report.json`／`profile-summary.json` 和训练 `report.json`／`profile-summary.json` **四份结果全部与原始记录／trace 重算一致**，预检、原始 rank 记录、脚本和 trace 均保持原哈希。另独立由各窗口八 rank 墙钟重算步时与配对点估计。审计记录：[audit.json](../../../../outputs/h100/phase6/ma9-results-audit-20261027/audit.json)，复算入口：[audit.py](../../../../outputs/h100/phase6/ma9-results-audit-20261027/audit.py)。本次回传不含 build 二进制，本地复核覆盖冻结源码、记录的库身份及执行证据，没有重新链接或加载服务器动态库。

下一步用第 6 节的 `--trials 2 --execute` 进行独立复现，继续使用原 CR3 对照和原 2% 门槛。当前可以记录“首轮达标”，不能把本轮两个半程写成“已两轮复现”；本页更新不修改 `baseline.md` 或历史批次证据。
