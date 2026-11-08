# RTX 5090 开源通信压缩六组 baseline：Megatron / Flux / INT8 / 交替排序 / 远端优先

更新：2026-11-02。目标机器为**单机 8×RTX 5090，TP8、DP1**。统一入口为 [run_open_compress.sh](../../../tools/5090/phase6/run_open_compress.sh)，协议为 `rtx5090-open-compress-int8-six-arm-v2`。本页沿用 [design-5090 baseline](../design-5090/baseline.md) 的模型、完整训练步和证据组织方式，建立独立实验；目录名保留 `design-5090-qutlass`，不表示运行时使用了 QUTLASS。

**当前状态：六组 v2 的 SL3061 批次已回传，六组算子检查和前三组 Flux 模型 smoke 通过，随后在 Megatron BF16 模型适配器初始化失败；配置修复已完成，完整六组训练及正式计时仍需重跑。** 本地源码审查、CPU 测试或 CUDA 编译只能证明相应检查通过，不能证明 RTX 5090 上的执行路径、八卡通信、数值误差、完整训练步耗时或显存已通过。旧 TACO、旧四组 Flux、H100 以及 QUTLASS 上游图表中的性能均不填入本页新六组结果。

## 1. 对比目标与算法选择

本阶段比较原生 Megatron 与 Flux 在同一训练负载下的实现，并分别衡量 INT8 通信路径；在 Flux＋INT8 中再比较原顺序、交替排序和远端优先。到达优先与选择性量化只保留为后续方向，不参与本阶段的量化范围、执行计划和收益归因。

### QUTLASS 评估

审查对象为本地 `../qutlass`，对应上游提交 **`e74319e3405ce6d71965732880f5dc1f52371f64`**，Apache-2.0 许可。它是支持 Blackwell 的量化 BLAS 库，已有 SM120a 和 RTX 5090 路径；**不选择它作为本阶段通信 codec，并非因为 RTX 5090 不支持它**。RTX 5090 的计算能力是 12.0，CUDA 12.8 开始提供 SM120 编译支持。[NVIDIA GPU 列表](https://developer.nvidia.com/cuda/gpus)、[CUDA 12.8 发布说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html#new-features)

| 候选 | 硬件与接口 | 对本任务的限制 | 本阶段决策 |
| --- | --- | --- | --- |
| QUTLASS MXFP4 / NVFP4 | Blackwell SM120a/SM100a；编码、旋转、scale 处理与低精度 GEMM；MXFP4 为 E2M1＋E8M0 | 不是现成的通信压缩/归约库；仍需实现 packet、解码/逆旋转、NCCL 传输及有序归约。换用其低精度 GEMM 会同时改变计算精度和后端，影响通信压缩归因。 | 保留为后续更高压缩率候选。 |
| 分组 absmax 对称 INT8 | 只需 BF16/FP32 标量运算、warp 归约及 INT8 存储，不依赖 FP4 Tensor Core；可编译为 SM120 | 有损；离群值会降低组内小值的精度；独立编码、All-to-All 和解码可能抵消载荷减少。 | **选择，独立 CUDA 实现并公开源码。** |

QUTLASS 当前 checkout 的实际构建约束高于 README 徽章：`setup.py` 要求 **PyTorch ≥2.11**，C++ 使用 `torch::stable::Tensor`；现有目标环境的 NGC 25.06 / PyTorch `2.8.0a0+5228986c39.nv25.06` 不能直接照 README 的“torch≥2.8”完成该提交的构建。`setup.py` 还在构建阶段查询可见 CUDA 设备。CUTLASS 子模块条目为 `b2ca083d2bb96c41d9b3c5a930637c641f6669bf`，本地评估时未初始化。若以后启用 QUTLASS，需单独冻结兼容提交或适配绑定、初始化相应 CUTLASS、核验 FP4 及 scale 布局，并重新跑完整数值和训练门禁；不能只降低版本判断。[冻结的 setup.py](https://github.com/IST-DASLab/qutlass/blob/e74319e3405ce6d71965732880f5dc1f52371f64/setup.py)、[README](https://github.com/IST-DASLab/qutlass/blob/e74319e3405ce6d71965732880f5dc1f52371f64/README.md)、[Python 接口](https://github.com/IST-DASLab/qutlass/blob/e74319e3405ce6d71965732880f5dc1f52371f64/qutlass/__init__.py)

MXFP4 的 32 个系数需要 16 字节数据＋1 字节 scale，相对 64 字节 BF16 的理想表示比为约 3.765×；实际 packet 还受 scale padding、布局、旋转与解码影响。上游的低精度 GEMM、推理或 QAT 收益不是通信 codec 或本训练协议的性能证据。更低位宽的误差也不能由短期 smoke 推导为收敛等价。

### 所选开源算法与实现边界

本实现参考 DeepSpeed 的**按组求 absmax、对称量化、发送整数与每组 scale、接收端反量化**思路，采用独立 BF16 CUDA 实现，受本仓库 Apache-2.0 许可覆盖。算法参考冻结为 DeepSpeed 提交 `2d80cedd2bda7223139d7dd2118950db59ab982c` 的 [quantization_utils.h](https://github.com/deepspeedai/DeepSpeed/blob/2d80cedd2bda7223139d7dd2118950db59ab982c/csrc/includes/quantization_utils.h) 与 [quantize.cu](https://github.com/deepspeedai/DeepSpeed/blob/2d80cedd2bda7223139d7dd2118950db59ab982c/csrc/quantization/quantize.cu)，上游为 Apache-2.0。

这里**没有调用或复制 DeepSpeed 的原生 kernel，也不兼容其位流**：所选协议使用 BF16 输入、每组 128 个数、`[-127,127]` 对称范围和 FP32 `maxabs/127`；上述 DeepSpeed 参考使用 FP16 输入，8-bit 对称路径采用 `128/maxabs` 的编码系数和 `[-128,127]` 范围。无需下载/安装 DeepSpeed、QUTLASS 或新 Python 包来运行本 codec。选择该适配的原因是能够保留 Flux BF16 计算，只替换通信表示，避免为第一轮对照同时升级 PyTorch 和引入 FP4 GEMM。

## 2. 六组实现与唯一变量

| 展示名称 | 脚本组名 | 目标投影实际路径 | GEMM 生产顺序 |
| --- | --- | --- | --- |
| 原始 Flux（5090 staged 对照） | `flux-bf16` | Flux BF16 GemmOnly → NCCL BF16 ReduceScatter | 原 GemmOnly 顺序 |
| Flux＋INT8 | `flux-int8` | 同精度 Flux GemmOnly → 独立 CUDA INT8 编码 → NCCL byte All-to-All → 本地解码/归约 | 原 GemmOnly 顺序 |
| Flux＋INT8＋交替排序 | `flux-int8-interleaved` | 与 `flux-int8` 相同的 codec、packet、通信及消费算法 | 在真正的 GemmOnly CTA tile 映射中交替目标分区 |
| Megatron 原生 | `megatron-bf16` | 原 `RowParallelLinear.forward` → 原本地 linear / cuBLAS → 原 sequence-parallel ReduceScatter | Megatron 原生调度 |
| Megatron＋INT8 | `megatron-int8` | 原 RowParallelLinear 和 local linear → 同一 INT8 codec / NCCL All-to-All / 解码归约 | Megatron 原生调度 |
| Flux＋INT8＋远端优先 | `flux-int8-remote-first` | 与 `flux-int8` 相同的 codec、packet、通信及消费算法 | 前 448 个逻辑 tile 槽位全部远端，最后 64 个本地 |

“原始 Flux”沿用当前 5090 中 GemmOnly＋NCCL 的可执行对照定义，不表示原 H100 的跨 GPU 直接访存或融合重叠得以保留。NVIDIA 已说明 GeForce RTX 50 系列不支持 GPU 间 P2P，旧目标机记录也为 0/56 跨卡有向链路可用；新批次仍须重新记录实际拓扑。[NVIDIA P2P 答复](https://forums.developer.nvidia.com/t/p2p-issue-using-two-rtx-5090-gpus/326776/8)

六组统一观测 32 层 Attention 输出投影和 MLP 输出投影，每 rank 共 64 个目标模块。Flux 组沿用原有行并行前向适配；Megatron 原生组保留原 `RowParallelLinear.forward` 的代码及原 linear、ReduceScatter 与反向实现，仅加审计/profile 包装。Megatron＋INT8 在该前向的 ReduceScatter 边界替换通信表示，保留原 linear 反向和 sequence-parallel AllGather 反向；使用每实例私有函数 globals，不修改进程全局 Megatron 函数。压缩只作用于这些模块**前向生成的远端输出贡献**，本地贡献保留 BF16；QKV、FC1、反向梯度通信、优化器等继续原路径，不增加 MA8 激活缓存、MA9 零统计优化、MA11 softmax、CUDA Graph、误差反馈或 QAT。所有远端贡献统一量化，不加载到达表或选择 mask。

固定算子形状为 `M=2048, N=4096`，Attention 的 `K_local=512`，MLP 的 `K_local=2048`。每来源产生 `2048×4096` BF16 部分结果，每目标持有 256 行，接收八个来源对该分区的贡献。

`flux-bf16` 与 `flux-int8` 的差值同时包含 codec、集合通信算法以及归约舍入差异，是整个压缩路径的端到端对照，不能仅归因于字节数。`flux-int8-interleaved` 与 `flux-int8` 共用量化范围、编码分组、物理布局和来源归约顺序，只改变 GEMM tile 的逻辑生产映射；这才是本轮交替排序比较。原顺序、交替排序和远端优先三套 Flux 库隔离构建，记录源码与二进制身份；两个 Megatron 组不加载 Flux 动态库。Flux/Megatron 的差值包含 GEMM 后端、前后向适配与提交开销，不能单独归因于通信压缩。

### 交替顺序的含义

交替沿用 [idea.md](../design-h100/idea.md) 的目标分区轮转思想。两组固定 Flux GemmOnly V2 BF16 RCR、`128×128×32` tile、stage 3、StreamK；本次输出为 `16×32=512` 个 tile。来源 `s` 的映射为：

```python
destination = (s + i % 8) % 8
within = i // 8
tile_m = destination * 2 + within // 32
tile_n = within % 32
physical_tile = tile_m * 32 + tile_n
```

映射必须完整覆盖物理 tile，不能重复、遗漏或改变目标归属。它发生在 GEMM producer 中，不是在 GEMM 完成后仅重新排列编码 packet。构建中保留的 Flux 注册键 `Sm89` 不表示输出是 SM89 二进制；实际 CUDA 编译目标为 `sm_120`，仍需目标机运行验证。

排序不能改变量化身份：block128 分组和发送地址始终由物理输出坐标确定，接收端仍按 source `0→7` 归约。逻辑 CTA 槽位不等于实际完成时间；CUDA 的 CTA 调度、资源竞争和缓存行为仍会影响完成顺序。StreamK 的 tile 分配可能改变 split-K 归约组合与舍入，因此两组接受独立参考与误差预算验证，不预先承诺 GEMM 输出或训练轨迹逐位一致。

本版本在**完整本地 GEMM、编码完成后**提交 NCCL，接收完成后解码。它尚未在 tile 生成期间发送，也未引入到达优先消费。因此交替排序可能改变 GEMM 局部性和耗时，也可能没有收益甚至变慢；不能把结果解释为已恢复 H100 的 tile 到达重叠。

### 远端 tile 优先

`flux-int8-remote-first` 在真实 GEMM CTA 映射中，将目标按相对 rank 顺序排列为 `s+1, s+2, …, s+7, s`（模 8），每目标连续 64 个物理 tile：

```python
destination = (s + 1 + i // 64) % 8
within = i % 64
tile_m = destination * 2 + within // 32
tile_n = within % 32
```

因此槽位 `0..447` 都归远端，`448..511` 都归本地；目标内按物理 tile 行优先。GPU 映射探针必须验证全部 512 个坐标、无重复/遗漏、远端/本地边界，以及实际加载的 `OpenCompressRemoteFirstSwizzle` kernel。它改变生产映射，不改变编码布局或 NCCL peer 顺序，也不是到达优先策略；不读取到达表。该严格定义与 H100 历史 `remote_first` 的 StreamK 坐标偏移实现分开命名和记录。

远端优先与交替排序都可能改变 StreamK 的归约分配，不承诺逐位相同。两者都在完整 GEMM 之后才发送，逻辑远端槽位靠前不保证所有远端 tile 物理完成更早，也不表示通信重叠已经实现。

## 3. 编码、传输、解码与显存

对每行连续 128 个 BF16 值，在 FP32 域计算：

```text
a = max(abs(x[0:128]))
scale = 1                              if a == 0
        round_fp32(a / 127)            otherwise
q[i] = clamp(round_to_nearest_even(x[i] / scale), -127, 127)
decoded[i] = round_bf16(float(q[i]) * scale)
```

输入域为有限 BF16 值；全零、随机、离群和边界输入分别测试。smoke 对实际 GEMM partial 检查有限性；编码 kernel 若发现组内非有限值则写入 NaN scale，使错误传播至解码及训练门禁，避免 `fmaxf` 忽略 NaN 后生成看似正常的值。解码后每个来源的贡献先舍入到 BF16，再按来源 `0→7` 依次相加，每次加法保留 BF16 舍入；本地 source 插在其原位置。bias/residual 和后续 LayerNorm 沿用原训练路径。

发送缓冲按目标 rank 升序排列，跳过 self；每目标为 INT8 系数区＋FP32 scale 区。NCCL self split 为 0，只交换远端字节，接收缓冲按来源 rank 排列。任何 rank 的输出均由本卡接收缓冲和本地贡献产生，不访问远端 CUDA 指针。发送后等待 Work 的 stream 依赖，消费完成后才能复用 scratch；输出存储独立，后续调用不得覆盖仍被调用者持有的结果。

| 单 rank、单次目标投影的逻辑数据量 | 数值 |
| --- | ---: |
| 每远端目标的原 BF16 贡献 | 2 MiB |
| 每远端目标的 INT8＋FP32 scale | 1.03125 MiB |
| 七个远端的原 BF16 贡献 | 14 MiB |
| 七个远端的 INT8＋FP32 scale | 7.21875 MiB |
| 本地 BF16 贡献 | 2 MiB |
| 每 128 值表示比 | `256 / (128+4) = 1.93939×` |
| 远端逻辑载荷减少 | 48.4375% |

这些是固定协议的尺寸计算，**不是实测带宽或 NCCL 链路流量**；尤其不能用 14 MiB 直接代表 BF16 ReduceScatter 的实际总链路传输。协议不压缩本地输出，scale、临时缓冲、编码解码和 NCCL 工作区均有成本。

六组冻结 `NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=0`、`NCCL_NVLS_ENABLE=0` 并记录其他通信配置。环境设置不等于实际使用 SHM 的证据，也不保证所有拓扑都具有相同性能。算子与完整模型分别记录 Torch `allocated`、`reserved` 峰值，以及显式 native workspace；先逐 rank 合计 allocated＋native 再取最大，不能将 allocated 与 reserved 相加。该合计仍不含 CUDA context、NCCL 等全部内部占用。

## 4. 目标机准备与一键执行

### 需要同步的内容

目标机按以下同级目录组织：

```text
/workspace/
├── TempName-New/          # 更新后的完整源码，包括 tools、env、src、include、python
│   └── 3rdparty/          # 本仓库锁定的 CUTLASS、NCCL 子模块
├── Megatron-LM/           # 3ea68ad6042cc1204386ae9364358f7c4de1bc37
└── Load-5090/             # 新结果目录；运行进程需要写权限
```

同步**整个更新后的 `TempName-New` 源码及其固定子模块**，不能只复制 shell 或新 `.py/.cu`。新 worker 继续使用本仓库 `tools/h100/phase6`、`tools/a800` 中的冻结模型、训练循环和辅助模板；它们是生成输入，不意味着执行 H100 二进制或 TACO codec。Megatron 必须为上述精确提交；缺失子模块时可在目标机源码根目录运行 `git submodule update --init --recursive`。保留目标机已有结果目录，不覆盖历史实验。

本阶段不需要同步 `qutlass`、`COCCL_TACO` 或已有 TACO `.so`。不把开发机或 H100 的 Flux/CUDA 二进制搬到 5090 复用；入口会在目标环境构建原顺序、交替顺序和远端优先 Flux 以及独立 INT8 codec。`--build` 可复用身份验证通过的完整 v2 构建；传入已完成的 v1 三组构建时，在新批次建立只读引用，并仅增量构建远端优先 Flux，不改写旧 manifest、二进制或实验数据。原 v1 构建、固定依赖与 ABI 环境必须仍可访问且通过校验；未知版本拒绝复用。

完整构建在同一兼容环境内搬迁后，旧清单中的 NCCL 绝对路径按当前构建目录内的 `nccl-source/build/include/nccl.h` 和 `nccl-source/build/lib/libnccl_static.a` 解析，并继续逐文件检查原 SHA256；新构建使用相对路径记录。旧清单不会被修改。搬迁时需保留这两个文件、全部二进制和正确的依赖链接，Python/CUDA ABI 和依赖身份仍必须匹配。引用 v1 父构建的 overlay 仍要求其清单所记录的父构建路径和只读链接可访问，此修复不自动搬迁父构建或重写链接。

构建依赖为支持 SM120 的 CUDA Toolkit（至少 12.8，现有目标为 12.9）、C++ 编译器、CMake/构建工具、PyTorch CUDA 和 C++ 扩展开发环境，以及固定 Megatron 所需的 Apex、Transformer Engine 和 Python 依赖。推荐复用现有 `nvcr.io/nvidia/pytorch:25.06-py3` 环境；所有原生扩展应与当前 PyTorch/CUDA 匹配。需要八卡可见、足够主机共享内存以及无其他训练任务占用 GPU。详见 [5090 环境说明](../../../env/README_5090.md)。

```bash
cd /workspace/TempName-New
bash env/setup_5090.sh
source env/activate_5090.sh
```

已有正确配置的环境可直接激活。不要用 `setup_h100.sh` 替换 5090 的 PyTorch/CUDA；新协议也不要求为 QUTLASS 升级到 PyTorch 2.11。

### 统一入口

```bash
# 仅预览；不构建、不查询 GPU、不启动训练。
bash tools/5090/phase6/run_open_compress.sh --place 61 --trials 1 --blocks 1

# 用户要求的最小六组完整批次；默认前台执行。
bash tools/5090/phase6/run_open_compress.sh \
  --place 61 --trials 1 --blocks 1 --execute

# 默认 6 个完整 block：六组 Williams 平衡顺序，每组在每个位置各出现一次。
bash tools/5090/phase6/run_open_compress.sh \
  --place 61 --trials 1 --blocks 6 --execute

# 只检查硬件、环境与真实八 rank NCCL。
bash tools/5090/phase6/run_open_compress.sh --preflight-only --execute

# 构建并完成独立 codec、通信、算子与 profile 检查后停止。
bash tools/5090/phase6/run_open_compress.sh --operator-only --execute

# 复用已完成的 v1 三组构建；只在新批次增量构建远端优先库。
bash tools/5090/phase6/run_open_compress.sh \
  --build /workspace/Load-5090/logs/5090/phase6/<已有批次>/build \
  --place 61 --trials 1 --blocks 6 --execute
```

`--trials` 表示重复实验次数，`--blocks` 是每个 trial 的完整六组 block 数，不是 CUDA thread block 数。`--blocks 1` 用于验收和单个实验点，只有每组一个正式窗口，不能支撑置信区间或稳定收益结论。至少 6 个完整块才报告配对 bootstrap 区间；增加 blocks 的具体顺序以保存的计划为准。

`--jobs` 默认取当前环境的 `nproc`，等价于显式传入 `--jobs "$(nproc)"`；仍可用 `--jobs N` 覆盖。所有当前 5090 编译入口/构建函数及根 `build.sh` 使用此默认值，H100 历史实验源码保持冻结。解析后的并行数写入 launch/plan，预览也显示 `jobs`。

每次执行使用新的批次目录，`--place` 选择批次的直接父目录：

| 参数 | 容器中的输出目录 |
| --- | --- |
| `--place 60` | `/workspace/Load-5090/SL3060/<批次>/` |
| `--place 61` | `/workspace/Load-5090/SL3061/<批次>/` |
| 不传 `--place` | 保留原 `/workspace/Load-5090/logs/5090/phase6/<批次>/` |

路径以 `TempName-New` 的同级 `Load-5090` 为根，适用于其他工作区挂载路径；显式 `--out /绝对路径/新批次目录` 优先于 `--place`。本参数只选择保存位置，不切换远程主机，实际主机仍由硬件预检记录。脚本打印批次路径，控制台阶段输出与批次内日志用于定位失败；前台运行时，需保持终端/容器存活，长任务可放在 `tmux` 内运行。也可由用户用 `nohup` 包裹完整入口并重定向输出，本入口不另建后台 supervisor。

查看结果与只读复核：

```bash
cat /workspace/Load-5090/SL3061/<批次>/batch-state.json
cat /workspace/Load-5090/SL3061/<批次>/operator-summary.json
cat /workspace/Load-5090/SL3061/<批次>/report.md

bash tools/5090/phase6/run_open_compress.sh \
  --report --out /workspace/Load-5090/SL3061/<批次>
```

只完成 preflight 或 operator 检查不代表完整训练通过；失败批次保留现场，不能继续混入已完成报告。`--report` 用于读取并复核已有完整批次，不重跑训练，也不向失败批次补写性能结论。

### 2026-11-02 失败批次修复与重跑

批次 [`open-compress-int8-six-arm-v2-1x-1blocks-20261102-171639-582516`](../../../../Load-5090/logs/5090/phase6/open-compress-int8-six-arm-v2-1x-1blocks-20261102-171639-582516/batch-state.json) 实际运行主机为 **SL3061**，launch 记录 `jobs=128`。八 rank 的 [Megatron BF16 日志](../../../../Load-5090/logs/5090/phase6/open-compress-int8-six-arm-v2-1x-1blocks-20261102-171639-582516/results/smoke-megatron-bf16/process.log) 均在 `Adapter(raw, POLICY, SPEC['shapes'], SPEC['ring_reduction'])` 处报 `Unsupported native adapter policy`。

原因是生成配置继承了 Flux GemmRS 专属的 `ring_reduction=True`，而 Megatron 原生适配器拒绝该标志。独立算子探针直接构造 `MegatronProjection`，没有经过该适配器，所以算子通过不代表模型初始化通过。现仅为 `megatron-bf16`、`megatron-int8` 设置 `ring_reduction=False`，Flux 四组保持原值；保留运行时门禁、原训练函数、原生前后向和数值预算。回归测试使用真实生成配置初始化两个原生组的 64 个模块，覆盖先前测试漏掉的实际参数传递。

已离线重新核验六组 **48 份算子记录、48 份原始 GPU trace**和三组 Flux **24 份完整模型 smoke 记录**；其路径摘要与原记录一致，96 个算子 rank/trace 文件哈希匹配，见[回传复核记录](failure-20261102-171639-review.json)。八卡硬件/NCCL 预检通过。该批次尚无完整训练 profile 或正式计时窗口，因此不能报告完整步收益。回传未包含 `build/` 和 `.so`，本地未验证目标二进制。

修复后 65 项开源压缩专项测试、7 项编译默认值/旧配方兼容测试通过；入口预览、shell 语法和 diff 检查通过，见[本次验证记录](local-validation-fix-20261102.json)。5090 全量回归执行 422 项、跳过 1 项，唯一错误来自测试发现阶段已加载的旧 subprocess mock；该测试已明确传入编译并行数，在后续专项回归通过。未在本地重跑 GPU，修复后的完整训练仍需以下目标机命令验证。

本次没有修改 CUDA codec 或 GEMM 源码。已登记失败批次对应的旧 v2 构建配方精确身份，允许目标机通过 `--build` 只读复用全部已编译库；仍核对源文件、二进制、依赖、注册和 Python/CUDA ABI，不改写原 manifest。同步更新后的完整 `tools/5090/phase6/`（含 `rtx5090_defaults.py`）和根 `build.sh`，在同一目标环境执行：

```bash
bash tools/5090/phase6/run_open_compress.sh \
  --place 61 --trials 1 --blocks 4 --execute \
  --build /workspace/Load-5090/logs/5090/phase6/open-compress-int8-six-arm-v2-1x-1blocks-20261102-171639-582516/build
```

新结果写入 `/workspace/Load-5090/SL3061/<新批次>/`；切换到 SL3060 保存位置时改为 `--place 60`。新批次重新执行所有门禁和六组同期窗口，原失败数据保留。四块为 24 个正式窗口、1200 个正式 optimizer step，尚不足六块，因此不生成置信区间。

## 5. 正确性、实际路径与测试方法

入口依次执行硬件/软件与八卡 NCCL 预检、三套 Flux＋codec 构建、独立算子门禁、六组完整模型 smoke、六组完整模型 profile，最后才运行正式计时。任一必要门禁失败均停止，不跳过失败组、不自动换成 Torch GEMM、不缩小模型，也不以 codec 单测替代完整步测试。

### 独立 codec 和算子门禁

- **量化与反量化：** 零值、固定随机输入、正负值、离群值及舍入边界，检查整数范围、scale、packet 布局和独立 oracle；分别报告编码误差和最终投影误差。INT8 是有损算法，不要求 `decode(encode(x)) == x`。应要求解码符合冻结量化协议，且误差门限通过。
- **八卡传输：** 覆盖 Attention/MLP 两种真实 K、所有来源/目标组合；发送与接收 packet 的哈希交叉匹配，self 不发送。NCCL 探针和压缩 packet 探针分别保留逐 rank 记录，不以本机 encode/decode 冒充跨卡验证。
- **GEMM 映射与输出：** 交替组调用实际注册 GEMM 的 mapping probe，检查全部物理 tile 覆盖和目标归属；原顺序组通过冻结源码/注册身份与 GPU kernel profile 核验。六组各自比较输出与独立矩阵乘法参考，不能只检查 Python 中的 `interleaved=true`。
- **生命周期与 stream：** 非默认 CUDA stream 上提交，等待通信依赖；连续调用并验证已返回输出不被复用缓冲覆盖。检查 retained output 与 encode/send/receive/decode 的数据关联。
- **实际执行：** GPU profile 关联真实 CUDA kernel 与 CPU launch，确认 Flux GEMM、INT8 encode、NCCL 字节交换及 decode/reduce 均执行。BF16 组必须出现 BF16 ReduceScatter，且不得出现 INT8 codec。排序组还必须通过其隔离库和 GEMM 映射证据验证。

算子计时在预热后分别测 GEMM、encode、NCCL exchange、独立 decode、融合 decode/reduce 以及完整投影，提供 CUDA event 时间与同步 wall 时间；BF16 组测 GEMM 和完整投影。每组件为 3 次预热，再采集 5 个窗口、每窗口 10 次；汇总先取同期八 rank 最大值，再对窗口取均值。NCCL 的事件通过 Work.wait 的 caller-stream 依赖覆盖实际通信。分段计时及 profiler 会扰动执行，不能把分段时间简单相加代替完整算子或完整训练步。另记录协议逻辑字节量及算子显存，不以理想压缩比推算加速比。

六组独立算子微基准统一在 `torch.no_grad` 下测前向，不把创建反向图的差异混入该表；Megatron 仍使用 trainable-weight 的原 linear 分支和真实 TP process group。完整训练步保留 autograd、原训练循环和各组对应的反向路径。原生组的审计包装与调用计数仍有主机开销，因此这里的 Megatron 数字对应本协议的可审计原生路径。

### 完整训练步门禁

模型固定为 GPT-6.7B 规模：32 层、hidden 4096、FFN 16384、32 heads、sequence 2048、micro batch 1、global batch 4、BF16、TP8/DP1；每 optimizer step 为 **8192 tokens**。使用原冻结训练、反向、优化器和计时函数，检查其 AST；固定初始参数、输入和 RNG 身份。新路径只注入指定行并行投影前向。

完整模型 smoke 检查全部 64 个目标模块覆盖、前向有限值、投影输出、loss、梯度范数与参数更新；记录真实 codec/集合通信调用和输出生命周期。对压缩组的投影比较采用独立 BF16 参考和独立 codec 参考，预算分别为相对 L2 **5%** 和 **1%**；这些是本协议短期数值门禁，不是训练收敛保证。原始 BF16 组单独对未量化矩阵乘法＋NCCL 参考进行验证。

六组每次窗口从相同初始化开始。量化与原 NCCL ReduceScatter 的归约舍入存在差别，因此不同组的 loss/梯度/参数轨迹不要求逐位相同；报告保留差值及指纹。各 INT8 组按相同量化定义验证，排序后的数学内容必须不变，轨迹差异需要如实呈现，不能为了通过而把明显误差解释为排序收益。

完整模型 profile 独立于正式计时，必须确认六组真实执行路径和目标模块次数，再进入性能阶段。不能仅用独立算子 profile 推断训练模型走了相同路径。

## 6. 计时、统计与结果说明

正式窗口每组采用 **10 步预热＋50 步正式 optimizer step**，计时包含前向、反向、通信、optimizer 更新和清梯度。每个完整 block 包含六组，同一 trial 内按计划配对，保留全部有效窗口和负收益。

| 指标 | 定义 |
| --- | --- |
| 每窗口步时 | `max(八 rank elapsed_seconds) × 1000 / 50` |
| 主表步时 | 该组所有正式窗口的算术均值；另报中位数和最慢窗口 |
| tokens/s | `8192000 / 平均 ms_per_step` |
| 压缩路径降幅 | `100 × (1 − mean(flux-int8) / mean(flux-bf16))` |
| Megatron 量化降幅 | `100 × (1 − mean(megatron-int8) / mean(megatron-bf16))` |
| 远端排序增量降幅 | `100 × (1 − mean(flux-int8-remote-first) / mean(flux-int8))` |
| 交替排序增量降幅 | `100 × (1 − mean(flux-int8-interleaved) / mean(flux-int8))` |
| 组合降幅 | `100 × (1 − mean(flux-int8-interleaved) / mean(flux-bf16))` |
| 配对不确定性 | 按完整 block 的对数步时比 bootstrap；至少 6 块，固定负载内的区间，不外推其他模型/拓扑 |
| 显存 | 所有有效正式窗口/八 rank 最大 allocated、reserved、native 及逐 rank allocated＋native |

`--trials 1 --blocks 1` 共 6 个正式窗口、48 份正式 rank 记录、300 个正式 optimizer step；`--trials 1 --blocks 6` 共 36 个正式窗口、288 份正式 rank 记录、1800 个正式 optimizer step。以上均不含预热、smoke、operator probe 和 profile。组件测量、profiling 与正式完整步数据分开保存和展示。

每六块采用 Williams 顺序：每组在每个位置各出现一次，每个有向相邻组对各出现一次。报告同时保留六组全部 15 对比较，均值比降幅与配对对数比降幅分列，置信区间不作多重比较校正；不能以其中最快的一次窗口替代全部窗口统计。

每批次保留配置/执行顺序、源文件与二进制摘要、硬件与软件身份、八卡通信预检、逐 rank operator 原始记录、packet 哈希、映射与数值结果、六组模型记录和 GPU trace，最终生成 `report.json` 与 `report.md`。主要文件如下：

| 文件/目录 | 用途 |
| --- | --- |
| `batch-state.json` | 当前阶段及 `completed` / `failed` / `preflight_completed` / `operator_completed` 状态 |
| `plan.json`、`workers.json`、`handoff/` | 执行计划、生成 worker 身份、controller 源码快照 |
| `environment-preflight.json`、`nccl-preflight/` | 硬件/软件、拓扑、八 rank 通信检查 |
| `build/open-compress-build.json`、`build/compile.log` | 三套 Flux 和 codec 的构建身份与编译日志；复用构建时路径见 `plan.json` |
| `operator/<arm>/rank*.json`、`rank*-profile.json.gz`、`process.log` | 独立算子的 oracle、packet、mapping、生命周期、计时与原始 profile |
| `operator-summary.json` | 算子各组件的 CUDA/wall 时间与显存汇总 |
| `results/smoke-<arm>/`、`results/profile-<arm>/` | 完整模型 smoke 和 profile 的 rank 记录与进程日志 |
| `results/compare<trial>/block<编号>-<arm>/`、`samples.json` | 正式窗口原始记录、`process.log` 与逐窗口汇总 |
| `profile-summary.json`、`report.json`、`report.md` | 训练路径审计、最终机器可读与 Markdown 报告 |

报告必须能够区分完成、失败和未运行，缺失 rank/窗口/路径证据时不得生成已完成性能结论。`--report` 重新核对 rank 记录、哈希、operator/profile 和统计，依赖原批次引用的构建/源码可访问；只回传 `report.md` 无法完成该复核。六组 v2 使用新协议和新目录，不把 v1 已跑窗口并入同期统计；v1 报告继续配合原 v1 源码版本复核，新入口对 v1 `--report` 给出明确版本提示。

### 当前证据表

| 项目 | 当前证据与状态 |
| --- | --- |
| QUTLASS 是否有 RTX 5090 路径 | 本地及冻结上游源码确认 SM120a、MXFP4/NVFP4 和 RTX 5090 benchmark；未在本地编译/运行。 |
| QUTLASS 与现有 PyTorch 2.8 环境 | 当前 checkout 的 PyTorch ≥2.11 构建门限和 stable C++ API 已确认；未降低门限或升级目标环境。 |
| INT8 逻辑载荷 | 按固定 packet 尺寸计算为远端载荷减少 48.4375%；不是实测链路收益。 |
| 新代码本地检查 | **v2 的 59 项 CPU 测试通过**，包含六组 worker 初始化、Megatron 原生函数及反向保留、远端 tile 映射、profile 拒绝错误路径、旧构建只读复用和六组统计。远端优先 header 的最小 CUDA kernel 用 CUDA 12.8 编译为 SM120 成功；入口预览、shell 语法及 diff 检查通过。命令、输出及源码摘要见 [v2 本地验证记录](local-validation-v2.json)。此处未安装 PyTorch、无可运行 GPU，完整 Flux 构建及 GPU 验证待目标机。v1 的 35 项检查记录保留在 [v1 本地验证记录](local-validation.json)，仅对应原版本。 |
| 原三组 v1 目标运行 | 用户确认运行完成，结果尚未拉取；没有在本地复核结果或新增实测数值。 |
| 新六组 codec、八卡 operator 与真实路径 | 上述 SL3061 批次的六组算子门禁与 profile 已通过并离线复核；完整六组训练路径仍待修复后执行。 |
| 新六组完整步耗时、吞吐与显存 | **待目标 RTX 5090 执行；没有可报告的 ms/step、GiB 或加速比。** |
| 压缩能降低完整步耗时 | 待验证假设；编码解码、staging 和集合通信选择可能使其变慢。 |
| 交替排序有额外收益 | 待验证假设；当前整 GEMM 后发包，不具有原 tile 级到达重叠前提。 |

历史 `design-5090` 与 `Load-5090` 的 TACO 结果仅解释目标机条件和实验方法，不与本协议合并统计。后续目标机回传新批次后，应同时登记批次路径、源码/构建身份、完整门禁状态、六组全部窗口、量化误差、配对结果和显存范围，再填写性能表。

本地可重复运行不依赖 GPU 的新增检查：

```bash
python3 -B -m unittest discover -s tools/5090/phase6 -p 'test_open_compress*.py' -v
bash -n tools/5090/phase6/run_open_compress.sh
bash tools/5090/phase6/run_open_compress.sh --place 61 --trials 1 --blocks 1
```

## 7. 后续组合方向

按 [idea.md](../design-h100/idea.md) 与 [idea-combine.md](../design-h100/idea-combine.md)，后续保留两个独立扩展点：生产映射使用 `(source, physical_tile)` 的稳定身份；通信格式显式记录每份物理贡献的表示，排序与表示选择不能混用逻辑槽位索引。

到达优先需要在 **5090 当前传输路径**上重新定义可观测到达事件并采样、留出验证，再冻结计划；当前整包 NCCL staging 不能直接复用 H100 的跨卡发布标记或到达表。若引入分块发送/消费，还要先验证线程/stream 生命周期与通信重叠。

选择性量化可在新协议中为来源/物理 tile 增加量化 mask、混合 BF16/INT8 packet 与匹配解码；仍保留本地 BF16，并单独冻结预算、精度与元数据成本。本阶段代码与报告中六组均不加载这类 mask，不把未来可能收益计入当前压缩或排序的结论。
