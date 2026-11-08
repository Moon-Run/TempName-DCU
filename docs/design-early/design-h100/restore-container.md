# 新容器恢复：复用持久化 conda，补齐 H100 编译环境

更新：2026-10-24。适用于原容器已删除、`/workspace/TempName` 和 `/workspace/conda-envs/flux-h100-ablation3-20261017` 仍从宿主机挂载到新容器的情况。入口为 [restore_container.py](../../../tools/h100/phase6/restore_container.py)。

当前报错 `ValueError: Build CUDA Toolkit missing` 来自启动器查找 `/usr/local/cuda-12.8/bin/nvcc`，发生在创建批次和编译之前。原环境使用 CUDA12.8 编译 Flux/TACO，CUDA12.4 用于 Apex 与训练；持久化 conda 中的 PyTorch/cu124 不等于容器内已经安装完整 CUDA12.8 Toolkit。`nvidia-smi` 显示的 CUDA 版本也不是已安装的 nvcc 版本。

## 1. 在新容器中执行

同步本次脚本后，在 `/workspace/TempName` 中运行：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/restore_container.py --execute
```

脚本不会启动训练。恢复成功后，运行生成的包装脚本，进行一次三组比较：

```bash
bash outputs/h100/container-recovery/run_a_baseline.sh --trials 1 --execute
```

该包装脚本先加载固定编译环境，再调用 `detach_a_baseline.py`；仍是 **36 个正式窗口、1800 个正式 optimizer step**。省略 `--trials 1` 时，实验默认进行两次比较。

仅检查已安装版本并查看恢复计划：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/restore_container.py
```

默认预览只执行 CPU 软件/文件检查，不安装、不创建输出、不查询 GPU。若指定 conda 解释器不存在，应先恢复其挂载或用 `--python` 指向正确环境；脚本不会删除、重建一个现有 conda 环境。

## 2. 自动检查与恢复范围

| 组件 | 行为 |
| --- | --- |
| Python | 核对原持久化 Python3.10.20，可不依赖 `conda` 命令 |
| CUDA12.4 / 12.8 | 检查实际 `nvcc --version`、开发头文件与 libcudart；优先复用，缺少时从 NVIDIA 软件源安装对应 Toolkit |
| 系统构建工具 | 补缺少的 GCC/G++11、Make、Git、curl、证书、pkg-config、binutils 等 |
| Python 包 | 按 [部署记录](deploy-tp8.md) 检查 PyTorch2.6.0+cu124、NumPy1.26.4、CMake3.30.0、Ninja1.11.1.3 及固定基础依赖；缺失才安装 |
| Apex | 三个编译扩展及 FusedAdam/FusedLayerNorm 可导入时复用；缺失或失效时，从固定提交重编所需扩展 |
| 源码 | 校验固定 CUTLASS/NCCL 子模块；缺失时初始化。Megatron 缺失时获取固定提交；已有版本错误或跟踪文件有改动时停止，不覆盖 |
| 编译检查 | 使用 CUDA12.8＋GCC11 编译一个 SM90 对象，不执行 kernel |
| GPU 检查 | 默认执行时检查驱动注入、八卡 H100/132SM 和两两 peer access，不启动训练 |

系统包自动安装支持 Ubuntu20.04/22.04/24.04 x86_64，并要求容器内 root。以原 Ubuntu22.04 环境为优先；如果某发行版没有 CUDA12.4 包，脚本会保留 APT 诊断并停止，不会以其他 CUDA 版本代替。已经安装完整依赖的环境无需安装系统包。

已有包版本与记录不一致时，默认报告差异。先确认指定的是原 conda 环境；确实需要恢复这些包时可以显式使用：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/restore_container.py --repair-python --execute
```

这会将不同版本的依赖恢复到固定版本；若安装/更改 PyTorch，则重新编译 Apex，避免沿用另一 Torch 版本的扩展。不会安装 PyPI 的同名 `apex` 包，Apex 使用固定源码 `e13873debc4699d39c6861074b9a3b2a02327f92`，保留 `apex_C`、`amp_C`、`fused_layer_norm_cuda` 三个扩展。

不需要额外安装 Transformer Engine、FlashAttention、torchvision、torchaudio、`megatron-core` wheel 或 NVSHMEM。a-baseline 构建器会从仓库 NCCL 子模块编译开发库，恢复脚本不以系统 `libnccl-dev` 替换 PyTorch wheel 的 NCCL。

## 3. CUDA 安装与编译变量

缺失 Toolkit 时使用版本明确的 `cuda-toolkit-12-4` / `cuda-toolkit-12-8`，按照 NVIDIA 的 keyring/APT 流程接入软件源。这类 Toolkit 包不包含驱动；脚本同时检查 APT 事务，不接受驱动包安装或软件包删除。参考 [NVIDIA CUDA12.8 Linux 安装指南](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/index.html#network-repo-installation-for-ubuntu)及其 [Toolkit 元包说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/index.html#meta-packages)。

容器内不安装或升级宿主机 NVIDIA 驱动。若 `nvidia-smi`、`libcuda.so.1` 或 GPU 注入缺失，需要在宿主机启动容器时恢复 GPU 挂载，例如沿用 `--gpus all --ipc=host`；这不能靠 pip 或容器内安装驱动修好。GPU 可见性检查也不代替后续实际通信与训练预检。

恢复成功后生成：

```text
outputs/h100/container-recovery/
  env.sh                  固定编译/运行变量与精确 Git 信任路径
  run_a_baseline.sh        加载 env.sh 后调用原实验启动器
  last-report.json         最近一次恢复状态
  runs/<北京时间戳>/
    install.log            安装、导入与编译检查输出
    report.json            本次计划、命令及完成/失败状态
    apt-plan.txt           需要安装时保存的 APT 事务
    compile-probe.cu/.o    SM90 编译探针
```

`env.sh` 设置 GCC/G++11、`DEBUG=0`、空 `MAKEFLAGS`、`PYTHONNOUSERSITE=1`、`MAX_JOBS=2`，并固定实际 CUDA12.4 运行路径。**不能只在恢复脚本的子进程中设置变量**，所以后续推荐使用包装脚本；也可以先 `source outputs/h100/container-recovery/env.sh` 再手动启动。

`DEBUG=release` 在固定 NCCL Makefile 中会被当作调试构建，可能产生 `-O0 -G`；生成的环境明确使用数字 `0`。脚本解析 Toolkit 的真实目录，防止安装12.8时通用 `/usr/local/cuda` 链接改向后，把 Apex/运行环境误切到12.8。

## 4. 自定义路径与失败处理

如依赖已安装在其他位置，可显式指定：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python tools/h100/phase6/restore_container.py \
  --cuda-home /path/to/cuda-12.4 \
  --build-cuda-home /path/to/cuda-12.8 \
  --megatron /workspace/Megatron-LM \
  --execute
```

显式路径必须已有正确且完整的 Toolkit；需要自动安装时不传对应路径。系统包下载需要新容器能访问 Ubuntu/NVIDIA 软件源；只有缺少 Python 包或 Apex/Megatron 源码时才需要相应 PyPI/PyTorch/GitHub 网络访问。

只恢复软件、暂不检查 GPU 时可增加 `--skip-gpu-check`，报告会明确 GPU 未检查。恢复失败时查看 `last-report.json` 和本次 `install.log`，修复具体问题后重新运行；每次创建新的日志目录，已满足要求的依赖会复用。

本地 **15 项 CPU 检查全部通过**，覆盖依赖复用、驱动包拒绝、CUDA 版本与软链接、失败停止、无副作用预览和实际 shell 参数转义。见 [验证登记](../../../outputs/h100/phase6/container-recovery-validation-20261024/validation.json)及[测试日志](../../../outputs/h100/phase6/container-recovery-validation-20261024/cpu-tests.log)。验证只使用 mock、临时文件和命令预览，不在当前开发主机运行 APT/pip 安装，也不代表已修好远端新容器。实际安装和 GPU 状态以用户在新容器执行产生的报告为准。
