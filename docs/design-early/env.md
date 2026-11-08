# 软件环境与新服务器配置

环境核验日期：2026-10-06；Megatron 版本与镜像选择说明补充于 2026-10-09。本文记录本仓库当前 Phase4–Phase6 实验实际使用的软件环境，以及新服务器 clone 后需要配置的依赖；不涉及 GPU 型号、卡数、拓扑或 Slurm 配置。本次只检查现有环境，没有安装软件或启动实验。

**主要环境是 Conda + Python 3.10.20 + PyTorch 2.6.0（cu124）+ NVIDIA Apex；编译本仓库的 Flux/TACO 扩展还需要 CUDA Toolkit、GCC/G++、CMake 和 Ninja。** 新机器可以使用 Miniconda 或 Miniforge，重点是环境内的版本，不要求使用旧机器的 Conda 安装目录。

**Megatron-LM 必须固定为 `core_v0.12.3`，精确提交为 `3ea68ad6042cc1204386ae9364358f7c4de1bc37`。** 获取与校验命令见第 5 节；74 服务器现有镜像的选择建议见第 8 节。

**2026-10-17，新服务器跑H100消融3：以 [H100 TP8 部署指南](design-h100/deploy-tp8.md) 为安装和执行主入口。** 本文保留公共依赖及A800历史环境；不要把两份安装脚本依次执行。H100采用Python 3.10.20、PyTorch 2.6.0+cu124、固定Apex/Megatron源码、GCC/G++ 11、CMake **3.30.0**、Ninja 1.11.1.3。下文A800示例的CMake 4.4.3不用于覆盖H100环境。容器基础镜像中的默认Python/PyTorch不能代替这些版本，安装后须实际导入Apex三个编译扩展并记录解释器路径。

H100的Flux构建使用CUDA 12.8，Apex及训练环境使用CUDA 12.4；编译前设置`DEBUG=0`、清空`MAKEFLAGS`，固定`CC`、`CXX`、`CUDAHOSTCXX`。`DEBUG=release`会被固定NCCL Makefile识别为调试构建。按部署指南构建一次NCCL 2.19.3 Release开发库，采样/base/arrival统一通过`--nccl-root`引用；这是项目编译依赖，与PyTorch wheel的NCCL 2.21.5运行依赖分开。新主机重新校准Attention和MLP并完整编译库，不能仅凭拉取的旧日志或源码判断环境就绪。

## 1. 当前实际使用的环境

| 用途 | 当前位置 | 主要版本与说明 |
| --- | --- | --- |
| Megatron 训练、完整 step 测量 | `/data/run01/scyb672/conda_envs/flux-megatron-a800` | Conda 环境；Python 3.10.20、PyTorch 2.6.0+cu124、NumPy 1.26.4、编译版 Apex |
| 历史 Flux/TACO 编译、CPU 分析 | 仓库内 `outputs/a800/venv` | Python 3.10.20；PyTorch 2.6.0+cu124；CMake 4.4.3、Ninja 1.13.0；分析包见第 6 节 |

第二个环境的 `pyvenv.cfg` 设置了 `include-system-site-packages = true`，底层 Python 来自 `/data/home/scyb672/.conda/envs/mmseg`，PyTorch 也从该环境继承。它不是独立、可直接复制的环境。**新服务器不需要安装 mmseg，建议新建一个独立 Conda 环境完成训练和扩展编译。** 分析/作图包可以单独安装。

训练环境中已核对的版本（按实际启动器设置 `PYTHONNOUSERSITE=1`，排除用户目录包）：

| 组件 | 实际版本 | 用途 / 备注 |
| --- | --- | --- |
| Python | 3.10.20 | conda-forge 构建 |
| PyTorch | 2.6.0+cu124 | `torch.version.cuda == "12.4"` |
| Megatron-LM / Megatron Core | `core_v0.12.3` | 使用固定提交的源码，完整 SHA 和校验方式见第 5 节 |
| Triton | 3.2.0 | 随该版本 PyTorch 安装 |
| NVIDIA Apex | 0.1，源码 tag `25.04` | 需要编译扩展，不能只安装 Python 部分 |
| NumPy | 1.26.4 | 训练环境版本 |
| einops / regex | 0.8.1 / 2024.11.6 | 模型及文本相关基础依赖 |
| psutil / PyYAML | 6.1.1 / 6.0.2 | 进程信息、配置读取 |
| tqdm / six | 4.67.1 / 1.17.0 | 基础依赖 |
| pybind11 | 2.13.6 | C++ / Python 扩展 |
| setuptools / wheel | 75.8.0 / 0.45.1 | 构建工具 |
| Ninja | 1.11.1.3 | 训练环境内的版本；旧编译 venv 为 1.13.0 |
| packaging / pip | 24.2 / 26.2.1 | 主 Conda 环境内的实际版本 |
| NCCL / cuDNN | 2.21.5 / 9.1.0.70 | PyTorch pip 依赖中的运行库 |
| PyTorch C++ ABI | `_GLIBCXX_USE_CXX11_ABI = False`（0） | 本地扩展必须与所用 PyTorch 匹配 |

`PYTHONNOUSERSITE=1` 需要保留。未设置时，旧机器会加载 `~/.local/lib/python3.10/site-packages` 中的 packaging 26.2 和 pandas 等包，且 `pip check` 会报 pandas 缺少 pytz；按训练入口禁用用户目录包后，主环境的 `pip check` 通过。这些用户目录包不属于需要迁移的训练依赖。

当前环境没有安装 `transformer-engine`、`flash-attn`、`megatron-core` wheel、`tensorboard`、`tiktoken` 或 `sentencepiece`。当前入口使用 Megatron 源码、`--transformer-impl local`、`--mock-data` 和 `NullTokenizer`，不需要为这一实验配置额外安装这些包；更换模型实现、数据或 tokenizer 时再补充相应依赖。

## 2. 新建主 Conda 环境

先安装并初始化 Miniconda / Miniforge，使 `conda` 命令可用。以下命令在新服务器执行；环境名称可以修改。

```bash
conda create -n flux-megatron-a800 -c conda-forge python=3.10.20 pip -y
conda activate flux-megatron-a800
export PYTHONNOUSERSITE=1

python -m pip install \
  pip==26.2.1 setuptools==75.8.0 wheel==0.45.1 packaging==24.2

python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install \
  numpy==1.26.4 einops==0.8.1 regex==2024.11.6 \
  psutil==6.1.1 PyYAML==6.0.2 tqdm==4.67.1 six==1.17.0 \
  pybind11==2.13.6 ninja==1.11.1.3

# A800历史配置；H100消融3改用deploy-tp8.md中的cmake==3.30.0。
python -m pip install cmake==4.4.3
```

PyTorch 2.6.0 的 cu124 安装源见 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。本仓库当前训练流程不需要 `torchvision`、`torchaudio`。NCCL、cuDNN、CUDA runtime 等 wheel 依赖由上述 PyTorch 安装自动解析，不另外手动替换它们的版本。

这里选择在主环境补齐编译工具，避免继承旧 mmseg 环境；这是一套新环境配置方案，不表示已经在新服务器完成编译验证。

## 3. CUDA Toolkit 与 C++ 编译工具

当前记录中有两套 CUDA Toolkit，作用不同：

| 软件 | 原机器位置 / 版本 | 实际用途 |
| --- | --- | --- |
| CUDA Toolkit 12.4 | `/data/apps/cuda/12.4`；nvcc V12.4.99 | Apex 编译、训练启动环境 |
| CUDA Toolkit 12.8 | `/data/apps/cuda/12.8`；nvcc V12.8.61 | 自定义 Flux/TACO CUDA 扩展编译 |
| GCC / G++ | `/usr/bin/gcc`、`/usr/bin/g++`；11.4.0 | C++ 与 CUDA host 编译器 |
| CMake | 历史 Flux 构建使用 4.4.3；系统另有 3.30.0 | 优先使用 Conda 环境内已安装的版本 |
| Ninja / Make / Git | Ninja 版本见上表；Make、Git 为系统工具 | 构建与源码获取 |

**PyTorch 的 cu124 wheel 不包含完整的 nvcc 编译工具链。** 安装 PyTorch 后仍需单独配置 CUDA Toolkit。`torch.version.cuda` 为 12.4，而编译 Flux 时 `nvcc --version` 为 12.8，是现有构建记录的实际组合，不应把两个值混为一谈。

设置本机路径，例如：

```bash
export CUDA124=/usr/local/cuda-12.4
export CUDA128=/usr/local/cuda-12.8

# 编译 Apex、启动训练时使用。
export CUDA_HOME="$CUDA124"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export MAX_JOBS=2
```

这些路径需要替换为新服务器的实际安装目录。Python 版本字符串中的 GCC 14.3.0 是 Python 自身的构建信息，不是本项目 CUDA 扩展实际使用的 G++ 版本。

## 4. 安装 NVIDIA Apex

源码仓库为 `NVIDIA/apex`，当前使用 tag `25.04` 对应的提交：

```text
e13873debc4699d39c6861074b9a3b2a02327f92
```

必须在上面已经安装 PyTorch 的同一个 Conda 环境中编译。不要用 `pip install apex` 从 PyPI 获取同名包。

```bash
# 将路径改为新服务器上 clone 后的仓库目录。
cd /path/to/TempName
export REPO="$PWD"
export APEX_SRC="$(dirname "$REPO")/apex-flux"

git clone https://github.com/NVIDIA/apex.git "$APEX_SRC"
git -C "$APEX_SRC" checkout e13873debc4699d39c6861074b9a3b2a02327f92

export CUDA_HOME="$CUDA124"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=2
```

已有训练环境使用的是裁剪编译，只保留 `apex_C`、`amp_C`、`fused_layer_norm_cuda` 三个扩展，为 `FusedAdam` 和 `FusedLayerNorm` 提供支持；标准安装还会编译其他 Apex 扩展。

若希望与原来的三扩展编译范围一致，可在执行 pip 安装之前，在刚 checkout 的 Apex 源码上执行以下补丁：

```bash
python - <<'PY'
import os
from pathlib import Path
p = Path(os.environ["APEX_SRC"]) / "setup.py"
s = p.read_text()
old = "    ext_modules=ext_modules,"
new = ('    ext_modules=[ext for ext in ext_modules if ext.name in '
       '{"apex_C", "amp_C", "fused_layer_norm_cuda"}],')
assert s.count(old) == 1, "Apex setup.py 与固定版本不一致"
p.write_text(s.replace(old, new))
PY
```

然后执行编译安装，参数参考 [NVIDIA Apex 官方安装说明](https://github.com/NVIDIA/apex)：

```bash
python -m pip install -v --disable-pip-version-check \
  --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" "$APEX_SRC"
```

旧 wheel 位于 `outputs/a800/megatron-e2e/wheelhouse/`，该目录不随 Git clone 迁移；新机器按固定源码重编译即可。

## 5. Megatron 与本仓库扩展

**本仓库当前实验固定使用 Megatron-LM 的 `core_v0.12.3` tag，并以提交 SHA 为最终校验依据：**

```text
3ea68ad6042cc1204386ae9364358f7c4de1bc37
```

2026-10-09 已核对相邻目录 `../Megatron-LM` 的 `HEAD` 和 `core_v0.12.3^{commit}`，二者均为上述提交。Megatron 通过源码目录加入 `PYTHONPATH`，不另外安装 `megatron-core` wheel，也不跟随 `main` 或自动升级到其他版本。现有 [实验配置](../../tools/a800/phase4/e2e/config.json) 固定了这个 SHA，[启动器](../../tools/a800/phase4/e2e/launch.py) 会检查源码 `HEAD`，版本不同会直接触发断言。

```bash
export MEGATRON_DIR="$(dirname "$REPO")/Megatron-LM"
git clone https://github.com/NVIDIA/Megatron-LM.git "$MEGATRON_DIR"
git -C "$MEGATRON_DIR" checkout --detach 3ea68ad6042cc1204386ae9364358f7c4de1bc37
test "$(git -C "$MEGATRON_DIR" rev-parse HEAD)" = \
  "3ea68ad6042cc1204386ae9364358f7c4de1bc37"

git -C "$REPO" submodule update --init --recursive
export PYTHONPATH="$MEGATRON_DIR:$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
```

仓库锁定的子模块为 CUTLASS `df8a550d3917b0e97f416b2ed8c2d786f7f686a3`、NCCL `8c6c5951854a57ba90c4424fa040497f6defac46`。这里的 NCCL 是编译所需的源码依赖，与 PyTorch wheel 携带的 NCCL 运行库应分别看待。

编译本仓库的 Flux/TACO 时，在已激活的主环境中切换工具链：

```bash
export CUDA_HOME="$CUDA128"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export FLUX_FORCE_BUILD=1
export FLUX_SHM_USE_NVSHMEM=0
export CMAKE_BUILD_PARALLEL_LEVEL=2
export MAX_JOBS=2
```

现有自定义构建使用 `ENABLE_NVSHMEM=OFF`、`WITH_PROTOBUF=OFF`，不需要额外安装 NVSHMEM 或 protobuf。后续编译应保留这些设置，并使用同一 PyTorch 环境重新生成 `.so`；不能仅安装上游 `byte-flux` wheel 来替代本仓库修改后的扩展。

构建流程见 [gemm_rs_validation/build.sh](../../tools/a800/gemm_rs_validation/build.sh) 和 [Phase6 构建说明](../../tools/a800/phase6/README.md)。这些脚本含旧机器的路径，不能原样当作跨机器安装器。根目录 `build.sh` 最后使用 `setup.py develop --user`，在隔离 Conda 环境中需调整这一步；现有自定义构建使用 `python setup.py build_ext --inplace`，然后通过所构建目录的 `python/` 加载包。只运行这条 Python 命令不会代替前面的 CMake/NCCL 构建。

完成编译后，训练入口恢复 `CUDA_HOME` / `CUDACXX` 到 12.4，并使用对应扩展目录设置 `PYTHONPATH`、`LD_LIBRARY_PATH`；Phase6 启动器已有这些设置。常用运行变量是 `PYTHONNOUSERSITE=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`CUDA_DEVICE_MAX_CONNECTIONS=1`。

## 6. 可选的数据分析环境

旧 `outputs/a800/venv` 默认可见的分析包为 NumPy 2.2.6、SciPy 1.15.3、pandas 2.3.3、Matplotlib 3.10.9。其中 NumPy/SciPy 来自底层 mmseg 环境，pandas/Matplotlib 来自用户目录，并非全部安装在该 venv 内。纯 CPU 数据整理、统计和作图可另建独立环境，避免改变训练环境的 NumPy 版本：

```bash
conda create -n flux-analysis -c conda-forge python=3.10.20 pip -y
conda activate flux-analysis
export PYTHONNOUSERSITE=1
python -m pip install \
  numpy==2.2.6 scipy==1.15.3 pandas==2.3.3 matplotlib==3.10.9
```

此环境只供分析使用；需要导入 torch、Apex 或 Flux 扩展的检查仍使用主环境。`pytest` 不在已核对的两个环境中，日常训练无需为了复刻环境而安装它。

## 7. 安装后检查

以下只检查软件版本与导入，不运行训练实验：

```bash
conda activate flux-megatron-a800
export PYTHONNOUSERSITE=1
python -m pip check
python - <<'PY'
import sys
import torch
import numpy
import apex_C, amp_C, fused_layer_norm_cuda
from apex.optimizers import FusedAdam
from apex.normalization import FusedLayerNorm
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("C++ ABI:", torch._C._GLIBCXX_USE_CXX11_ABI)
print("NumPy:", numpy.__version__)
print("Apex compiled extensions: OK")
PY
"$CUDA124/bin/nvcc" --version
"$CUDA128/bin/nvcc" --version
g++ --version
cmake --version
ninja --version
```

预期核心值：Python 3.10.20、PyTorch 2.6.0+cu124、PyTorch CUDA 12.4、C++ ABI 为 `False`、NumPy 1.26.4，三个 Apex 扩展成功导入。本次已在旧训练环境核验这些值和 Apex 导入，并确认禁用用户目录包后 `pip check` 通过；新环境安装及扩展重编译尚未执行。

`outputs/`、`logs/`、虚拟环境和编译产物均被 Git 忽略，clone 不会带上它们。部分实验启动器还固定了 `conda_envs/flux-megatron-a800/bin/python` 相对位置，迁移时需改为新环境的 `$CONDA_PREFIX/bin/python`；配置好依赖不等于旧实验的冻结库、mask 和结果目录已经恢复。

本页依据：现有两个 Python 环境查询、`outputs/a800/megatron-e2e/requirements-resolved.txt`、Apex 裁剪补丁、Flux 的 CMakeCache，以及仓库 [训练环境设置](../../tools/a800/phase4/e2e/campaign.py)、[训练入口](../../tools/a800/phase4/e2e/launch.py)、[构建脚本](../../tools/a800/gemm_rs_validation/build.sh)。旧输出目录中的快照只作为本机核验依据，不是新 clone 的必需文件。

## 8. env-74.md 中的镜像怎么选

**以复现本仓库现有软件版本、减少环境改动为目标，首选列表中的这个基础镜像：**

```text
swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

用户此前提供的 `env-74.md` 镜像清单记录的 IMAGE ID 为 `05d1b981bb5b`，大小 13.3 GB。该原始清单当前未包含在此文档目录中，本节保留已整理的选择依据。这里仅根据用户提供的镜像名称、tag 和公开版本说明作选择，未访问另一台服务器，也未核验镜像内部实际内容。

判断依据是标签明确标出了 CUDA 12.4、cuDNN 9 和 `devel` 开发环境，与当前 PyTorch cu124 及需要编译 Apex/Flux 的用途较接近。**镜像自带的 PyTorch 2.5.1 仍不等于目标版本；它适合作为配置起点，不能直接视作完整复现环境。**

选定后仍需完成以下软件配置：

1. 在镜像内使用 Conda 新建 Python 3.10.20 环境，按第 2 节安装 `torch==2.6.0` 的 cu124 wheel 和其他固定依赖，不继承镜像原来的 Python 包。
2. 保留或补齐 CUDA Toolkit 12.4，并另行配置 CUDA Toolkit 12.8；仅凭镜像标签不能认为 12.8 已存在。
3. 按第 4 节编译固定版本 Apex，按第 5 节获取固定提交的 Megatron-LM，并在新环境中重新编译本仓库扩展。

其他候选的取舍：

| 镜像 / 类别 | 对当前仓库的判断 |
| --- | --- |
| `nvcr.io/nvidia/pytorch:25.04-py3` | 可作为开发基础，但不是现有版本组合。官方配置为 Python 3.12、PyTorch 2.7.0a0、CUDA 12.9.0；复现时需要调整更多依赖，不作为首选 |
| `nvcr.io/nvidia/pytorch:24.10-py3` | 官方配置为 PyTorch 2.5.0a0、CUDA 12.6.2，也不如上述 CUDA 12.4 的 devel 镜像贴近现有环境 |
| `1sci:*`、`pcged-training:latest` 等自定义镜像 | 名称不足以确定 Python、CUDA Toolkit、Apex 和 PyTorch 的完整组合，仅看列表不优先选择 |
| 带 `vllm` / `sglang` 的镜像 | 名称指向推理服务用途；当前任务需要固定 Megatron 训练及自定义编译环境，不优先选择这些带额外服务依赖的镜像 |

NGC 版本差异依据 [NVIDIA PyTorch 25.04 发布说明及历史版本表](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-04.html)。Apex 的源码 tag `25.04` 与 NGC 镜像 tag `25.04-py3` 是不同组件的版本标识，不需要因为 Apex 的版本号而选择同名月份的镜像。
