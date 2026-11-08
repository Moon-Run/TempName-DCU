# 新服务器：单机 8 张 H100 部署与九组消融实验3

更新：2026-10-17。本文用于在**新服务器**配置环境、检查运行条件，并按任务范围执行 [消融实验3](消融实验/消融实验3.md)：三种策略 × MLP、Attention、MLP＋Attention 三种作用范围，共九组。模型为 GPT 6.7B、单节点8×H100、TP8，每卡必须为132 SM。无需Slurm；受调度器管理时，在已分配的计算节点内执行GPU步骤。

**本指南已替换原六策略运行流程，全程使用 `ablation3.py`。本次按用户要求只跑一轮：`prepare --blocks 9 --rounds 1`，81个正式窗口，另有18个预检窗口。** CLI不传`--rounds`时仍默认两轮162窗口，以兼容旧入口；本页启动命令均显式选择一轮。详细分组和统计口径见 [消融实验3](消融实验/消融实验3.md)，CLI说明及开发验证见 [README-ablation3.md](../../../tools/h100/phase6/README-ablation3.md)。

新服务器必须重新校准 Attention 和 MLP，重新编译实际运行库。本文不复用旧机器的到达计划、mask或二进制。原H100服务器已完成完整构建、九组预检，并回传124个完整正式窗口；第二轮异常中断，尚未完成162窗口，见 [已回传结果及边界](消融实验/base-phase6-ablation3.md)。因此“尚无GPU验证”已不是当前状态，但旧结果不能代替新机器的预检。模型定义见 [模型说明](../model-gpt-6.7b.md)，交付规范见 [执行端说明](instruction-h100.md)。

## 阅读顺序与任务范围

| 文档 | 新H100服务器上的用途 |
| --- | --- |
| 本文 `deploy-tp8.md` | 环境、源码、双角色校准、Release构建和九组运行的主流程 |
| [env.md](../env.md) | 公共依赖与固定Megatron提交；H100编译版本及命令以本文为准 |
| [消融实验3.md](消融实验/消融实验3.md)、[README-ablation3.md](../../../tools/h100/phase6/README-ablation3.md) | 九组定义、CLI、计时和报告协议 |
| [instruction-h100.md](instruction-h100.md) | H100执行端的数据保存与回传约定 |
| [instruction.md](instruction.md)、[A800说明](../design-a800/instruction.md) | 开发分析入口、历史实现背景；不作为本次部署或启动命令 |

本次任务范围为部署后实际完成**一轮81个正式窗口**，不自动追加第二轮或独立复测。如果另一任务只要求“配置环境、查看能否运行”，完成本机双角色校准、完整编译、mapping和九组smoke/profile后停止，交付预检结果；不自动开始正式计时。两种范围都使用消融3入口，不因仓库新增Phase7而切换实验。

## 可直接交给新服务器执行者的指令

> 在实际TempName仓库根目录工作，先阅读 `docs/design/design-h100/deploy-tp8.md`、`docs/design/env.md`、`docs/design/design-h100/消融实验/消融实验3.md`、`tools/h100/phase6/README-ablation3.md` 和 `docs/design/design-h100/instruction-h100.md`。配置新服务器的实验环境，然后实际运行消融3，本次只跑一轮。允许GPU身份检查、双角色BF16校准、mapping、九组smoke/profile及正式计时。核对固定软件版本、源码和子模块、单节点8张132 SM H100、拓扑及空闲状态。新建独立会话，本机重新校准Attention和MLP，显式使用两份新计划；按部署指南以 `DEBUG=0` 构建并共用NCCL Release开发库，完整编译采样库与base/arrival候选库，不复用旧机器二进制。使用 `ablation3.py prepare --blocks 9 --rounds 1`，dry-run确认九组、18个预检和81个正式窗口，共99个worker窗口，再用 `run --stage all --execute` 完成预检及一轮计时。生成严格报告，交付648份正式rank、144份预检rank、72份profile trace和12项角色mapping。不追加第二轮或独立复测；GPU忙时说明原因，不终止其他任务、不创建自动等待队列；运行期间不周期查询GPU占用。失败保留日志、状态和原始记录，不降低检查标准。最后报告环境、会话路径、完成数量、结果摘要及缺项。

若只做可运行性检查，将上述执行阶段改为`run --stage preflight --execute`及`report --preflight`后停止。若以后明确要求两轮，在新compare准备时改为`--rounds 2`；不能对已冻结的一轮或两轮compare修改schedule/manifest切换轮数。第7节给出实际命令。

以上是给**有H100的新服务器**的任务模板。本次开发端只更新指南，不启动GPU实验。

用户在IDE中的 `hjr/TempName/` 是本地工作区前缀；进入新服务器实际仓库根目录后，使用上述 `docs/design/...` 相对路径。文档中有日期的“本次开发未进行GPU测试”描述历史开发交付，不限制后来明确安排的H100测试任务。实验分组以消融实验3为准，不执行旧五组、四组或六策略入口。

## 1. 获取完整源码

先将九组新增代码和本文提交并同步到新服务器使用的仓库版本；开发端未提交的新文件不会随 `git pull` 自动迁移。不要只复制 `tools/h100/`：构建器还读取 `src/`、`include/`、`python/`、共享A800工具、固定候选源码及其manifest。

目录可以按下面的布局安排，实际路径可自行修改：

```text
~/data/gyd-h/
├── TempName/
├── Megatron-LM/
└── apex-flux/
```

```bash
mkdir -p "$HOME/data/gyd-h"
cd "$HOME/data/gyd-h"

# 将占位内容替换成你自己的仓库地址；已有 clone 时跳过这条。
git clone --recurse-submodules '<你的仓库地址>' TempName
cd TempName
export REPO="$PWD"
git submodule update --init --recursive
test -f tools/h100/phase6/ablation3.py
test -f tools/h100/phase6/README-ablation3.md
test -f outputs/h100/phase6/frozen-candidate-v1-20261013/manifest.json
test -f outputs/h100/phase6/gpt67-tp8-20261010-112249/candidates-20261010-153517/manifest.json

export MEGATRON_DIR="$(dirname "$REPO")/Megatron-LM"
# 已有 Megatron-LM 目录时跳过 clone，在干净工作区中核对固定提交。
git clone https://github.com/NVIDIA/Megatron-LM.git "$MEGATRON_DIR"
git -C "$MEGATRON_DIR" checkout --detach \
  3ea68ad6042cc1204386ae9364358f7c4de1bc37
git -C "$MEGATRON_DIR" rev-parse HEAD
git -C "$MEGATRON_DIR" status --short
```

Megatron 必须是上述精确提交，受版本控制的文件保持未修改。Git已携带部分H100日志、trace、配置与冻结源码；编译库、构建缓存和Conda环境通常不会随clone迁移。拉到历史结果不等于新机器已有可运行环境，旧机器的 `.so`、诊断产物和A800到达计划都不作为新机器的运行输入。

容器挂载导致Git报 `detected dubious ownership` 时，先确认目录确属本次可信源码。只对报错的仓库及子模块设置精确的 `safe.directory`，不要使用全局通配符 `*`。例如在当前shell为已确认的四个路径追加配置，随后重试失败的Git步骤；这些环境变量也会传给构建器：

```bash
export GIT_CONFIG_COUNT="${GIT_CONFIG_COUNT:-0}"
for H100_TRUST_DIR in "$REPO" "$REPO/3rdparty/cutlass" "$REPO/3rdparty/nccl" "$MEGATRON_DIR"; do
  export "GIT_CONFIG_KEY_${GIT_CONFIG_COUNT}=safe.directory"
  export "GIT_CONFIG_VALUE_${GIT_CONFIG_COUNT}=$H100_TRUST_DIR"
  GIT_CONFIG_COUNT=$((GIT_CONFIG_COUNT + 1))
done
```

九组构建还需要下列已跟踪文本来源，不能因为位于 `outputs/` 而省略：

- `outputs/h100/phase6/frozen-candidate-v1-20261013/` 的冻结登记及源码副本。
- `outputs/h100/phase6/gpt67-tp8-20261010-112249/candidates-20261010-153517/` 的manifest，以及其中 `remote_arrival_selective/`、`interleaved_arrival_selective/` 两套登记源码。构建器会核验两套来源，即使本轮只测试交替顺序。
- 固定版本的CUTLASS、NCCL子模块及维护代码。链接脚本、`.clang-format`、已登记的Triton输入等也须按manifest保留。

这些历史文件提供可核验的**计算源码模板**，不表示复用旧机器的运行库或校准计划。缺少文件时补齐对应提交的原文件，不改manifest绕过检查。

## 2. 配置软件环境

以已经过本机编译检查的软件组合为起点：

| 组件 | 版本 / 要求 |
| --- | --- |
| Python | 3.10.20，独立 Conda 环境 |
| PyTorch | 2.6.0+cu124 |
| Apex | `e13873debc4699d39c6861074b9a3b2a02327f92`，编译 C++/CUDA 扩展 |
| CUDA Toolkit | 12.4 用于 Apex/运行环境；12.8 用于 Flux/TACO 编译 |
| C/C++ 工具 | GCC/G++ 11、Make、Git、Ninja |
| CMake | 3.30.0；本次 H100 编译使用此版本 |
| Megatron | `core_v0.12.3`，以上述 SHA 为准 |

先让系统具备 Git、GCC/G++、Make 和两套 Toolkit。Ubuntu 22.04 已配置 NVIDIA CUDA 软件源时，Toolkit 软件包名称为 `cuda-toolkit-12-4`、`cuda-toolkit-12-8`；软件源安装方法见 [NVIDIA Linux 安装指南](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/index.html)。已有 Toolkit 时直接使用其实际路径。`nvidia-smi` 显示的 CUDA 版本不能替代 `nvcc --version` 检查。

```bash
conda create -n flux-h100 -c conda-forge python=3.10.20 pip -y
conda activate flux-h100
export PYTHONNOUSERSITE=1

python -m pip install setuptools==75.8.0 wheel==0.45.1 packaging==24.2
python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install \
  numpy==1.26.4 einops==0.8.1 regex==2024.11.6 \
  psutil==6.1.1 PyYAML==6.0.2 tqdm==4.67.1 six==1.17.0 \
  pybind11==2.13.6 ninja==1.11.1.3 cmake==3.30.0

# 按新服务器实际安装位置修改。
export CUDA124=/usr/local/cuda-12.4
export CUDA128=/usr/local/cuda-12.8
export CUDA_HOME="$CUDA124"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX="$CXX"
export MAX_JOBS=2
export DEBUG=0
export MAKEFLAGS=

"$CUDA124/bin/nvcc" --version
"$CUDA128/bin/nvcc" --version
"$CXX" --version
cmake --version
ninja --version
```

PyTorch 安装源与版本见 [官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。这里不需要安装 `torchvision`、`torchaudio`、Transformer Engine、FlashAttention 或额外的 `megatron-core` wheel；当前模型使用源码 Megatron、local transformer、合成数据及 NullTokenizer。

**`DEBUG`必须是数字`0`，不能写成`release`。** 固定NCCL子模块的Makefile把所有非`0`值都当成调试构建，启用`-O0 -G`；原H100部署曾因此重建。CMake的`Release`配置不覆盖这个Make变量。清空继承的`MAKEFLAGS`，用本节固定的编译器，并按第4.1节构建一次共用的NCCL开发库。若已经用错误参数构建，应保留日志、换新构建目录；入口会跳过已标记`built`的子库，仅改环境重跑不足以保证重建。

在同一环境编译 Apex，固定源码并保留实验实际需要的三个扩展：

```bash
export APEX_SRC="$(dirname "$REPO")/apex-flux"
git clone https://github.com/NVIDIA/apex.git "$APEX_SRC"
git -C "$APEX_SRC" checkout --detach \
  e13873debc4699d39c6861074b9a3b2a02327f92

python - <<'PY'
import os
from pathlib import Path
p = Path(os.environ['APEX_SRC']) / 'setup.py'
s = p.read_text()
old = '    ext_modules=ext_modules,'
new = ('    ext_modules=[ext for ext in ext_modules if ext.name in '
       '{"apex_C", "amp_C", "fused_layer_norm_cuda"}],')
assert s.count(old) == 1, 'Apex source differs from the pinned revision'
p.write_text(s.replace(old, new))
PY

CUDA_VISIBLE_DEVICES='' TORCH_CUDA_ARCH_LIST='9.0' \
  python -m pip install -v --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" "$APEX_SRC"

python -m pip check
python - <<'PY'
import sys
import numpy as np
import torch
import apex_C, amp_C, fused_layer_norm_cuda
from apex.optimizers import FusedAdam
from apex.normalization import FusedLayerNorm
assert sys.version_info[:3] == (3, 10, 20), sys.version
assert torch.__version__ == '2.6.0+cu124', torch.__version__
assert torch.version.cuda == '12.4', torch.version.cuda
assert torch._C._GLIBCXX_USE_CXX11_ABI is False, 'Expected the recorded PyTorch ABI=0 environment'
assert np.__version__ == '1.26.4', np.__version__
print('Python:', sys.executable)
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('NumPy:', np.__version__, 'C++ ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)
print('Apex compiled extensions: OK')
PY

# 固定后续worker解释器；重新连接时先激活同一Conda环境，再恢复该变量。
export H100_PYTHON="$(python -c 'import sys; print(sys.executable)')"
```

Apex 编译方式依据 [固定提交的 NVIDIA Apex 说明](https://github.com/NVIDIA/apex/tree/e13873debc4699d39c6861074b9a3b2a02327f92)。已有符合要求的环境可跳过安装步骤；详细历史依赖见 [公共软件环境说明](../env.md)。

已有环境跳过安装时仍执行上述导入/版本检查，并设置 `H100_PYTHON`。这些检查不创建CUDA张量；Torch/Apex导入成功只说明依赖可加载，GPU可用性在第3节检查，实际候选运行库仍需后续完整编译及预检。编译和执行始终使用这个Conda环境，不能在链接扩展后切换PyTorch或ABI。

如果使用 Docker，上述配置在容器内完成。容器需要能访问 8 张 GPU、支持 GPU IPC，并同时挂载 TempName 与 Megatron-LM。可以沿用 `--gpus all --ipc=host --network=host` 和父目录到 `/workspace` 的挂载方式，此时 `REPO=/workspace/TempName`、`MEGATRON_DIR=/workspace/Megatron-LM`；后续整个流程保持在同一容器/路径下执行。镜像自带的 PyTorch 2.5.1 或 2.7.0 不等于本实验的目标环境。

## 3. 核对八卡配置

从这里开始的设备检查、校准及测试命令会访问新服务器的 GPU。裸机且八卡均分配给本次实验时，可显式选择设备：

```bash
cd "$REPO"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nvidia-smi
nvidia-smi topo -m

python - <<'PY'
import torch
assert torch.cuda.device_count() == 8, 'Exactly eight visible GPUs are required'
counts = set()
for i in range(8):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, (p.major, p.minor), p.multi_processor_count, p.uuid)
    assert 'H100' in p.name and (p.major, p.minor) == (9, 0)
    counts.add(p.multi_processor_count)
assert counts == {132}, 'Ablation3 requires eight 132-SM H100s'
assert all(torch.cuda.can_device_access_peer(i, j)
           for i in range(8) for j in range(8) if i != j)
print('CUDA peer access: all pairs OK')
PY
```

调度器已经设置 `CUDA_VISIBLE_DEVICES` 时保留其分配；容器内使用实际可见的设备编号。当前路径要求完整 H100、SM90、同质八卡、NVLink 路径和两两 peer access。单纯“插了八张 H100”并不足以保证任意 PCIe 拓扑受支持。

九组入口读取冻结模型配置，固定TP8/132 SM。目前不支持把 `sm_count` 改为114后直接运行本实验；114 SM、MIG或不满足peer access的设备应停止并说明，不修改冻结配置伪装成支持。GPU身份、可见顺序、容器和产物绝对路径在校准及正式实验期间保持一致。

## 4. CPU来源核验与新会话目录

在上述Conda环境、仓库根目录执行。下面的来源检查不导入Torch、不查询GPU：

```bash
cd "$REPO"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python - <<'PY_CHECK'
import sys
sys.path.insert(0, 'tools/h100/phase6')
from candidate import sources
freeze, root, manifest = sources()
print('Frozen compute sources verified:', root)
print('Measured source commit:', manifest['source_commit'])
PY_CHECK

export H100_A3_SESSION="ablation3-gpt67-tp8-scope-$(date +%Y%m%d-%H%M%S)"
export H100_A3_OUT="$REPO/outputs/h100/phase6/$H100_A3_SESSION"
export H100_A3_LOG="$REPO/logs/h100/phase6/$H100_A3_SESSION"
export H100_A3_PLANS="$H100_A3_OUT/plans/measured"
export H100_A3_NCCL_SRC="$H100_A3_OUT/nccl-source"
export H100_A3_NCCL_ROOT="$H100_A3_OUT/nccl-release"
printf 'session=%s\nbuild=%s\nlog=%s\n' "$H100_A3_SESSION" "$H100_A3_OUT" "$H100_A3_LOG"

# 先记录本次会话的路径和环境，便于断开连接后找到同一套产物。
test ! -e "$H100_A3_OUT"
test ! -e "$H100_A3_LOG"
mkdir -p "$H100_A3_OUT" "$H100_A3_LOG/handoff"
python - <<'PY_SESSION'
import json
import os
from pathlib import Path
names = ('REPO', 'MEGATRON_DIR', 'CUDA124', 'CUDA128', 'H100_PYTHON',
         'H100_A3_SESSION', 'H100_A3_OUT', 'H100_A3_LOG', 'H100_A3_PLANS',
         'H100_A3_NCCL_SRC', 'H100_A3_NCCL_ROOT', 'CC', 'CXX', 'CUDAHOSTCXX',
         'DEBUG', 'CUDA_VISIBLE_DEVICES')
values = {name: os.environ.get(name) for name in names}
assert all(values[name] for name in names if name != 'CUDA_VISIBLE_DEVICES'), 'Missing session environment variable'
assert values['DEBUG'] == '0', 'NCCL must use DEBUG=0'
(Path(os.environ['H100_A3_LOG']) / 'handoff/session-paths.json').write_text(
    json.dumps(values, ensure_ascii=False, indent=2) + '\n')
PY_SESSION
git rev-parse HEAD > "$H100_A3_LOG/handoff/tempname-commit.txt"
git submodule status --recursive > "$H100_A3_LOG/handoff/submodules.txt"
git -C "$MEGATRON_DIR" rev-parse HEAD > "$H100_A3_LOG/handoff/megatron-commit.txt"
python -m pip freeze > "$H100_A3_LOG/handoff/pip-freeze.txt"
```

所有阶段输出目录必须是新目录；每一步成功后才执行下一步。`handoff/session-paths.json` 记录路径、解释器、编译器和设备配置，不代表对应阶段已完成。重新连接后激活同一环境，按该文件恢复原会话变量并清空`MAKEFLAGS`，不重新生成会话名，也不对已启动的校准/compare目录再次执行整套任务；先查看阶段状态和原进程是否还在运行。可先用以下命令做CPU回归：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s tools/h100/phase6 -p 'test_*.py' -v
```

### 4.1 构建一次NCCL Release开发库

本项目固定子模块的NCCL开发库为2.19.3；它与PyTorch wheel携带的NCCL 2.21.5运行依赖是两项不同来源，不替换PyTorch依赖。采样库和两套候选统一使用本次构建的头文件、静态库。新服务器一般没有原机器被gitignore的`nccl-release/`，在本机会话目录复制固定子模块源码并编译，不污染子模块工作区：

```bash
python - <<'PY_NCCL'
import os
import shutil
from pathlib import Path
src = Path(os.environ['REPO']) / '3rdparty/nccl'
dst = Path(os.environ['H100_A3_NCCL_SRC'])
assert not dst.exists()
assert not Path(os.environ['H100_A3_NCCL_ROOT']).exists()
shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
    '.git', 'build', '__pycache__', '*.pyc', '*.o', '*.a', '*.so', '*.so.*'))
PY_NCCL

git -C "$REPO/3rdparty/nccl" rev-parse HEAD > "$H100_A3_LOG/handoff/nccl-commit.txt"
CUDA_VISIBLE_DEVICES='' make -C "$H100_A3_NCCL_SRC" -j2 src.build \
  CUDA_HOME="$CUDA128" BUILDDIR="$H100_A3_NCCL_ROOT" \
  CC="$CC" CXX="$CXX" DEBUG=0 VERBOSE=1 \
  'NVCC_GENCODE=-gencode=arch=compute_90,code=sm_90' \
  > "$H100_A3_LOG/handoff/nccl-release-build.log" 2>&1
test -f "$H100_A3_NCCL_ROOT/include/nccl.h"
test -f "$H100_A3_NCCL_ROOT/lib/libnccl_static.a"
sha256sum "$H100_A3_NCCL_ROOT/include/nccl.h" \
  "$H100_A3_NCCL_ROOT/lib/libnccl_static.a" \
  > "$H100_A3_LOG/handoff/nccl-release-sha256.txt"
```

完成后检查退出码和编译日志中的优化参数，不能带CUDA调试选项`-G`或`-O0`。完整日志、固定源码提交和实际库哈希均保留。跨机器编译的库哈希不必等于旧机器，但本次采样/base/arrival三处必须引用同一份本机开发库；后续编译统一传`--nccl-root`。

## 5. 完整编译双shape采样库，并在新机器校准两角色

```bash
python tools/h100/phase6/ablation3.py calibration-build "$H100_A3_OUT/calibration" \
  --roles attn mlp
CUDA_VISIBLE_DEVICES='' python tools/h100/phase6/ablation3.py compile "$H100_A3_OUT/calibration" \
  --cuda-home "$CUDA128" --jobs 2 --nccl-root "$H100_A3_NCCL_ROOT"
python tools/h100/phase6/ablation3.py calibration-prepare "$H100_A3_LOG/calibration" \
  --build "$H100_A3_OUT/calibration"

# 先打印计划，不启动GPU。
python tools/h100/phase6/ablation3.py calibration-run "$H100_A3_LOG/calibration"

# 以下命令实际启动本机八卡BF16校准。
python tools/h100/phase6/ablation3.py calibration-run "$H100_A3_LOG/calibration" \
  --python "$H100_PYTHON" --cuda-home "$CUDA124" --execute
python tools/h100/phase6/ablation3.py fit "$H100_A3_LOG/calibration" "$H100_A3_PLANS"
```

采样库要完成动态库、Python扩展和mapping程序的完整编译，manifest达到 `status=built`；仅 `compile-check` 的对象编译不能替代。编译阶段的空 `CUDA_VISIBLE_DEVICES` 只作用于单条命令，不改变当前shell的八卡选择。`--nccl-root`须指向第4.1节含头文件与静态库的开发安装，不能使用PyTorch的NCCL wheel目录。

Attention使用 `[2048,4096,4096]`、本地K512；MLP使用 `[2048,4096,16384]`、本地K2048。两角色分别拟合，pass0/1训练、pass2诊断，输出本机 `attn.json` 与 `mlp.json`。新服务器两份都必须重建，不采用README中的旧服务器计划复用示例。

启动前工具会核对GPU状态，忙时退出，不自动等待或重试。失败记录保留；修复需要变更的环境/源码后，在新目录重新准备对应阶段，不在旧目录改哈希或补写完成标记。

## 6. 从本机计划完整编译九组候选

```bash
python tools/h100/phase6/ablation3.py build-prepare "$H100_A3_OUT/candidates" \
  --attn-plan "$H100_A3_PLANS/attn.json" \
  --mlp-plan "$H100_A3_PLANS/mlp.json"
CUDA_VISIBLE_DEVICES='' python tools/h100/phase6/ablation3.py compile "$H100_A3_OUT/candidates" \
  --cuda-home "$CUDA128" --jobs 2 --nccl-root "$H100_A3_NCCL_ROOT"
```

**新服务器必须显式传入上述两份新计划。** 不省略 `--mlp-plan`，因为省略时入口会尝试使用旧冻结MLP计划。正式会话不传 `--source-only`；该标记只用于CPU演练，生成的worker永久拒绝GPU执行，不能靠修改标记转为实测。

构建生成两套双shape库：`candidates/base/` 对应基础交替，`candidates/arrival/` 对应到达置换。A组共用base库，B/C共用arrival库，通过全远端/定向mask区分。两套库均需完成完整编译链接及Python扩展构建；编译日志分别位于两个目录的 `compile.log`，两份manifest都须为 `status=built`。不要使用旧六组的 `build.py prepare` 或 `prepare.py` 替代本节命令。

## 7. 准备并执行一次九组对比

```bash
python tools/h100/phase6/ablation3.py prepare "$H100_A3_LOG/compare" \
  --build "$H100_A3_OUT/candidates" --megatron "$MEGATRON_DIR" --blocks 9 --rounds 1

# 本次一轮dry-run：打印12项角色mapping和99个worker任务，不使用GPU。
python tools/h100/phase6/ablation3.py run "$H100_A3_LOG/compare"
```

`--rounds`是`prepare`参数，写入submission、准备记录和worker配置，run/report按冻结轮数执行。**一轮不是九组中的一组，也不是一次前向：本次仍测全部九组，每组9个窗口，每窗口10步预热＋20步完整训练。** 只保留原第一轮循环移位调度，所有策略在九个位置各出现一次；没有第二轮反转验证。旧两轮目录按旧协议保留，缺第二轮不能算一轮成功；更改轮数须另建compare。

**只核验能否运行：** 执行以下分支，完成后停止。控制器日志另存到`handoff/`，同时保留工具生成的每个worker日志。

```bash
set -o pipefail
python tools/h100/phase6/ablation3.py run "$H100_A3_LOG/compare" \
  --stage preflight --python "$H100_PYTHON" --cuda-home "$CUDA124" --execute \
  2>&1 | tee "$H100_A3_LOG/handoff/preflight-controller.log"
python tools/h100/phase6/ablation3.py report "$H100_A3_LOG/compare" --preflight
```

**本次实际执行一轮：** 对尚未启动的compare选择以下分支，预检通过后继续81个正式窗口。不要在同一compare先跑上面的预检分支再跑`--stage all`。

```bash
set -o pipefail
python tools/h100/phase6/ablation3.py run "$H100_A3_LOG/compare" \
  --stage all --python "$H100_PYTHON" --cuda-home "$CUDA124" --execute \
  2>&1 | tee "$H100_A3_LOG/handoff/all-controller.log"

# all成功时已生成报告；这条命令在原环境重新做CPU核验并生成报告。
python tools/h100/phase6/ablation3.py report "$H100_A3_LOG/compare"
```

如果此前已完成独立预检、后来明确要求正式计时，保持同一submission和全部冻结输入，使用`--stage timing`衔接；该分支不会重复启动前GPU占用检查，执行者须先确认八卡仍分配给本任务且空闲，另存本次检查记录。不要重复预检、覆盖已有状态文件或把已中断的计时当作可直接续跑；入口没有通用断点续跑开关。

本次一轮实验的规模为：

| 项目 | 数量或定义 |
| --- | --- |
| 策略 | A：交替＋完整量化；B：交替＋到达优先＋完整量化；C：交替＋到达优先＋选择性量化 |
| 作用范围 | 仅MLP、仅Attention、MLP＋Attention；三策略×三范围＝九组 |
| 正式窗口 | 九组同块，9块×1轮＝81窗口；以后显式选择两轮时为162窗口 |
| 预检 | 18个smoke/profile窗口、12个策略/角色mapping组合 |
| 正式计时 | 每窗口10步预热＋20步完整optimizer step；本次共1620个正式step |
| 完整step | 整个32层模型、4个微批的前向与反向、通信/同步、一次参数更新、学习率调度及清梯度 |

完整量化只针对所选模块全部远端贡献；GEMM、本地贡献和未选中模块仍为BF16。Attention仅指 `self_attention.linear_proj`，MLP仅指 `mlp.linear_fc2`；未选中模块统一保留原生Megatron BF16。每组均由独立进程执行，控制器管理库路径与八个rank，不需额外套 `torchrun`。

九组不增加原生Megatron、朴素量化、固定融合v2或其他对照；不套用旧六组的4%门槛。所有ms/step均是完整训练步，报告保留全样本及负收益；逐轮给出九项同范围策略比较和九项同策略范围比较。慢窗口筛选仅为附表，不能替代主报告或删改原始数据。

预检失败时不会进入正式计时；GPU忙时停止并说明，不建立等待队列。一套运行中不周期查询GPU占用，不额外启动独立复测。显存、数值及性能以这台新机器的实际预检和计时为准，不能用开发端CPU检查或旧服务器结果替代。

## 8. 核验结果与交付

完整执行时检查 `$H100_A3_LOG/compare/report.json`、`report.md`、`preflight.json` 和 `all-state.json`（分阶段执行则检查`preflight-state.json`、`timing-state.json`）；本次报告应为`round_count=1`、`timing_windows=81`且完成状态为真。`combined`与第1轮是同一批9块，不是额外重复样本。只做可运行性检查时核验`preflight.json`及`preflight-state.json`，应有144份预检rank、72份profile trace、12项角色mapping，正式窗口为0。新服务器首次部署包含安装、完整链接、双角色校准，本次单轮对比另有99个worker窗口，不承诺固定分钟数内完成。

失败时查看相应阶段状态、`handoff/`控制器日志与`results/`各窗口的`process.log`，保留失败目录及原因。持续任务应在可保持会话的终端中运行并记录控制器PID、完整命令和启动时间。状态文件停在`running`不能证明进程仍有效运行：原服务器曾出现控制器/通信异常而worker仍占显存的情况。遇到日志长期不前进，应核对原进程、父子关系、窗口路径及错误；不盲目重启第二套、不自动终止其他任务，也不将部分结果标记为完成。

本次一轮完整交付至少包含：

- 本机两角色校准原始记录、mapping、profile、完成状态与 `attn.json`/`mlp.json`。
- 采样库和base/arrival候选库的manifest、源码快照、完整编译日志、实际运行库哈希及依赖版本。
- 共用NCCL开发库的固定源码提交、`DEBUG=0`编译日志、头文件/静态库哈希；静态库本体留在执行端。
- 九组submission、冻结scripts、schedule、12项角色mapping与preflight。
- 648份正式计时rank、144份预检rank、72份压缩profile trace，以及命令和进程日志；两轮模式才是1296份正式rank。
- 全样本报告、敏感性附表（如生成）、阶段状态和 `handoff/` 中的执行说明，包括源码提交/未提交补丁、环境、会话路径、完成状态及缺项。

正式实验文本和trace按 [交付规范](instruction-h100.md#测试完成或中断后交什么) 回传；运行库和编译缓存由执行端保留，不提交`.so/.o/.a`、环境或checkpoint。缺项如实列出，不修改manifest绕过验证。将数据复制回开发端分析时保留记录中的原绝对路径；接收端可用 `ablation3.py report <compare目录> --offline` 核验文本证据，但这不替代原环境的运行库核验。
