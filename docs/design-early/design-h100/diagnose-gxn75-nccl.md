# 75：Fabric Manager 检查与纯 NCCL 对照

历史回传 `adb9ef4e` 中，75 的 `TTention-MLP-gpt67-tp8-20261017-154456` 在校准
`worker.py:58` 的 `flux.create_tensor_list` 失败：PyTorch NCCL 2.21.5、CUDA 401，
内核记录 `NV_ERR_FABRIC_STATE_OUT_OF_SYNC`。

后续 [独立 NCCL 对照记录](../../../logs/h100/nccl-probe-20261017T161905Z/) 已回传：
default 退出码 1，`nvls-off` 退出码 0、八 rank 全部 PASS。75 随后在关闭 NVLS 的配置下
完成校准及五套消融 3，见 [结果汇总](消融实验/base-phase6-ablation3.md)。以下命令保留作诊断复现说明；
宿主 Fabric Manager 的具体状态仍需实际宿主日志确认，不能由容器状态推断。

2026-10-18 清理失败会话后，原失败构建不再作为下列命令的路径来源；改用保留的
`ablation3-gpt67-tp8-scope-20261017-163146` 校准库。独立 NCCL 对照记录继续保留。

## 1. 在 75 宿主机采集服务与驱动信息

通过平时的 SSH 入口登录 **75 宿主机**，不要在 `/workspace` 实验容器里查询宿主服务。
先看 `hostname`、`ps -p 1 -o comm=` 和 `systemd-detect-virt --container`。
容器可能沿用宿主机 hostname，因此仅看到 `gxn75` 不够。若只有容器账号，把本节交给管理员。

以下命令在 Bash 执行；仅查询并保存日志，不重启服务或重置 GPU：

```bash
H100_HOST_DIAG=$(mktemp -d /tmp/gxn75-fabric.XXXXXX)
{
  date -Is
  hostname
  ps -p 1 -o comm=
  systemd-detect-virt --container || true
  nvidia-smi
  cat /proc/driver/nvidia/version
  cat /sys/module/nvidia/version
  cat /sys/module/nvidia_nvswitch/version
  command -v nv-fabricmanager
  nv-fabricmanager --version
  systemctl status nvidia-fabricmanager --no-pager -l
  systemctl show nvidia-fabricmanager -p ActiveState -p SubState -p ExecStart
} > "$H100_HOST_DIAG/host-status.txt" 2>&1

sudo journalctl -b -u nvidia-fabricmanager --no-pager -n 500 -o short-iso \
  > "$H100_HOST_DIAG/fabric-journal.txt" 2>&1
sudo journalctl -k -b --no-pager -n 2000 -o short-iso \
  > "$H100_HOST_DIAG/kernel-journal.txt" 2>&1
sudo tail -n 500 /var/log/fabricmanager.log \
  > "$H100_HOST_DIAG/fabricmanager-file.txt" 2>&1
nvidia-smi -q > "$H100_HOST_DIAG/nvidia-smi-q.txt" 2>&1
nvidia-smi topo -m > "$H100_HOST_DIAG/topology.txt" 2>&1
nvidia-smi nvlink -s > "$H100_HOST_DIAG/nvlink-status.txt" 2>&1
nvidia-smi nvlink -e > "$H100_HOST_DIAG/nvlink-errors.txt" 2>&1
cat "$H100_HOST_DIAG/host-status.txt"
echo "Host evidence: $H100_HOST_DIAG"
```

重点看服务是否 `active (running)`，FM 实际版本与已加载驱动是否匹配，以及日志中的
`version mismatch`、初始化失败、NVSwitch、Xid/SXid、fabric 错误。75 回传的已加载驱动是
`580.65.06`；以本次现场读取为准。不要只比主版本号 `580`，也不要把磁盘上的新驱动包版本
当作已加载模块版本。默认日志文件不存在时，检查 `ExecStart` 指向的配置及 journal；
缺少单个日志文件不等于服务未安装。

`nvidia-smi` 的 `CUDA Version` 是驱动支持能力，不是 Conda 中 PyTorch 的 CUDA 构建版本。
Fabric 的 `Completed/Success` 和 peer-access 可达也不能替代实际 NCCL 通信测试。
服务/版本检查依据 [NVIDIA Fabric Manager 官方文档](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/#checking-the-fabric-manager-status)。

## 2. 在原实验容器、原 Python 环境准备独立测试

先把本仓库新增的 `tools/h100/diagnostics/nccl_probe.py` 同步到 75。
不要为这个测试重装 torch 或另建 Python 环境。进入原实验容器，在 Bash 执行：

```bash
cd /workspace/TempName
export H100_PYTHON=/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$(dirname "$H100_PYTHON"):$CUDA_HOME/bin:$PATH"

# 使用保留的 75 校准库目录；测试程序不导入 Flux。
H100_PROBE_LIB="$PWD/outputs/h100/phase6/ablation3-gpt67-tp8-scope-20261017-163146/calibration/scope-sampler"
export LD_LIBRARY_PATH="$H100_PROBE_LIB/python/flux/lib:$H100_PROBE_LIB:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL
H100_NCCL_DIAG=$(mktemp -d "$PWD/logs/h100/nccl-probe-XXXXXXXX")
cp tools/h100/diagnostics/nccl_probe.py "$H100_NCCL_DIAG/nccl_probe.py"

"$H100_PYTHON" -c 'import sys,torch; print(sys.executable); print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())'
for H100_NCCL_CONFIG in /etc/nccl.conf "$HOME/.nccl.conf"; do
  if [ -f "$H100_NCCL_CONFIG" ]; then
    echo "$H100_NCCL_CONFIG"
    cat "$H100_NCCL_CONFIG"
  fi
done > "$H100_NCCL_DIAG/nccl-config.txt"
cat "$H100_NCCL_DIAG/nccl-config.txt"
```

预期 Python 是上述路径、torch 为 `2.6.0+cu124`、torch CUDA 为 `12.4`、NCCL 为
`(2, 21, 5)`。**先检查配置文件是否固定了 `NCCL_NVLS_ENABLE`**：下面 A 组只去掉进程
环境中的该变量，配置文件仍然有效。如果配置文件已有覆盖，先记录这一情况，不能把 A
称为默认自动选择。不要为测试直接覆盖系统配置。

原失败记录未完整保存继承的 `LD_LIBRARY_PATH`，因此这里只对齐已知设置，并由 probe
记录实际加载的库路径；不能宣称与历史进程环境完全相同。

## 3. 顺序运行 A/B，各一次

脚本在新 NCCL group 上先执行每 rank 64 字节的 uint8 all-gather，模拟原来 IPC 句柄交换的
通信形状；再执行 float32 all-reduce，预期和为 36。它不加载模型，也不导入 Flux。
每个进程会记录版本、设备 UUID、环境变量和动态库路径。

```bash
for H100_PROBE_CASE in default nvls-off; do
  H100_PROBE_DIR="$H100_NCCL_DIAG/$H100_PROBE_CASE"
  mkdir "$H100_PROBE_DIR"
  # 每次启动前检查一次。已有任务或上次有残留时停止，不等待、不杀其他任务。
  if ! nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
      --format=csv,noheader > "$H100_PROBE_DIR/compute-apps-before.csv"; then
    echo "GPU query failed; stopping."
    break
  fi
  if [ -s "$H100_PROBE_DIR/compute-apps-before.csv" ]; then
    cat "$H100_PROBE_DIR/compute-apps-before.csv"
    echo "GPU compute processes exist; stopping."
    break
  fi
  if [ "$H100_PROBE_CASE" = default ]; then
    H100_NVLS_ENV=(-u NCCL_NVLS_ENABLE)
  else
    H100_NVLS_ENV=(NCCL_NVLS_ENABLE=0)
  fi
  if env "${H100_NVLS_ENV[@]}" \
      NCCL_DEBUG_FILE="$H100_PROBE_DIR/nccl-%h-%p.log" \
      timeout --signal=TERM --kill-after=15s 180s \
      "$H100_PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
      --nproc_per_node=8 --max_restarts=0 \
      "$H100_NCCL_DIAG/nccl_probe.py" --execute \
      > "$H100_PROBE_DIR/process.log" 2>&1; then
    H100_PROBE_RC=0
  else
    H100_PROBE_RC=$?
  fi
  echo "$H100_PROBE_RC" > "$H100_PROBE_DIR/exit-code.txt"
  echo "$H100_PROBE_CASE exit=$H100_PROBE_RC"
  if [ "$H100_PROBE_RC" -eq 124 ] || [ "$H100_PROBE_RC" -eq 137 ]; then
    echo "Timed out/killed; confirm this probe's torchrun/workers exited before any next test."
    break
  fi
done
echo "NCCL evidence: $H100_NCCL_DIAG"
```

A/B 用独立进程，功能设置只改变 `NCCL_NVLS_ENABLE`；日志路径也各自独立。
不要同时添加 `NCCL_P2P_DISABLE`、`NCCL_CUMEM_ENABLE` 等开关。
`NCCL_NVLS_ENABLE=0` 的含义是关闭 NVLink SHARP，见
[NCCL 2.21.5 官方说明](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/env.html#nccl-nvls-enable)。

## 4. 判断并回传

成功要求 `exit-code.txt` 为 0，且 `process.log` 中有八个不同 rank 的 `"event": "PASS"`；
看到版本、创建 group 或 `all_gather_pass` 都不代表整项通过。退出码 124 表示外层超时，
137 可能是强制终止，需要结合日志；八个 rank 通常报告同一故障，优先找第一条 CUDA/NCCL 错误。
超时后循环停止；即使 GPU 暂时空闲，也需确认本次 torchrun/worker 已退出，再单独运行 B。

| A：默认配置 | B：NVLS=0 | 解释与下一步 |
| --- | --- | --- |
| 失败 | 通过 | 强烈支持 NVLS/Fabric 路径触发；结合驱动日志检查宿主通信栈。B 通过不代表 Fabric 已修复。 |
| 失败 | 失败 | 基础 NCCL 已可独立复现问题，先根据首个错误查主机、运行库和通信配置。 |
| 通过 | 通过 | 基础 NCCL 当前可用；再查间歇状态、Flux 导入后的库路径和 IPC 创建，不能直接断言历史环境正常。 |
| 通过 | 失败 | 检查 B 的首个错误及替代通信路径，不能套用“NVLS 故障”的结论。 |

测试后在宿主机重新采集一次 kernel journal，保留时区和时间戳。此前 process 与 `dmesg -T`
时间固定差 100 秒，不能仅凭格式化时间相同与否判定关联。

回传完整 `$H100_HOST_DIAG` 和 `$H100_NCCL_DIAG` 两个目录，包含每 rank NCCL INFO 文件；
不要只截最后的 `ChildFailedError`。这些测试是诊断，不计入正式性能结果。
本地开发端仅做语法、无执行入口及静态依赖检查；GPU 是否通过以 75 的新日志为准。
