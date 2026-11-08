# OP1 A/B/C：RTX 5090 的 FLUX 交替排序、分块到达消费与选择性量化

更新：2026-11-02。目标为单机 **8×RTX 5090，TP8/DP1**。新增独立协议 **`rtx5090-phase6-op1-abc-staged-v1`** 和入口 [run_op1_abc.sh](../../../tools/5090/phase6/run_op1_abc.sh)，同场比较下面三种组合及 [baseline-flux.md](baseline-flux.md) 的 **FLUX＋改进量化**。默认已按要求缩短为 **2 个 block**。

**状态：目标机批次 `20261102-000347-679838` 已完成构建和四组 native 探针，首组模型 smoke 在控制器校验阶段失败；尚无正式计时或本轮加速比。** 原因是生成 worker 过早采集通信状态，修复与复用构建重跑方法见第 9 节。新 codec 此前已在本地用 CUDA 12.8 编译为 SM120 动态库；CPU 检查不等于 GPU 正确性或性能验证。原 H100、5090 九组 baseline、OP1-A 和 Flux 四组的源码构建及历史日志保持各自协议，不向旧结果目录追加新数据。

## 1. 从已有结果确定改进方向

[5090 baseline](baseline.md) 已记录目标机器跨卡 peer access 为 **0/56**。原 H100 的远端地址写入、tile ready 标志和到达表不能直接照搬。旧 staged 路径先完成整个 GEMM，再执行 NCCL，交替产出不同目的 tile 并不能让这些 tile 提前跨卡到达。

[已完成 OP1-A 批次](../../../../Load-5090/logs/5090/phase6/op1-a-direct-wire-v1-1x-50steps-20261101-185132-330063/report.md) 中，FLUX＋基础量化平均 **967.327 ms/step**，加交替排序后 **967.302 ms/step**，配对区间跨零。它支持继续优化通信/消费边界，而不能作为交替排序已有明确收益的证据。

[已完成 Flux 四组批次](../../../../Load-5090/logs/5090/phase6/flux-quant-four-arm-v1-1x-50steps-20261101-203719-996417/report.md) 中，改进量化平均 **1016.920 ms/step**，基础量化 **1029.783 ms/step**，两者配对降幅区间 **[-2.480%, +5.081%]**。IDE 中打开的 [214204 批次报告](../../../../Load-5090/logs/5090/phase6/flux-quant-four-arm-v1-1x-50steps-20261101-214204-981363/results/compare1/report.md) 只有四个完整块，是中断批次的阶段性观察。不同批次不能合并或挑最快窗口作为新方案的对照。

本轮具体处理三项成本：基础量化的远端 BF16 `torch.cat` 和每次 packet 分配；整包通信结束才开始解码的串行等待；H100 选择表与当前 staged 传输边界不对应。保留模型、反向、optimizer 和正式计时范围。

## 2. 四组实际定义

本页 **A/B/C 是三组新 FLUX 组合的简称**，不是旧九组的 native `a/c`，也不是 OP1-A 中的融合 producer A。

| 本页组名 | 脚本 ID | 实际路径 | 远端量化 |
| --- | --- | --- | --- |
| **A：FLUX＋基础量化＋交替排序** | `flux-basic-interleaved` | 交替排序 GemmOnly → 直接 packet 编码 → 一次整包 NCCL All-to-All → 融合解码/归约 | 全部远端 H128 group |
| **B：A＋到达优先** | `flux-basic-arrival` | 相同 GEMM/codec；拆成四个输出行块，每块通信完成后立即提交解码，独立解码 stream 可与后续块通信并行 | 与 A 相同 |
| **C：B＋选择性量化** | `flux-basic-selective` | 与 B 相同的分块和 stream 依赖；raw group 直接写 BF16，selected group 写原 H128/FP8 | 每个目的 packet 最后 **1/64** group；其余 **63/64 保持 BF16** |
| **FLUX＋改进量化** | `flux-improved` | 原 Flux rank-offset 排序；旧 C 融合 producer → pack → NCCL → unpack → warp decoder；保留实例配置、双工作区及独立输出 | 全部远端 group；不引入本轮分块或选择策略 |

A/B/C 共用同一份 SM89/RTX5090 GemmOnly V2 / Stream-K 构建，机器码为 **SM120**。交替映射复用 [op1_a_swizzle.hpp](../../../tools/5090/phase6/op1_a_swizzle.hpp)：目的 rank 随线性 tile 编号交替，实际 mapping 必须由注册 GEMM 参数导出并检查，profile 必须出现 `Op1AInterleavedSwizzle`。这只是逻辑 tile 映射，不能保证硬件 CTA 启动顺序。

改进量化的实现直接复用 [flux_baseline_build.py](../../../tools/5090/phase6/flux_baseline_build.py) 登记的原排序 C；保留它原有的 GEMM、融合方式、decoder 和工作区差异。四组整体性能差异不能全部归因于某一个 kernel；A/B 是分块调度对照，B/C 是本轮选择性量化策略对照。

## 3. A：直接写最终 wire，保留基础量化算术

[op1_abc_native.py](../../../tools/5090/phase6/op1_abc_native.py) 从已固定来源的 TACO 上游 H128 encoder 生成新寻址版本；上游归约、Hadamard、adaptive scale、E4M3 转换和舍入公式保持不变。上游来源仍由仓库的 `baseline_upstream/provenance.json` 和源码哈希验证，不从未登记的 `COCCL_TACO` 工作目录随意抽取另一版公式。

新 encoder 直接读取完整 GEMM partial 中对应目的 rank 的连续行，并写最终发送缓冲，去掉前置 `torch.cat`。每个量化 H128 group 使用 **128 字节 FP8＋4 字节 quant scale＋4 字节 adaptive scale** 的 AoS 布局；raw group 使用 256 字节 BF16。不存在额外的 packet pack/unpack kernel。布局与旧基础量化 SoA 不同，但每组编码位串必须通过旧上游 encoder 的独立比对。

每个实例在初始化时分配发送/接收 packet，后续调用复用；每次返回独立输出，不返回下次会覆盖的 scratch。decoder 保留 `(rank+1)%8` 到 `(rank+7)%8` 的来源次序，最后加入本地 BF16；每个解码值和每次累加仍经过 BF16 舍入。NCCL 存储的来源升序与数值归约次序分开处理。

## 4. B：把到达边界改为 NCCL 完成的行块

每个目的输出为 `256×4096`，拆成 **四块，每块 64 行 / 2048 个 H128 group**。所有 rank 使用相同块号顺序和相互匹配的 split，各 rank 的 self split 恒为 0。

```text
caller stream：完整交替 GEMM → encode[0] → encode[1] → encode[2] → encode[3] → 等待最终 decoded event → 模型消费
                                  ↓event       ↓event       ↓event       ↓event
comm stream：                  NCCL[0]  →   NCCL[1]  →   NCCL[2]  →   NCCL[3]
                                  ↓完成        ↓完成        ↓完成        ↓完成
decode stream：                decode[0] →  decode[1] →  decode[2] →  decode[3] → event
```

代码在每块 `all_to_all_single(async_op=True)` 后调用 `Work.wait()`，在通信 stream 上建立 NCCL 完成依赖，之后才记录该块的 received event；decode stream 等待此事件再读接收缓冲。最后 caller 等待最终解码事件，bias/residual/LayerNorm 才能消费输出。不是用 Python 函数返回或提交时事件冒充通信完成。

这里的“到达优先”准确含义是 **某个完整行块到达后先消费该块，无须等其余行块全部交换完**。四次 collective 仍按固定块号执行，不声称实现了任意 peer 的先到先服务，也不读取 H100 到达表。**整个 GEMM 仍先完成，本轮没有恢复 GEMM tile 生产与跨卡通信重叠。**

profile 检查实际 CUDA launch 归属、每块 encoder→NCCL→decoder、独立解码 stream、最终消费者依赖，并记录解码与后续 NCCL kernel 的观测重叠量。允许重叠量为零；NCCL 资源争用和四次 collective 的启动成本可能抵消收益，结果保留负收益。

固定 caller stream 是实例契约；同进程组和设备复用通信/解码 stream，每实例 packet 独立。正常返回前已提交完整依赖链，下一次复用排在上次完成之后；异常使实例和对应 pipeline 禁止继续提交，并保留仍可能被异步工作引用的张量。

## 5. C：按当前分块尾部选择量化，明确预算含义

C 固定量化每个目的 packet 的 **最后 128/8192 个 H128 group，即最后四行**。其余 group 直接发送 BF16，decoder 对 raw group 跳过 Hadamard/FP8 解码，仍使用同样的来源累加顺序。raw/quant 边界由形状推导，记录在 `chunks[].raw_groups`，每个 rank 都能确定通信大小，无需交换动态 mask。

这是新的 **确定性 staged 尾部策略**：压缩最后提交块中的少量数据，避免把旧 H100 的到达评分当成 5090 校准。`calibration_used=false`；没有声称最后四行是测量得到的最优选择，也没有声称它等价于 H100 的冻结物理 tile mask。原 H100 按远端 tile 的预算上限选择，本轮按每个目的 packet 的 H128 group 固定选择，比例相同不意味着被选中的数据相同。

**1/64 是量化比例，不是 BF16 例外比例。** 对固定形状，每 rank 每次投影的逻辑发送字节如下：

| 方案 | 每目的 packet 字节 | 七个远端总字节 | 相对全 BF16 减少 |
| --- | ---: | ---: | ---: |
| A/B 和改进量化，全远端量化 | `8192×136 = 1,114,112` | **7,798,784** | 46.875% |
| C，最后 1/64 量化 | `8064×256 + 128×136 = 2,081,792` | **14,572,544** | 0.732421875% |
| 全 BF16 逻辑贡献，供说明 | `8192×256 = 2,097,152` | **14,680,064** | 0% |

C 发送的数据多于 A/B；可能通过减少 codec 运算获益，也可能受通信开销拖累，必须完整 step 实测。上表不是 NCCL 硬件链路计数，也不包含 NCCL 内部流量。第三行不是本轮第五个测试组。

## 6. 一键测试与构建复用

在目标 5090 容器的 `TempName-New` 根目录运行，沿用现有 `env/setup_5090.sh` / `env/activate_5090.sh` 环境。同步整个 `tools/5090/phase6/`，不要只复制入口 shell。

```bash
# 无副作用预览，不查询 GPU、不创建批次。
bash tools/5090/phase6/run_op1_abc.sh

# 四组比较，默认 2 个 block，后台运行。
bash tools/5090/phase6/run_op1_abc.sh --execute

# 如需完整的八块排序平衡周期，显式指定。
bash tools/5090/phase6/run_op1_abc.sh --execute --blocks 8

# 推荐：复用 baseline-flux 已完成的构建，保留同一份改进量化二进制。
bash tools/5090/phase6/run_op1_abc.sh --execute \
  --flux-build /path/to/completed-flux-batch/build

# 只有旧九组构建时，以其作为冻结父构建。
bash tools/5090/phase6/run_op1_abc.sh --execute \
  --baseline-build /path/to/verified-baseline/build

# 只检查硬件、Apex 实际运算和八 rank NCCL。
bash tools/5090/phase6/run_op1_abc.sh --execute --preflight-only

# 构建后只运行 mapping、算子数值、packet、输出所有权检查。
bash tools/5090/phase6/run_op1_abc.sh --execute --foreground \
  --transport-smoke-only --flux-build /path/to/completed-flux-batch/build

# 后续新批次复用本轮完整构建；仍重跑全部 GPU 门禁和计时。
bash tools/5090/phase6/run_op1_abc.sh --execute \
  --abc-build /path/to/completed-op1-abc-batch/build

# 只读重放完整批次，校验原始证据和报告。
bash tools/5090/phase6/run_op1_abc.sh --report --out /path/to/completed-op1-abc-batch
```

三个 `--*-build` 复用参数互斥。路径必须是目标机器上仍完整保存、通过源码和二进制哈希检查的构建；只有回传日志而没有二进制的 `Load-5090` 归档不能当成构建。复用构建也必须保留其原父构建路径和相同 Python/CUDA ABI。

新 [op1_abc_build.py](../../../tools/5090/phase6/op1_abc_build.py) 组合已有 OP1 交替排序构建、原排序 Flux 对照构建和新的 codec。已有构建只读；生成文件写入新目录。为使用现有完整构建验证器，内部 OP1/Flux 构建可能编译未计入本轮四组的附带变体，这些变体不进入本轮训练/统计。`--flux-build` 可直接复用对照部分，`--abc-build` 可复用整套构建。

默认结果目录：

```text
../Load-5090/logs/5090/phase6/op1-abc-staged-v1-1x-50steps-<时间>/
```

状态看 `batch-state.json`，后台日志看 `handoff/supervisor.log`，完整完成后生成 `report.md` 和 `report.json`。参数 `--foreground`、`--python`、`--cuda-home`、`--build-cuda-home`、`--megatron`、`--jobs`、`--worker-timeout` 和 `--out` 沿用原入口的用法。后台容器需持续存活。

## 7. 公平比较与失败门禁

模型保持 **32 层、hidden 4096、FFN 16384、32 heads、sequence 2048、micro batch 1、global batch 4、BF16、TP8/DP1**，每 optimizer step 为 **8192 tokens**。四组覆盖全部 64 个 Attention/MLP 输出投影，反向、optimizer、清梯度及全部通信计入墙钟，不启用 CUDA Graph 或后续 MA8/MA9/MA11 优化。

默认 **1 trial × 2 blocks × 4 组**；每窗口 **10 步预热＋50 步正式计时**，共 **8 个正式窗口、64 份 rank 记录、400 个正式 optimizer step**。`results/compare1/` 下只有 `block00-<组名>`、`block01-<组名>`，每个 block 各有四个组别目录。一个窗口是“一个 block 中的一组运行”，不是一个 block；此前的 32 个窗口对应八个 block，并非 `block00` 到 `block31`。

两块短测使用原八块排序的前两块，四组每块都运行，但尚未完成位置和前后关系的平衡周期；报告只给描述统计与配对点估计，**不计算 95% CI**。指定 `--blocks 8` 可恢复原 **32 个窗口、1600 个正式 step** 的平衡比较和 bootstrap 区间。`--blocks` 接受 2、3 或 8 的正整数倍（保留三块兼容），`--trials` 支持 1 或 2；每个 trial 独立汇总。

执行前先检查冻结训练源码、八卡硬件/Apex/NCCL，再构建和运行四组 native probe，之后依次运行四组模型 smoke、四组八 rank profile，全部通过才进入正式计时。仍固定 `NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=0`、`NCCL_NVLS_ENABLE=0`。

新增检查包括：

- 两个真实 K 形状、八 rank 的实际交替 GEMM mapping；改进量化仍检查原排序 mapping。
- 零值、随机、离群输入；独立 matmul/上游 codec 参考；相对 BF16 的 L2 上限 5%，相对同策略 codec 参考上限 1%。
- 用未改动的上游 codec 对直接 wire 位串与接收后的 BF16 fold 独立核对；56 条有向通信链路 packet 哈希相互匹配。
- A/B 原生探针输入、权重、packet 和输出逐位相同；分块改变调度，不能改变同一基础 codec 的结果。C 和改进量化使用各自独立数值门禁。
- A/B/C 的非默认 stream 检查；四组的 rank 延迟、七次复用后的六个独立输出；逐模块通信/等待/解码计数、split、工作区所有权和失败状态。
- 四组完整模型的初始权重/RNG/token 身份、有限 loss/gradient norm、64 模块覆盖、零 fallback、smoke 梯度和更新证据；profile 必须关联真实 CUDA launch 与模型消费者。

不同量化策略或 GEMM 归约路径不要求训练轨迹逐位相同；报告保留六组配对的数值差异观测。短程 smoke 和 50 步窗口不是收敛或模型质量等价证明。

报告从全部原始 rank 记录重算：窗口步时为 `max(八 rank elapsed_seconds) × 1000 / 50`，单位 ms/step。主表给所有窗口均值、中位数、最慢窗口、tokens/s、相对同期改进量化的均值降幅；六项配对使用完整 block 的对数比。两块或三块短测仅给点估计；八块或更多的完整平衡周期另给 10,000 次 bootstrap 区间，不做多重比较校正。不筛除慢窗口或负收益。显存包括 Torch allocated/reserved 和独立 native workspace，allocated＋native 不含 CUDA context/NCCL 内部占用。

若正式计时中断，保留已完成 profile 和原始窗口，但不产生伪完整报告；也不会复用历史窗口补齐本轮。预检 CUDA OOM 的处理仍参考 [baseline-flux 第 8 节](baseline-flux.md#8-2026-11-01-回传失败硬件预检-cuda-oom)，本入口不会自动停止其他 GPU 任务。

## 8. 本地验证

初次实现新增 **30 项专项测试**，当时整个 `tools/5090/phase6` 测试集 **217 项全部通过**；独立入口预览、shell 语法、Python 语法、文档文件链接和 whitespace 检查也通过。此前默认三块改动另增加短测调度与报告回归，覆盖不生成额外 block、不调用 bootstrap 和完整八块模式保留；**10 项调度/入口测试全部通过**，三块及八块入口预览也通过。CPU 回归命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tools/5090/phase6 -p 'test_op1_abc*.py' -v

# 全部 5090 回归，含已有基线和本轮新增测试。
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tools/5090/phase6 -p 'test_*.py' -v
```

覆盖构建来源及篡改拒绝、生成 worker、分块布局与字节预算、stream 依赖与失败后禁止复用、探针记录验证、profile 的 CUDA launch/完成依赖、八块平衡顺序、六组配对和中断状态。fixture 只用于验证检查器，不作为 GPU 测量数据。

已用本地 CUDA 12.8 将新 codec 的全部编码/解码入口编译为 SM120 动态库。当前环境没有可见 GPU，未在本地执行 native 探针、模型 smoke 或八卡计时。目标机运行上述完整入口后，才可判断本轮三组对 5090 的实际收益。

## 9. 2026-11-02 回传失败：首次前向前采集了空通信状态

批次 `op1-abc-staged-v1-1x-50steps-20261102-000347-679838` 的 [supervisor.log](../../../../Load-5090/logs/5090/phase6/op1-abc-staged-v1-1x-50steps-20261102-000347-679838/handoff/supervisor.log) 在 `smoke-flux-basic-interleaved` 阶段报 `Missing per-module ABC transport evidence`。这是本次代码生成器的错误。

冻结的 `worker.py` 第 117 行在首次前向之前执行 `report['op1_abc_transport'] = {...adapter.scope_ops...}`。算子实例在第一次前向中才延迟创建，此时 `scope_ops` 为空，字典推导式保存的是空快照，之后不会自动更新。八个 rank 的实际记录都含 64 个目标模块、每模块 20 次 codec 调用，即 `(2 步预热＋3 步 smoke)×4 microsteps`，但通信状态均为 `{}`。worker 自身记录 `completed=true / passed=true`，控制器仍必须因证据缺失拒绝该窗口，不能追填状态或把本批标记为通过。

[op1_abc_sources.py](../../../tools/5090/phase6/op1_abc_sources.py) 已将 A/B/C 的动态状态采集移到全部训练/profile step 结束后的最终报告区，与原生 transport 状态相邻，且在 rank JSON 落盘之前。新增 worker 断言要求覆盖全部目标模块，调用次数必须包含预热及正式/profile 步；控制器继续严格校验等待、stream、split、缓冲区和调用计数。改进量化对照仍使用其原生状态，ABC 状态保持空。诊断错误现会列出期望数量、实际数量、缺失和额外模块。

原测试手工构造了已经填好的状态，只验证记录校验器，没有执行生成 worker 的状态采集时机。新增生命周期回归执行实际生成代码的预热循环、smoke/profile/timing 分支及最终 JSON 写入，用 CPU 替身模拟首次前向创建 64 个实例、随后累计调用次数；把采集故意移回启动阶段必须失败。这些替身不会生成或冒充 GPU 实验结果。

修复后 **35 项 OP1-ABC 回归全部通过**。同一生命周期测试执行回传旧 worker，复现“5 步、64 个目标模块、0 条通信状态、每模块 20 次调用”；执行修复后的 worker 得到“5 步、64 个目标模块、64 条通信状态、每模块 20 次调用”。新增四项测试覆盖三组全部模式、改进量化原生状态、启动时空快照，以及仅预热后采样导致的过期计数。修复后尚未重跑八卡模型。

本地只读重验了四组 native 探针的逐 rank 记录，重新计算的汇总与 [native-probe/report.json](../../../../Load-5090/logs/5090/phase6/op1-abc-staged-v1-1x-50steps-20261102-000347-679838/native-probe/report.json) 一致。回传目录没有完整 build 二进制，未在本地重跑 GPU，也未执行整个批次的成功报告重放。后续三组模型 smoke、四组 profile 和正式计时尚未执行。

修复只涉及 Python worker 生成与校验诊断；与本批冻结源码核对，native builder、CUDA codec、运行时及其构建依赖未改变。同步更新 `tools/5090/phase6/` 后，在目标机 `TempName-New` 根目录复用本批已完成构建，**无需重编译 CUDA**：

```bash
bash tools/5090/phase6/run_op1_abc.sh --execute --foreground --blocks 2 \
  --abc-build /workspace/Load-5090/logs/5090/phase6/op1-abc-staged-v1-1x-50steps-20261102-000347-679838/build
```

该路径来自本批 `plan.json`，必须仍存在并通过原有构建身份验证。命令创建新批次，重新运行全部门禁及两块比较，不向失败目录追加或修改记录。
