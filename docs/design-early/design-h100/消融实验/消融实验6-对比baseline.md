# H100 消融实验6：对齐朴素量化的Flux与Megatron baseline

更新：2026-10-19。入口为 [`ablation6.py`](../../../../tools/h100/phase6/ablation6.py)，协议 `h100-gpt67-tp8-ablation6-v1`。按确认的方案，C/A/B沿用消融4实现，将Flux baseline改为 **原生Flux GemmOnly＋与Megatron共用的朴素Tensor TACO**。每窗口 **预热10步、正式50步**。本页记录源码核对和执行方案，尚无消融6 GPU结果。

## 1. 核对结论：同一量化协议，不是同一执行实现

消融4的融合Flux与Megatron＋量化都采用固定H128/E4M3 TACO规则，但前者不是把后者的Python张量代码直接放入Flux；不能把“同协议”写成“同一份实现”或“逐位相同”。

| 项目 | 消融4 Flux融合v2 | Megatron朴素量化 |
| --- | --- | --- |
| 量化对象 | BF16 GEMM的远端来源贡献；本地贡献不量化 | 相同 |
| 分组与变换 | 每128元素、归一化Hadamard、adaptive scale | 相同数学规则 |
| FP8与载荷 | E4M3、最大有限值448；128字节payload＋两份FP32 scale＝136字节 | 相同格式与边界 |
| 编码执行 | GEMM epilogue内的warp CUDA实现 | 多个PyTorch张量算子、临时张量及类型转换 |
| 解码与累加 | CTA CUDA解码；来源按rank+1…rank，每次BF16舍入 | 张量Hadamard解码后转BF16；按相同来源顺序逐次BF16相加 |
| 通信 | 融合Flux的IPC packet写入与同步 | `dist.all_to_all_single`，本地分片不发送 |
| 数值一致性 | CUDA归约树、`rsqrtf`等运算细节独立实现 | PyTorch归约、变换及转换；不保证与CUDA路径逐位相同 |
| GEMM / 反向 | Flux GemmV2，原生形式的BF16/STE反向适配 | Megatron原生GEMM及linear反向，量化通信以AllGather反向近似 |

源码依据：

- 朴素路径：[TensorTacoRS、pack_rows、unpack_rows](../../../../tools/a800/phase5/e2e/megatron_taco.py)，[Hadamard/独立参考](../../../../tools/a800/phase4/reference.py)。
- 融合v2：[CUDA codec](../../../../tools/h100/phase6/fused_v2/src/gemm_rs/taco_codec.cuh)、[融合epilogue](../../../../tools/h100/phase6/fused_v2/src/gemm_rs/epilogue_evt.hpp)、[解码/环形累加](../../../../tools/h100/phase6/fused_v2/src/gemm_rs/taco_runtime.cu)、[固定Python前端](../../../../tools/h100/phase6/fused_v2/python/flux/gemm_rs_taco.py)。
- A/B共用融合v2、C沿用候选实现的分派：[ablation4_sources.py](../../../../tools/h100/phase6/ablation4_sources.py)。H100构建只做既定SM90适配，并保留冻结来源核验。

这说明原消融4中Flux/Megatron的性能与loss差异同时包含GEMM和量化/通信执行实现的差异，不能仅归因于GEMM，也不能仅归因于量化误差。本实验把两个baseline的量化/通信代码对齐。

## 2. 五组定义

| 组 | 策略ID | GEMM / 调度 | 量化和通信实现 |
| --- | --- | --- | --- |
| C：完整版（选择性量化） | `interleaved_arrival_selective_attn_mlp` | 原C的交替＋到达表、SM90编译GemmV2 | 原C优化codec、双工作区；原1/64选择性mask |
| A：交替＋完整量化 | `interleaved_all_attn_mlp` | 原A交替顺序、SM90编译GemmV2 | 原固定融合v2、全远端、单工作区、输出clone |
| B：交替＋到达＋完整量化 | `interleaved_arrival_all_attn_mlp` | 原B交替＋到达表、SM90编译GemmV2 | 与A相同固定融合v2，全远端 |
| Flux＋朴素量化 | `flux_tensor_taco_attn_mlp` | **原生Hopper GemmV3的GemmOnly**，先产生本地BF16 partial | **同一份TensorTacoRS**：编码→All-to-All→解码→BF16累加 |
| Megatron＋朴素量化 | `native_taco_attn_mlp` | 原生Megatron GEMM | **同一份TensorTacoRS**；保留原生linear forward/反向 |

五组均覆盖MLP＋Attention：32个 `mlp.linear_fc2` 与32个 `self_attention.linear_proj`。C的1/64是每来源448个远端tile的预算上限，最多7个，保留原校准mask和到达表；不使用消融5中更大的预算。A/B/C共用已验证的原消融4构建，未重写其CUDA实现。

新Flux组不调用 `GemmRS`：不能先完成一次BF16 ReduceScatter再调用Tensor TACO，那会重复通信并改变量化对象。实际实现见 [ablation6_flux_tensor.py](../../../../tools/h100/phase6/ablation6_flux_tensor.py)。两个baseline从同一会话文件导入同一个 `TensorTacoRS` 类、`pack_rows`、`unpack_rows` 和同一份 `reference.py`；除了TP8适配和调用计数，朴素codec规则保持原样。

**相同codec不等于整步结果必然逐位一致。** 两种GEMM可能产生不同的BF16 partial；Flux采用公共GEMM/RS反向适配，Megatron保留原生linear路径。新的Flux/Megatron对比固定了量化/通信实现，仍包含GEMM与接入路径差异。B/A继续比较到达优先；C/B仍同时改变mask与候选实现；A/新Flux同时改变GEMM算法、融合方式和排序，不是纯交替顺序的单因素比较。本次不把消融4的融合v2作为第六组。

## 3. 模型、步数与样本规模

固定GPT6.7B、32层、H4096/FFN16384、32 heads、S2048、micro batch1/global batch4、TP8/DP1/PP1/CP1、BF16，seed1234、token_seed4321、padded vocab33792。Attention完整shape `[2048,4096,4096]`、本地K512；MLP `[2048,4096,16384]`、本地K2048。每正式optimizer step包含4个微批、8192 token。

本协议独立冻结 `execution_timing={warmup_steps:10,timed_steps:50}`，worker、记录校验和报告分母统一50。smoke/profile仍用既有预检设置。原消融4/5的新会话默认60步及既有20步历史报告不变，不修改旧compare配置。

本次总脚本安排三次独立消融6，分别新建 `compare1`、`compare2`、`compare3`，各自 `--blocks 5 --rounds 2 --nvls-enable 0`；全部通过后再执行三次消融7。第二轮反转块顺序和块内策略顺序。**50步是每个正式窗口的步数，不是整个实验只有50步。**

| 项目 | compare1 | compare2 | compare3 | 消融6合计 |
| --- | ---: | ---: | ---: | ---: |
| 正式窗口 | 50 | 50 | 50 | 150 |
| 正式optimizer step（不重复计rank） | 2500 | 2500 | 2500 | 7500 |
| 正式rank记录 | 400 | 400 | 400 | 1200 |
| smoke/profile窗口 / rank | 10 / 80 | 10 / 80 | 10 / 80 | 30 / 240 |
| profile trace | 40 | 40 | 40 | 120 |
| 策略/角色mapping | 6 | 6 | 6 | 18 |

只有A/B/C需要到达/顺序mapping，共3组×2角色＝6项；Flux GemmOnly没有ReduceScatter tile调度，因此不伪造其mapping。三次各自重新预检，每份严格报告完整通过后才准备和执行下一次。

## 4. 需要新增的GemmOnly构建

现有消融4运行库的构建脚本只保留 `gemm_rs` 及公共组件，没有编译 `comm_none/GemmOnly`。不能直接拿旧融合库当作GemmOnly库，也不能悄悄回退成 `torch.matmul`。

[ablation6_build.py](../../../../tools/h100/phase6/ablation6_build.py) 在**新目录**准备独立源码快照，只编译 `comm_none` 与公共CUDA/THS组件，注册原生SM90、132 SM、BF16 RCR GemmV3，保留上游cooperative/ping-pong与default/StreamK选择。它不启用TACO宏或ReduceScatter算子；不更改上游GemmOnly计算代码。构建来源、注册器、源码、二进制及Python扩展均记录哈希。新库只供Flux朴素组使用。

构建CUDA默认12.8，运行CUDA默认12.4，与原容器约定一致。第一次执行需要编译这一个新库；A/B/C沿用已有库。可用 `--flux-build` 复用已经成功编译的消融6专用构建，不能指定原消融4的 `fused/taco_fused`。

## 5. 容器内一键先执行消融6三次，再执行消融7三次

总入口为 [detach_ablation6_then7.py](../../../../tools/h100/phase6/detach_ablation6_then7.py)。先将本次 `tools/h100/phase6/` 的新增和修改代码同步到原75容器。在仓库根目录执行：

```bash
cd /workspace/TempName
export H100_A6_PYTHON=/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX="$CXX"
export DEBUG=0
export MAKEFLAGS=

# 只打印计划，不创建目录、不编译、不查询GPU。
"$H100_A6_PYTHON" tools/h100/phase6/detach_ablation6_then7.py

# 新建批次，后台依次构建、预检、执行六次独立比较。
"$H100_A6_PYTHON" tools/h100/phase6/detach_ablation6_then7.py --execute
```

固定串行顺序为 `ablation6/compare1 → compare2 → compare3 → ablation7/compare1 → compare2 → compare3`。先编译一次消融6 GemmOnly，再完成三次消融6的独立预检、计时和报告；全部通过后才派生消融7计划、编译一次改进B，并执行三次消融7。每次新建目录与独立run_id，同一实验三次核验相同源码、运行库、计划、通信配置及测量环境。

| 实验 | 每次块数×轮数×组数 | 次数 | 正式窗口总数 | 正式optimizer step总数 |
| --- | --- | ---: | ---: | ---: |
| 消融6 | 5×2×5 | 3 | 150 | 7500 |
| 消融7 | 6×2×3 | 3 | 108 | 5400 |
| 合计 | 每窗口10步预热＋50步计时，NVLS=0 | 6 | **258** | **12900** |

消融7保留三组均衡调度所需的6块，而不是套用消融6的5块；其C为1/16预算、B为改进到达调度，详见[消融7定义](消融实验7-改进到达优先.md)。本脚本不改变消融6的A/B/C定义，也不将两个实验拼成同场配对报告。

默认父构建根目录为 `outputs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/build`。脚本从该构建的 `fused/manifest.json` 读取并核验原NCCL开发文件路径，用于两个新库链接；路径已迁移时用 `--nccl-root` 指定同时含 `include/nccl.h` 和 `lib/libnccl_static.a` 的目录。可通过 `--build`、`--python`、`--megatron`、`--build-cuda-home`、`--cuda-home`、`--jobs` 指定环境。`--flux-build` 和 `--ablation7-build` 可分别复用已编译的消融6/7专用构建；消融7构建必须来自同一个 `--build` 父构建。复用构建仍会重新执行六次完整预检。

默认输出为 `logs/h100/phase6/ablation6-then7-3x-50steps-北京时间戳/`：

```text
batch-state.json
handoff/launch.json
handoff/supervisor.json
handoff/container-supervisor.pid
handoff/container-supervisor.log
ablation6/flux-build/{manifest.json,compile.log,flux-gemm-only/}
ablation6/compare{1,2,3}/{submission,preflight,report}.json
ablation6/compare{1,2,3}/report.md
ablation6/compare{1,2,3}-controller.log
ablation7/build/{ablation7-manifest.json,plans.json,improved/}
ablation7/compare{1,2,3}/{submission,preflight,report}.json
ablation7/compare{1,2,3}/report.md
ablation7/compare{1,2,3}-controller.log
```

`batch-state.json`记录当前阶段、每次run_id、已完成报告路径及SHA256。已有输出目录拒绝复用；任一构建、准备、预检、正式窗口或报告失败即停止，保留日志，不自动重试。

采用与前批相同的双重fork托管，确认PPID=1后返回启动状态；stdin接 `/dev/null`，stdout/stderr写监督日志，关闭VS Code或终端不影响已托管的进程。启动成功不表示实验完成，以顶层 `batch-state.json.status=completed` 和六份严格报告为准。`--foreground --execute` 可用于前台诊断。运行仍依赖容器和主机存活。

原 [detach_ablation6_run_two.py](../../../../tools/h100/phase6/detach_ablation6_run_two.py) 保留原来的“仅消融6两次”行为；本次3＋3批次请使用新总入口。

## 6. 分步入口

```bash
export H100_A6_BUILD=/workspace/TempName/outputs/h100/phase6/ablation6-gemm-only-new
export H100_A6_COMPARE=/workspace/TempName/logs/h100/phase6/ablation6-compare-new
export H100_A6_PARENT=/workspace/TempName/outputs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/build

"$H100_A6_PYTHON" tools/h100/phase6/ablation6.py build-prepare "$H100_A6_BUILD"
"$H100_A6_PYTHON" tools/h100/phase6/ablation6.py compile "$H100_A6_BUILD" \
  --cuda-home /usr/local/cuda-12.8 --jobs 2 \
  --nccl-root /workspace/TempName/outputs/h100/phase6/ablation3-gpt67-tp8-scope-20261017-163146/nccl-release
"$H100_A6_PYTHON" tools/h100/phase6/ablation6.py prepare "$H100_A6_COMPARE" \
  --build "$H100_A6_PARENT" --flux-build "$H100_A6_BUILD" \
  --blocks 5 --rounds 2 --nvls-enable 0
"$H100_A6_PYTHON" tools/h100/phase6/ablation6.py run "$H100_A6_COMPARE" \
  --python "$H100_A6_PYTHON" --cuda-home /usr/local/cuda-12.4
"$H100_A6_PYTHON" tools/h100/phase6/ablation6.py run "$H100_A6_COMPARE" \
  --stage all --python "$H100_A6_PYTHON" --cuda-home /usr/local/cuda-12.4 --execute
"$H100_A6_PYTHON" tools/h100/phase6/ablation6.py report "$H100_A6_COMPARE"
```

`build-prepare/prepare --source-only`仅供CPU演练，worker永久拒绝GPU执行，不能修改标签升级为真实实验。已有A/B/C和新GemmOnly构建必须都是真实构建，或者都为source-only演练。

## 7. 严格报告与解释边界

- 两个baseline冻结并核验相同朴素codec和独立参考源码哈希；worker还记录实际 `TensorTacoRS.reduce` 调用数。每投影正式50×4＝200次，预热另计；不能只声明相同协议却调用不同实现。
- profile核验两角色各128次投影调用；两个baseline均须出现编码、All-to-All、解码/累加。Flux另须出现256次GemmOnly调用及SM90 GEMM kernel，拒绝融合解码或额外ReduceScatter。
- 核验八rank齐全、50个正式loss/grad_norm、正确库/路由/工作区、零skip/fallback、相同初始化/RNG/token/运行环境，以及独立预检哈希；相对BF16 L2≤5%、相对codec参考L2≤1%的门槛不变。
- `ms/step = max(八rank elapsed_seconds) × 1000 / 50`。平均训练loss只计rank0的50个正式值；最终loss取每窗口第50个正式step记录的值，再跨窗口平均，不含预热或预检。
- 主表保留全部样本。按完整五组块配对bootstrap10000次、seed20261009，报告全部10项配对；区间不作多重比较校正，无4%收益门槛。敏感性分析沿用本轮同组中位数×1.10阈值并排除整块。
- 固定合成微批重复训练仅用于该性能/数值协议，不代表真实语料泛化；50步结果不能直接混入消融4/5的60步均值或末步loss。

报告主表列顺序为：

| 指标 | C：完整版（选择性量化） | A：交替＋完整量化 | B：交替＋到达＋完整量化 | Flux＋朴素量化 | Megatron＋朴素量化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 平均耗时（ms/step） | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 平均训练loss | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 最终训练loss的平均 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |

## 8. 开发验证状态

已通过11项消融6 CPU检查，覆盖五组平衡/反转调度、A/B/C定义保持一致、真实生成适配器的audit前后分派、GemmOnly后直接调用共享codec、实际调用计数、源码/步数/构建漂移拒绝、50窗口完整模拟报告、缺rank/截短轨迹/错误profile拒绝、两次顺序与失败停止、只读计划。另有36项scope、消融4、消融5及历史步数兼容回归通过，合计47项。文档本地链接、shell示例及Python语法检查通过。

新增3＋3总脚本另通过8项CPU测试，覆盖六次串行顺序、两种构建失败、正式执行失败、第三次报告/输入漂移拒绝、独立run_id、复用构建仍重新预检、NCCL记录校验和默认只读预览。测试使用模拟子进程与报告元数据，不启动GPU训练。

上述初版检查仅覆盖CPU源码准备与模拟证据，未覆盖真实生成器编译；后续首次构建失败和修复验证见第9节。**新baseline尚未通过H100实测预检。** 一键脚本在原容器先编译，再通过数值与profile门槛后才进入正式窗口；CPU检查和编译通过均不等同于GPU数值或性能验证。

## 9. 2026-10-19首次3＋3批次编译失败及修复

回传批次 `ablation6-then7-3x-50steps-20261019-122720-434246` 的顶层状态为 `failed`，停止于 `ablation6/compile`。后台托管成功，消融6的 `comparisons` 仍为空，消融7为 `pending`：**六次实验均未进入预检或正式训练，没有可汇总的性能/loss结果。**

当时的GemmOnly构建日志显示：`gen_comm_none.cc:29` 向 `main_template` 传参时出现 `invalid initialization of reference`。日志已识别 GNU11.4.0 和 CUDA12.8.93；编译环境检查通过后，新增生成器中的未限定 `make_tuple(...)` 因 `std::vector/std::string` 参数触发ADL，选中了 `std::make_tuple`，而接口要求 `cute::tuple`。该失败会话目录已按用户要求清理，根因保留在本节。

[ablation6_gemm_only_generator.cc](../../../../tools/h100/phase6/ablation6_gemm_only_generator.cc) 已显式使用 `cute::make_tuple(...)`，与上游生成器一致。这是构建注册器的类型修复，保留原BF16/GemmV3参数空间、共享Tensor TACO、A/B/C实现、10＋50步和3＋3执行顺序。

此前Python模拟测试没有覆盖真实C++编译。新增 [test_ablation6_generator.py](../../../../tools/h100/phase6/test_ablation6_generator.py) 使用实际CUDA头文件和C++编译器，编译并运行生成器，核验仅生成一个SM90/H800/BF16/RCR注册文件及6个GemmOnly配置；不调用GPU。修复前错误已在本地GCC11重现，修复后该编译测试通过。可在容器复核：

```bash
CUDA_HOME=/usr/local/cuda-12.8 CXX=/usr/bin/g++-11 \
  /workspace/conda-envs/flux-h100-ablation3-20261017/bin/python \
  -m unittest discover -s tools/h100/phase6 -p test_ablation6_generator.py -v
```

本次复核 **19项既有消融6/3＋3启动器测试＋1项真实生成器编译测试全部通过**。另在本地GNU11.4.0、CUDA12.8.61环境完成全新GemmOnly源码快照的CMake配置、6个配置的SM90内核编译、`libflux_cuda.so` / `libflux_cuda_ths_op.so`链接安装及 `setup.py build_ext --inplace`，最终manifest状态为 `built`。临时产物位于 `/tmp/ablation6-fixed-build-rce5pqpd/build/`，保持 `source_only=true`、`gpu_validated=false`，不能作为正式实验构建复用。

本机没有NVIDIA驱动库 `libcuda.so.1`，按正式运行环境进行的动态加载检查在该依赖处停止；这不属于已回传批次的生成器编译错误，也未通过改动加载器掩盖。**原H100容器的实际加载、八卡数值/profile预检及训练仍待重新执行。**

将修复同步至原容器的 `tools/h100/phase6/` 后，在仓库根目录重新执行总入口即可。默认自动创建新时间戳批次；不要复用已清理的失败会话目录或覆盖历史manifest哈希。

```bash
cd /workspace/TempName
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python \
  tools/h100/phase6/detach_ablation6_then7.py --execute
```

## 10. 第二次会话准备失败：解析原校准的冻结源码副本

批次 `ablation6-then7-3x-50steps-20261019-152811-461354` 的GemmOnly构建已完成（manifest为 `built`），之后在 `ablation6/compare1-prepare` 停止，尚未执行预检和正式窗口。错误为旧校准记录中的 `tools/h100/phase6/scope_calibrate.py` 无法按原SHA256解析。

消融7开发更新了当前的 `scope_calibrate.py` 与 `plan.py`，旧校准计划仍登记它们生成证据时的版本。原校准会话同时保存并登记了 `calibration/controllers/` 副本，但旧解析器只检查工作树和最早的candidate冻结目录，因此没有找到已经存在的正确副本。

| 原文件 | 旧计划登记的SHA256 |
| --- | --- |
| `scope_calibrate.py` | `c215e1926c928b2fc3189ee19b8adc10edd75ac6be07fb8fe15c437d4c894c00` |
| `plan.py` | `e110229caedff76e28110271c7a468d4dd3d18f000b1c7a59b73d57c293715f1` |

修复位于 [ablation_plan.py](../../../../tools/h100/phase6/ablation_plan.py) 和 [scope_plan.py](../../../../tools/h100/phase6/scope_plan.py)：校验当前文件或最早冻结版本失败时，可使用**同一旧计划已登记、同名且期望SHA256相同**的 `controllers/` 副本，并重新计算文件SHA256。只允许phase6控制器源码使用此路径；测量数据、运行库、未知文件以及缺失/被修改的副本仍拒绝。历史副本只用于核对证据，不导入执行；当前消融7仍运行新的到达调度代码。未改写旧计划、旧源码、二进制或预检门槛。

新增 [test_frozen_input_resolution.py](../../../../tools/h100/phase6/test_frozen_input_resolution.py) 覆盖7项检查：已登记副本的精确解析、当前文件优先、本地路径迁移、旧冻结源兼容、未登记/错误名称/错误哈希/缺失副本拒绝、篡改拒绝及原始数据不得借用控制器副本，并核对真实Attention/MLP历史记录中的两份源码。

本次7项新增检查及56项既有回归全部通过，合计 **63项**。另外逐项审计两角色各79份登记输入：本地各78份通过，包括先前失败的两份控制器；对实际raw数据进行CPU重放后，原Attention/MLP mask保持一致，改进到达算法仍均为0个接受交换组。

本地回传档案缺少原校准登记的一份 `runtime.cpython-310.pyc`，因此完整可执行父构建的准入仍需在原容器验证；不会为了离线通过而跳过该文件。当前修复针对的是已确认的两份源码路径解析问题，不能把CPU回放或合成测试当作八卡执行验收。

按本次要求，已清理本地 `logs/h100/phase6/` 下两份状态明确为 `failed` 的3＋3会话：`ablation6-then7-3x-50steps-20261019-122720-434246`、`ablation6-then7-3x-50steps-20261019-152811-461354`。原始消融3校准档案、消融4/5成功结果和父构建均保留。此处清理针对本地拉回副本，不表示已经操作远端容器。
