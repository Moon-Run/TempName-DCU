# H100 到达优先 op3 v2：降低发布成本，限制提前消费拖尾

实现记录：2026-10-20；实测状态更新：2026-10-21。本页记录v2方案；旧方案原文保存在 [op3-v1.md](op3-v1.md)，旧版实测见 [base-op3-v1.md](base-op3-v1.md)。**v2回传实验已完成：辅助扫描拖尾明显缩短，但O/A两次完整训练分别慢0.928%、0.844%，1%目标未通过。完整结果与证据见 [base-op3.md](base-op3.md)。**

## 1. 针对旧版暴露的问题改什么

旧版虽然出现有效提前消费，但两次完整训练分别慢0.739%、0.952%。诊断中，辅助扫描在有消费的调用里经常晚于 GEMM 结束约200 μs；独立算子中新协议的生产也明显变重。profile 是诊断证据，不能把这些时间直接拆成无采样训练的退化量。

本轮同时处理三项实际开销：

| 位置 | op3 v1 | op3 v2 |
| --- | --- | --- |
| 每 fragment 发布 | 所有写线程做 system fence，再 CTA 同步、leader release | CTA 同步汇集写入，再由单个 leader system-release；去掉每线程显式 fence |
| H128 解码 | 128-thread CTA、共享 scratch 和多次 CTA barrier | 每 warp 独立处理一个 H128 group，寄存器＋shuffle 完成变换 |
| 提前消费范围 | 每轮512个CTA，可能消费全部8192组，每CTA串行16组 | 每轮64个CTA，四warp并行；全部轮次合计最多消费1024组 |
| GEMM 结束后的工作 | 辅助消费者可以继续解码整批任务 | 本地 GEMM 后发布 stop epoch，warp 观察到后不再领取下一组 |
| 完成标记 | 每 fragment 一个 done | 每 H128 group 一个 done，允许安全地部分完成 fragment |

A 的生产映射、全远端量化对象、编码算法、来源累加次序和逐次 BF16 舍入都保留。本轮没有重新拟合离线到达排序；提前消费由实际 ready 标记触发。优化对象是生产发布与消费执行路径，不能将总收益全部解释成离线调度收益。

## 2. 发布优化的内存顺序依据

新发布链为：

```text
所有载荷/scale 写线程完成当前 fragment 的写入
→ 所有写线程参加 __syncthreads()
→ thread 0 执行 st.release.sys，写 ready[source][fragment] = epoch
→ 接收端 thread 0 执行 ld.acquire.sys，检查八个来源
→ 接收 CTA 同步
→ 各 warp 读取载荷、解码
```

这里利用的是**同步关系的传递性与 system release 的累积性**。PTX 定义 CTA barrier 建立线程间同步，因果关系可以经同步链传递；CUDA 的 memory-fence 说明指出，system release 还需涵盖通过同步对当前线程可见的其他线程写入。依据为与工具链匹配的 [CUDA12.8 PTX 内存模型](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#memory-consistency-model) 和 [Memory Fence Interference](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#memory-fence-interference)。

旧文“不能只让线程0做fence”的表述需要区分条件：**没有汇集其他写线程的同步时，leader 单独 fence 不足够；保留 CTA barrier 后再 system-release 是不同的完整协议。** v2 没有删除同步，也没有假定其他线程的写入已经自然完成。remote epilogue 原有的末尾 CTA barrier 与 publish 内的 barrier 合并，避免重复同步。

发布仍然以当前回调拥有的16行为单位，不等待其他 CTA 的 fragment，不把 Stream-K 的跨 CTA 关系变成局部 barrier。消费者仍使用 system acquire，最终复用屏障仍保留。跨 GPU 可见性和连续调用正确性必须由新的多输入、延迟rank实验确认。

实现见 [arrival_op3_v2_cuda.cuh](../../../../tools/h100/phase6/arrival_op3_v2_cuda.cuh) 和 [源码变换](../../../../tools/h100/phase6/arrival_op3_v2_sources.py)。

## 3. 快速解码保留哪些数值语义

一个 warp 的 lane `l` 持有原128线程算法中的 `l, l+32, l+64, l+96` 四个元素。Hadamard 的 d=1、2、4、8、16 阶段通过 warp shuffle 完成，d=32、64 在四个寄存器间完成，保持原加减顺序。解码函数内部不使用共享 scratch 或 CTA barrier。

仍执行原计算链：E4M3转浮点、乘 quant scale、H128、原归一化、finite处理、除 adaptive scale。每个来源解码后先舍入 BF16，再按 `rank+1,…,rank` 累加，每次相加后再舍入 BF16；没有改成“全部FP32相加后只舍入一次”。四个warp可并行解码四组。

CPU 数学顺序模型验证了逐级计算映射和舍入，并加入换序、少舍入的负例。最终 GPU 编译器行为与 A 的输出逐位一致性仍由实验入口强制检查；独立 codec 参考检查也保留。实现见 [arrival_op3_v2_decode.cuh](../../../../tools/h100/phase6/arrival_op3_v2_decode.cuh)。

## 4. 提前消费有硬工作量上限，并响应本地完成信号

当前形状为 TP8、`M=2048,N=4096,K_local=512或2048`。每个接收rank拥有64个tile、512个fragment、8192个H128 group。

新 try 默认仍发起两轮，但每轮仅64个CTA，访问同一组 `fragment 0…63`，对应本目标分区中基础生产顺序较早的8个tile。每CTA四个warp并行，每warp最多处理四组；每处理下一组之前检查 stop 和该组的 done。两轮同stream顺序执行，已完成的组不会再次解码：

```text
64 CTA × 16 group = 最多1024个不同group = 全部8192组的12.5%
```

没有扩大这部分预算的在线队列。未就绪的 fragment 直接跳过，不在辅助 stream 轮询等待。第二轮仍是相同集合，不会把上限翻倍。其余工作由 tail 处理。

执行关系为：

```text
主 stream：输入就绪事件 → GEMM/fragment发布 → stop(epoch)
                                               ↓
                                      等待辅助 stream 完成
                                               ↓
                                  warp tail补齐 → 复用屏障

辅助 stream：等待输入就绪事件 → 有限try轮次
                                每组之前检查stop与done
```

stop kernel 在主 stream 的本地 GEMM 之后，独立于辅助 stream 的完成事件。**warp 观察到本代 stop 后不再领取下一组，已经开始的组允许完成。** stop 的跨SM可见性存在传播延迟，因此不能宣称 GPU 在某个绝对时刻立即停止全部工作；硬保证是最多1024个不同group及有限轮次，不是一个已实测的微秒级尾部上限。

主 stream 的 join 仍然存在，用来保证 tail 不与 try 重复写同一输出。改进目标是让 join 前的剩余工作变少，而不是未经所有权协议直接删除 join。诊断新增 `stop_ns`、`groups_finished_after_stop`，用于观察实际结束位置；这些统计不进入正式计时。

## 5. 部分完成与工作区复用

done 改为8192个组级epoch。一个fragment只完成几组时，tail仅跳过这些组，其余组正常消费；不会因为一个warp完成就跳过整块16行。

控制区仍复用原packet中未使用的本地来源slot：

| 内容 | 大小 |
| --- | ---: |
| 8来源×512 fragment ready | 16,384字节 |
| 8192个 group done | 32,768字节 |
| stop epoch及对齐padding | 8字节 |
| 7个uint64诊断字段 | 56字节 |
| 合计 | **49,216字节** |

packet分配尺寸不变。ready、done、stop全部按实例epoch隔离，只在构造时清零，不在每次调用清控制区；拒绝epoch回绕。所有rank完成本代消费的复用屏障仍在，下一代才能覆盖相同工作区。

tail仍只在本rank的生产与有限try结束后运行；其远端等待不会阻止自己的本代生产。没有并发调度时，try也能有限退出，随后tail补齐。局部stop不会要求其他rank提前结束生产。

## 6. 用四组对照区分收益来源

| 组别 | 运行库 | 发布/消费路径 | 对照意义 |
| --- | --- | --- | --- |
| A | A | 原生产、原两道屏障、原scalar CTA解码 | 保留准确基线 |
| D | D | 原生产、原两道屏障；仅换warp解码 | 快速解码的独立收益 |
| S | O，serial | 新fragment发布和warp tail，无辅助try | 新协议串行控制 |
| O | O，overlap | 新发布、受限warp try、stop、warp tail | 本轮候选 |

三套库独立构建，每个实验窗口使用全新TP8进程。A与原登记源码一致；D只修改runtime中的解码发起并新增decoder helper；O包含新的发布/控制区/stop与前端协议。旧op3 v1源码与启动器保留，不使用旧O二进制替代新候选。

报告分别给出 `O/A`、`O/S`、`D/A`、`O/D` 等比较。三个结果字段互不替代：

- `target_validated`：两次完整比较中，O/A的点估计和配对95%区间下界均至少为1%。
- `arrival_benefit_validated`：两次O/S的点估计和95%区间下界均大于0，才支持提前消费的额外收益。
- `decoder_control_passed`：两次D/A的点估计和95%区间下界均大于0，记录快速解码控制组的收益。

如果O/A达到1%，但O/S没有确认正收益，必须报告“整体路径达标，提前消费的增量未证实”。D/A达标也不会被记为O/A达标。

## 7. 算子诊断完成后执行完整训练

一键入口依次执行构建、四组算子正确性/计时、独立profile和完整训练。数值、执行来源与证据完整性检查通过后，自动继续完整训练；算子收益规模和有效重叠不作为准入条件。

算子阶段默认6个完整A/D/S/O块，每窗口两角色各100次预热＋200次正式 `forward().clone()`。保留多seed、多次改变输入/权重、重复调用、rank0/rank7延迟生产，以及独立codec与逐位输出对照。所有组固定连接数8、`NCCL_NVLS_ENABLE=0`，计时关闭诊断。

以下指标保留在报告中用于分析：

1. 有效提前消费：Attention、MLP各至少一个**同一次调用**具有不同stream的GEMM/try交集，且诊断显示至少一组解码完成早于本地最后生产发布。默认每次调用有两轮try，按调用分组匹配，不把其他调用的重叠拿来凑数。
2. 两角色的O/A耗时与配对收益统计。
3. 按完整块配对计算的节省估算 `micro_savings_estimate`：

   ```text
   projected_ms = 128 × [(A_attn_us − O_attn_us) + (A_mlp_us − O_mlp_us)] / 1000
   ```

   报告均值与bootstrap95%区间，不设收益门槛。

**该估算不是测到的整步节省，也不保证迁移到训练。** 即使没有有效重叠、某个角色变慢、估算节省为负或置信区间跨零，也继续完整训练。取消原先5.37 ms和有效重叠的预筛门槛及 `completed_no_candidate` 提前退出分支；数值错误、来源不匹配、证据损坏或运行失败仍会终止任务并记录错误。

完整训练固定GPT6.7B、TP8、10步预热＋50步正式计时、两次独立比较、每次12个四组块，共 **96个正式窗口、768份rank记录、4800个累计optimizer step**。各窗口取八rank最大墙钟，按完整块配对统计；最终1%只与本批A比较。

Megatron仍使用隔离源码副本，四组共用连接数检查补丁；原仓库不改。离线准备、编译、预热和独立profile时间不计入正式50步耗时；每步实际发生的try、stop、join和tail开销必须计入。

## 8. 另一台服务器的一键启动

同步本轮代码后，在原8×H100服务器/容器中执行新的入口：

```bash
cd /workspace/TempName
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python \
  tools/h100/phase6/detach_arrival_op3_v2.py --execute
```

默认后台运行、fresh重建A/D/O；CUDA12.8编译、CUDA12.4运行、jobs2、相邻目录 `../Megatron-LM`。不传 `--nccl-root` 时在新目录构建NCCL。沿用原服务器已验证的编译器设置；路径不同可显式传入：

```bash
/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python \
  tools/h100/phase6/detach_arrival_op3_v2.py --execute \
  --build-cuda-home /usr/local/cuda-12.8 \
  --cuda-home /usr/local/cuda-12.4 \
  --megatron /workspace/Megatron-LM
```

去掉 `--execute` 仅预览，不创建目录、启动子进程或查询GPU；加 `--foreground` 可前台运行。`--op3-build` 只能复用完整可执行的**v2 A/D/O**构建，旧v1构建会被协议核验拒绝。

默认结果目录为 `logs/h100/phase6/arrival-op3-v2-2x-50steps-<时间>/`，以启动时打印的 `Batch:` 为准。重点查看：

```text
batch-state.json                      # 阶段、三个验收字段、算子诊断
handoff/container-supervisor.log      # 后台总日志
operator/report.json                  # 四组算子统计和micro_savings_estimate
operator/profile-summary.json         # 两角色逐调用有效提前消费
fullstep/report.json                  # 只有进入并完成整步阶段后才存在
```

旧入口 `detach_arrival_op3.py` 仍对应v1；此次验证请使用带 `_v2` 的新脚本。实验超时或报错会终止本次worker进程组并保留失败记录，不重试已部分发出的实例。

## 9. 当前验证边界与实现文件

新实现由以下文件组成：

- [源码变换](../../../../tools/h100/phase6/arrival_op3_v2_sources.py)、[发布/控制区协议](../../../../tools/h100/phase6/arrival_op3_v2_cuda.cuh)、[warp decoder](../../../../tools/h100/phase6/arrival_op3_v2_decode.cuh)。
- [构建与封存](../../../../tools/h100/phase6/arrival_op3_v2_build.py)、[算子验证](../../../../tools/h100/phase6/arrival_op3_v2_bench.py)、[完整训练验证](../../../../tools/h100/phase6/arrival_op3_v2_fullstep.py)、[一键后台入口](../../../../tools/h100/phase6/detach_arrival_op3_v2.py)。

CPU测试覆盖数值顺序、组级唯一覆盖、stop传播延迟、partial done、跨代复用、同步反例、D与O隔离、旧库拒绝、四组配对统计、双角色多轮profile匹配、诊断不阻止训练及两次正式验收。真实SM90交叉编译检查包括A/D/O的mapping、runtime、完整GEMM实例与C++ wrapper，证据保存到本轮独立输出目录。

实现阶段的本地验证结果（2026-10-20）：

- 取消预筛门槛前，**131项CPU测试通过**，其中v2新增78项、旧op3回归53项。见 [统一日志](../../../../outputs/h100/phase6/arrival-op3-v2-validation-20261020/all-cpu-tests.log) 和 [输入哈希与状态](../../../../outputs/h100/phase6/arrival-op3-v2-validation-20261020/cpu-validation.json)。
- 取消预筛门槛后，重新执行 `python3 -m unittest discover -s tools/h100/phase6 -p 'test_arrival_op3_v2_*.py' -v`，**78项v2 CPU测试通过**；覆盖无有效重叠、低/零/负估算收益仍继续完整训练，以及最终验收独立于算子诊断。新入口无副作用预览通过；此项仅验证调度与报告逻辑，没有执行GPU训练。
- fresh A/D/O已封存，源码数量分别为351/352/353。见 [构建登记](../../../../outputs/h100/phase6/arrival-op3-v2-20261020/arrival-op3-v2-manifest.json)；A保持原源码，D仅改runtime并新增decoder helper，O修改5个既有文件并新增2个helper。
- **27条真实编译/检查命令、12个对象全部通过**，覆盖三个变体的mapping、runtime、完整GEMM模板实例和THS wrapper。见 [SM90编译结果](../../../../outputs/h100/phase6/arrival-op3-v2-20261020-compile/validation.json) 和 [日志](../../../../outputs/h100/phase6/arrival-op3-v2-20261020-compile/compile.log)。记录为 `completed=true,gpu_used=false,linked_runtime=false`，不是完整动态库链接或GPU运行通过。
- 新入口已执行无副作用预览；Python语法、文档本地链接检查通过。旧源码及旧GPU结果未修改，旧结果页仅将历史设计链接指向归档文件。

实现阶段本机没有执行v2 GPU性能实验。2026-10-21已复核另一台服务器回传的完整四组实验：算子逐位正确性和模型数值检查通过，43/64次O诊断调用存在有效提前消费，但O/A与O/S两次整步比较均退化。详见 [最新实测结果](base-op3.md)；本页上述CPU测试与编译记录仍仅代表实现阶段验证，不能写成“已经提升1%”。
