# H100 GPT 6.7B TP8 Attention/MLP作用范围实测

更新：2026-10-17，补充消融实验3的策略、平均耗时及平均训练loss对照。原作用范围实验的测试a、b、c、d分别对应本批次repeat1、repeat2、repeat3、repeat4，每次两轮、32个正式窗口，合计128个正式窗口。

单节点8×H100，GPT 6.7B、32层、H4096/FFN16384、32 heads、S2048、micro batch1/global batch4、BF16、TP8。每窗口10步预热＋20步完整optimizer step，取八rank最大elapsed/20；原作用范围实验逐轮表内ms/step为每轮4个配对块的中位数，使用全部样本。新增消融3对照表的平均耗时口径在表后单独说明。

Attention候选为 `interleaved_full_attn`，Attention＋MLP候选为 `interleaved_full_attn_mlp`；对应Megatron＋量化基线分别为 `native_taco_attn` 和 `native_taco_attn_mlp`。两个候选共享双shape库和Attention计划；Attention-only组的MLP保留原生Megatron BF16。

## 完整step耗时对照

| 测试 | 轮次 | 配对块数 | Attention＋MLP ms/step | Attention ms/step | Megatron＋量化 Attention ms/step | Megatron＋量化 Attention＋MLP ms/step |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| a | 1 | 4 | 543.400 | 559.068 | 819.842 | 1049.767 |
| a | 2 | 4 | 541.356 | 560.551 | 818.547 | 1047.405 |
| b | 1 | 4 | 541.307 | 559.332 | 816.824 | 1042.903 |
| b | 2 | 4 | 539.588 | 557.113 | 814.340 | 1053.032 |
| c | 1 | 4 | 553.379 | 572.083 | 825.740 | 1059.926 |
| c | 2 | 4 | 549.061 | 569.658 | 828.329 | 1056.232 |
| d | 1 | 4 | 546.801 | 563.982 | 822.816 | 1050.362 |
| d | 2 | 4 | 549.437 | 567.232 | 826.003 | 1061.608 |

## 配对耗时下降

平均耗时取表内各轮ms/step的算术平均。Attention四列取a–d两轮共8行；交替组合候选及其余三个参考列取 [base-phase6.md](base-phase6.md) 的a–h轮。

| 指标 | Attention＋MLP | Attention | MLP | Megatron＋量化 Attention | Megatron＋量化 MLP | Megatron＋量化 Attention＋MLP | 原生 Megatron | 原生 Flux |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 平均耗时（ms/step） | 545.541 | 563.627 | 537.642 | 821.555 | 823.203 | 1052.654 | 575.986 | 542.284 |
| Attention＋MLP相对耗时下降 | 0.000% | +3.209% | -1.469% | +33.597% | +33.729% | +48.175% | +5.286% | -0.601% |
| 平均训练loss | 7.631617 | 7.763821 | 7.593046 | 6.707388 | 7.410106 | 6.720901 | 7.339843 | 7.339843 |
| 最终训练loss的平均 | 4.076094 | 4.721410 | 4.151009 | 2.504091 | 2.743493 | 1.926777 | 3.302796 | 3.302796 |

“Attention＋MLP相对耗时下降”一行为Attention＋MLP相对各列的耗时下降百分比，按 `100×(1−545.541/该列平均耗时)` 计算；正值表示更快，负值表示更慢。

**平均训练loss直接取原始正式计时记录中的算术平均，不取中位数，不包含预热、smoke或profile。** 每窗口先核对八rank的 `losses` 序列一致，再只计rank0的20个正式optimizer step；不能将同一TP组的八份记录当成八次独立样本。设该列有效窗口数为W：

```text
平均训练loss = sum_w(sum_s(losses_rank0[w][s])) / (W × 20)
最终训练loss的平均 = sum_w(losses_rank0[w][-1]) / W
```

“最终训练loss”指每个窗口第20个正式训练step记录的loss，即 `losses[-1]`；先取各窗口末步值，再跨窗口求算术平均。每窗口提供一个末步值，不取整套实验最后一个窗口代替其他窗口，也不表示模型已经收敛。

历史Attention、Attention＋MLP及对应两个Megatron＋量化列各使用a–d两轮共32个窗口、640个正式step；历史MLP、Megatron＋量化MLP、原生Megatron和原生Flux各使用a–h共48个窗口、960个正式step。这些历史loss使用对应全部正式窗口，不按耗时筛选，也不只取产生中位耗时的窗口。

补充 [消融实验3的策略与平均耗时对照](消融实验/base-phase6-ablation3.md#1-当前结论)：

| 策略 | 基础顺序 | 到达优先 | 远端量化 |
| --- | --- | --- | --- |
| A | interleaved | 关闭 | 全远端物理贡献，候选侧融合H128/E4M3 |
| B | interleaved | 开启 | 与A相同的全远端mask |
| C | interleaved | 与B相同 | 定向选择性mask，每来源最多7个远端tile |

| 作用范围 | 指标 | A平均 | B平均 | C与历史两均值平均（各50%） |
| --- | --- | ---: | ---: | ---: |
| MLP | 平均耗时（ms/step） | 569.967 | 569.603 | 553.775 |
| MLP | 平均训练loss | 7.332271 | 7.332271 | 7.617070 |
| MLP | 最终训练loss的平均 | 4.100240 | 4.100240 | 4.707765 |
| Attention | 平均耗时（ms/step） | 569.514 | 569.527 | 566.730 |
| Attention | 平均训练loss | 8.907540 | 8.907540 | 7.763821 |
| Attention | 最终训练loss的平均 | 5.284326 | 5.284326 | 4.721410 |
| MLP＋Attention | 平均耗时（ms/step） | 547.801 | 549.597 | 547.043 |
| MLP＋Attention | 平均训练loss | 8.344758 | 8.344758 | 7.631617 |
| MLP＋Attention | 最终训练loss的平均 | 6.573645 | 6.573645 | 4.076094 |

“平均耗时”行均为整个模型的完整step耗时；“平均训练loss”行是整个模型的训练loss，不是单个MLP或Attention层的局部loss。A/B来自消融3第一轮9块与第二轮前4块，直接对完整块内全部窗口取算术平均，不做慢窗口筛选。每组13个窗口、260个正式step，两轮按9∶4加权。此前筛选没有剔除任何完整块，因此取消A/B筛选后耗时数值不变。

最后一列的耗时为 `（本次C筛选后均值 + 上方同作用范围的历史完整候选均值）/ 2`，两项各占50%；C继续沿用同轮同组中位数1.10倍阈值的整块筛选规则，本次无触发项。其中历史MLP、Attention、MLP＋Attention分别为537.642、563.627、545.541 ms/step。**loss行沿用同样的跨批次权重**：先分别计算本次C保留窗口与历史完整候选全部窗口的平均训练loss，再各取50%，不是按两批窗口数加权。本次C的MLP、Attention、双模块平均loss分别为7.641095、7.763821、7.631617；与历史对应loss平均后得到表内7.617070、7.763821、7.631617，计算使用未舍入数值，最终保留六位小数。

“最终训练loss的平均”行也使用同一权重：本次A/B各对13个窗口末步loss求平均；C列先分别对本次13个窗口及历史同范围32个或48个窗口的末步loss求平均，再将两项均值各取50%。所有末步均值均从未舍入原始 `losses[-1]` 计算，保留六位小数。

最后一列仅作跨批次汇总值的平均，不是本次C的合并实测或同期配对结果；两边耗时统计口径不同，且旧MLP候选的未选中Attention走Hopper V3，本次为原生Megatron BF16。原始C均值、逐轮结果及来源见上述消融3文档。两表的loss均为固定合成token上的短程训练记录，各窗口重新初始化模型；不能串接为长期训练曲线，也不能用它代替真实验证集loss、准确率或收敛结论。

以下继续列出原作用范围实验a–d的配对结果。下降百分比采用 `100×(1−exp(mean(log(T候选/T参考))))`，95% CI按每轮4个完整配对块bootstrap 10000次、seed20261009计算。正值表示更快。两种作用范围分别对照各自Megatron＋量化基线；扩展MLP净下降以同轮Attention候选为参考。

| 测试 | 轮次 | Attention＋MLP ms/step | Attention ms/step | Attention 对同范围 Megatron＋量化下降% [95% CI] | Attention＋MLP 对同范围 Megatron＋量化下降% [95% CI] | 扩展MLP候选净下降% [95% CI] |
| --- | --- | ---: | ---: | --- | --- | --- |
| a | 1 | 543.400 | 559.068 | +31.803 [+31.686, +31.921] | +48.272 [+48.150, +48.454] | +2.858 [+2.211, +3.630] |
| a | 2 | 541.356 | 560.551 | +31.508 [+31.301, +31.714] | +48.413 [+48.113, +48.593] | +3.430 [+3.324, +3.525] |
| b | 1 | 541.307 | 559.332 | +31.476 [+31.257, +31.681] | +48.134 [+47.867, +48.509] | +3.220 [+2.701, +3.657] |
| b | 2 | 539.588 | 557.113 | +31.720 [+31.477, +32.059] | +48.669 [+48.413, +49.013] | +2.923 [+2.003, +3.723] |
| c | 1 | 553.379 | 572.083 | +30.856 [+30.553, +31.111] | +47.861 [+47.671, +48.112] | +3.333 [+3.238, +3.438] |
| c | 2 | 549.061 | 569.658 | +31.444 [+30.903, +31.832] | +48.020 [+47.952, +48.087] | +3.457 [+2.913, +3.909] |
| d | 1 | 546.801 | 563.982 | +31.458 [+31.205, +31.704] | +47.723 [+47.327, +48.116] | +2.867 [+2.360, +3.371] |
| d | 2 | 549.437 | 567.232 | +27.658 [+19.497, +31.519] | +48.189 [+47.634, +48.749] | +12.007 [+2.720, +27.535] |

测试d第2轮的block3包含慢窗口：Megatron＋量化Attention为992.445 ms/step、Attention候选为842.824 ms/step。上表保留该块；原报告的慢窗口敏感性附表另列筛选结果。

## 数据来源

原始 `results/.../rank*.json` 还保留正式step的 `losses`、`grad_norms`、预热首末loss，以及smoke阶段的 `forward_checks` 数值误差；字段含义及消融3核验情况见 [loss与数值精度记录](消融实验/base-phase6-ablation3.md#8-已保留的loss与数值精度记录)。这些记录用于短程训练和数值检查，不等同于真实验证集准确率。

本次loss复算共核对437个窗口、3496份rank记录：旧a–h四套会话中取交替组合与三个参考组，共192个窗口；旧作用范围a–d四套会话四组共128个窗口；消融3取13个完整配对块、九组共117个窗口。完成标记、运行身份、20步loss/梯度长度及有限值检查通过，同一窗口八rank的loss序列完全相同，统计时只计一次。A/B对应作用范围的loss均值相同来自原始记录，并非复制耗时或沿用另一组数值。未启动GPU实验，也未修改原始报告。

- 测试a：[TTention-MLP-gpt67-tp8-20261015-012238-repeat1](../../../logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-012238-repeat1/compare/report.json)，轮次1/2对应原报告round1/round2。
- 测试b：[TTention-MLP-gpt67-tp8-20261015-012238-repeat2](../../../logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-012238-repeat2/compare/report.json)，轮次1/2对应原报告round1/round2。
- 测试c：[TTention-MLP-gpt67-tp8-20261015-012238-repeat3](../../../logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-012238-repeat3/compare/report.json)，轮次1/2对应原报告round1/round2。
- 测试d：[TTention-MLP-gpt67-tp8-20261015-012238-repeat4](../../../logs/h100/phase6/TTention-MLP-gpt67-tp8-20261015-012238-repeat4/compare/report.json)，轮次1/2对应原报告round1/round2。

第5次因启动时GPU繁忙停止，正式窗口为0；第6、7次尚未启动。

## Phase7 开发前的原版保存与复现（2026-10-17）

Phase7 使用独立的 `tools/h100/phase7/` 和新输出目录。原 Phase6 控制代码、实测候选源码、计划、构建 manifest 及历史日志保持原位；本节不修改前面的实测结果。

已生成 [Phase6 原版快照（本地）](../../../outputs/h100/phase7/phase6-baseline-20261017/README.md)，其 [manifest（本地）](../../../outputs/h100/phase7/phase6-baseline-20261017/manifest.json) 记录维护 HEAD `aa5d3d933319e5e9f38cb11fca5758d4781f9da0`、子模块版本、实测产物身份和逐文件 SHA256。快照 `repo/` 下按仓库相对路径保存 **3999 个文件、61,625,410 字节（约 58.77 MiB）**，包括当前 `src/include/cmake/python`、全部 Phase6 控制代码、A800 依赖代码、冻结候选源、两种 shape 的候选与采样器源码、消融3两种顺序源码、实际 worker/adapter、计划以及四次历史作用范围会话的报告与 provenance。未修改文件权限；通过工具拒绝覆盖和校验和保留原版含义，后续不要在快照内编译或编辑。

**`outputs/h100/phase7/phase6-baseline-20261017/` 仅保留本地，不随 Git 同步，克隆仓库不会自动获得此快照。** 下列快照链接和校验命令以本地文件已存在为前提；异机复现须另行复制快照及所需的外部校准记录。

几个来源不能混淆：

| 保存对象 | 原始来源 | 复现含义 |
| --- | --- | --- |
| Phase6 维护源码 | 上述 HEAD；快照记录 `tools/h100/phase6`、`tools/a800` 的工作树状态 | 创建、校验和调度实验的控制代码 |
| 冻结计算候选 | `outputs/h100/phase6/gpt67-tp8-20261010-112249/candidates-20261010-153517` | 原始计算源码提交 `ff38f5b1ff1c80cb831f58c64b95f42b0379879a`；由 `frozen-candidate-v1-20261013` 固定来源 |
| 本文测试 a–d 共用双 shape 构建 | `outputs/h100/phase6/TTention-MLP-gpt67-tp8-20261014-231236/candidates` | 构建 manifest SHA256 为 `3c58ebc180a07a7a66298f10406ebe46767619f55783000a6e87d247806ba135`；各 repeat 的 `reused-build.json` 指向它，不能以 repeat 输出目录当作独立编译版本 |
| Attention/MLP 实测计划 | 同会话 `plans/measured/attn.json`、`mlp.json` 及 `candidates/plans.json` | 两角色独立计划；快照保留各自校准身份和输入哈希 |
| 消融3 A 与 B/C 的来源 | `outputs/h100/phase6/ablation3-gpt67-tp8-scope-20261016-221018-release1/candidates/base`、`arrival` | 两份 manifest 和全部注册源文件单独保存；B/C 共用 arrival 顺序，mask 由策略区分 |

快照不复制 `.so/.o/.a`、构建树、环境、第三方 Git 仓库和全量训练/profile 日志。保留的历史 manifest 仍包含原 H100 端 `/workspace/TempName/...` 路径及库哈希，它们是身份记录，不表示本机存在这些库。91 项去重后的校准输入均已与原文件核验，其中24项较大的原始校准记录仅以 `plan_inputs` 中的 `repository_path`、字节数及 SHA256 引用保存，**迁移到另一台机器时必须同时携带这些原始记录，才能通过旧计划的证据验证**。快照不能单独替代完整的实测数据交付。

校验命令只读取文本与源码，不导入 Torch、不查询 GPU：

```bash
# 在仓库根目录执行；仅校验快照本身。
python3 tools/h100/phase7/snapshot.py verify \
  outputs/h100/phase7/phase6-baseline-20261017

# 另核对当前原版源码和外部保留的校准证据。
python3 tools/h100/phase7/snapshot.py verify \
  outputs/h100/phase7/phase6-baseline-20261017 --original
```

`create` 对已存在目录明确报错，不能覆盖；新的保存时点须使用新的目录名。`verify --original` 在以后有意修改 Phase6 源码时会失败，但普通 `verify` 仍可用于检查保存版本是否完整。SHA256 清单同时覆盖 manifest 和快照 README。

复建时，在上述 HEAD 的**独立 checkout** 中恢复 `repo/` 的相对目录，并恢复未归档校准输入；不要覆盖当前 Phase7 工作目录。CUTLASS 固定 `df8a550d3917b0e97f416b2ed8c2d786f7f686a3`，NCCL 固定 `8c6c5951854a57ba90c4424fa040497f6defac46`，Megatron 固定 `3ea68ad6042cc1204386ae9364358f7c4de1bc37`。准备好原兼容环境后，按快照内 [README-mlp-attention.md（本地副本）](../../../outputs/h100/phase7/phase6-baseline-20261017/repo/tools/h100/phase6/README-mlp-attention.md) 新建候选、编译及 compare；消融3按 [README-ablation3.md（本地副本）](../../../outputs/h100/phase7/phase6-baseline-20261017/repo/tools/h100/phase6/README-ablation3.md) 操作。

同硬件身份及顺序下复用旧双角色计划的源码准备示例（输出目录必须不存在）：

```bash
export H100_PHASE6_REPRO="$PWD/outputs/h100/phase6/repro-$(date +%Y%m%d-%H%M%S)"
export H100_PHASE6_OLD="$PWD/outputs/h100/phase6/TTention-MLP-gpt67-tp8-20261014-231236"
CUDA_VISIBLE_DEVICES='' python3 tools/h100/phase6/mlp_attention.py build-prepare \
  "$H100_PHASE6_REPRO/candidates" \
  --attn-plan "$H100_PHASE6_OLD/plans/measured/attn.json" \
  --mlp-plan "$H100_PHASE6_OLD/plans/measured/mlp.json"
```

后续按原 README 的 `compile`、`prepare`、预检和正式计时步骤执行；更换 host、GPU UUID/顺序或计算实现时先重新校准两种 shape。重新编译的库形成新的构建身份，不能修改历史 manifest 或沿用历史计时冒充复测。本次仅完成源码/证据保存及 CPU 校验，没有重新编译旧动态库或启动 GPU 实验。
