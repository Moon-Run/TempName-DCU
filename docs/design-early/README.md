# 设计与实验文档

A800 文档现统一放在 `docs/design/design-a800/`，接续入口为 [A800 instruction.md](design-a800/instruction.md)。

H100 单节点代码移植单独维护在 [H100 instruction.md](design-h100/instruction.md)。A800/H100后续主模型已改为[GPT 6.7B](model-gpt-6.7b.md)，本次未测试；旧0.64B的H100实测保存在[历史记录](design-h100/base-phase6.md)。

| 内容 | 入口 |
| --- | --- |
| 当前6.7B模型、A800/H100准备与历史边界 | [model-gpt-6.7b.md](model-gpt-6.7b.md) |
| 当前实验结论、baseline、验收规则与资源快照 | [A800 接续说明](design-a800/instruction.md) |
| H100 单节点移植、构建流程与无 GPU 验证 | [H100 接续说明](design-h100/instruction.md) |
| 软件版本、Megatron 固定提交与环境配置 | [env.md](env.md) |
| 基础顺序、到达优先与选择性量化的研究思路 | [idea.md](design-a800/idea.md) |
| Phase6 单节点 TP4 | [实验记录](design-a800/base-phase6-tp4.md) · [冻结存档与复跑](design-a800/tp4-archive.md) |
| Phase6 单节点 TP8 | [实验记录](design-a800/base-phase6-tp8.md) · [限时优化记录](design-a800/optimize/tp8/optimize1.md) |
| Phase6 双节点 TP4/DP2 | [实验记录](design-a800/base-phase6-tp4dp2.md) |
| Phase6 双节点 TP8 | [实验记录](design-a800/base-phase6-tp8双节点.md) |

文档纳入 Git；`logs/a800/` 和 `outputs/a800/` 中的实验记录、冻结库与构建产物仍按 [.gitignore](../../.gitignore) 忽略。文档中的这些链接指向原实验工作区，新 clone 不会自动带上对应文件；已发现缺失的历史原始记录在正文保留路径并注明状态。

2026-10-09：A800 文档完成路径和引用维护；新增 H100 单节点移植及 CPU/编译验证记录，没有重新测量性能或更新 GPU 资源状态。
