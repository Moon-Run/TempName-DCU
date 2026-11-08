# TempName-DCU

海光深算三号 BW1000 的五组对照：Flux-DCU、TACO、交替顺序、到达优先，以及选择性量化。

本仓库使用DTK/HIP PyTorch BF16 BLAS、RCCL大块流水和原生HIP TACO codec；这是Flux设计的DCU适配实现，不是直接编译上游CUDA/CUTLASS Flux。交替组改变实际生产顺序，所有rank使用论文的共同映射。

在已分配的单节点八卡DTK容器内：

```bash
# 预览，无GPU或文件写入。
bash tools/dcu/phase6/run_init.sh --trials 1 --blocks 1

# 预检、独立校准、算子检查后，执行五组GPT6.7B TP8完整训练短测。
bash tools/dcu/phase6/run_init.sh --trials 1 --blocks 1 --execute
```

完整模型、镜像、数值门禁、十块平衡测试及报告说明见 [初始实验文档](docs/design-DCU/init.md)。`--blocks 1`没有置信区间；本地CPU测试通过不等于已在DCU编译或实测通过。

新增到达组使用实测选择的共同生产顺序及双lane独立消费；选择性组按完整输出块闭合量化。计划先校准并冻结，再用于完整训练；未通过独立确认会明确回退。统一入口为 `run_init.sh`。
