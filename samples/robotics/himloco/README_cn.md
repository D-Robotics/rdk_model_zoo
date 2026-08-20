[English](./README.md) | [简体中文](./README_cn.md)

# RDK X5 HIMLoco

本示例为 Unitree Go2 HIMLoco 策略提供融合模型导出、PTQ 转换、数值评测以及
RDK X5 Python/C++ 离线推理。

## 算法介绍

[HIMLoco](https://github.com/OpenRobotLab/HIMLoco) 是基于混合内部模型的足式机器人
学习控制器。estimator 输入 6 帧、每帧 45 维的堆叠观测，估计速度和归一化 latent，
再与当前帧观测共同输入 actor。

本示例策略使用 [himloco_lab](https://github.com/IsaacZH/himloco_lab) 训练，它是原始
Isaac Gym 实现向 Isaac Lab 的移植。两者实现相同的策略思想，但独立训练的 checkpoint
不会共享权重。本示例以导出的 himloco_lab Go2 策略接口和产物为准。

## 功能

- 从 `policy.pt` 导出单一固定 shape ONNX，并与 TorchScript 做数值检查；
- 从原生 rollout `.pt` 生成代表性 Mapper 校准输入；
- 使用 MIX PTQ 构建 RDK X5 Bayes-e `.bin` 并记录编译证据；
- 在带原始行号的样本上评估 JIT/ONNX 和 X5/JIT 数值一致性；
- 提供相同 I/O 合同的 Python 和 C++ BPU Runtime。

## 平台兼容性

| 平台 | Runtime 模型 | Python | C++ |
| --- | --- | --- | --- |
| RDK X5 | `.bin` | 支持 | 支持 |

板端推理已在 RDK OS 3.5.0-beta、DNN Runtime 1.24.5 和 HBRT 3.15.55 上验证。
Python 必须使用 BSP 随附、且与板端 `libdnn` 匹配的 `hbm_runtime`。

## 模型合同

| 项目 | 取值 |
| --- | --- |
| 机器人 | Unitree Go2 |
| 源模型 | 融合 TorchScript `policy.pt` |
| 输入 | `obs_history`，float32 `[1,270]` |
| 输出 | `actions`，float32 `[1,12]` |
| 历史顺序 | 当前 45 维观测，随后是前 5 帧观测 |
| 目标平台 | RDK X5，`march: bayes-e` |
| Runtime 输入 | DDR featuremap，不额外预处理 |
| 动作应用 | `default_joint_position + 0.25 * actions` |

融合图包含 estimator、速度/latent 处理和 actor，并以一个 opset-11 ONNX 替代旧的
`encoder.onnx` 与 `policy.onnx` 双模型部署边界。

## 目录结构

```text
himloco/
├── conversion/           # JIT 导出、校准准备和 X5 PTQ
├── evaluator/            # 浮点模型与 X5 数值评测
├── model/                # X5 模型下载脚本与说明
├── runtime/
│   ├── cpp/              # X5 BPU SDK C++ 离线推理
│   └── python/           # X5 Python 离线推理
├── test_data/            # 21 个带源行号的离线 Runtime 输入
├── README.md
└── README_cn.md
```

训练权重、ONNX、校准数据、编译工作目录、rollout、action dump 和报告必须存放在仓库
外部。下载的 `.bin` 位于被忽略的 `model/bayes-e/` 下。

## 快速开始

下载 RDK X5 模型，或按 [模型转换说明](conversion/README_cn.md) 复现模型转换：

```bash
cd model
bash download_model.sh
```

在 RDK X5 上使用显式的模型、输入、输出和报告路径运行任一 Runtime：

```bash
cd runtime/python
bash run.sh --help

cd ../cpp
bash run.sh --help
```

随附的 `test_data/obs_history` 包含独立原生 rollout 中编号 0 至 20 的样本。更多 Runtime
输入可通过 `evaluator/prepare_runtime_inputs.py` 生成。解释板端输出前请先阅读
[评测说明](evaluator/README_cn.md)。

## 模型转换

支持的转换链路为：

```text
policy.pt -> 融合 ONNX -> 代表性校准 -> MIX PTQ -> Bayes-e BIN
```

`conversion/mapper.py` 会为每次 run 生成完整的实际生效 YAML，不需要单独维护 PTQ
模板文件。

## Runtime 推理

- [Python Runtime](runtime/python/README_cn.md)
- [C++ Runtime](runtime/cpp/README_cn.md)

两个实现都会检查单输入/单输出模型合同，并写出 12 维 float32 action 文件。这些 action
dump 是 evaluator 使用的临时数值证据，不是模型资产，不能提交到仓库。

## 评测与性能

MIX PTQ 编译输出 cosine 为 `0.999606`。下表使用相同模型、100 个输入和 10 次预热：

| Runtime | 计时范围 | 平均延迟 | 顺序吞吐 |
| --- | --- | ---: | ---: |
| Python | 同步 `HB_HBMRuntime.run` | 0.885 ms | 1129.37 FPS |
| C++ | `hbDNNInfer` + `hbDNNWaitTaskDone` | 0.350 ms | 2853.09 FPS |

两者计时范围不同，因此不能据此声称 C++ 改变了 BPU 图执行速度。编译器估计为
0.063 ms；对这个小模型，Runtime 提交、调度和同步开销占主要部分。独立评测集上的
板端精度仍是单独门禁，详见 [评测说明](evaluator/README_cn.md)。

## 源码参考

按照 [源码文档生成说明](../../../docs/source_reference/README.md) 生成并浏览 API 文档。

## 注意事项

- 校准 rollout 与独立评测 rollout 不得重叠；
- 离线 tensor 一致不能验证观测构造、控制周期、关节映射、执行器限制或闭环稳定性；
- 必须先在仿真中验证；早期真机测试应提供物理支撑、急停并核对命令和限制；

## 许可证

遵循 Model Zoo 顶层 License。
