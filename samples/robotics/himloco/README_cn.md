# HIMLoco 融合 Go2 策略

[English](README.md)

<a id="overview"></a>
## 概述

HIMLoco 从六帧 45 维观测估计内部状态，输出 12 维策略动作。本例使用
[himloco_lab](https://github.com/IsaacZH/himloco_lab) 导出的融合 estimator／actor，
该项目是 [HIMLoco](https://github.com/OpenRobotLab/HIMLoco) 的 Isaac Lab 实现。
独立训练的检查点不能互换。迁移源为 X5 提交
`ac115717197920355fc390bb04299b20e6436864`。

统一 Python SDK 入口、显式模型准备、离线输入以及源转换／评测工具和双语说明已迁入；C++ 纯阶段核心已完成主机检查，SDK／CLI 迁移仍在推进。量化说明继承现有源方案，本次文档工作不要求重新实跑验证，尚未完成整套独立验收。

输入 `obs_history` 为 float32 `[1,270]`，当前帧在前；输出 `actions` 为 float32
`[1,12]`。不额外归一化、不更新历史、不缩放输出、不发送机器人指令。
源控制器在模型边界之外应用 `default_joint_position + 0.25 * actions`。

<a id="support-matrix"></a>
## 支持矩阵

| 目标 | 制品 | 统一 Python | 统一 C++ |
| --- | --- | --- | --- |
| X5 | Bayes-e BIN | 已实现；主机 SDK 替身测试，板端未运行 | 纯阶段核心已验证；SDK／CLI 待迁移，板端未运行 |
| S100／S100P／S600 | 无匹配发布制品 | 不支持 | 不支持 |

源板测环境为 RDK OS 3.5.0-beta、DNN Runtime 1.24.5、HBRT 3.15.55。
这些是历史源证据，不是统一版本重新验证的环境结论。

<a id="prerequisites"></a>
## 前提

主机预览与核心测试需要 Python、NumPy、PyYAML；真实推理需要 X5、匹配的 BSP
`hbm_runtime` 和准确的发布 BIN，不安装 PyPI 上无关的同名包。使用发布模型不需要
转换工具链、Torch 或训练环境。以下命令均在仓库根目录运行，虚拟环境可通过
`PYTHON` 指定 shell 包装器解释器。

<a id="quickstart"></a>
## 快速开始

```bash
python samples/robotics/himloco/runtime/python/main.py --list-models
python samples/robotics/himloco/runtime/python/main.py --target x5 --dry-run
bash samples/robotics/himloco/model/download_model.sh --target x5 --dry-run
```

预览不加载板端 SDK、不下载、不写结果。显式准备发布模型后，在 X5 上执行离线输入：

```bash
bash samples/robotics/himloco/model/download_model.sh --target x5
python samples/robotics/himloco/runtime/python/main.py --target x5 \
  --output-dir outputs/himloco
```

每次使用新输出目录。推理不隐式下载；板型或模型摘要不符，在 SDK 创建前失败。
单文件输入、外部模型身份、调度、报告位置、预热和库集成见 Python 说明。

<a id="expected-results"></a>
## 预期结果与历史测量

默认预热 10 次，处理 21 条源索引观测，生成 `000000.bin` 到 `000020.bin`，每个
含 12 个小端 float32 动作，并写入 `report.json`。完整结果为 `status: completed`；
可写的失败报告保留部分文件与失败索引。输出是数值证据，不是模型制品或执行器指令，
本例不包含实时控制环。

源记录 MIX PTQ 输出余弦相似度为 `0.999606`；同一模型、100 条输入、10 次预热的
历史性能如下：

| 源运行时 | 计时范围 | 平均值 | 顺序吞吐 |
| --- | --- | --- | --- |
| Python | `HB_HBMRuntime.run` | 0.885 ms | 1129.37 FPS |
| C++ | `hbDNNInfer` + `hbDNNWaitTaskDone` | 0.350 ms | 2853.09 FPS |

编译器估计为 0.063 ms。以上均继承源记录，不是本轮实测；计时范围不同，新 Python
任务计时还包含适配器校验／复制，不能当作纯设备耗时比较。完整源百分位数据与评测
口径见[评测说明](evaluator/README_cn.md)。
离线动作一致不能证明观测构造、关节映射、控制环行为或闭环稳定性。

<a id="directory"></a>
## 目录

- `model/`：显式获取固定摘要的 X5 BIN。
- `runtime/python/`：CLI／应用、输入来源、绑定／共享 runner 与纯策略阶段。
- `test_data/`：21 个原样观测文件和源清单。
- `tests/`：核心、元数据和 CLI 检查，明确使用模型／SDK 夹具。
- `conversion/`：原融合导出、校准和 Mapper 方案及双语说明。
- `evaluator/`：浮点格式／动作对照、输入准备与历史测量。
- `runtime/cpp/`：纯策略阶段及主机检查；SDK／CLI 仍待迁移。

<a id="entry-points"></a>
## 入口

- [模型包](model/README_cn.md)：身份、路径、准备与摘要。
- [Python 运行](runtime/python/README_cn.md)：命令、全部选项、输出与公开 API。
- [输入来源](test_data/README_cn.md)：源索引、字节格式与摘要检查。
- [C++ 核心](runtime/cpp/README_cn.md)：阶段接口、主机构建、资源边界及迁移状态。
- [模型转换](conversion/README_cn.md)：融合导出／校准／MIX 配方。
- [模型评测](evaluator/README_cn.md)：数据、命令、指标及源参考值。

<a id="license"></a>
## 许可

Sample 代码遵循仓库 [Apache-2.0 许可](../../../LICENSE)。上游策略与 rollout 数据
保留各自条款，代码许可不能替代外部模型或数据集的权利说明。
