[English](README.md) | 简体中文

# VLA 与机械臂策略集成

在支持的板卡上使用 ACT 或 Pi0 运行机械臂策略。先按下表选择策略和板卡，再获取源码并准备模型。

| 能力 | 检出目录／入口 | 源环境 | 说明 |
| --- | --- | --- | --- |
| S100 ACT | `act/` | 旧版 LeRobot、v2.1 数据集、SO101 控制 | [ACT](guides/act_cn.md) |
| S600 ACT | `pi0/models/act/` | LeRobot v0.5.2、v3.0 数据集、SO100 | [ACT](guides/act_cn.md) |
| S600 Pi0 | `pi0/models/pi0/` | LeRobot v0.5.2、S600 SDK 1.0.2、SO100 双相机 | [Pi0](guides/pi0_cn.md) |

S100 ACT 使用 `act/`；S600 ACT 和 Pi0 使用 `pi0/`。

## 获取源码

在 Model Zoo 仓库根目录运行：

```bash
git submodule sync -- samples/vla/act samples/vla/pi0
git submodule update --init --checkout samples/vla/act samples/vla/pi0
git submodule status -- samples/vla/act samples/vla/pi0
```

源码版本：

- ACT：`326ea043be204de25223d95c7d918efe8672dc66`。
- Pi0／S600 工具：`a32de276bc1681a2b1531012de111eaa1c16acb6`。

两个子模块均来自 `D-Robotics/rdk_LeRobot_tools`。上述命令会下载[集成清单](integrations.json)中指定的版本。获取源码后，按以下指南安装依赖并准备模型。

## 阅读与运行

上游文档中的命令在对应子模块目录（`act/` 或 `pi0/`）运行。请准备相互匹配的训练权重、归一化统计、相机名称和机器人校准数据。

- [ACT 指南](guides/act_cn.md)：S100/S600 环境、模型导出和推理。
- [Pi0 指南](guides/pi0_cn.md)：模型准备、离线推理和机器人控制。
- [ACT README](act/README_CN.md) 与 [S600 工具 README](pi0/README_CN.md)：初始化子模块后可在本地阅读的源码说明。

运行实机控制前，按上游指南准备硬件和校准数据。

## 许可

两个子模块采用 Apache-2.0，详见[许可文件](act/LICENSE)。权重、数据集和厂商 SDK 遵循各自许可。
