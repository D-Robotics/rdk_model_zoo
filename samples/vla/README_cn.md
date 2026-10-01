# VLA 与机械臂策略集成

[English](README.md)

本目录以 Git 子模块保留完整上游仓库。先选择策略及对应板型／版本，再准备环境。
它们与离线 [HIMLoco 足式策略](../robotics/himloco/README_cn.md) 独立。
发布清单将 ACT／Pi0 标记为 manual 集成，没有 Model Zoo 发布制品或自动模型下载。

| 能力 | 检出目录／入口 | 源环境 | 说明 |
| --- | --- | --- | --- |
| S100 ACT | `act/` | 旧版 LeRobot、v2.1 数据集、SO101 控制 | [ACT](guides/act_cn.md) |
| S600 ACT | `pi0/models/act/` | LeRobot v0.5.2、v3.0 数据集、SO100 | [ACT](guides/act_cn.md) |
| S600 Pi0 | `pi0/models/pi0/` | LeRobot v0.5.2、S600 SDK 1.0.2、SO100 双相机 | [Pi0](guides/pi0_cn.md) |

`pi0/` 是第二个集成入口的名称，其仓库也包含 S600 ACT；只初始化 `act/` 不会获得 S600 ACT。


## 初始化准确源码

在 Model Zoo 仓库根目录运行：

```bash
git submodule sync -- samples/vla/act samples/vla/pi0
git submodule update --init --checkout samples/vla/act samples/vla/pi0
git submodule status -- samples/vla/act samples/vla/pi0
```

预期提交：

- ACT：`326ea043be204de25223d95c7d918efe8672dc66`。
- Pi0／S600 工具：`a32de276bc1681a2b1531012de111eaa1c16acb6`。

两者均来自 `D-Robotics/rdk_LeRobot_tools`，父仓库 Git tree 和
[集成清单](integrations.json) 固定版本。复现时不使用 `git submodule update --remote`，
也不切换到移动分支。初始化只获取源码，不安装依赖、不下载模型、不构建运行时、不连接机器人。
普通 clone 未初始化子模块时，源码目录为空是正常状态。

## 阅读与运行

下列指南区分源码准备、模型转换、离线板端推理与实机控制。上游文档中的“仓库根目录”
指对应子模块根，而非 Model Zoo 根目录。训练权重、归一化统计、相机名称和机器人校准
必须是匹配的一组，由使用者提供，本仓库不包含这些部署资源。

- [ACT 指南](guides/act_cn.md)：S100／S600 差异、导出输入、运行入口、历史测量及完整流程。
- [Pi0 指南](guides/pi0_cn.md)：三模型链、固定部署配置、离线输入输出、源结果和控制边界。
- 初始化后可在本地阅读[上游 ACT README](act/README_CN.md) 与
  [S600 工具 README](pi0/README_CN.md)，完整代码、演示图和工作流文档均保留。

本轮核对了提交、源码可获取性、文档路径与清单注册，未执行板端推理或机器人控制。
上游历史测量保留原口径，未重跑量化方案。第三方仓库保留自身完整结构，不以不完整的
Model Zoo 运行包装层替代。集成检查命令为
`python -m unittest discover -s samples/_shared/tests -p test_vla_integration.py`。

## 许可与修改

两个固定版本均包含 Apache-2.0 [许可](act/LICENSE)，权重、数据集、厂商 SDK 和机器人硬件
保留各自条款。修改子模块需要明确的新上游提交及父仓库 pin 更新。旧
旧 `platforms/s/samples/vla/` 路径已随历史目录移除（固定提交 `d2d2a4e0`）；请初始化并使用上述统一路径。
