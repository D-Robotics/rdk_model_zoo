# ACT：选择 S100 或 S600 源版本

[English](act.md) · [集成总览](../README_cn.md)

## 版本与前提

| 目标 | 准确检出 | LeRobot／数据 | 运行入口 |
| --- | --- | --- | --- |
| S100 | `samples/vla/act`，`326ea043be204de25223d95c7d918efe8672dc66` | D-Robotics 旧 fork；v2.1 | 根 `bpu_control_robot.py`，默认 SO101 |
| S600 | `samples/vla/pi0`，`a32de276bc1681a2b1531012de111eaa1c16acb6` | Hugging Face LeRobot v0.5.2；v3.0 | `models/act/bpu_control_robot.py`，SO100Follower |

S600 源明确不使用旧 D-Robotics fork。配置中能填写某种 march 不代表该板型已验证。
使用匹配板卡镜像的 BPU runtime 和所选指南的环境。板型支持结论继承源说明，本轮未执行板测。

## 初始化与阅读

在 Model Zoo 根目录：

```bash
git submodule update --init --checkout samples/vla/act samples/vla/pi0
git -C samples/vla/act rev-parse HEAD
git -C samples/vla/pi0 rev-parse HEAD
```

随后阅读完整 [S100 指南](../act/README_CN.md)、
[S100 工作流](../act/doc/WORKFLOW_GUIDE_CN.md) 或 [S600 ACT 指南](../pi0/models/act/README_CN.md)。
演示图和训练样例保留在子模块中。下文各条命令均从对应子模块根目录执行。

## 源导出／编译方案

准备训练 ACT 目录（`config.json`、`model.safetensors`）、匹配数据集与校准统计，并在
对应 YAML 中设置路径。S100 用 `bpu_export_config.yaml`；S600 用
`bpu_export_config_s600_calfix.yaml`（`nash-p`）。S600 树中的通用模板默认 `nash-e`，
不能当成 S600 配方。

```bash
# S100，cwd samples/vla/act
python export_bpu_actpolicy.py --config bpu_export_config.yaml
# S600，cwd samples/vla/pi0
python models/act/export_bpu_actpolicy.py --config bpu_export_config_s600_calfix.yaml
```

按源指南在对应 OE 环境中运行生成的 `build_all.sh`。最终 VisionEncoder／TransformerLayers
两个 HBM 和所有归一化 `.npy` 文件要作为一组保留在 `bpu_output/`，统计中的相机名称
必须与运行时名称一致。S600 源要求 `uint8 → /255 → (image-mean)/std`，不能混用旧校准假设。
这里保留可信源方案，本轮未执行。

## 运行与输出

该入口是**机器人控制应用**，不是离线图像分类器。板端环境、模型、机器人及相机配置就绪后，
源运行命令为：

```bash
# S100，cwd samples/vla/act；源使用 SO101
python bpu_control_robot.py --bpu-act-path /data/bpu_output
# S600，cwd samples/vla/pi0；源使用 SO100Follower
python models/act/bpu_control_robot.py --bpu-act-path /data/bpu_output \
  --robot-port /dev/ttyACM0 --camera-index 0 --camera-name front \
  --fps 30 --inference-time 60
```

`/data/bpu_output` 指使用者准备的模型目录。这些命令打开机器人／相机设备并发送动作，
不能当作主机 smoke test。端口、校准与机器人型号必须匹配对应源版本。S600 ACT 输出
100 步动作块，源文档明确不使用强制单动作步数来绕过部署问题。
完整源指南保留安装流程和全部运行参数；本轮未运行转换或执行器命令。

## 历史测量与边界

S600 源记录 20 次预热、200 次 BPU 测量：VisionEncoder 3.92 ms、TransformerLayers
2.29 ms、合计 6.20 ms（161.2 次推理／秒）。这是 BPU 推理吞吐，不是相机／控制环 FPS，
也不是任务成功率。S100 pin 记录部署验证，但未发布数值 benchmark。
这些结论不外推到 S100P／X5、其他检查点或其他机器人。

代码许可保留于各子模块 LICENSE，模型／数据权利和 SDK 条款独立。版本固定检查见
[集成说明](../README_cn.md)。
