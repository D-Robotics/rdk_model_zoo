# S600 Pi0：源码、部署与离线入口

[English](pi0.md) · [集成总览](../README_cn.md)

## 准确源码与范围

`samples/vla/pi0` 固定 `D-Robotics/rdk_LeRobot_tools` 提交
`a32de276bc1681a2b1531012de111eaa1c16acb6`。模型实现位于 `models/pi0/`，
同一检出还包含 [S600 ACT](act_cn.md)。在 Model Zoo 根目录：

```bash
git submodule update --init --checkout samples/vla/pi0
git -C samples/vla/pi0 rev-parse HEAD
```

初始化后阅读完整[上游指南](../pi0/models/pi0/README_CN.md)。固定源使用 LeRobot v0.5.2、
六维关节角度（度）的 SO100Follower、两路真实相机和一个掩蔽空图槽。
独立运行时依赖 D-Robotics LLM S600 SDK 1.0.2（`nash-p`），运行时不依赖 `libxlm.so`。
Model Zoo 没有对应自动下载模型包，也没有已验证的 X5／S100／S100P 入口。
训练检查点、三个匹配的 HBM、归一化统计、机器人校准和图像由使用者按源指南准备。

## 模型链与源方案

正面／侧面图像和六维关节状态依次经过 SigLIP（三槽）、PaliGemma（36 个 KV tensor）、
Action Expert（十次 flow-matching），输出 `[50,6]` 绝对关节目标。
PaliGemma KV 只属于本次图像对／请求，在同一次请求的十个 Expert 去噪步骤间复用。

源方案采用串联校准：SigLIP HBM 输出用于 PaliGemma 校准，PaliGemma 真实 HBM KV 输出
用于 Expert 校准。完整训练／转换工具及约束保留于上游指南，本轮未重跑。
不要将单独重编译的一个 HBM 混入固定部署包。

相对 `models/pi0/` 的默认部署文件为
`configs/deployments/pi0_full_v5_2cam_positionfp16_siglip_fixed16_paligemma_hbmkv_expert_20260801.json`。
配套 manifest 记录文件身份和大小。请检查资源位置，并另存适合本地环境的部署配置；
源配置保留原 `/root` 和 `/home/sunrise` 路径假设。

## 构建与离线板端推理

以下保留 S600 流程，**不是纯主机测试**。先按上游文档准备 SDK 和依赖，从 Model Zoo 根目录：

```bash
cd samples/vla/pi0
export D_ROBOTICS_LLM_SDK_ROOT=/root/D-Robotics_LLM_S600_1.0.2_SDK/oellm_runtime
bash models/pi0/native/build_standalone_pi0.sh
cd models/pi0
/home/sunrise/lerobot/.venv/bin/python validate_pi0_config.py \
  configs/deployments/pi0_full_v5_2cam_positionfp16_siglip_fixed16_paligemma_hbmkv_expert_20260801.json
/home/sunrise/lerobot/.venv/bin/python pi0_standalone_offline.py \
  --front /data/front.jpg --side /data/side.jpg --state 0 0 0 0 0 0 \
  --output-dir /data/pi0_offline_result
```

替换为真实图像路径，并使用新输出目录。这个入口不打开机器人串口，但会启动本地原生
BPU 引擎并进行 TCP 交换，生成 `actions.npy`（`[50,6]`）、`result.json`（输入路径／状态／
任务、首个动作、形状和请求耗时）及 `engine.log`。请求耗时包含同步消息交换，不是纯 BPU
benchmark。退出 0 表示脚本完成，具体结果仍以本次报告和引擎日志为准。

| 离线参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--front`、`--side` | 必需 | 正面／侧面真实图像文件 |
| `--state` | 必需，六个浮点数 | 源坐标约定下的关节位置 |
| `--output-dir` | 必需，新目录 | 动作数组、结果、引擎日志 |
| `--task` | `Place the RDK camera box on top of the black MCU box.` | 固定自然语言任务 |
| `--config` | 上述部署 JSON | 文件／引擎配置 |
| `--engine-runner` | 脚本同目录的 `run_pi0_standalone_config.sh` | 引擎启动器 |
| `--fixed-noise-file` | `configs/fixed_noise_cv_12345678_fp16.bin` | 固定推理噪声 |
| `--connect-timeout-s` | `120.0` | 引擎连接超时 |

构建产物为 `models/pi0/native/install/bin/pi0_standalone_sdk102`。
`D_ROBOTICS_LLM_SDK_ROOT` 指定 SDK 头文件／库；`PI0_STANDALONE_BIN` 可指定引擎二进制。
配置启动器在执行引擎前校验制品。SDK 路径不符、模型包缺失或摘要错误，应在准备阶段解决，
不能通过重命名模型掩盖。

## 实机控制与历史结果

`models/pi0/run_live_sync.sh` 是源实机入口，会打开 SO100 和相机，以 30 Hz 调用
`send_action()`。它使用同步 50 步动作块，`prefetch_steps=0` 且启用
`--force-model-actions`，不添加相对目标限幅或动作块融合。不要将上述离线命令和这套
实机流程混淆；选择实机运行前需遵循上游完整配置、校准和人工监督说明。

源记录包含 64 个同步实机动作块及一组固定真实输入 BF16／HBM 对照：MAE 0.4405°、
RMSE 0.5903°、最大误差 1.6888°、相对 L2 0.852%、余弦相似度 0.999981858。
这是单输入链路一致性证据，不是任务成功率。本轮未执行板端推理、实机控制、训练或量化。
上游 Apache-2.0 LICENSE 及独立的模型／数据／SDK 条款继续适用。
