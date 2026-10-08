[English](README.md) | [简体中文](README_cn.md)

# DiffusionDrive 规划示例

<a id="overview"></a>
## 概述

DiffusionDrive 组合三相机 RGB 全景、LiDAR BEV 直方图、自车状态和显式扩散噪声进行轨迹规划。源描述为两步截断扩散解码器，输出八个未来自车位姿，并带有 Agent 状态和七类 BEV 辅助头。本 sample 消费已准备的 NAVSIM 特征，提供 Python 推理、可视化、五案例运行与浮点参考对照；不准备原始传感器数据、不计算完整 NAVSIM 分数、不执行车辆控制指令。

源算法参考：[官方 DiffusionDrive 项目](https://github.com/hustvl/DiffusionDrive)、[CVPR2025 论文](https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html)、[NAVSIM](https://github.com/autonomousvision/navsim)。这些链接用于了解算法；可部署资产的身份由[模型说明](model/README_cn.md)中的发布 HBM 校验和定义。

<a id="directory"></a>
## 目录结构

```text
diffusiondrive/
├── conversion/  # 导出与量化配置
├── evaluator/  # 评估程序与指标
├── model/  # 模型文件与下载脚本
├── runtime/  # 推理程序
├── test_data/  # 示例输入
├── tests/  # 自动化测试
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="support-matrix"></a>
## 支持矩阵

| 目标 | 已发布 HBM | 运行入口 | 准备 |
| --- | --- | --- | --- |
| S100P / nash-m | `s100p/diffusiondrive_r34_256x1024_s100p.hbm` | Python / hbm_runtime | 支持 |
| S600 / nash-p | `s600/diffusiondrive_r34_256x1024_s600.hbm` | Python / hbm_runtime | 支持 |
| S100 / X5 | 无 | 显式拒绝 | 无回退 |

本 sample 没有原生 C++ 源码。保留两个发布 HBM 摘要，下载/加载均校验。`auto` 要求可识别本机身份或显式资产身份，未知主机会被拒绝而不是静默选择 S600。随附主机检查以合成数据验证浮点参考对照与运行元数据契约；真实 HBM 推理需按下文准备 S100P/S600 运行环境。

<a id="prerequisites"></a>
## 前置条件

实际推理需准备匹配 S100P/S600 的 `hbm_runtime`、Python、NumPy、OpenCV，并显式下载目标模型。[模型说明](model/README_cn.md)记录路径与校验和。主机检查与离线比较不需要 SDK。运行包装入口不自动安装依赖或下载模型。

随附 NPZ 已包含 camera/lidar/status/noise 特征，不可替换为原始传感器图像或点云，仓库未打包原始 NAVSIM 特征构建器。使用发布资产无需自行转换；[转换说明](conversion/README_cn.md)列出缺失的导出/校准前提。

<a id="quickstart"></a>
## 快速开始

从仓库根目录检查模型与目标契约，不执行 SDK：

```bash
python3 -m samples.vision.diffusiondrive.runtime.python.main --list-models
python3 -m samples.vision.diffusiondrive.runtime.python.main --target s600 --dry-run
```

显式准备一个模型：

```bash
bash samples/vision/diffusiondrive/model/download.sh --target s600
```

在准备好的 S600 上，以新目录运行默认案例：

```bash
bash samples/vision/diffusiondrive/runtime/python/run.sh --target s600 --output outputs/diffusiondrive
```

执行源五案例；追加 `--dry-run` 可在主机只检查命令和输入：

```bash
bash samples/vision/diffusiondrive/runtime/python/run_all_cases.sh --target s600 --output outputs/diffusiondrive_cases
```

S100P 的下载和推理均选择 `--target s100p`，它使用独立 HBM，修改文件名不能改变模型目标。使用外部路径前请阅读[运行参数与集成](runtime/python/README_cn.md)。

<a id="expected-results"></a>
## 预期结果

每次保存物理量化输入、物理原始输出、解码 `outputs.npz`、`result.png` 与溯源报告。解码结果包含轨迹 `[1,8,3]`、三十个 Agent 状态/概率/掩码，以及七类 BEV 预测。噪声始终来自调用者。原始和解码张量契约不同，诊断量化时应同时保留。

可视化组合相机全景、BEV 语义与 LiDAR 栅格，橙色轨迹、红色筛选后 Agent、蓝色自车。灰色代表道路，大面积灰色不自动意味着色表错误。坐标和类别详情见[测试数据](test_data/README_cn.md)。

S600 参考可视化（源记录）：

![参考 S600 DiffusionDrive 显示](test_data/reference_result.png)

| 参考 case_017 | 参考 case_042 |
| --- | --- |
| ![路口](test_data/case_017/result.png) | ![密集交通](test_data/case_042/result.png) |
| 参考 case_073 | 参考 case_099 |
| ![大道](test_data/case_073/result.png) | ![宽阔路口](test_data/case_099/result.png) |

六组输入/参考和六张结果图均逐字节保留。[评估说明](evaluator/README_cn.md#reference-results)完整保留原 S100P/S600 精度/性能表，含单线程延迟、双线程总吞吐和 S100P 五案例均值；[测试数据说明](test_data/README_cn.md)保留 S600 五案例表。源记录条件：数值对照使用 case_000，profiling 使用 case_017；源记录所有分段 CPU 0.0 ms、全部 BPU 执行。

<a id="entry-points"></a>
## 人与 Agent 的入口

使用 `DiffusionDriveTask.predict`，或由它组合的 `preprocess` → `infer` → `postprocess` 阶段。任务类只处理规划张量语义；SDK 加载/调度、NPZ 读写、下载、绘图和指标均在其外。共享 `NamedArrayRunner` 保留全部具名物理张量并检查板卡/资产身份。[完整 API 示例](runtime/python/README_cn.md#integration-example)包括变量和输入加载过程。

量化处理核对逐轴 scale 与标量零点，拒绝畸形或负 scale，并在整数转换前完成裁剪。评估器要求形状一致；任一向量范数为零时，余弦相似度为未定义。

<a id="license"></a>
## 许可证

Sample 代码遵循仓库 [Apache-2.0 许可证](../../../LICENSE)。DiffusionDrive 与 NAVSIM 资产仍受各自原条款约束。随附示例不构成完整的已授权 NAVSIM 数据集或认证驾驶系统。
