[English](README.md) | [简体中文](README_cn.md)

# DiffusionDrive 规划示例

<a id="overview"></a>
## 概述

DiffusionDrive 组合三相机 RGB 全景、LiDAR BEV 直方图、自车状态和显式扩散噪声进行轨迹规划。源描述为两步截断扩散解码器，输出八个未来自车位姿，并带有 Agent 状态和七类 BEV 辅助头。本 sample 消费已准备的 NAVSIM 特征，保留 S 分支的 Python 推理、可视化、五案例运行与浮点参考对照能力，不准备原始传感器数据、不计算完整 NAVSIM 分数、不执行车辆控制指令。

源算法参考：[官方 DiffusionDrive 项目](https://github.com/hustvl/DiffusionDrive)、[CVPR2025 论文](https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html)、[NAVSIM](https://github.com/autonomousvision/navsim)。源未固定精确上游 checkpoint/导出提交，这些链接是参考资料，不是资产来源证明。

<a id="support-matrix"></a>
## 支持矩阵

| 目标 | 已发布 HBM | 运行入口 | 迁移验证 |
| --- | --- | --- | --- |
| S100P / nash-m | `s100p/diffusiondrive_r34_256x1024_s100p.hbm` | Python / hbm_runtime | 主机夹具通过；板测 not-run |
| S600 / nash-p | `s600/diffusiondrive_r34_256x1024_s600.hbm` | Python / hbm_runtime | 主机夹具通过；板测 not-run |
| S100 / X5 | 无 | 显式拒绝 | 不回退 |

本 sample 没有原生 C++ 源码。保留两个发布 HBM 摘要，下载/加载均校验。`auto` 要求可识别本机身份或显式资产身份，未知主机不再静默选择 S600。主机对照使用随附浮点数组与合成运行元数据，不是真实 HBM 推理。

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

以下源 S600 图片为历史结果，不是本次迁移测量：

![历史 S600 DiffusionDrive 显示](test_data/reference_result.png)

| 历史 case_017 | 历史 case_042 |
| --- | --- |
| ![路口](test_data/case_017/result.png) | ![密集交通](test_data/case_042/result.png) |
| 历史 case_073 | 历史 case_099 |
| ![大道](test_data/case_073/result.png) | ![宽阔路口](test_data/case_099/result.png) |

六组输入/参考和六张结果图均逐字节保留。[评估说明](evaluator/README_cn.md#reference-results)完整保留原 S100P/S600 精度/性能表，含单线程延迟、双线程总吞吐和 S100P 五案例均值；[测试数据说明](test_data/README_cn.md)保留 S600 五案例表。数值对照使用 case_000，profiling 使用 case_017，本次均未重测。源 CPU0ms/全 BPU 声明仍标记为历史记录。

<a id="directory"></a>
## 目录职责

| 目录 | 内容 |
| --- | --- |
| [model](model/README_cn.md) | 精确目标资产、SHA256SUMS、显式下载 |
| [runtime/python](runtime/python/README_cn.md) | CLI、批量入口、严格四输入/四输出绑定、任务阶段与独立绘图 |
| [conversion](conversion/README_cn.md) | 两份保留的 OE3.7.0 PTQ 配置及缺失导出/校准前提 |
| [evaluator](evaluator/README_cn.md) | 严格离线解码/浮点对照，含形状与有限值检查 |
| [test_data](test_data/README_cn.md) | 默认与五个确定性 NAVSIM 特征/参考案例、历史图片 |
| [tests](tests) | 主机源对照、量化、CLI、批量和评估测试 |

<a id="entry-points"></a>
## 人与 Agent 的入口

使用 `DiffusionDriveTask.pre_process`、`forward`、`post_process` 或 `predict`。任务类只处理规划张量语义；SDK 加载/调度、NPZ 读写、下载、绘图和指标均在其外。共享 `NamedArrayRunner` 保留全部具名物理张量并检查板卡/资产身份。[完整 API 示例](runtime/python/README_cn.md#integration-example)包括变量和输入加载过程。

重构修正源逐轴 scale/单零点处理，拒绝畸形或负 scale，防止整数饱和上界回绕，并拒绝评估形状广播；零范数余弦明确记为未定义。主机对照验证源输出算术和六案例绘图，不能替代板端验证。原实现保留于 `platforms/s/samples/vision/diffusiondrive`。

<a id="license"></a>
## 许可证

Sample 代码遵循仓库 [Apache-2.0 许可证](../../../LICENSE)。DiffusionDrive 与 NAVSIM 资产仍受各自原条款约束。随附示例不构成完整的已授权 NAVSIM 数据集或认证驾驶系统。
