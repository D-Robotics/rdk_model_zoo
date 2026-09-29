[English](README.md) | 简体中文

# PointNet 椅子点云部件分割

<a id="overview"></a>
## 算法与来源

PointNet 为每个 XYZ 点预测四类椅子部件之一：`back`、`seat`、`leg`、`arm`。
共享 MLP 提取点特征，最大值聚合形成全局特征。本例保持输入点顺序，不是整幅点云分类、
目标检测，也不代表支持 ShapeNet 全部类别。
[论文](https://arxiv.org/abs/1612.00593)与[官方实现](https://github.com/charlesq34/pointnet)
说明算法；交付模型参考[S100 PointNet 项目](https://gitee.com/chenguanzhong/rdk_-s100_-point-net_-official)。
HBM 需单独下载，Git 仓库没有附带模型文件。

<a id="support-matrix"></a>
## 支持与实测矩阵

| Target | 变体 | Python | C++ |
| --- | --- | --- | --- |
| s100 | chair，四类部件 | supported-not-run | not-supported |
| x5 / s100p / s600 | 无已发布制品 | not-supported | not-supported |

主机 fixture 测试覆盖阶段接口、精确目标选择、metadata 校验和源前处理对照，不证明 HBM
或板端 SDK 已验证。当前统一入口板测为 not-run；下方图像和性能均为原始记录。

<a id="prerequisites"></a>
## 环境前提

使用 RDK S100 及其配套 `hbm_runtime`，不要用 PyPI 同名包替代板端 SDK。
入口要求 Python 3.10+、NumPy、PyYAML；绘图另需 matplotlib。源资料未固定最低固件/SDK
版本，也没有测得的内存/磁盘预算；这些条件仍待统一入口板测确认。预留 HBM 和结果文件空间。
主机 help/list/dry-run 不需要模型或 SDK。

<a id="quickstart"></a>
## 快速体验

在 S100 的仓库根目录执行。下载是显式步骤，`predict` 不会下载模型。
输入 `test_data/chair.pts` 已随仓库提供。
```bash
# cwd: repository root
python3 -m pip install numpy PyYAML matplotlib
bash samples/vision/pointnet/model/download.sh --target s100
python3 samples/vision/pointnet/runtime/python/main.py --target s100
```

退出码 0 且生成 `outputs/pointnet/result.json` 表示命令完成。对照查看 `result.png` 和
`result_orig.png`；图轴沿用源可视化的 X/Z/Y 排列。运行时校验点数与编译模型一致，不静默采样或补点。

<a id="expected-results"></a>
## 预期结果

`labels.npy` 是 int32 `(N,)` 部件标签，顺序对应输入行；`result.json` 包含各类点数、归一化
信息、制品身份和实际 metadata。计数之和为 N，但不要求任意输入都出现全部四类。
图像看起来合理不能代替精度评估。

![历史椅子分割结果](test_data/readme_img/chair_res.png)

<a id="directory"></a>
## 目录职责

- `model/`：显式下载及制品身份。
- `runtime/python/`：CLI、绘图、独立 binding/runner 和四阶段模型类。
- `conversion/`：保留网络/算子说明，并列出转换缺失前提。
- `evaluator/`：功能检查与历史性能，不是数据集评估器。
- `test_data/`：原始椅子点云和参考图。
- `tests/`：不依赖 SDK 的数值与异常边界测试。

<a id="entry-points"></a>
## 入口索引

[模型准备](model/README_cn.md) · [Python 参数与 API](runtime/python/README_cn.md)
· [转换说明](conversion/README_cn.md) · [评估说明](evaluator/README_cn.md)。本例没有 C++ 实现。

<a id="license"></a>
## 许可

代码遵循仓库 [Apache-2.0](../../../LICENSE)。训练项目和模型权重保留各自条款；
发布清单未提供独立权重许可声明，不能把示例代码许可直接当作权重许可。
