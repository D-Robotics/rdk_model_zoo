[English](./README.md) | 简体中文

# 3D ResNet-18（R3D-18）视频动作分类

<a id="overview"></a>
## 算法与来源

R3D-18 用于短视频片段的动作识别。它把 ResNet-18 的卷积扩展为三维卷积，同时学习空间与时间特征，预测 Kinetics 的 400 个动作类别。

参考：[A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248), [torchvision r3d_18](https://pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html).

<a id="directory"></a>
## 目录结构

```text
3dresnet/
├── conversion/  # 导出与量化配置
├── evaluator/  # 评估程序与指标
├── model/  # 模型文件与下载脚本
├── runtime/  # 推理程序
├── test_data/  # 示例输入
├── tests/  # 自动化测试
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── requirements-host.txt  # 源码或数据文件
```

本 sample 没有转换脚本、C++ runtime 或视频解码器。

<a id="support-matrix"></a>
## 支持与实测矩阵

| Variant | x5 | s100 | s100p | s600 |
| --- | --- | --- | --- | --- |
| R3D-18 / `r3d_18.hbm` | not-supported | supported | not-supported | not-supported |

| 语言 | 支持 |
| --- | --- |
| Python | supported（S100） |
| C++ | not-supported；未提供 C++ 实现 |

板端执行需要安装 `hbm_runtime` Python 包的 S100 板卡。已发布的延迟记录及其测量条件见[评测说明](evaluator/README_cn.md)。

<a id="prerequisites"></a>
## 环境前提

- 板卡：RDK S100，使用提供 `hbm_runtime` Python 包的 S 系列系统镜像；镜像与 runtime 版本由部署环境选择。
- 转换：x86 Linux 主机上的 OpenExplorer 3.5.0（见[转换说明](conversion/README_cn.md)）；未包含完整的导出、校准和编译配方。
- 磁盘：需要保存下载的 HBM 和随附的 2.4 MB `video0.npy`。

<a id="quickstart"></a>
## 快速体验

下面是完整的显式流程。第二条命令要求 S100 板卡，不会隐式下载模型。

```bash
# cwd：仓库根目录
bash samples/vision/3dresnet/model/download.sh s100
# 预期：samples/vision/3dresnet/model/s100/r3d_18.hbm

# cwd：仓库根目录；在安装 hbm_runtime 的 S100 板卡上执行
python3 samples/vision/3dresnet/runtime/python/main.py \
  --target s100 \
  --asset-id s:3dresnet:s100/r3d_18.hbm
# 预期：退出码 0，输出含 asset_id、target 和 5 条 predictions 的 JSON；
# 参考：source 记录 video0.npy 的 Top-1 动作类别为 archery。
```

准备模型后也可使用便捷入口：

```bash
# cwd：samples/vision/3dresnet/runtime/python
bash run.sh --target s100 --asset-id s:3dresnet:s100/r3d_18.hbm
```

<a id="expected-results"></a>
## 预期结果

默认片段是 `test_data/video0.npy`；source 记录其 Top-1 类别为 `archery`。CLI 输出如下形式的 JSON 结果：

```json
{
  "asset_id": "s:3dresnet:s100/r3d_18.hbm",
  "target": "s100",
  "clip": ".../test_data/video0.npy",
  "predictions": [
    {"class_id": 5, "score": 0.0, "label": "archery"}
  ]
}
```

score 数值取决于编译产物，上面的数字仅示意字段结构。实际列表包含 `--top-k` 条结果；label 从 `test_data` 中的 400 条 Kinetics 映射读取（加载时会去掉原始标签名称中内嵌的引号字符）。

运行时读取准备好的 RGB float32 片段 `test_data/video0.npy`，形状为 `(1,3,16,112,112)`。调用前完成视频解码、抽帧、缩放和归一化。

<a id="entry-points"></a>
## 入口索引

- 模型准备：[`model/README_cn.md`](model/README_cn.md) — 一个精确的 S100 HBM 制品和显式下载命令。
- Python runtime：[`runtime/python/README_cn.md`](runtime/python/README_cn.md) — CLI 与 `R3D18Classifier` API。
- 转换：[`conversion/README_cn.md`](conversion/README_cn.md) — source 转换说明、截图及缺失配方边界。
- 评测：[`evaluator/README_cn.md`](evaluator/README_cn.md) — 功能参考与已发布的性能记录。
- Runtime 语言：Python。

<a id="license"></a>
## 许可

本 sample 代码使用仓库 Apache-2.0 许可，与 source 代码的 Apache 许可头一致。`r3d_18.hbm` 未记录 publisher SHA-256 和独立权重许可；再次分发前应向发布方确认权重许可和分发条款。
