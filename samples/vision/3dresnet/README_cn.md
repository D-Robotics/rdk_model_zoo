[English](./README.md) | 简体中文

# 3D ResNet-18（R3D-18）视频动作分类

<a id="overview"></a>
## 算法与来源

R3D-18 将预处理后的 16 帧视频片段分类为 Kinetics-400 的 400 个动作类别之一。模型把 ResNet-18 的二维卷积扩展为三维卷积，同时建模空间和时间特征；统一 runtime 接收已经归一化的 RGB float32 NumPy 片段，并按 source 语义执行 softmax Top-K。

- 论文：[A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248)
- 参考实现：[torchvision r3d_18](https://pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html)
- 仓库位置：`samples/vision/3dresnet`
- source 基线：`platforms/s/samples/vision/3dresnet`，source inventory 为 `380e1a2bf42041af54be6f34935e50197cfadff9`

输入不是视频文件。`test_data/video0.npy` 是已经准备好的 `(1, 3, 16, 112, 112)` 片段；视频解码、抽帧、缩放和归一化不属于本 sample。

<a id="support-matrix"></a>
## 支持与实测矩阵

| Variant | x5 | s100 | s100p | s600 |
| --- | --- | --- | --- | --- |
| R3D-18 / `r3d_18.hbm` | not-supported | supported-not-run | not-supported | not-supported |

| 语言 | 状态 |
| --- | --- |
| Python | S100 supported-not-run；主机 fixture tests 已通过 |
| C++ | not-supported；未提供 C++ 实现 |

本轮没有在 S100 板卡执行 Python runtime。主机测试不证明 `hbm_runtime`、HBM 推理、延迟或板端输出。source 性能记录及其状态见[评测说明](evaluator/README_cn.md)。

<a id="prerequisites"></a>
## 环境前提

- 板卡：发布制品对应 RDK S100；source 没有记录板端镜像和 `hbm_runtime` 版本，因此板端状态为 supported-not-run。
- 主机检查：仓库 `.venv`、Python 3.14.7、`numpy` 和 `PyYAML`；命令见 [Python runtime](runtime/python/README_cn.md)。
- 转换：source 记录使用 x86 Linux 主机上的 OpenExplorer 3.5.0，但没有完整导出、校准和编译配方。
- 磁盘：需要保存下载的 HBM 和随附的 2.4 MB `video0.npy`；没有其他内存约束记录。

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
# source 对 video0.npy 的 Top-1 功能结果描述为 archery。
```

准备模型后也可使用便捷入口：

```bash
# cwd：samples/vision/3dresnet/runtime/python
bash run.sh --target s100 --asset-id s:3dresnet:s100/r3d_18.hbm
```

<a id="expected-results"></a>
## 预期结果

默认片段是 `test_data/video0.npy`。source 将其功能性 Top-1 类别记录为 `archery`；这只是 source 参考，不是本轮板端实测。统一 CLI 输出 JSON，而不是旧版格式化文本：

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

上面的 score 只是字段示例，不断言板端数值。实际列表包含 `--top-k` 条结果；label 从 400 条 JSON 映射读取，并按 source 行为去掉名称中的字面双引号。

<a id="directory"></a>
## 目录职责

```text
3dresnet/
├── conversion/                 # 转换说明和保留的 source 截图
├── model/                      # 显式、manifest 驱动的 HBM 准备
├── runtime/python/             # binding、懒加载 runner、task、labels、CLI、run.sh
├── evaluator/                  # source 功能/性能记录与边界
├── test_data/                  # 预处理片段、400 条标签和 source 截图
├── requirements-host.txt       # 主机测试依赖
├── README.md                   # 英文说明
└── README_cn.md                # 本文档
```

本 sample 没有转换脚本、C++ runtime 或视频解码器。

<a id="entry-points"></a>
## 入口索引

- 模型准备：[`model/README_cn.md`](model/README_cn.md) — 一个精确的 S100 HBM 制品和显式下载命令。
- Python runtime：[`runtime/python/README_cn.md`](runtime/python/README_cn.md) — CLI 与四阶段 `VideoClassificationTask` API。
- 转换：[`conversion/README_cn.md`](conversion/README_cn.md) — source 转换说明、截图及缺失配方边界。
- 评测：[`evaluator/README_cn.md`](evaluator/README_cn.md) — 功能参考、source 性能表和 not-run 状态。
- C++ runtime：未提供，因此不宣称 C++ 支持。

<a id="license"></a>
## 许可

迁移后的 sample 代码使用仓库 Apache-2.0 许可；source 代码也带有相同的 Apache 许可头。manifest 没有记录 publisher SHA-256，也没有记录 `r3d_18.hbm` 的独立权重许可；再次分发前应向发布方确认权重许可和分发条款。
