[English](README.md) | 简体中文

# 模型转换 — CLIP X5

<a id="source-model"></a>
## 源模型

发布制品对由 X5 BPU 图像 encoder（`img_encoder.bin`）和 CPU ONNX 文本 encoder（`text_encoder.onnx`）组成。未随附上游 checkpoint 版本、导出脚本、转换 YAML 或校准数据集；原始 CLIP BPE 词表和模型协议保持不变。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>
## 工具链与目标

重新生成图像 `.bin` 需要 OpenExplorer Docker 或对应的 OE 包编译环境。没有记录编译器版本、`march` 配置、文本 ONNX 导出版本或按 target 的 YAML。发布目标只有 X5。

| Target | march | OE 版本 | 配置 |
| --- | --- | --- | --- |
| x5 | 未记录 | 未记录 | 未随附 YAML；使用 OpenExplorer/OE 包环境 |

地瓜开发者社区有离线镜像讨论：<https://forum.d-robotics.cc/t/topic/35229>。

<a id="export"></a>
## 导出（ONNX）

没有提供导出命令或源 checkpoint。文本 encoder 保持发布的 ONNX 制品，图像 encoder 保持发布的 `.bin`。已知部署契约为：

| Encoder | 输入 | 输出 |
| --- | --- | --- |
| Image BPU | float32 RGB NCHW `(1,3,224,224)` | float32 `(1,512)` |
| Text CPU ONNX | int32 token `(N,77)` | float32 `(N,512)` |

<a id="calibration"></a>
## 校准

没有随附校准数据集、样本数、量化配置或校准脚本。因此无法从本目录复现校准。

<a id="compile"></a>
## 编译

没有发布两个 encoder 的转换 YAML 或编译命令。模型下载器只准备已发布的制品，不负责编译。

<a id="validation"></a>
## 转换后验证

验证路径通过 `runtime/python/main.py` 运行发布制品对，计算 prompt cosine 分数，并单独写出可视化；命令见 [Python runtime](../runtime/python/README_cn.md)。

<a id="artifacts"></a>
## 产物

| 产物 | Target | 落盘路径 |
| --- | --- | --- |
| `img_encoder.bin` | x5 图像 BPU | `samples/vision/clip/model/` |
| `text_encoder.onnx` | x5 文本 CPU ONNX | `samples/vision/clip/model/` |

<a id="known-gaps"></a>
## 补充准备

- 没有源 checkpoint、图像 ONNX 导出脚本、文本 ONNX 导出脚本、转换 YAML、编译器/版本矩阵或校准数据集。
- 可依据 manifest 准备已发布 `.bin`、`.onnx`，但无法从本 sample 复现转换过程。
- active manifest 中两个制品的 SHA-256 均为 `sha256: null (unknown)`。

## 许可

转换文档遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。发布制品沿用其发布来源。
