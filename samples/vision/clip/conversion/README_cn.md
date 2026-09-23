[English](./README.md) | 简体中文

# 模型转换 — CLIP X5

<a id="source-model"></a>
## 源模型

发布制品对由 X5 BPU 图像 encoder（`img_encoder.bin`）和 CPU ONNX 文本 encoder（`text_encoder.onnx`）组成。固定源没有提供上游 checkpoint 版本、导出脚本、转换 YAML 或校准数据集。它保留原始 CLIP BPE 词表和模型协议；本文不新增转换断言。

<a id="toolchain-targets"></a>
## 工具链与目标

源转换说明指出，重新生成图像 `.bin` 应使用 OpenExplorer Docker 或对应 OE 包编译环境，但没有记录编译器版本、`march` 配置、文本 ONNX 导出版本或按 target 的 YAML。发布目标只有 X5。

| Target | march | OE 版本 | 配置 |
| --- | --- | --- | --- |
| x5 | 源未记录 | 源未记录 | 源无 YAML；使用 OpenExplorer/OE 包环境 |

保留源转换说明中的资源指针：地瓜开发者社区离线镜像讨论 <https://forum.d-robotics.cc/t/topic/35229>。该链接是资源指针，不是转换结果。

<a id="export"></a>
## 导出（ONNX）

没有提供导出命令或源 checkpoint。文本 encoder 保持发布的 ONNX 制品，图像 encoder 保持发布的 `.bin`；通用导出命令超出源证据范围。已知部署契约为：

| Encoder | 输入 | 输出 |
| --- | --- | --- |
| Image BPU | float32 RGB NCHW `(1,3,224,224)` | float32 `(1,512)` |
| Text CPU ONNX | int32 token `(N,77)` | float32 `(N,512)` |

<a id="calibration"></a>
## 校准

源中没有校准数据集、样本数、量化配置或校准脚本。因此无法从本目录复现校准，也不提供校准命令。

<a id="compile"></a>
## 编译

源没有发布两个 encoder 的转换 YAML 或编译命令。模型下载器只准备已发布的制品，不负责编译。不要把通用 OpenExplorer 调用伪装成已验证配方。

<a id="validation"></a>
## 转换后验证

维护的验证路径通过 `runtime/python/main.py` 运行发布制品对，计算 prompt cosine 分数，并单独写出可视化。本轮未执行转换或板端验证：状态 `not-run`。

<a id="artifacts"></a>
## 产物

| 产物 | Target | 落盘路径 |
| --- | --- | --- |
| `img_encoder.bin` | x5 图像 BPU | `samples/vision/clip/model/` |
| `text_encoder.onnx` | x5 文本 CPU ONNX | `samples/vision/clip/model/` |

<a id="known-gaps"></a>
## 缺失项

- 没有源 checkpoint、图像 ONNX 导出脚本、文本 ONNX 导出脚本、转换 YAML、编译器/版本矩阵或校准数据集。
- 可依据 manifest 准备已发布 `.bin`、`.onnx`，但无法从本 sample 复现转换过程。
- active manifest 中两个制品的 SHA-256 均为 `sha256: null (unknown)`。
- 转换和板端冒烟验证均为 not-run。

## 许可

转换文档遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。源制品沿用其发布来源。
