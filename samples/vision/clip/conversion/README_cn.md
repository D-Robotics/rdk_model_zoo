[English](README.md) | 简体中文

# 模型转换 — CLIP

<a id="source-model"></a>
## 源模型

发布制品对由 X5 BPU 图像编码器（`img_encoder.bin`）和 CPU ONNX 文本编码器（`text_encoder.onnx`）组成。未随附上游权重文件版本、导出脚本、转换 YAML 或校准数据集；原始 CLIP BPE 词表和模型协议保持不变。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>
## 工具链与目标

重新生成图像 `.bin` 需要 OpenExplorer Docker 或对应的 OE 包编译环境。没有记录编译器版本、`march` 配置、文本 ONNX 导出版本或按目标板卡的 YAML。发布目标只有 X5。

| Target | march | OE 版本 | 配置 |
| --- | --- | --- | --- |
| x5 | 未记录 | 未记录 | 未随附 YAML；使用 OpenExplorer/OE 包环境 |

地瓜开发者社区有离线镜像讨论：<https://forum.d-robotics.cc/t/topic/35229>。

<a id="export"></a>
## 导出（ONNX）

没有提供导出命令或源权重文件。文本编码器保持发布的 ONNX 制品，图像编码器保持发布的 `.bin`。已知部署契约为：

| Encoder | 输入 | 输出 |
| --- | --- | --- |
| Image BPU | float32 RGB NCHW `(1,3,224,224)` | float32 `(1,512)` |
| Text CPU ONNX | int32 token `(N,77)` | float32 `(N,512)` |

<a id="calibration"></a>
## 校准

没有随附校准数据集、样本数、量化配置或校准脚本。因此无法从本目录复现校准。

<a id="compile"></a>
## 编译

没有发布两个编码器的转换 YAML 或编译命令。模型下载器只准备已发布的制品，不负责编译。

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

- 没有源权重文件、图像 ONNX 导出脚本、文本 ONNX 导出脚本、转换 YAML、编译器/版本矩阵或校准数据集。
- 可依据清单准备已发布 `.bin`、`.onnx`，但无法从本示例复现转换过程。
- active 清单中两个制品的 SHA-256 均为 `sha256: null (unknown)`。

## 许可

转换文档遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。发布制品沿用其发布来源。

## 独立 ViT-B/32 导出流程

[上游声明](UPSTREAM_NOTICE.md) 固定官方源码 commit、完整 checkpoint SHA-256、
BPE 字典和 MIT 许可。旧 X5 默认模型保留原身份。

```bash
# 仓库根目录；使用现有 Torch 2.8.0、ONNX 1.17.0、ORT 1.19.2 导出环境。
python3 samples/vision/clip/conversion/export_pair.py \
  --source-dir /path/to/CLIP-d05afc436d78f1c48dc0dbf8e5980a9d471f35f6 \
  --checkpoint /path/to/ViT-B-32.pt --images /path/to/frozen/images \
  --opset 17 --output /path/to/new-run/export
```

严格加载权重，保留图像塔、文本塔和学习到的投影；三个真实图片及 seed-0 输入的
上游模型、原生 wrapper、两张 ONNX 和余弦匹配均需通过原阈值对照。原生 BPE token
也与官方实现逐项一致。图像输入为 RGB/NCHW float32 [0,1]，图内仅做一次官方
mean/std；生成 200 张真实图片的独立 `.npy` 校准张量，随机输入仅用于数值验证。
文本 ONNX 使用动态 int32 [prompts,77] 输入和 float32 [prompts,512] 输出，在 CPU 运行。

S SDK 仅编译图像编码器。登记前逐目标读取完整 HBM ABI、NodeInfo 和原始 OE；
模型包同时包含 BPU 图像模型、CPU 文本 ONNX、BPE/分词文件、许可、原始串联报告和
各自的 BPU 转换报告或 CPU 主机报告。原生 `--variant` 在任一 Runtime 加载前校验
两份模型摘要。板端精度、速度和完整流水线延迟尚未实测。

<!-- converted-all-20261103 -->

## 本轮 S 系列转换规格（2026-11-03）

每个目标使用自己的 PTQ、BC、完整 NodeInfo 和最终 OE。内部 INT16 配置不能替代实际公共 dtype；本轮按已读取的 float32 输出绑定。具体配方及 SHA 见 [converted_targets_20261103.json](converted_targets_20261103.json)。

- `s100 / vit-b32-native-int16-o1-20261103 / image`: [vit_b32_native_int16_o1_20261103_s100_image.yaml](recipes/vit_b32_native_int16_o1_20261103_s100_image.yaml)
- `s100 / vit-b32-native-int16-o1-20261103 / text`: [clip_vitb32_text_runtime_20261103.json](recipes/clip_vitb32_text_runtime_20261103.json)
- `s100p / vit-b32-native-int16-o1-20261103 / image`: [vit_b32_native_int16_o1_20261103_s100p_image.yaml](recipes/vit_b32_native_int16_o1_20261103_s100p_image.yaml)
- `s100p / vit-b32-native-int16-o1-20261103 / text`: [clip_vitb32_text_runtime_20261103.json](recipes/clip_vitb32_text_runtime_20261103.json)
- `s600 / vit-b32-native-int16-o1-20261103 / image`: [vit_b32_native_int16_o1_20261103_s600_image.yaml](recipes/vit_b32_native_int16_o1_20261103_s600_image.yaml)
- `s600 / vit-b32-native-int16-o1-20261103 / text`: [clip_vitb32_text_runtime_20261103.json](recipes/clip_vitb32_text_runtime_20261103.json)

原始浮点门禁的阈值、真实输入和 checkpoint 绑定保持不变。编译估算及首条校准输入的 NodeInfo 诊断不作为数据集精度或板端性能。
