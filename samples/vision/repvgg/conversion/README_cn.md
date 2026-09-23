# RepVGG 转换

<a id="source-model"></a>
## 源模型

源说明采用官方 RepVGG 训练权重，示例为 `create_RepVGG_B1g2(deploy=False)`，经 `repvgg_model_convert()` 后导出 ONNX。没有提供可执行导出脚本、源码修订、PyTorch 版本或权重摘要。训练形式权重必须先完成重参数化才能用于部署编译。

<a id="toolchain-targets"></a>
## 工具链与目标

固定源 rdk_x5 @ac11571 的 6 份原样 YAML，X5 march 为 `bayes-e`，无 S 配方。源未固定 OE 版本，重建时须记录实际环境。

| Config | ONNX input path | Working directory | Compiled basename |
| --- | --- | --- | --- |
| `RepVGG_A0_config.yaml` | `./RepVGG-A0.onnx` | `RepVGG-A0_224x224_nv12` | `RepVGG-A0_224x224_nv12.bin` |
| `RepVGG_A1_config.yaml` | `./RepVGG-A1.onnx` | `RepVGG-A1_224x224_nv12` | `RepVGG-A1_224x224_nv12.bin` |
| `RepVGG_A2_config.yaml` | `./RepVGG-A2.onnx` | `RepVGG-A2_224x224_nv12` | `RepVGG-A2_224x224_nv12.bin` |
| `RepVGG_B0_config.yaml` | `./RepVGG-B0.onnx` | `RepVGG-B0_224x224_nv12` | `RepVGG-B0_224x224_nv12.bin` |
| `RepVGG_B1g2_config.yaml` | `./RepVGG-B1g2.onnx` | `RepVGG-B1g2_224x224_nv12` | `RepVGG-B1g2_224x224_nv12.bin` |
| `RepVGG_B1g4_config.yaml` | `./RepVGG-B1g4.onnx` | `RepVGG-B1g4_224x224_nv12` | `RepVGG-B1g4_224x224_nv12.bin` |

<a id="export"></a>
## ONNX 导出

源说明采用官方 RepVGG 训练权重，示例为 `create_RepVGG_B1g2(deploy=False)`，经 `repvgg_model_convert()` 后导出 ONNX。没有提供可执行导出脚本、源码修订、PyTorch 版本或权重摘要。训练形式权重必须先完成重参数化才能用于部署编译。

本 sample 没有可执行且已验证的导出命令。须在上表路径准备匹配图，名义输入 RGB NCHW 1×3×224×224、输出 ImageNet-1k。YAML input_shape/input_name 为空，维度与名称从图读取，必须核对。

<a id="calibration"></a>
## 校准

所有 YAML 需要 `./calibration_data_rgb_f32`（float32、校准 default），训练输入 RGB/NCHW、运行输入 NV12。mean 为 123.675/116.28/103.53，scale 为 0.01712475/0.017507/0.01742919。缺少数据集选择/数量、准备脚本及实际校准文件。须核对浮点语义与图和归一化，不能给 NV12 目录改名冒充 RGB 校准数据。

<a id="compile"></a>
## 编译

补齐缺失图与校准前提后，才可在 OE 中执行以下条件命令。本次迁移未运行。

```bash
# cwd: repository root, then conversion directory
cd samples/vision/repvgg/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./RepVGG-A0.onnx
hb_mapper makertbin --model-type onnx --config RepVGG_A0_config.yaml
```

该配置预期输出为 `RepVGG-A0_224x224_nv12/RepVGG-A0_224x224_nv12.bin`，各配置均使用 latency/O3。各变体目录/前缀不同，形式如 `RepVGG-A0`。编译文件名用连字符，而发布文件名用下划线。每份 YAML 还配置删除 Quantize/Dequantize/Transpose/Cast/Reshape 节点并启用 dump_calibration_data；这些变换后须验证图语义，改文件名不能证明等价。

<a id="validation"></a>
## 转换后验证

状态 not-run。推理前核对 packed NV12 几何 224×224、squeeze 后为 (1000,) 的 F32 分数输出。首个变体示例：

```bash
# cwd: repository root on X5
python3 samples/vision/repvgg/runtime/python/main.py --target x5 \
  --asset-id x5:repvgg:RepVGG_A0_224x224_nv12.bin \
  --model-path samples/vision/repvgg/conversion/RepVGG-A0_224x224_nv12/RepVGG-A0_224x224_nv12.bin \
  --test-img samples/vision/repvgg/test_data/gooze.JPEG
```

准确引用用于选择契约，不证明重建字节等于发布字节。记录新哈希和图来源，对照源/统一输出，交付前评估精度。

<a id="artifacts"></a>
## 产物

编译路径见上表，逐变体发布文件与目标见[模型准备](../model/README_cn.md#artifacts)，下载落在 sample model 目录。移动已验证构建时保留变体与来源，改名本身不是修复。

<a id="known-gaps"></a>
## 已知缺口

缺少固定框架/OE/权重、可执行导出、校准准备及转换/板端精度证据。YAML 与原许可声明逐字节保留。可使用发布制品下载，不声明端到端转换可复现。
