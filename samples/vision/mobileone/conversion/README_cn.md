# MobileOne 转换

<a id="source-model"></a>
## 源模型

源引用 apple/ml-mobileone，先加载如 mobileone_s0_unfused.pth.tar 的未融合权重，再执行 `reparameterize_model(model)` 后导出和简化 ONNX。没有提供可执行导出脚本、上游修订、包版本或权重摘要。

<a id="toolchain-targets"></a>
## 工具链与目标

固定源 rdk_x5 @ac11571 的 5 份原样 YAML，X5 march 为 `bayes-e`，无 S 配方。源未固定 OE 版本，重建时须记录实际环境。

| Config | ONNX input path | Working directory | Compiled basename |
| --- | --- | --- | --- |
| `MobileOne_S0_config.yaml` | `./mobileone_s0.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S1_config.yaml` | `./mobileone_s1.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S2_config.yaml` | `./mobileone_s2.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S3_config.yaml` | `./mobileone_s3.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S4_config.yaml` | `./mobileone_s4.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |

<a id="export"></a>
## ONNX 导出

源引用 apple/ml-mobileone，先加载如 mobileone_s0_unfused.pth.tar 的未融合权重，再执行 `reparameterize_model(model)` 后导出和简化 ONNX。没有提供可执行导出脚本、上游修订、包版本或权重摘要。

本 sample 没有可执行且已验证的导出命令。须在上表路径准备匹配图，名义输入 RGB NCHW 1×3×224×224、输出 ImageNet-1k。YAML input_shape/input_name 为空，维度与名称从图读取，必须核对。

<a id="calibration"></a>
## 校准

所有 YAML 需要 `./calibration_data_rgb_f32`（float32、校准 default），训练输入 RGB/NCHW、运行输入 NV12。mean 为 123.675/116.28/103.53，scale 为 0.01712475/0.017507/0.01742919。缺少数据集选择/数量、准备脚本及实际校准文件。须核对浮点语义与图和归一化，不能给 NV12 目录改名冒充 RGB 校准数据。

<a id="compile"></a>
## 编译

补齐缺失图与校准前提后，才可在 OE 中执行以下条件命令。本次迁移未运行。

```bash
# cwd: repository root, then conversion directory
cd samples/vision/mobileone/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./mobileone_s0.onnx
hb_mapper makertbin --model-type onnx --config MobileOne_S0_config.yaml
```

该配置预期输出为 `MobileOne_224x224_nv12_int8/MobileOne_224x224_nv12.bin`，各配置均使用 latency/O3。所有变体共用 `MobileOne_224x224_nv12_int8` 目录及 `MobileOne_224x224_nv12` 前缀，必须隔离构建并保留变体身份。YAML 设置 32 个编译 jobs，需要匹配构建主机资源；未设置删节点覆盖项或 debug_mode。

<a id="validation"></a>
## 转换后验证

状态 not-run。推理前核对 packed NV12 几何 224×224、squeeze 后为 (1000,) 的 F32 分数输出。首个变体示例：

```bash
# cwd: repository root on X5
python3 samples/vision/mobileone/runtime/python/main.py --target x5 \
  --asset-id x5:mobileone:MobileOne_S0_224x224_nv12.bin \
  --model-path samples/vision/mobileone/conversion/MobileOne_224x224_nv12_int8/MobileOne_224x224_nv12.bin \
  --test-img samples/vision/mobileone/test_data/tiger_beetle.JPEG
```

准确引用用于选择契约，不证明重建字节等于发布字节。记录新哈希和图来源，对照源/统一输出，交付前评估精度。

<a id="artifacts"></a>
## 产物

编译路径见上表，逐变体发布文件与目标见[模型准备](../model/README_cn.md#artifacts)，下载落在 sample model 目录。移动已验证构建时保留变体与来源，改名本身不是修复。

<a id="known-gaps"></a>
## 已知缺口

缺少固定框架/OE/权重、可执行导出、校准准备及转换/板端精度证据。YAML 与原许可声明逐字节保留。可使用发布制品下载，不声明端到端转换可复现。
