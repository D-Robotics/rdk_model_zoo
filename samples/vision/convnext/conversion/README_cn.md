[English](README.md) | 简体中文

# ConvNeXt 模型转换

## 可复现的 Atto D2 X5 流程

维护中的 `convnext_atto_d2_x5.yaml` 从
[timm/convnext_atto.d2_in1k](https://huggingface.co/timm/convnext_atto.d2_in1k/tree/435c0a8d02d5b872e90c8cde36887465982cb399)
（Apache-2.0）构建独立新产物，固定版本为
`435c0a8d02d5b872e90c8cde36887465982cb399`，权重 SHA-256 为
`0f70bc49f1896c8c15af857d2d1d418abfaf097a62b4e29a35e6e65b752f5953`。
Atto/Femto/Pico/Nano 是 timm 对 [ConvNeXt](https://arxiv.org/abs/2201.03545)
架构扩展的小规格。

`export.py` 严格加载本地 safetensors，不通过 `pretrained=True` 动态下载权重。
导出固定 1×3×224×224 RGB float32 ONNX（opset 11），简化并检查图，使用固定随机输入
在 ONNX Runtime CPU 与 PyTorch 间对照 logits。相邻 `.export.json` 记录输入、输出、
哈希及环境版本。这是主机图对照，不是 ImageNet 精度或板端推理。

```bash
# Repository root, export environment: torch 2.8.0, timm 1.0.20,
# safetensors, onnx 1.17.0, onnxsim 0.4.36, onnxruntime 1.19.2.
python3 samples/vision/convnext/conversion/export.py --variant atto \
  --checkpoint /path/to/pinned/model.safetensors \
  --output /path/to/run/convnext_atto_d2_224x224.onnx
python3 samples/vision/convnext/conversion/prepare_calibration.py \
  --images /path/to/frozen/images --count 200 \
  --output /path/to/run/calibration_rgb_f32
cp samples/vision/convnext/conversion/convnext_atto_d2_x5.yaml /path/to/run/compile.yaml
# X5 OE container, cwd /path/to/run; resolve recipe paths inside this container.
hb_mapper checker --model convnext_atto_d2_224x224.onnx --model-type onnx --march bayes-e
hb_mapper makertbin --model-type onnx --config compile.yaml
hb_model_info compiled/convnext_atto_d2_bayese_224x224_nv12.bin
hb_perf compiled/convnext_atto_d2_bayese_224x224_nv12.bin
```

在 x86 Linux 的既有 X5 工具链镜像
`registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8` 中转换。
实测工具版本为 hb_mapper 1.24.3、hbdk 3.49.15、HBRT 3.15.55.0。
目标为 `bayes-e`（X5），本次没有声明 S100/S100P/S600 适配。

## 校准与 Runtime 契约

`prepare_calibration.py` 按排序冻结图片，使用 `numpy.tofile` 写原始 float32
RGB/NCHW 二进制张量（`.rgb`，不是 `.npy`），保持 0–255，不提前归一化。
使用与原生 `atto-d2` Runtime 相同的 Pillow bicubic 短边 256、四舍五入居中裁剪 224。
ImageNet mean/scale 由 Mapper 融合，避免重复归一化。相邻 manifest 记录图片与张量
哈希、数量和生成脚本身份。校准图片及张量留在内部，不上传公开目录。

配置声明 BT.601 video-range NV12 Runtime 输入，与 OpenCV 转换匹配；输出为
1,000 类 float32 logits。采用 default PTQ、latency/O3 和独立前缀
`convnext_atto_d2_bayese_224x224_nv12`。生成的 BIN 对应原生 `atto-d2` 规格。
见[模型目录](../model/README_cn.md)和[Runtime](../runtime/python/README_cn.md)。

## 验收与发布

2026-11-02 的 checker/compiler、主机图对照已通过。当前为 **converted**：
板端冒烟、精度、Runtime 1/2 并发、C++ 1/2 流均 **not-run**。
`hb_perf` 和 OE HTML 是编译器估计，不得写入板端性能指标。
BIN 与 SHA256SUMS 放在 `rdk_model_zoo_web/rdk_x5/convnext/`；版本化附属材料放在
`rdk_model_zoo_web/assets/rdk_x5/convnext/<model-stem>/v1/`。

## 历史参考配置

`ConvNeXt_atto.yaml`、`ConvNeXt_femto.yaml`、`ConvNeXt_nano.yaml` 保持
`rdk_x5@ac11571` 原始字节。ONNX 路径存在交叉（atto → femto、femto → 外部 atto、
nano → pico），输出前缀均为 `ConvNeXt-deploy_224x224_nv12`，不能据此识别本次新 BIN。
四个历史模型及测量记录保留原样，历史指标不得套用到新产物。

<!-- converted-batch-20261102 -->
## 固定版本转换配方

[Shared export/calibration instructions](../../../../utils/tools/classification_conversion/README.md).

```bash
python3 utils/tools/classification_conversion/export.py --model convnext_femto.d1_in1k --checkpoint /path/to/pinned/model.safetensors --calibration-sample /path/to/run/calibration_rgb_f32/0000.rgb --output /path/to/run/model.onnx
cp samples/vision/convnext/conversion/recipes/timm-femto-d1-in1k_x5.yaml /path/to/run/compile.yaml
# In the X5 OE container, cwd /path/to/run:
hb_mapper checker --model model.onnx --model-type onnx --march bayes-e
hb_mapper makertbin --model-type onnx --config compile.yaml
```

各规格使用 converted.json 中对应的模型、固定权重及 recipes 配方，输出到全新目录。校准为原始 RGB float32 二进制张量，不重复归一化，不套用历史板端数据。
<!-- /converted-batch-20261102 -->

<!-- converted-all-20261103 -->

## 本轮 S 系列转换规格（2026-11-03）

每个目标使用自己的 PTQ、BC、完整 NodeInfo 和最终 OE。内部 INT16 配置不能替代实际公共 dtype；本轮按已读取的 float32 输出绑定。具体配方及 SHA 见 [converted_targets_20261103.json](converted_targets_20261103.json)。

- `s100 / timm-femto-d1-in1k-s-int16-o1-20261103 / model`: [timm_femto_d1_in1k_s_int16_o1_20261103_s100_model.yaml](recipes/timm_femto_d1_in1k_s_int16_o1_20261103_s100_model.yaml)
- `s100p / timm-femto-d1-in1k-s-int16-o1-20261103 / model`: [timm_femto_d1_in1k_s_int16_o1_20261103_s100p_model.yaml](recipes/timm_femto_d1_in1k_s_int16_o1_20261103_s100p_model.yaml)
- `s600 / timm-femto-d1-in1k-s-int16-o1-20261103 / model`: [timm_femto_d1_in1k_s_int16_o1_20261103_s600_model.yaml](recipes/timm_femto_d1_in1k_s_int16_o1_20261103_s600_model.yaml)
- `s100 / timm-nano-d1h-in1k-s-int16-o1-20261103 / model`: [timm_nano_d1h_in1k_s_int16_o1_20261103_s100_model.yaml](recipes/timm_nano_d1h_in1k_s_int16_o1_20261103_s100_model.yaml)

S 校准必须使用 ONNX 输入域的归一化 `.npy`；NV12 Runtime 的 mean/scale 仍写在配方中。公共工具的 `--target` 和 `--normalization-json` 从同一 checkpoint 导出记录读取实际归一化。

[Shared classification workflow](../../../../utils/tools/classification_conversion/README.md).

原始浮点门禁的阈值、真实输入和 checkpoint 绑定保持不变。编译估算及首条校准输入的 NodeInfo 诊断不作为数据集精度或板端性能。
