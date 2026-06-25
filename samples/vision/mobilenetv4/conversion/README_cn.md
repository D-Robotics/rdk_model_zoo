[English](./README.md) | 简体中文

# MobileNetV4 模型转换

本目录提供 MobileNetV4 的转换配置、校准数据准备和 HBM 编译说明，并使用当前
sample 目录结构。Model Zoo 已在 `../model/` 提供预编译的 S100 和 S600 HBM 模型。

> 仓库附带的 `mobilenetv4_{small,medium}_config.yaml` 针对 **S100**
> （`march: "nash-e"`）。发布到 `rdk_s600/MobileNet/` 的 S600 版本基于同一份
> 源 ONNX 和同一套量化配置，只需把 `march` 改成 `nash-p` 即可。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `mobilenetv4_small_config.yaml` | `mobilenetv4_conv_small.onnx` 的 S100 转换配置 |
| `mobilenetv4_medium_config.yaml` | `mobilenetv4_conv_medium.onnx` 的 S100 转换配置 |
| `get_mobilenetv4_onnx.py` | 从 timm 导出 MobileNetV4 small 和 medium ONNX |
| `timm2onnx_local.py` | 原版本地权重导出模板；使用前需修改模型名和权重路径 |
| `get_calibration_data.py` | 生成 float32 BGR 校准 `.npy` 文件 |
| `x86_medium_inference.py` | 原版 x86 参考推理脚本，用于转换侧验证 |

## 源模型

原始 sample 从 timm 导出 ONNX：

- `mobilenetv4_conv_small`，输入 `1x3x224x224`
- `mobilenetv4_conv_medium`，输入 `1x3x256x256`

安装导出依赖：

```bash
pip install timm onnx onnxsim
```

如果需要从 HuggingFace 下载预训练权重，先登录：

```bash
huggingface-cli login
```

导出 ONNX：

```bash
python3 get_mobilenetv4_onnx.py
```

原始脚本记录的导出信息如下：

```text
Processing mobilenetv4_conv_small...
input: (3, 224, 224)
mean (0.485, 0.456, 0.406)
std (0.229, 0.224, 0.225)
Simplified model saved to mobilenetv4_conv_small.onnx
Total number of parameters in the model: 3761480

Processing mobilenetv4_conv_medium...
input: (3, 256, 256)
mean (0.485, 0.456, 0.406)
std (0.229, 0.224, 0.225)
Simplified model saved to mobilenetv4_conv_medium.onnx
Total number of parameters in the model: 9681560
```

如需使用本地权重，先编辑 `timm2onnx_local.py` 再运行。该文件按原版 sample
保留，用作参考模板。

## 校准数据

模型使用 ImageNet 校准图片。原始 sample 期望 100 张
`ILSVRC2012_val_*.JPEG` 图片。

```bash
python3 get_calibration_data.py
```

脚本输出 float32 校准数据。使用前按目标模型在脚本中选择尺寸：

- small 模型使用 `calibration_data_bgr_224`
- medium 模型使用 `calibration_data_bgr_256`

## 编译

快速验证 ONNX：

```bash
hb_compile --model mobilenetv4_conv_small.onnx --march nash-e
hb_compile --model mobilenetv4_conv_medium.onnx --march nash-e
```

使用 YAML 编译：

```bash
hb_compile --config mobilenetv4_small_config.yaml
hb_compile --config mobilenetv4_medium_config.yaml
```

关键配置：

| 项目 | Small | Medium |
| --- | --- | --- |
| 源模型 | `mobilenetv4_conv_small.onnx` | `mobilenetv4_conv_medium.onnx` |
| 运行时输入 | NV12 | NV12 |
| 训练输入 | BGR / NCHW | BGR / NCHW |
| 校准数据 | `calibration_data_bgr_224` | `calibration_data_bgr_256` |
| 输出前缀 | `mobilenetv4_small_224x224_nv12` | `mobilenetv4_medium_256x256_nv12` |
| March | `nash-e` | `nash-e` |

原版 `mobilenetv4_medium_config.yaml` 中输出前缀写为 224，但已发布模型和
README 使用 `mobilenetv4_medium_256x256_nv12.hbm`。当前 YAML 按已发布
S100 模型名称和导出脚本的 256x256 medium 输入修正。

## 原始量化记录

```text
mobilenetv4_medium:
Calibrated Cosine: 0.999759
Quantized Cosine: 0.999863

mobilenetv4_small:
Calibrated Cosine: 0.999892
Quantized Cosine: 0.99988
```

## 原始工具链性能记录

```text
mobilenetv4_medium:
FPS (1 core): 2468.07
latency: 0.41 ms (405.2 us)
BPU conv original OPs per run: 2,160,488,448

mobilenetv4_small:
FPS (1 core): 5698.18
latency: 0.18 ms (175.5 us)
BPU conv original OPs per run: 372,011,136
```

## 模型说明

本 sample 使用公开 S100 HBM 模型。如需重新生成模型，请参考上面的转换说明。

## 转换参考

- ONNX 导出
- PTQ 配置生成

## OE 资源入口

模型转换请在 x86 Linux 主机的 RDK S100 OpenExplore 环境中完成，不建议在板端执行转换。

- OE 资源入口（docker+OE开发包）：<https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE 工具链在线手册：<https://toolchain.d-robotics.cc/>

请从 OE 资源入口获取适配 RDK S100/S100P 的 OpenExplore CPU Docker 镜像，并按实际文件名加载：

```bash
sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar
sudo docker images
```

启动容器时建议挂载当前仓库并增大共享内存：

```bash
sudo docker run -it --rm \
  --network host \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  --workdir /workspace \
  <docker-image-name> /bin/bash
```
