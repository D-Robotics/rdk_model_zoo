[English](./README.md) | 简体中文

# 模型转换

本目录提供 MobileNetV3 的转换配置、校准数据准备和 HBM 编译说明，
重新整理命令。Model Zoo 已在 `../model/` 提供预编译的 S100 和 S600 HBM 模型。

> 仓库附带的 `mobilenetv3_config.yaml` 针对 **S100**（`march: "nash-e"`）。
> 发布到 `rdk_s600/MobileNet/` 的 S600 版本基于同一份源 ONNX 和同一套量化配置，
> 只需把 `march` 改成 `nash-p` 即可。

## 文件

| 文件 | 说明 |
| --- | --- |
| `mobilenetv3_config.yaml` | 面向 NV12 运行时输入的 S100 转换配置 |
| `get_mobilenetv3_onnx.py` | 从 timm 导出 `mobilenetv3_large_100` ONNX |
| `timm2onnx_local.py` | 原版本地权重导出模板；使用前需手动修改模型名和权重路径 |
| `get_calibration_data.py` | 生成 float32 BGR 校准 `.npy` 文件 |

## ONNX 导出

原始 sample 使用 timm 实现：

- 模型：`mobilenetv3_large_100`
- HuggingFace 模型页：`timm/mobilenetv3_large_100.ra_in1k`
- 依赖：`timm`、`onnx`、`onnxsim`、`torch`

在线导出：

```bash
pip install timm onnx onnxsim
huggingface-cli login
python3 get_mobilenetv3_onnx.py
```

脚本会输出：

```text
input: (3, 224, 224)
mean (0.485, 0.456, 0.406)
std (0.229, 0.224, 0.225)
Simplified model is valid.
Simplified model saved to mobilenetv3_large_100.onnx
Total number of parameters in the model: 5470832
```

如需使用本地权重，先编辑 `timm2onnx_local.py` 再运行。该文件按原版 sample
保留为模板，需要用户设置模型名和权重路径。

## 校准数据

原始 sample 使用 ImageNet 校准数据。准备 100 张校准图片后生成 BGR float32
`.npy` 文件：

```bash
python3 get_calibration_data.py
```

脚本输出目录：

```text
./calibration_data_bgr/
```

## 编译

验证 ONNX 模型：

```bash
hb_compile --model mobilenetv3_large_100.onnx --march nash-e
```

使用提供的 YAML 编译：

```bash
hb_compile --config mobilenetv3_config.yaml
```

关键转换配置：

| 项目 | 配置 |
| --- | --- |
| 源模型 | `mobilenetv3_large_100.onnx` |
| 运行时输入 | NV12，两个输入：Y plane 和 UV plane |
| 训练输入 | BGR，NCHW |
| 校准数据 | `./calibration_data_bgr`，float32 |
| 目标 march | S100 使用 `nash-e` |
| 输出前缀 | `mobilenetv3_224x224_nv12` |

原始 YAML 还保留了 `node_info` 配置，用于将指定节点放到 BPU 且使用 int16
输入/输出。除非源 ONNX 图发生变化，否则不要删除这些配置。

原始量化记录：

```text
TensorName: output
Calibrated Cosine: 0.911233
Quantized Cosine: 0.909042
```

原始工具链性能参考：

```text
FPS (1 core): 2616.81
latency: 0.38 ms (382.1 us)
BPU conv original OPs per run: 433,179,520
```

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
