[English](./README.md) | 简体中文

# ViT 模型转换

本目录提供 CIFAR-10 ViT 模型的 OpenExplore 转换配置、校准数据要求和 HBM
编译步骤。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `config_vit_nv12.yaml` | CIFAR-10 ViT NV12 HBM 模型的 OpenExplore 转换配置。 |
| `hb_compile.log` | 当前配置对应的参考编译日志。 |

## 源模型

源模型是基于 PyTorch 实现训练并导出的 CIFAR-10 Vision Transformer。

- 论文：[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- 项目参考：<https://github.com/xiongqi123123/ViT_PyTorch.git>
- ONNX 模型名：`vit_cifar10_batch1.onnx`

PyTorch 训练环境用于训练或微调 CIFAR-10 ViT 模型，并导出
`vit_cifar10_batch1.onnx`。运行 OpenExplore 编译前，请将导出的 ONNX 文件放在本目录。

## 校准数据

转换配置使用 CIFAR-10 RGB 校准数据。请将校准图片准备为 `float32` NumPy 文件，并放在：

```text
./calibration_data_rgb
```

发布的参考模型使用 PTQ 和 50 张校准图片。运行时输入为 NV12，训练输入为 RGB
NCHW 布局。

## OE 资源入口

模型转换请在 x86 Linux 主机的 RDK S100 OpenExplore 环境中完成，不建议在板端执行转换。

- OE 资源入口（docker+OE开发包）：<https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE 工具链在线手册：<https://toolchain.d-robotics.cc/>

请按照 Docker 官方文档安装 Docker，并验证安装：

```bash
sudo docker --version
sudo docker run --rm hello-world
```

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

## 编译

在本目录执行：

```bash
hb_compile --config config_vit_nv12.yaml
```

预期输出前缀为：

```text
vit_cifar10_batch1
```

核心配置如下：

```yaml
model_parameters:
  onnx_model: './vit_cifar10_batch1.onnx'
  march: "nash-e"
  layer_out_dump: False
  working_dir: './vit_cifar10_batch1'
  output_model_file_prefix: 'vit_cifar10_batch1'
input_parameters:
  input_name: ""
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  norm_type: 'data_mean_and_scale'
  mean_value: 0.4914 0.4822 0.4465
  scale_value: 4.943153707865546 5.014042553191489 4.975124378109453
calibration_parameters:
  cal_data_dir: './calibration_data_rgb'
  cal_data_type: 'float32'
  quant_config: {"op_config": {"softmax": {"qtype": "int32"}}}
compiler_parameters:
  extra_params: {'input_no_padding': True, 'output_no_padding': True}
  compile_mode: 'latency'
  debug: False
  jobs: 8
  optimize_level: 'O2'
  advice: 1
```

## 运行时接口

编译后的 HBM 模型使用两个 NV12 输入 tensor：

- Y plane
- UV plane

重新生成模型时需要保持该输入协议不变，因为 Python runtime 会显式准备并传入这两个平面。

## 参考精度

当前发布配置的 CIFAR-10 参考精度如下：

| 模型 | Top-1 | Top-5 |
| --- | --- | --- |
| ONNX | `74.54%` | `98.36%` |
| HBM | `72.62%` | `98.03%` |
