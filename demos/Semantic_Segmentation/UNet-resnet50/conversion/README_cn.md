# UNet-resnet50 模型转换

[English](./README.md) | 简体中文

## 概述

本目录包含将 UNet-resnet50 从 PyTorch/ONNX 转换到 Horizon BPU（.bin 格式）的工具链配置。

## 目录结构

```bash
.
|-- ptq_yamls/              # PTQ 量化 YAML 配置文件
|   |-- unet_resnet50_bayese_512x512_nv12.yaml      # Bayes-e (X5) PTQ 配置
|   `-- unet_resnet50_bernoulli2_512x512_nv12.yaml  # Bernoulli2 (X3) PTQ 配置
`-- README.md               # 本文件
```

## 转换流程

### 1. 环境准备

下载 UNet PyTorch 实现：

```bash
git clone https://github.com/bubbliiiing/unet-pytorch.git
cd unet-pytorch
pip install -r requirements.txt
```

### 2. 导出 ONNX

在 `predict.py` 中将 `mode` 修改为 `export_onnx`，然后运行：

```bash
python predict.py
```

### 3. PTQ 量化与编译

使用 Horizon OE 工具链检查模型：

```bash
hb_mapper checker --model-type onnx --march bayes-e --model UNet_11.onnx
```

编译模型：

```bash
hb_mapper makertbin --model-type onnx --config unet_resnet50_bernoulli2_512x512_nv12.yaml
```

### 4. 移除输出反量化节点（可选但推荐）

检查可移除的反量化节点：

```bash
hb_model_modifier unet_resnet50_bayese_512x512_nv12.bin
```

移除输出反量化节点以提升性能：

```bash
hb_model_modifier unet_resnet50_bayese_512x512_nv12.bin \
    -r "/final/Conv_output_0_HzDequantize"
```

### 5. 验证

可视化编译后的模型：

```bash
hb_perf unet_resnet50_bernoulli2_512x512_nv12.bin
```

查看模型 I/O 信息：

```bash
hrt_model_exec model_info --model_file unet_resnet50_bernoulli2_512x512_nv12.bin
```

## 注意事项

- 已提供预转换好的模型，普通用户可以跳过转换步骤。
- 导出时节点名称可能会有差异，请仔细根据日志进行核对。
- 详细的工具链文档请参考 Horizon Open Explorer 手册。
