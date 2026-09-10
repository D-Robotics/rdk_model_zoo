# UNet-resnet50 模型下载

[English](./README.md) | 简体中文

## 预转换模型

UNet-resnet50 的预转换 BPU 模型已提供下载。这些模型已针对 Horizon 平台优化，可直接用于推理。

## 模型命名

模型文件遵循以下命名规范：

```
<model_name>_<input_resolution>_<chip_name>.bin
```

示例：

```
unet_resnet50_512x512_x3.bin
unet_resnet50_512x512_x5.bin
```

## 下载

请通过以下渠道获取模型：

1. **Model Zoo 官方下载页面**（推荐）
2. 目标设备上的**系统默认路径**：
   - RDK X3 / X3 Module：`/opt/hobot/model/basic/unet_resnet50_512x512_x3.bin`
   - RDK X5 / X5 Module：`/opt/hobot/model/basic/unet_resnet50_512x512_x5.bin`

## 模型信息

| 属性             | 值                            |
| ---------------- | ----------------------------- |
| 模型             | UNet-resnet50                 |
| 输入分辨率       | 512×512                       |
| 输入格式         | NV12                          |
| 输出形状         | [1, 21, 512, 512]             |
| 输出类型         | INT32（含反量化系数）         |
| 类别数           | 21（VOC 格式，background=0）  |
