[English](./README.md) | [简体中文](./README_cn.md)

# UNet 模型文件

本目录提供 UNet ResNet 系列在 RDK X5（`bayes-e`）上的预编译 `.bin` 模型下载入口。
模型文件不会提交到 Git 仓库，下载脚本会把产物保存到本目录。

## 可用模型

| Backbone | 文件名 | 状态 | SHA256 |
| --- | --- | --- | --- |
| ResNet18 | `unet_resnet18_voc_512x512_nv12.bin` | 已发布 | `d082ff055532081d14326d96fb2bb8ac85a0f1edc46e868cbbbea0259bc36b5f` |
| ResNet34 | `unet_resnet34_voc_512x512_nv12.bin` | 已发布 | `9d758822b2de4d5aaa24b4c02479c9f742c4b4e4af075389d921c12396194ac0` |
| ResNet50 | `unet_resnet50_voc_512x512_nv12.bin` | 已发布 | `22ea4eec82328d34dc963e091f3cb4e134c8a432d7c6f92d74522b998b7bd23a` |
| ResNet101 | `unet_resnet101_voc_512x512_nv12.bin` | 已发布 | `04031417d3d4098bceac0bb1b731a7aed099dca5c8ee0cd30527c2f6494c7215` |
| ResNet152 | `unet_resnet152_voc_512x512_nv12.bin` | 已发布 | `990855473e5411c2996bd7f161591dc7ba479402bcfe40c36d3fd2b10edbb32a` |

## 下载

默认下载已发布的 ResNet18：

```bash
cd samples/vision/unet/model
./download_model.sh
```

直接下载地址：
[unet_resnet18_voc_512x512_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/unet/unet_resnet18_voc_512x512_nv12.bin)

[unet_resnet34_voc_512x512_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/unet/unet_resnet34_voc_512x512_nv12.bin)

[unet_resnet50_voc_512x512_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/unet/unet_resnet50_voc_512x512_nv12.bin)

[unet_resnet101_voc_512x512_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/unet/unet_resnet101_voc_512x512_nv12.bin)

[unet_resnet152_voc_512x512_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/unet/unet_resnet152_voc_512x512_nv12.bin)

也可以显式指定一个或多个 backbone：

```bash
./download_model.sh resnet18
./download_model.sh resnet34 resnet101
./download_model.sh all
```

每个模型下载后都会强制校验已发布的 SHA256，只有校验通过后才会把临时文件移动到
目标路径；下载或校验失败不会覆盖已有模型。

模型只适用于 RDK X5，不能用于 RDK S100/S600。自行转换模型的方法见
[转换文档](../conversion/README_cn.md)。
