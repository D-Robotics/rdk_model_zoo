[English](./README.md) | [简体中文](./README_cn.md)

# UNet 精度评测

`eval_unet.py` 是五个 UNet ResNet 变体的统一精度入口，可以使用同一份
Pascal VOC 图像与标签清单评测 PyTorch checkpoint、浮点 ONNX 或 RDK X5
`bayes-e` `.bin` 模型。

评测脚本会把 RGB 图像和类别索引 mask 缩放到 `512x512`，保留 VOC 忽略标签
`255`，对 21 类 logits 执行 `argmax`，并输出 mIoU、Pixel Accuracy 和每类
IoU。每次运行都会把模型和 manifest 哈希写入新的 JSON 报告。

## Manifest 格式

每个非空行包含一个绝对图像路径和一个 mask 路径，中间使用单个 Tab 分隔：

```text
/data/VOC2012/JPEGImages/2007_000033.jpg\t/data/VOC2012/SegmentationClass/2007_000033.png
```

VOC 调色板 mask 必须按类别索引读取，评测前不能转换为灰度图。

## PyTorch checkpoint

在装有 PyTorch 和 Pillow 的开发机上执行：

```bash
python eval_unet.py \
  --model /models/unet_resnet18_voc_best.pth \
  --backbone resnet18 \
  --manifest /data/unet/val.tsv \
  --report /reports/unet_resnet18_pytorch.json \
  --min-miou 0.50
```

## ONNX

在主机评测环境中安装 `onnxruntime` 后执行：

```bash
python eval_unet.py \
  --model /models/unet_resnet18_voc_512x512.onnx \
  --manifest /data/unet/val.tsv \
  --report /reports/unet_resnet18_onnx.json \
  --min-miou 0.50
```

## RDK X5 `.bin`

`.bin` 后端必须在 aarch64 RDK X5 本机运行，并且 RDK OS 不低于 3.5.0。
必须使用板卡系统随附的 X5 `hbm_runtime`，不要从 PyPI 安装其他平台的同名包。
编译模型必须提供一个 packed NV12 输入和一个 21 类 logits 输出。

```bash
python eval_unet.py \
  --model /models/unet_resnet18_voc_512x512_nv12.bin \
  --manifest /data/unet/val.tsv \
  --report /reports/unet_resnet18_x5.json \
  --min-miou 0.50
```

可以用 `--limit` 执行小样本冒烟测试。如果结果低于 `--min-miou`，JSON 报告
仍会正常生成，但程序退出码为 2。
