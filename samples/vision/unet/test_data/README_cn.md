[English](./README.md) | [简体中文](./README_cn.md)

# UNet 测试数据

`2007_000033.jpg` 是 Python Runtime 默认使用的 Pascal VOC 2012 验证图片。它仅
用于快速检查图片读取、NV12 预处理、BPU 推理和可视化输出是否可以连通。

- 原始尺寸：500 × 366
- SHA256：`23b51ccd1a19c6f1f75573b1903e19015bf98c159b03d497efa8e912f8ffbe8e`
- 数据集：[Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

完整精度评测需要单独下载 Pascal VOC 数据集并使用 `evaluator/eval_unet.py`。
