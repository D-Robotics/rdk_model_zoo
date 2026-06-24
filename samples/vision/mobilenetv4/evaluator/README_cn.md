[English](./README.md) | 简体中文

# MobileNetV4 评测

本目录记录 MobileNetV4 的验证方式。当前 sample 未提供独立精度
评测脚本。

## 功能验证

使用 Python runtime：

```bash
cd ../runtime/python
bash run.sh
bash run.sh medium
```

直接入口示例（把 `<soc>` 替换为 `s100` 或 `s600`）：

```bash
python3 main.py \
  --model-variant small \
  --model-path ../../model/<soc>/mobilenetv4_small_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

```bash
python3 main.py \
  --model-variant medium \
  --model-path ../../model/<soc>/mobilenetv4_medium_256x256_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

只有 Top-1 或 Top-5 与测试图语义匹配时，才认为结果正确。对于
`zebra_cls.jpg`，`zebra` 应出现在结果中，且置信度为有限非零值。

## 精度评测

完整 ImageNet 评测需使用验证集，并保持 `../conversion/` 中记录的预处理：
BGR 输入、NCHW 训练布局、ImageNet mean/scale，以及 NV12 运行时输入。

## 参考记录

原始 conversion 文档记录了量化 cosine 和工具链性能。详见
`../conversion/README_cn.md`。
