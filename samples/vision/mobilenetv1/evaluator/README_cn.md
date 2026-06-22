[English](./README.md) | 简体中文

# 模型评估

本目录记录 MobileNetV1 分类模型的评估入口。当前 sample 以单张 ImageNet
测试图片做功能验证；完整 ImageNet 精度评测应复用
`runtime/python/mobilenetv1.py` 中相同的前处理和后处理逻辑。

## 功能检查

```bash
cd ../runtime/python
python3 main.py \
  --model-path ../../model/s100/mobilenetv1_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names
```

通过标准：

- Top-5 包含 `zebra`。
- 分数为有限值，且不是全 0。
- 同一模型和图片重复运行时输出稳定。
