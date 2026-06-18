[English](./README.md) | 简体中文

# ResNet50 评测

本目录记录 ResNet50 在 RDK S100 / RDK S600 上的验证方式。当前 sample 提供
notebook 和 runtime 脚本，但没有独立精度评测脚本。

## 功能验证

```bash
cd ../runtime/python
bash run.sh
```

Python 直接入口：

```bash
python3 main.py \
  --model-path ../../model/s100/resnet50_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

RDK S600 用户请将 `--model-path` 改为 `../../model/s600/resnet50_224x224_nv12.hbm`。

只有 Top-1 或 Top-5 与测试图语义匹配时，才认为结果正确。对于
`zebra_cls.jpg`，`zebra` 应出现在结果中，且置信度为有限非零值。

## 精度评测

完整 ImageNet 评测需使用验证集，并保持转换参考中的模型预处理：224x224 输入
和 NV12 运行时输入。

原始推理截图已保留为 `../test_data/result.png`。
