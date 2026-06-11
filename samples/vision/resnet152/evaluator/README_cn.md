[English](./README.md) | 简体中文

# ResNet152 评测说明

本目录提供 runtime 轻量功能检查材料，不包含完整 ImageNet 精度评测流程。

## 功能检查

运行默认示例：

```bash
cd ../runtime/python
bash run.sh
```

直接运行入口脚本：

```bash
python3 main.py \
  --model-path ../../model/s100/resnet152_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

对于 `test_data/zebra_cls.jpg`，Top-K 输出应包含与 zebra 相关的 ImageNet 类别，并具有较高置信度。输出数值应为有限值、非全 0，并且同一输入下重复运行结果稳定。

## 原始记录

参考记录了以下参考值：

| 项目 | 数值 |
| --- | --- |
| 单线程 Frame total latency | `426.180 ms` |
| 单线程 Average latency | `2.131 ms` |
| 单线程 Frame rate | `463.021 FPS` |
| 三线程 Frame total latency | `1100.839 ms` |
| 三线程 Average latency | `5.504 ms` |
| 三线程 Frame rate | `539.012 FPS` |

参考结果图保存在 `test_data/result.png`。
