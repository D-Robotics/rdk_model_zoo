[English](./README.md) | 简体中文

# EfficientNet-Lite 评测说明

本目录提供轻量功能检查材料，不包含完整 ImageNet 精度评测流程。

## 功能检查

运行默认示例：

```bash
cd ../runtime/python
bash run.sh
```

直接运行入口脚本：

```bash
python3 main.py \
  --model-path /opt/hobot/model/s100/basic/efficientnet_lite0_224x224_nv12.hbm \
  --test-img ../../test_data/Scottish_deerhound.JPEG \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

对于 `test_data/Scottish_deerhound.JPEG`，Top-K 输出应包含与 dog 相关的 ImageNet 类别，并具有较高置信度。输出数值应为有限值、非全 0，并且同一输入下重复运行结果稳定。

## 性能记录

| 变体 | 单线程延迟 | 单线程 FPS | 多线程延迟 | 多线程 FPS |
| --- | --- | --- | --- | --- |
| Lite0 | `0.448 ms` | `2107.815` | `0.591 ms` | `4827.886` |
| Lite1 | `0.489 ms` | `1948.957` | `0.708 ms` | `4086.470` |
| Lite2 | `0.565 ms` | `1702.519` | `0.935 ms` | `3123.682` |
| Lite3 | `0.668 ms` | `1451.031` | `1.249 ms` | `2345.518` |
| Lite4 | `0.915 ms` | `1064.339` | `1.979 ms` | `1487.055` |

参考结果图保存在 `test_data/result.png`。
