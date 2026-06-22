[English](./README.md) | 简体中文

# ViT 评测说明

本目录记录 ViT runtime 的 CIFAR-10 精度数据和功能结果检查方法。

## 功能检查

运行默认示例：

```bash
cd ../runtime/python
bash run.sh int8
```

直接运行入口脚本：

```bash
python3 main.py \
  --model-path ../../model/s100/vit_cifar10_batch1_int8.hbm \
  --test-img ../../test_data/airplane_0000.png \
  --label-file ../../test_data/cifar10_classes.names \
  --top-k 5
```

对于 `test_data/airplane_0000.png`，Top-K 输出应包含 CIFAR-10 的 `airplane`
类别，并具有较高置信度。输出数值应为有限值、非全 0，并且同一输入下重复运行结果稳定。

## 测试图片

`test_data/` 中包含 CIFAR-10 每个类别的一张图片：

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

## 精度数据

| 模型 | Top-1 | Top-5 |
| --- | --- | --- |
| ONNX | `74.54%` | `98.36%` |
| HBM | `72.62%` | `98.03%` |

## 精度测试说明

1. BPU 模型在量化 NCHW RGB888 输入并转换为 YUV420SP (NV12) 输入后，会有一部分精度损失，这是色彩空间转换带来的误差。在训练时加入这种色彩空间转换损失可以降低该影响。
2. Python 接口和 C/C++ 接口的精度结果可能有细微差异，主要来自内存拷贝和浮点数转换过程中的处理方式差异。
3. 批量评测脚本可参考 RDK Model Zoo 评测工具：<https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/tools/eval_pycocotools>
4. 表格结果使用 PTQ 和 50 张图片进行校准，用于模拟普通开发者第一次直接编译的精度情况；未进行精度调优或 QAT，不代表精度上限。
