[English](README.md) | 简体中文

# UNetMobileNet 验证

<a id="dataset"></a>
## 数据集

Cityscapes 定义 19 类任务，但源未提供带标签验证 split 或数据集执行器。segmentation.png 为冒烟输入，result.jpg 为源记录示意。数据集评估需要有许可的图片／标签、明确 train-ID 映射、忽略标签策略及记录的 split。

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="environment"></a>
## 环境

使用匹配的 S100 或 S600 板端镜像、模型和 SDK。Python 需要 Python 3.10+、NumPy、OpenCV 与 PyYAML；C++ 需要 C++17、CMake、OpenCV 开发库及板端 DNN/UCP 头文件与库。

<a id="command"></a>
## 运行命令

在仓库根目录准备 S100 模型，然后对相同图片运行两个实现：

```bash
bash samples/vision/unetmobilenet/model/download.sh --target s100
python3 samples/vision/unetmobilenet/runtime/python/main.py --target s100 \
  --test-img samples/vision/unetmobilenet/test_data/segmentation.png \
  --mask-save-path outputs/unetmobilenet/python-labels.npy
bash samples/vision/unetmobilenet/runtime/cpp/run.sh --target s100 --build \
  --test-img samples/vision/unetmobilenet/test_data/segmentation.png \
  --mask-save-path outputs/unetmobilenet/cpp-labels.png
```

比较时保持板卡、模型和输入一致。使用 S600 时，将全部命令的目标改为 `--target s600`。

<a id="metrics"></a>
## 指标

将 Python NPY 和 C++ PNG 读取为整数类别 ID 数组，报告逐像素一致率。数据集 mIoU 需要 Cityscapes 真值 mask 与评估协议；类别 ID 不适合计算 logits 余弦相似度。

<a id="outputs"></a>
## 输出

Python 输出 int32 NPY；C++ 输出无损 uint8 PNG ID，API mask 仍为 int32。两侧均保留原图尺寸，输出元数据报告和叠加图。应在可视化前比较数值 mask；JPEG 压缩不适合叠加图的逐像素精确比较。

<a id="reference-results"></a>
## 参考结果

本示例源中没有 mIoU/FPS/延迟表。[参考效果图](../test_data/result.jpg) 来自源记录。

<a id="boundaries"></a>
## 适用范围

使用上述 Python 和 C++ Runtime 命令生成相同输入的单图结果，再按输入格式与指标说明比较整数类别掩码和量化输出。整数输出通过模型的 SCALE 元数据解码。
