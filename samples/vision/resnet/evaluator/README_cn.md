[English](README.md) | 简体中文

# ResNet 评估（ResNet18/50/152）

使用随仓图片进行板端分类。数据集精度评估需准备匹配的带标注验证集，并按下述指标统计预测；参考表列出发布测量及其运行条件。

<a id="dataset"></a>

## 数据集

单图分类使用随仓测试图执行。进行数据集级评估时，按
[datasets/imagenet](../../../../datasets/imagenet/README_cn.md) 准备
ImageNet 验证集（50,000 张，ILSVRC2012 val）；数据集由用户自备（本目录
不附带下载脚本）。

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="environment"></a>
## 环境

在目标板卡上运行，准备对应的 `hbm_runtime` 镜像、NumPy、OpenCV-Python、PyYAML、模型文件和标签文件。依赖安装见运行说明；记录数据集结果时附上板卡镜像与工具链版本。

<a id="command"></a>
## 评估命令

X5 板端分类（前置：`bash samples/vision/resnet/model/download.sh x5 resnet18`；
输出：退出码 0 并打印 Top-K 类别和分数）：

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

S100 板端分类（S600 将资产引用与制品路径中的 `s100` 换为 `s600`）：

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target s100 \
  --asset-id s:resnet18:s100/resnet18_224x224_nv12.hbm \
  --model-path samples/vision/resnet/model/s100/resnet18_224x224_nv12.hbm \
  --test-img samples/vision/resnet/test_data/zebra_cls.jpg \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

ResNet50/152（S100/S600）使用相同的命令，替换为对应
`s:resnet50|resnet152:<target>/...` 引用、制品路径并指定
`--variant resnet50|resnet152`。S 系列 C++ 检查执行
`bash samples/vision/resnet/runtime/cpp/run.sh`。

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| Top-K 输出 | 打印随仓图片的类别 ID、分数与标签；重复运行结果一致 | 同板、同制品字节、同图、同缩放、同 Top-K |
| Top-1 精度 | argmax 正确预测占比 | 在准备好的 ImageNet val 上统计；随结果记录数据划分 |
| 延迟 / FPS | 推理耗时 | 在目标板卡上测量；随结果记录批量/线程设置（参考条件见下） |

<a id="outputs"></a>
## 输出

板端命令在终端打印 Top-K 类别 ID、分数和标签。使用 `--img-save-path` 保存标注图。比较结果时，一并记录板卡、模型、命令、输入图像、缩放方式和原始分数。

<a id="reference-results"></a>
## 参考结果

X5 发布（x5-v1.1.3）的 ResNet18 记录：

| 模型 | 尺寸 | 类别数 | 参数量 (M) | 浮点 Top-1 | 量化 Top-1 | 延迟 (ms) | FPS | 线程条件 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | 224x224 | 1000 | 11.2 | 71.5% | 70.5% | 2.95 | 449+ | —（源未注明） |

ImageNet val 精度与延迟按上文准备数据集后测量，并随结果记录工具链、批量与
线程设置。S 发布未提供 ResNet18/50/152 的精度或延迟数据。

### RDK X5 / X5 Module 性能

数据版本：`rdk_x5_legacy @ cb86079ae5befcef9ca50fb46c8a6d8980106dec`.

X5 的 BPU 为 1×Bayes-e。下文八线程条件对应 X3 数据。

以下表格是在 RDK X5 & RDK X5 Module 上实际测试得到的性能数据


| 模型          | 尺寸(像素)  | 类别数  | 参数量(M) | 浮点Top-1  | 量化Top-1  | 延迟/吞吐量(单线程) | 延迟/吞吐量(多线程) | 帧率     |
| ----------- | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------ |
| ResNet18 | 224x224 | 1000 | 11.2    | 71.49 | 70.50 | 2.95        | 8.81        | 448.79 |


说明:
1. X5：CPU 8xA55@1.8G，全核 Performance 调度；BPU 1xBayes-e@1G，10TOPS INT8 算力。
2. 单线程延迟统计单帧、单线程、单 BPU 核的推理时间。
3. 四线程帧率使用四个线程并发提交任务，BPU 利用率为 100%。
4. X3 八线程数据使用八个线程，向两个 BPU 核并发提交任务。
5. 浮点/定点Top-1：浮点Top-1使用的是模型未量化前onnx的 Top-1 推理精度，量化Top-1则为量化后模型实际推理的精度。

### RDK X3 / X3 Module 性能

数据版本：`rdk_x3 @ 0eb344ba8bed76923a6bd696e468fd82489cf46e`.

此表对应 X3 板卡与工具链；当前运行入口的适用板卡见样例根目录的支持表。

以下表格是在 RDK X3 & RDK X3 Module 上实际测试得到的性能数据


| 模型          | 尺寸(像素)  | 类别数  | 参数量(M) | 浮点Top-1  | 量化Top-1  | 延迟/吞吐量(单线程) | 延迟/吞吐量(多线程) | 帧率     |
| ----------- | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------ |
| ResNet18 | 224x224 | 1000 | 11.2    | 71.49 | 70.50 | 8.87        | 17.07        | 232.74 |


说明:
1. X3的状态为最佳状态：CPU为4xA53@1.5G, 全核心Performance调度, BPU为2xBernoulli@1G, 共5TOPS等效int8算力。
2. 单线程延迟统计单帧、单线程、单 BPU 核的推理时间。
3. 四线程帧率使用四个线程并发提交任务，BPU 利用率为 100%。
4. X3 八线程数据使用八个线程，向两个 BPU 核并发提交任务。
5. 浮点/定点Top-1：浮点Top-1使用的是模型未量化前onnx的 Top-1 推理精度，量化Top-1则为量化后模型实际推理的精度。

<a id="boundaries"></a>
## 适用范围

在准备好的 ImageNet 验证集上，通过自行编写的评估循环统计精度和延迟。每份结果附上板卡、模型、工具链、批量大小和线程设置。