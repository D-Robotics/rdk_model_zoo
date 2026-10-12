[English](README.md) | 简体中文

# FasterNet 评估
使用随附图片进行单图分类。计算数据集精度时，准备对应验证集及逐图真值类别索引，并将其与运行时返回的 Top-1 类别 ID 对照。

<a id="dataset"></a>

## 数据集
单图分类使用随附测试图。数据集级精度使用 ImageNet ILSVRC2012 验证集（50,000 张、1,000 类）。准备逐图到模型零起始类别索引的真值映射，并与运行时返回的 Top-1 类别 ID 对照。`datasets/imagenet/imagenet_classes.names` 将输出索引映射为显示名称；逐图真值取自数据集标注。参见 [ImageNet 数据准备](../../../../datasets/imagenet/README_cn.md)。

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
## 命令

X5 板端分类（前置：
`bash samples/vision/fasternet/model/download.sh x5 s`；输出：
退出码 0 且 Top-5 含鸭相关类别）：

```bash
python3 samples/vision/fasternet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:fasternet:FasterNet_S_224x224_nv12.bin \
  --model-path samples/vision/fasternet/model/FasterNet_S_224x224_nv12.bin \
  --test-img samples/vision/fasternet/test_data/drake.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

同板前后对照：固定同一图像、制品字节、标签、resize 类型和 Top-K，在标签格式化之前比较类别 ID 和 Top-K 分数；输出应为有限、非零，且同一输入下重复运行保持稳定。相同输入重复运行时输出应当
有限、非零且稳定。

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| Top-K 一致 | 同一制品重复运行 softmax 后 Top-K 类别 ID 相同，分数差在 1e-5 内 | 同板、同制品字节、同图像、同 resize、同 Top-K |
| Top-1 精度 | argmax 正确的样本比例 | 在准备好的 ImageNet ILSVRC2012 验证集上度量 |
| 延迟 / FPS | 在匹配板卡上的推理计时 | 与[参考结果](#reference-results)的已发布数值按其声明的条件对照 |

<a id="outputs"></a>
## 输出

板端命令在终端打印 Top-K 类别 ID、分数和标签。使用 `--img-save-path` 保存标注图。比较结果时，一并记录板卡、模型、命令、输入图像、缩放方式和原始分数。

<a id="reference-results"></a>
## 参考结果

X5 发布（x5-v1.1.3）的已发布数值（Float Top-1 为量化前 ONNX 结果，Quant Top-1 为部署模型结果，延迟为
单帧单线程单核，FPS 为多线程；CPU 8xA55@1.8GHz 性能模式，BPU
1xBayes-e@1GHz）：

| 模型 | 尺寸 | 参数量 (M) | Float Top-1 | Quant Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| FasterNet-S | 224x224 | 31.1 | 77.04% | 76.15% | 6.73 | 162.83 |
| FasterNet-T2 | 224x224 | 15.0 | 76.50% | 76.05% | 3.39 | 342.48 |
| FasterNet-T1 | 224x224 | 7.6 | 74.29% | 71.25% | 1.96 | 708.40 |
| FasterNet-T0 | 224x224 | 3.9 | 71.75% | 68.50% | 1.41 | 1135.13 |

已发布参数量 (M) 列与上游论文的模型尺寸并非全部吻合。

### RDK X5 / X5 Module 性能

数据版本：`rdk_x5_legacy @ cb86079ae5befcef9ca50fb46c8a6d8980106dec`.

X5 的 BPU 为 1×Bayes-e。下文八线程条件对应 X3 数据。

RDK X5 与 RDK X5 Module 的发布性能：


| 模型           | 尺寸(像素)  | 类别数  | 参数量(M) | 浮点Top-1  | 量化Top-1  | 延迟/吞吐量(单线程) | 延迟/吞吐量(多线程) | 帧率      |
| ------------ | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------- |
| FasterNet_S  | 224x224 | 1000 | 31.1   | 77.04 | 76.15 | 6.73        | 24.34       | 162.83  |
| FasterNet_T2 | 224x224 | 1000 | 15.0   | 76.50 | 76.05 | 3.39        | 11.56       | 342.48  |
| FasterNet_T1 | 224x224 | 1000 | 7.6    | 74.29 | 71.25 | 1.96        | 5.58        | 708.40  |
| FasterNet_T0 | 224x224 | 1000 | 3.9    | 71.75 | 68.50 | 1.41        | 3.48        | 1135.13 |


说明:
1. X5：CPU 8xA55@1.8G，全核 Performance 调度；BPU 1xBayes-e@1G，10TOPS INT8 算力。
2. 单线程延迟统计单帧、单线程、单 BPU 核的推理时间。
3. 四线程帧率使用四个线程并发提交任务，BPU 利用率为 100%。
4. X3 八线程数据使用八个线程，向两个 BPU 核并发提交任务。
5. 浮点/定点Top-1：浮点Top-1使用的是模型未量化前onnx的 Top-1 推理精度，量化Top-1则为量化后模型实际推理的精度。

<a id="boundaries"></a>
## 数据集级评估

计算数据集 Top-1 精度时，将每张验证图像通过 `--test-img` 传给运行时入口，把返回的 Top-1 类别 ID 与该图像的模型真值索引对照，再用正确预测数除以已评测的带标签图像数。对照运行时固定制品、resize 模式、Top-K、板卡镜像和调度设置。测量延迟或 FPS 时，在匹配板卡上计时推理阶段，并记录线程数和工作模式。
