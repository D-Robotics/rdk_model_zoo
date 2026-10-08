# ResNeXt 评测
使用随附图片进行单图分类检查。计算数据集精度时，准备对应验证集及逐图真值类别索引，并将其与运行时返回的 Top-1 类别 ID 对照。

<a id="dataset"></a>

## 数据集
功能检查使用随附测试图。数据集级精度使用 ImageNet ILSVRC2012 验证集（50,000 张、1,000 类）。准备逐图到模型零起始类别索引的真值映射，并与运行时返回的 Top-1 类别 ID 对照。`datasets/imagenet/imagenet_classes.names` 将输出索引映射为显示名称；逐图真值取自数据集标注。参见 [ImageNet 数据准备](../../../../datasets/imagenet/README_cn.md)。

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="environment"></a>
## 环境

主机：sample requirements，源对照额外使用 SciPy。板端：runtime README 所述 X5 环境。检查复用同一分类任务；本目录提供操作说明，没有另一个基准可执行程序。

<a id="command"></a>
## 命令

```bash
# cwd: repository root; expected: unittest OK, exit 0
python3 -m unittest discover -s samples/vision/resnext/tests -v
```

板端功能检查：

```bash
# cwd: repository root
bash samples/vision/resnext/model/download.sh x5 50_32x4d
python3 samples/vision/resnext/runtime/python/main.py \
  --target x5 --variant 50_32x4d \
  --test-img samples/vision/resnext/test_data/bee_eater.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

在匹配板卡上对每个已发布变体分别执行。同板多次运行对照时，保持
模型字节、图像、resize 类型、Top-K 和调度参数一致，在标签格式化之前
比较类别 ID 与原始分数；预期类别 ID 相同、分数差在 1e-5 内。当 Top-K
边界出现完全平局时，核对逐 ID 分数而不是放宽容差。

<a id="metrics"></a>
## 指标

在匹配板卡上运行功能检查，并按[参考结果](#reference-results)中的测量条件与已发布数值对照。计算数据集 Top-1 精度时，将逐图真值类别 ID 与运行时返回的 Top-1 ID 比较；测量延迟或 FPS 时记录推理阶段、线程数、并发量和工作模式。

<a id="outputs"></a>
## 输出

板端 CLI 打印推理结果，并可选保存可视化。运行数据集评测时使用验证集标注中的逐图真值类别 ID，并将其与运行时 Top-1 ID 对照。

<a id="reference-results"></a>
## 参考结果

下表来自已发布的源 evaluator。

源条件：X5 CPU 8×A55@1.8GHz 性能模式、BPU Bayes-e@1GHz。Float Top-1 为量化前 ONNX，Quant Top-1 为部署结果；单线程延迟为单帧单 BPU 核，多线程延迟和 FPS 使用并发提交。开展新对照时，使用相同数据子集，并记录预热、重复次数、板卡模式和并发设置。

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Single-thread Latency (ms) | Multi-thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNeXt50_32x4d | 224x224 | 24.99 | 76.25% | 76.00% | 5.89 | 20.90 | 189.61 |

### RDK X5 / X5 Module 性能

数据版本：`rdk_x5_legacy @ cb86079ae5befcef9ca50fb46c8a6d8980106dec`.

表后四线程/八线程说明沿用源数据的测试条件。其中双核和 X3 八线程说明对应 X3；X5 的硬件配置为 1×Bayes-e。

以下表格是在 RDK X5 & RDK X5 Module 上实际测试得到的性能数据


| 模型          | 尺寸(像素)  | 类别数  | 参数量(M) | 浮点Top-1  | 量化Top-1  | 延迟/吞吐量(单线程) | 延迟/吞吐量(多线程) | 帧率     |
| ----------- | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------ |
| ResNeXt50_32x4d  | 224x224 | 1000 | 24.99  | 76.25 | 76.00 | 5.89   | 20.90       | 189.61 |


说明:
1. X5的状态为最佳状态：CPU为8xA55@1.8G, 全核心Performance调度, BPU为1xBayes-e@1G, 共10TOPS等效int8算力。
2. 单线程延迟为单帧，单线程，单BPU核心的延迟，BPU推理一个任务最理想的情况。
3. 4线程工程帧率为4个线程同时向双核心BPU塞任务，一般工程中4个线程可以控制单帧延迟较小，同时吃满所有BPU到100%，在吞吐量(FPS)和帧延迟间得到一个较好的平衡。
4. 8线程极限帧率为8个线程同时向X3的双核心BPU塞任务，目的是为了测试BPU的极限性能，一般来说4核心已经占满，如果8线程比4线程还要好很多，说明模型结构需要提高"计算/访存"比，或者编译时选择优化DDR带宽。
5. 浮点/定点Top-1：浮点Top-1使用的是模型未量化前onnx的 Top-1 推理精度，量化Top-1则为量化后模型实际推理的精度。

<a id="boundaries"></a>
## 数据集级评估

计算数据集 Top-1 精度时，将每张验证图像通过 `--test-img` 传给运行时入口，把返回的 Top-1 类别 ID 与该图像的模型真值索引对照，再用正确预测数除以已评测的带标签图像数。对照运行时固定制品、resize 模式、Top-K、板卡镜像和调度设置。测量延迟或 FPS 时，在匹配板卡上计时推理阶段，并记录线程数和工作模式。
