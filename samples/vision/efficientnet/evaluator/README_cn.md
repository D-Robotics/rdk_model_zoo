[English](README.md) | 简体中文

# EfficientNet 评估
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

主机检查需要仓库的用户态 Python 依赖（sample 根的
`requirements-host.txt`），无需板端 SDK。板上功能检查需要目标板卡及
其 `hbm_runtime` 镜像、已准备的制品和标签文件。数据集级评估还需在
结果中一并声明所用的 OE/板端工具链。

<a id="command"></a>
## 命令

主机检查（cwd：仓库根目录；成功判据：全部测试 OK，退出码 0）：

```bash
python3 -m unittest discover -s samples/vision/efficientnet/tests -v
```

X5 板上功能检查（前置：
`bash samples/vision/efficientnet/model/download.sh x5 b2`；成功判据：
退出码 0 且 Top-5 含鹿猎犬相关类别）：

```bash
python3 samples/vision/efficientnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:efficientnet:EfficientNet_B2_224x224_nv12.bin \
  --model-path samples/vision/efficientnet/model/EfficientNet_B2_224x224_nv12.bin \
  --test-img samples/vision/efficientnet/test_data/Scottish_deerhound.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

S100/S600 换用 `s:` 引用与 `s100/`/`s600/` 制品路径（例如 240x240 的
lite1：`s:efficientnet:s100/efficientnet_lite1_240x240_nv12.hbm`）；标签
文件共用。同板多次运行对照时，固定同一图像、制品字节、标签、resize
类型和 Top-K，在标签格式化之前对比类别 ID 与 Top-K 分数；预期类别 ID
相同、分数差在 1e-5 内。对照双方使用相同板型。相同输入
重复运行时输出应当有限、非零且稳定。

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| 契约通过 | 运行时接受制品，张量名/形状/dtype 与绑定一致，返回一个 F32 分数向量 | 任何已准备制品在匹配板卡上 |
| Top-K 一致 | 同一制品重复运行 softmax 后 Top-K 类别 ID 相同，分数差在 1e-5 内 | 同板、同制品字节、同图像、同 resize、同 Top-K |
| Top-1 精度 | argmax 正确的样本比例 | 在准备好的 ImageNet ILSVRC2012 验证集上度量 |
| 延迟 / FPS | 在匹配板卡上的推理计时 | 与[参考结果](#reference-results)的已发布数值按其声明的条件对照 |

<a id="outputs"></a>
## 输出

主机检查打印 unittest 结果。板上功能检查在 stdout 打印 Top-K（类别
ID、分数、标签），可选通过 `--img-save-path` 写标注图像。留存证据时
请保存：板卡身份、模型引用、运行时 metadata、原始 F32 分数张量、
Top-K 输出、图像路径、resize 类型与命令行。

<a id="reference-results"></a>
## 参考结果

已发布性能记录。

X5 源发布（x5-v1.1.3；Float Top-1 为量化前
ONNX 结果，Quant Top-1 为部署模型结果，延迟为单帧单线程单核，FPS 为
多线程；CPU 8xA55@1.8GHz 性能模式，BPU 1xBayes-e@1GHz）：

| 模型 | 尺寸 | 参数量 (M) | Float Top-1 | Quant Top-1 | 单线程延迟 (ms) | 多线程延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientNet-B4 | 224x224 | 19.27 | 74.25% | 71.75% | 5.44 | 18.63 | 212.75 |
| EfficientNet-B3 | 224x224 | 12.19 | 76.22% | 74.05% | 3.96 | 12.76 | 310.30 |
| EfficientNet-B2 | 224x224 | 9.07 | 76.50% | 73.25% | 3.31 | 10.51 | 376.77 |

S 源发布（s-v1.1.2；除表格外条件未声明）：

| 变体 | 单线程延迟 | 单线程 FPS | 多线程延迟 | 多线程 FPS |
| --- | --- | --- | --- | --- |
| Lite0 | 0.448 ms | 2107.815 | 0.591 ms | 4827.886 |
| Lite1 | 0.489 ms | 1948.957 | 0.708 ms | 4086.470 |
| Lite2 | 0.565 ms | 1702.519 | 0.935 ms | 3123.682 |
| Lite3 | 0.668 ms | 1451.031 | 1.249 ms | 2345.518 |
| Lite4 | 0.915 ms | 1064.339 | 1.979 ms | 1487.055 |

### RDK X5 / X5 Module 性能

数据版本：`rdk_x5_legacy @ cb86079ae5befcef9ca50fb46c8a6d8980106dec`.

表后四线程/八线程说明沿用源数据的测试条件。其中双核和 X3 八线程说明对应 X3；X5 的硬件配置为 1×Bayes-e。

以下表格是在 RDK X5 & RDK X5 Module 上实际测试得到的性能数据，可以根据自己推理实际需要的性能和精度，对模型的大小做权衡取舍。


| 模型           | 尺寸(像素)  | 类别数  | 参数量(M) | 浮点Top-1  | 量化Top-1  | 延迟/吞吐量(单线程) | 延迟/吞吐量(多线程) | 帧率      |
| ------------ | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------- |
| Efficientnet_B4   | 224x224     | 1000     | 19.27     | 74.25     | 71.75     | 5.44        | 18.63       | 212.75      |
| Efficientnet_B3   | 224x224     | 1000     | 12.19     | 76.22     | 74.05     | 3.96        | 12.76       | 310.30      |
| Efficientnet_B2   | 224x224     | 1000     | 9.07      | 76.50     | 73.25     | 3.31        | 10.51       | 376.77      |


说明:
1. X5的状态为最佳状态：CPU为8xA55@1.8G, 全核心Performance调度, BPU为1xBayes-e@1G, 共10TOPS等效int8算力。
2. 单线程延迟为单帧，单线程，单BPU核心的延迟，BPU推理一个任务最理想的情况。
3. 4线程工程帧率为4个线程同时向双核心BPU塞任务，一般工程中4个线程可以控制单帧延迟较小，同时吃满所有BPU到100%，在吞吐量(FPS)和帧延迟间得到一个较好的平衡。
4. 8线程极限帧率为8个线程同时向X3的双核心BPU塞任务，目的是为了测试BPU的极限性能，一般来说4核心已经占满，如果8线程比4线程还要好很多，说明模型结构需要提高"计算/访存"比，或者编译时选择优化DDR带宽。
5. 浮点/定点Top-1：浮点Top-1使用的是模型未量化前onnx的 Top-1 推理精度，量化Top-1则为量化后模型实际推理的精度。

<a id="boundaries"></a>
## 数据集级评估

计算数据集 Top-1 精度时，将每张验证图像通过 `--test-img` 传给运行时入口，把返回的 Top-1 类别 ID 与该图像的模型真值索引对照，再用正确预测数除以已评测的带标签图像数。对照运行时固定制品、resize 模式、Top-K、板卡镜像和调度设置。测量延迟或 FPS 时，在匹配板卡上计时推理阶段，并记录线程数和工作模式。
