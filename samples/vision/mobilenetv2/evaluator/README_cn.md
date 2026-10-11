[English](README.md) | 简体中文

# MobileNetV2 评估

固定版本 timm checkpoint 流程使用 [`evaluate.py`](evaluate.py)，命令见[公共主机流程](../../../../utils/tools/mobilenet/README_cn.md)。该流程明确记录权重、中心裁剪预处理、batch=1 logits 合同和全量评测输入；下文既有制品命令按各自合同使用。
使用随附图片进行单图分类检查。计算数据集精度时，准备对应验证集及逐图真值类别索引，并将其与运行时返回的 Top-1 类别 ID 对照。


## 固定制品的板端评测

`evaluate_board.py` 加载一次模型，对冻结的 ImageNetV2 MatchedFrequency 清单逐图评测。
使用与板卡匹配的制品（X5 `.bin`；S100、S100P 或 S600 `.hbm`），输入为
`bt601_video` NV12，输出为 float32 logits，并用 `--variant`（`v2-100`、`v2-140`）选择检查点的预处理合同。前处理为 PIL bicubic 短边缩放
（100 为 256，140 为 256）、中心裁剪到模型输入尺寸（224），再调用样例的 OpenCV NV12 转换；
这与 runtime 使用 `--resize-type 2` 时的几何相同。campaign 固定标签顺序、
几何变换和完整 10,000 张图片数量；程序校验每张图片与模型的 SHA256。

```bash
python3 samples/vision/mobilenetv2/evaluator/evaluate_board.py \
  --target s100 --variant v2-100 \
  --model samples/vision/mobilenetv2/model/s100/mobilenetv2_100_nashe_224x224_nv12.hbm \
  --model-sha256 <model/README_cn.md 中的 sha256> \
  --campaign /path/to/campaign.json --manifest /path/to/manifest.json \
  --data-root /path/to/imagenetv2/images --output /path/to/new-evaluation
```

新输出目录包含全量预测、张量元信息、三个固定输入的输出，以及记录 Top-1/Top-5 和源码哈希的
`evaluation.json`。C++ 性能计时与输入边界见[评测说明](../../../../utils/tools/mobilenet/cpp/README.md)。

<a id="dataset"></a>

## 数据集
功能检查使用随附测试图。数据集级精度使用 ImageNet ILSVRC2012 验证集（50,000 张、1,000 类）。准备逐图到模型零起始类别索引的真值映射，并与运行时返回的 Top-1 类别 ID 对照。`datasets/imagenet/imagenet_classes.names` 将输出索引映射为显示名称；逐图真值取自数据集标注。参见 [ImageNet 数据准备](../../../../datasets/imagenet/README_cn.md)。

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── evaluate.py  # FP32 ONNX 评测（公共主机流程）
└── evaluate_board.py  # 编译制品的板端评测
```

<a id="environment"></a>
## 环境

主机检查需要 sample 根目录 `requirements-host.txt` 的用户态 Python 依赖，
不需要板卡 SDK。功能板卡检查需要目标板卡及其 `hbm_runtime` 镜像、已准备
制品与标签文件。数据集级评估还需随结果声明 OE/板卡工具链。

<a id="command"></a>
## 命令

主机检查（cwd：仓库根目录；成功判据：全部 OK，退出码 0）：

```bash
python3 -m unittest discover -s samples/vision/mobilenetv2/tests -v
```

X5 功能板卡检查（前置：`bash samples/vision/mobilenetv2/model/download.sh x5 100`；成功判据：退出码 0 且 Top-K 符合预期）：

```bash
python3 samples/vision/mobilenetv2/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv2:mobilenetv2_100_bayese_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv2/model/mobilenetv2_100_bayese_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv2/test_data/Scottish_deerhound.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

S100、S100P、S600 替换 `s:` 引用与 `s100/`、`s100p/`、`s600/` 制品路径；标签文件共用。同板多次运行对照时，固定同一图像、制品字节、标签、resize 类型与
Top-K，在标签格式化之前比较类别 ID 与原始分数；预期类别 ID 相同、
分数差在 1e-5 内。对照双方使用相同板型。

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| 契约通过 | 运行时接受制品，张量名称/shape/dtype 与绑定一致，返回一个 F32 分数向量 | 匹配板卡上的任一已准备制品 |
| Top-K 一致 | 同一制品重复运行 softmax 后 Top-K 类别 ID 相同，分数差在 1e-5 内 | 同板、同制品字节、同图、同缩放、同 Top-K |
| Top-1 精度 | argmax 正确比例 | 在准备好的 ImageNet ILSVRC2012 验证集上度量 |
| 延迟 / FPS | 在匹配板卡上的推理计时 | 与[参考结果](#reference-results)的已发布数值按其声明的条件对照 |

<a id="outputs"></a>
## 输出

主机检查打印 unittest 结果。功能板卡检查在 stdout 打印 Top-K（类别 ID、
分数、标签），可选以 `--img-save-path` 写标注图。作为证据需保存：板卡
身份、模型引用、运行时 metadata、raw F32 分数张量、Top-K 输出、图像路径、
缩放方式与命令行。

<a id="reference-results"></a>
## 参考结果

已发布模型的精度，用上面的固定板端评测在完整 ImageNetV2 MatchedFrequency 集合
（10,000 张图像）上测得。它不是 ILSVRC2012 验证集，因此数值不能与 ImageNet-1k
验证集的结果相比。“FP32”是同一检查点的 ONNX 导出在相同裁剪图上的结果。

| 模型 | Target | FP32 Top-1 | 板端 Top-1 | FP32 Top-5 | 板端 Top-5 |
| --- | --- | --- | --- | --- | --- |
| MobileNetV2-100 | X5 | 60.18% | 59.55% | 82.05% | 81.48% |
| MobileNetV2-100 | S100 | 60.18% | 59.61% | 82.05% | 81.61% |
| MobileNetV2-100 | S100P | 60.18% | 59.61% | 82.05% | 81.61% |
| MobileNetV2-100 | S600 | 60.18% | 59.55% | 82.05% | 81.42% |
| MobileNetV2-140 | X5 | 63.71% | 63.11% | 84.68% | 84.44% |
| MobileNetV2-140 | S100 | 63.71% | 63.13% | 84.68% | 84.41% |
| MobileNetV2-140 | S100P | 63.71% | 63.13% | 84.68% | 84.41% |
| MobileNetV2-140 | S600 | 63.71% | 63.09% | 84.68% | 84.51% |

相同制品的延迟与吞吐及测量条件见 [sample README](../README_cn.md#performance)。
下面的 X5 与 X3 数据属于早期构建（权重与前处理不同），仅作历史保留。

### RDK X5 / X5 Module 性能

数据版本：`rdk_x5_legacy @ cb86079ae5befcef9ca50fb46c8a6d8980106dec`.

表后四线程/八线程说明沿用源数据的测试条件。其中双核和 X3 八线程说明对应 X3；X5 的硬件配置为 1×Bayes-e。

以下表格是在 RDK X5 & RDK X5 Module 上实际测试得到的性能数据


| 模型          | 尺寸(像素)  | 类别数  | 参数量(M) | 浮点Top-1  | 量化Top-1  | 延迟/吞吐量(单线程) | 延迟/吞吐量(多线程) | 帧率     |
| ----------- | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------ |
| Mobilenetv2 | 224x224 | 1000 | 3.4    | 72.0 | 68.17 | 1.42        | 3.43        | 1152.07 |


说明:
1. X5的状态为最佳状态：CPU为8xA55@1.8G, 全核心Performance调度, BPU为1xBayes-e@1G, 共10TOPS等效int8算力。
2. 单线程延迟为单帧，单线程，单BPU核心的延迟，BPU推理一个任务最理想的情况。
3. 4线程工程帧率为4个线程同时向双核心BPU塞任务，一般工程中4个线程可以控制单帧延迟较小，同时吃满所有BPU到100%，在吞吐量(FPS)和帧延迟间得到一个较好的平衡。
4. 8线程极限帧率为8个线程同时向X3的双核心BPU塞任务，目的是为了测试BPU的极限性能，一般来说4核心已经占满，如果8线程比4线程还要好很多，说明模型结构需要提高"计算/访存"比，或者编译时选择优化DDR带宽。
5. 浮点/定点Top-1：浮点Top-1使用的是模型未量化前onnx的 Top-1 推理精度，量化Top-1则为量化后模型实际推理的精度。

### RDK X3 / X3 Module 性能

数据版本：`rdk_x3 @ 0eb344ba8bed76923a6bd696e468fd82489cf46e`.

此表对应 X3 板卡与工具链；当前运行入口的适用板卡见样例根目录的支持表。

以下表格是在 RDK X3 & RDK X3 Module 上实际测试得到的性能数据


| 模型          | 尺寸(像素)  | 类别数  | 参数量(M) | 浮点Top-1  | 量化Top-1  | 延迟/吞吐量(单线程) | 延迟/吞吐量(多线程) | 帧率     |
| ----------- | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------ |
| Mobilenetv2 | 224x224 | 1000 | 3.4    | 72.0 | 68.17 | 2.41        | 4.42        | 890.99 |


说明:
1. X3的状态为最佳状态：CPU为4xA53@1.5G, 全核心Performance调度, BPU为2xBernoulli@1G, 共5TOPS等效int8算力。
2. 单线程延迟为单帧，单线程，单BPU核心的延迟，BPU推理一个任务最理想的情况。
3. 4线程工程帧率为4个线程同时向双核心BPU塞任务，一般工程中4个线程可以控制单帧延迟较小，同时吃满所有BPU到100%，在吞吐量(FPS)和帧延迟间得到一个较好的平衡。
4. 8线程极限帧率为8个线程同时向X3的双核心BPU塞任务，目的是为了测试BPU的极限性能，一般来说4核心已经占满，如果8线程比4线程还要好很多，说明模型结构需要提高"计算/访存"比，或者编译时选择优化DDR带宽。
5. 浮点/定点Top-1：浮点Top-1使用的是模型未量化前onnx的 Top-1 推理精度，量化Top-1则为量化后模型实际推理的精度。

<a id="boundaries"></a>
## 数据集级评估

计算数据集 Top-1 精度时，将每张验证图像通过 `--test-img` 传给运行时入口，把返回的 Top-1 类别 ID 与该图像的模型真值索引对照，再用正确预测数除以已评测的带标签图像数。对照运行时固定制品、resize 模式、Top-K、板卡镜像和调度设置。测量延迟或 FPS 时，在匹配板卡上计时推理阶段，并记录线程数和工作模式。
