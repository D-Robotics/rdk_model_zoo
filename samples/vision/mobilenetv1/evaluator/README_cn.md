[English](README.md) | 简体中文

# MobileNetV1 评估

使用 [`evaluate.py`](evaluate.py) 在主机上评估指定 timm 权重导出的 FP32 ONNX。安装和运行步骤见[公共主机流程](../../../../utils/tools/classification/README_cn.md)；使用对应权重的中心裁剪预处理，以 batch=1 生成 logits 并统计全量数据集精度。
使用随附图片进行单图分类。计算数据集精度时，准备对应验证集及逐图真值类别索引，并将其与运行时返回的 Top-1 类别 ID 对照。


## 固定制品的板端评测

`evaluate_board.py` 加载一次模型，对ImageNetV2 MatchedFrequency 清单逐图评测。
使用与板卡匹配的制品（X5 `.bin`；S100、S100P 或 S600 `.hbm`），输入为
`bt601_video` NV12，输出为 float32 logits，并用 `--variant`（`v1-100`、`v1-125`）选择检查点的预处理配置。前处理为 PIL bicubic 短边缩放
（100 为 256，125 为 248）、中心裁剪到模型输入尺寸（224），再调用样例的 OpenCV NV12 转换；
这与 runtime 使用 `--resize-type 2` 时的几何相同。campaign 固定标签顺序、
几何变换和完整 10,000 张图片数量；程序校验每张图片与模型的 SHA256。

```bash
python3 samples/vision/mobilenetv1/evaluator/evaluate_board.py \
  --target s100 --variant v1-100 \
  --model samples/vision/mobilenetv1/model/s100/mobilenetv1_100_nashe_224x224_nv12.hbm \
  --model-sha256 <model/README_cn.md 中的 sha256> \
  --campaign /path/to/campaign.json --manifest /path/to/manifest.json \
  --data-root /path/to/imagenetv2/images --output /path/to/new-evaluation
```

新输出目录包含全量预测、张量元信息、三个固定输入的输出，以及记录 Top-1/Top-5 和源码哈希的
`evaluation.json`。C++ 性能计时与输入边界见[评测说明](../../../../utils/tools/classification/cpp/README.md)。

<a id="dataset"></a>

## 数据集
单图分类使用随附测试图。上面的 `evaluate_board.py` 使用 ImageNetV2 MatchedFrequency（10,000 张）。下文逐图统计方法使用 ImageNet ILSVRC2012 验证集（50,000 张、1,000 类）。准备逐图到模型零起始类别索引的真值映射，并与运行时返回的 Top-1 类别 ID 对照。`datasets/imagenet/imagenet_classes.names` 将输出索引映射为显示名称；逐图真值取自数据集标注。参见 [ImageNet 数据准备](../../../../datasets/imagenet/README_cn.md)。

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

主机 FP32 ONNX 评估使用 `evaluate.py`，需要 Python 3.10 和 `utils/tools/classification/requirements.lock.txt` 中的依赖；安装与运行方法见[公共主机流程](../../../../utils/tools/classification/README_cn.md)。

板端分类与 `evaluate_board.py` 需要目标板卡的 `hbm_runtime` 镜像、NumPy、OpenCV-Python、Pillow、PyYAML、模型文件和标签文件。依赖安装见[Python 运行说明](../runtime/python/README_cn.md#environment)。

<a id="command"></a>
## 命令

X5 板端分类（前置：`bash samples/vision/mobilenetv1/model/download.sh x5 100`；输出：退出码 0 并打印 Top-K 类别和分数）：

```bash
python3 samples/vision/mobilenetv1/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv1:mobilenetv1_100_bayese_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv1/model/mobilenetv1_100_bayese_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv1/test_data/bulbul.JPEG \
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
| Top-K 一致 | 同一制品重复运行 softmax 后 Top-K 类别 ID 相同，分数差在 1e-5 内 | 同板、同制品字节、同图、同缩放、同 Top-K |
| Top-1 精度 | argmax 正确比例 | 在准备好的 ImageNet ILSVRC2012 验证集上度量 |
| 延迟 / FPS | 在匹配板卡上的推理计时 | 与[参考结果](#reference-results)的已发布数值按其声明的条件对照 |

<a id="outputs"></a>
## 输出

板端命令在终端打印 Top-K 类别 ID、分数和标签。使用 `--img-save-path` 保存标注图。比较结果时，一并记录板卡、模型、命令、输入图像、缩放方式和原始分数。

<a id="reference-results"></a>
## 参考结果

已发布模型的精度，用上面的固定板端评测在完整 ImageNetV2 MatchedFrequency 集合
（10,000 张图像）上测得。它不是 ILSVRC2012 验证集，因此数值不能与 ImageNet-1k
验证集的结果相比。“FP32”是同一检查点的 ONNX 导出在相同裁剪图上的结果。

| 模型 | Target | FP32 Top-1 | 板端 Top-1 | FP32 Top-5 | 板端 Top-5 |
| --- | --- | --- | --- | --- | --- |
| MobileNetV1-100 | X5 | 62.86% | 59.45%* | 84.37% | 81.53% |
| MobileNetV1-100 | S100 | 62.86% | 62.00% | 84.37% | 83.51% |
| MobileNetV1-100 | S100P | 62.86% | 62.00% | 84.37% | 83.51% |
| MobileNetV1-100 | S600 | 62.86% | 62.04% | 84.37% | 83.66% |
| MobileNetV1-125 | X5 | 64.25% | 63.14% | 85.23% | 84.17% |
| MobileNetV1-125 | S100 | 64.25% | 63.19% | 85.23% | 84.27% |
| MobileNetV1-125 | S100P | 64.25% | 63.19% | 85.23% | 84.27% |
| MobileNetV1-125 | S600 | 64.25% | 62.95% | 85.23% | 84.12% |

\* MobileNetV1-100 在 X5 上相对 FP32 的 Top-1 损失为 5.4%；见 [sample README](../README_cn.md#performance)。

相同制品的延迟与吞吐及测量条件见 [sample README](../README_cn.md#performance)。
下面的 X5 与 X3 数据属于早期构建（权重与前处理不同），请按各表的权重和预处理条件比较。

### RDK X5 / X5 Module 性能

数据版本：`rdk_x5_legacy @ cb86079ae5befcef9ca50fb46c8a6d8980106dec`.

X5 的 BPU 为 1×Bayes-e。下文八线程条件对应 X3 数据。

以下表格是在 RDK X5 & RDK X5 Module 上实际测试得到的性能数据


| 模型          | 尺寸(像素)  | 类别数  | 参数量(M) | 浮点Top-1  | 量化Top-1  | 延迟/吞吐量(单线程) | 延迟/吞吐量(多线程) | 帧率     |
| ----------- | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------ |
| MobileNetv1 | 224x224 | 1000 | 1.33   | 71.74 | 65.36 | 1.27        | 2.90        | 1356.25 |


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
| MobileNetv1 | 224x224 | 1000 | 1.33   | 71.74 | 65.36 | 3.44        | 6.10        | 647.83 |


说明:
1. X3的状态为最佳状态：CPU为4xA53@1.5G, 全核心Performance调度, BPU为2xBernoulli@1G, 共5TOPS等效int8算力。
2. 单线程延迟统计单帧、单线程、单 BPU 核的推理时间。
3. 四线程帧率使用四个线程并发提交任务，BPU 利用率为 100%。
4. X3 八线程数据使用八个线程，向两个 BPU 核并发提交任务。
5. 浮点/定点Top-1：浮点Top-1使用的是模型未量化前onnx的 Top-1 推理精度，量化Top-1则为量化后模型实际推理的精度。

<a id="boundaries"></a>
## 数据集级评估

计算数据集 Top-1 精度时，将每张验证图像通过 `--test-img` 传给运行时入口，把返回的 Top-1 类别 ID 与该图像的模型真值索引对照，再用正确预测数除以已评测的带标签图像数。对照运行时固定制品、resize 模式、Top-K、板卡镜像和调度设置。测量延迟或 FPS 时，在匹配板卡上计时推理阶段，并记录线程数和工作模式。
