[English](README.md) | 简体中文

# EfficientViT 评估
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
`requirements-host.txt`），不需要板端 SDK。功能板测需要带
`hbm_runtime` 镜像的 X5 板卡、已准备的制品和标签文件。数据集级评估
还需随结果声明 OE/板端工具链。

<a id="command"></a>
## 命令

主机检查（cwd：仓库根目录；成功判据：全部 OK，退出码 0）：

```bash
python3 -m unittest discover -s samples/vision/efficientvit/tests -v
```

X5 功能板测（前置：`bash samples/vision/efficientvit/model/download.sh x5`；
成功判据：退出码 0 且 Top-5 含挂钩相关类别）：

```bash
python3 samples/vision/efficientvit/runtime/python/main.py \
  --target x5 \
  --asset-id x5:efficientvit:EfficientViT_m5_224x224_nv12.bin \
  --model-path samples/vision/efficientvit/model/EfficientViT_m5_224x224_nv12.bin \
  --test-img samples/vision/efficientvit/test_data/hook.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

同板多次运行对照时，固定同一图像、制品字节、标签、resize 类型和
Top-K，在标签格式化之前比较类别 ID 和 Top-K 分数；预期类别 ID 相同、
分数差在 1e-5 内，输出有限、非零且稳定。

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| 契约通过 | 运行时接受制品，张量名/形状/dtype 与绑定一致，返回一个 F32 分数向量 | 任一制品在其匹配板卡上 |
| Top-K 一致 | 同一制品重复运行 softmax 后 Top-K 类别 ID 相同，分数差在 1e-5 内 | 同板、同制品字节、同图、同 resize、同 Top-K |
| Top-1 精度 | argmax 正确的比例 | 在准备好的 ImageNet ILSVRC2012 验证集上度量 |
| 延迟 / FPS | 在匹配板卡上的推理计时 | 与[参考结果](#reference-results)的已发布数值按其声明的条件对照 |

<a id="outputs"></a>
## 输出

主机检查打印 unittest 结果。功能板测在 stdout 打印 Top-K（类别 ID、
分数、标签），可用 `--img-save-path` 可选写标注图。取证时保存：板卡
身份、模型引用、运行时 metadata、原始 F32 分数张量、Top-K 输出、图像
路径、resize 类型和完整命令行。

<a id="reference-results"></a>
## 参考结果

X5 发布（x5-v1.1.3）的已发布数值（Float Top-1 为量化前 ONNX 结果，Quant Top-1 为部署模型结果；源表未声明
延迟的线程条件）：

| 模型 | 尺寸 | 类别数 | 参数量 (M) | Float Top-1 | Quant Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientViT_m5 | 224x224 | 1000 | 12.4 | 73.75% | 72.50% | 6.34 | 174.70 |

已发布记录中量化 Top-1（72.50%）与 float 值（73.75%）接近。

### RDK X5 性能

数据版本：x5-v1.1.3。使用 `.bin` 模型与 `hbm_runtime`，CPU 8×A55@1.8GHz、全核 Performance 调度，BPU 1×Bayes-e@1GHz，10TOPS INT8。单线程延迟为单帧、单线程、单 BPU 核推理时间，多线程延迟和 FPS 对应并发提交任务。

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Single-thread Latency (ms) | Multi-thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientViT_m5 | 224x224 | 12.4 | 73.75% | 72.50% | 6.34 | 22.69 | 174.70 |

<a id="boundaries"></a>
## 数据集级评估

计算数据集 Top-1 精度时，将每张验证图像通过 `--test-img` 传给运行时入口，把返回的 Top-1 类别 ID 与该图像的模型真值索引对照，再用正确预测数除以已评测的带标签图像数。对照运行时固定制品、resize 模式、Top-K、板卡镜像和调度设置。测量延迟或 FPS 时，在匹配板卡上计时推理阶段，并记录线程数和工作模式。
