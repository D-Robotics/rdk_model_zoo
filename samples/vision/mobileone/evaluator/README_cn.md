# MobileOne 评测
使用随附图片进行单图分类检查。计算数据集精度时，准备对应验证集及逐图真值类别索引，并将其与运行时返回的 Top-1 类别 ID 对照。

<a id="dataset"></a>

## 数据集
功能检查使用随附测试图。数据集级精度使用 ImageNet ILSVRC2012 验证集（50,000 张、1,000 类）。准备逐图到模型零起始类别索引的真值映射，并与运行时返回的 Top-1 类别 ID 对照。`datasets/imagenet/imagenet_classes.names` 将输出索引映射为显示名称；逐图真值取自数据集标注。参见 [ImageNet 数据准备](../../../../datasets/imagenet/README_cn.md)。

<a id="environment"></a>
## 环境

主机：sample requirements，源对照额外使用 SciPy。板端：runtime README 所述 X5 环境。检查复用同一分类任务；本目录提供操作说明，没有另一个基准可执行程序。

<a id="command"></a>
## 命令

```bash
# cwd: repository root; expected: unittest OK, exit 0
python3 -m unittest discover -s samples/vision/mobileone/tests -v
```

板端功能检查：

```bash
# cwd: repository root
bash samples/vision/mobileone/model/download.sh x5 s0
python3 samples/vision/mobileone/runtime/python/main.py \
  --target x5 --variant s0 \
  --test-img samples/vision/mobileone/test_data/tiger_beetle.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

在匹配板卡上对每个已发布变体分别执行。同板多次运行对照时，保持
模型字节、图像、resize 类型、Top-K 和调度参数一致，在标签格式化之前
比较类别 ID 与原始分数；预期类别 ID 相同、分数差在 1e-5 内。当 Top-K
边界出现完全平局时，核对逐 ID 分数而不是放宽容差。

<a id="metrics"></a>
## 指标

Top-1 精度按正确类别 ID 数除以已标注图像数计算；Top-K 对照使用运行时返回的类别 ID 与原始分数。延迟和 FPS 需在匹配板卡上测量推理阶段，并记录线程数、并发量和工作模式。

<a id="outputs"></a>
## 输出

板端 CLI 打印推理结果，并可选保存可视化。运行数据集评测时使用验证集标注中的逐图真值类别 ID，并将其与运行时 Top-1 ID 对照。

<a id="reference-results"></a>
## 参考结果

下表来自已发布的源 evaluator。

源条件：X5 CPU 8×A55@1.8GHz 性能模式、BPU Bayes-e@1GHz。Float Top-1 为量化前 ONNX，Quant Top-1 为部署结果；单线程延迟为单帧单 BPU 核，多线程延迟和 FPS 使用并发提交。开展新对照时，使用相同数据子集，并记录预热、重复次数、板卡模式和并发设置。

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Single-thread Latency (ms) | Multi-thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileOne_S4 | 224x224 | 14.8 | 78.75% | 76.50% | 4.58 | 15.44 | 256.52 |
| MobileOne_S3 | 224x224 | 10.1 | 77.27% | 75.75% | 2.93 | 9.04 | 437.85 |
| MobileOne_S2 | 224x224 | 7.8 | 74.75% | 71.25% | 2.11 | 6.04 | 653.68 |
| MobileOne_S1 | 224x224 | 4.8 | 72.31% | 70.45% | 1.31 | 3.69 | 1066.95 |
| MobileOne_S0 | 224x224 | 2.1 | 69.25% | 67.58% | 0.80 | 1.59 | 2453.17 |

<a id="boundaries"></a>
## 数据集级评估

计算数据集 Top-1 精度时，将每张验证图像通过 `--test-img` 传给运行时入口，把返回的 Top-1 类别 ID 与该图像的模型真值索引对照，再用正确预测数除以已评测的带标签图像数。对照运行时固定制品、resize 模式、Top-K、板卡镜像和调度设置。测量延迟或 FPS 时，在匹配板卡上计时推理阶段，并记录线程数和工作模式。
