[English](README.md) | 简体中文

# ViT 评测
使用随附图片进行单图分类检查。计算数据集精度时，准备对应验证集及逐图真值类别索引，并将其与运行时返回的 Top-1 类别 ID 对照。

<a id="dataset"></a>

## 数据集
功能检查使用随附的十张 CIFAR-10 图片，每类一张。数据集级精度使用完整 CIFAR-10 测试集。为每张测试图准备其真值类别索引（0–9），并与运行时返回的 Top-1 类别 ID 对照；随附示例覆盖每个类别一张图片。

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="environment"></a>
## 环境

[运行时前提](../runtime/python/README_cn.md#environment)。板端功能检查
需要带 `hbm_runtime` 的 S100 板卡与已准备的制品；评估不需要 OE 环境。

<a id="command"></a>
## 命令

```bash
# cwd: repository root; expected: unittest OK, exit 0
python3 -m unittest discover -s samples/vision/vit/tests -v
```

准备模型后在匹配板卡上执行功能检查；换用 int16 制品并遍历其余随附
图片可扩大功能覆盖：

```bash
# cwd: repository root
bash samples/vision/vit/model/download.sh s100 int8
python3 samples/vision/vit/runtime/python/main.py --target s100 --variant int8 \
  --test-img samples/vision/vit/test_data/airplane_0000.png \
  --label-file samples/vision/vit/test_data/cifar10_classes.names --top-k 5
```

同板多次运行对照时，固定制品字节、图像、resize 类型与 Top-K，在标签
格式化之前比较类别 ID 与原始分数；预期类别 ID 相同、分数差在 1e-5 内。
当 Top-K 边界出现完全平局时，核对逐 ID 分数而不是放宽容差。

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| 契约通过 | 运行时接受制品，张量名/形状/dtype 与绑定一致，返回一个 F32 分数向量 | 已准备制品在匹配的 S100 板卡上 |
| Top-K 一致 | 同一制品重复运行 softmax 后 Top-K 类别 ID 相同，分数差在 1e-5 内 | 同板、同制品字节、同图、同 resize、同 Top-K |
| Top-1 / Top-5 精度 | 真值分别为 rank-1 / 落在选中的 K 个类别中 | 在用户准备的 CIFAR-10 测试集上度量 |
| 延迟 / FPS | 在匹配板卡上的推理计时 | 使用匹配板卡运行样例入口并记录实测值 |

<a id="outputs"></a>
## 输出

主机测试输出 unittest 结果。板端功能检查在 stdout 打印 Top-K（类别
ID、分数、标签），可用 `--img-save-path` 可选写标注图。记录运行时，
保存板卡身份、模型引用、命令行、原始 F32 分数张量与 Top-K 输出，
并附图像路径与 resize 类型。

<a id="reference-results"></a>
## 参考结果

原 ViT evaluator 记录（CIFAR-10）发布的数值：

| Model | Top-1 | Top-5 |
| --- | --- | --- |
| ONNX | 74.54% | 98.36% |
| HBM | 72.62% | 98.03% |

发布记录说明 PTQ 使用 50 张校准图、无 QAT；记录未区分 int8/int16。

<a id="boundaries"></a>
## 数据集级评估

计算数据集 Top-1 精度时，将每张验证图像通过 `--test-img` 传给运行时入口，把返回的 Top-1 类别 ID 与该图像的模型真值索引对照，再用正确预测数除以已评测的带标签图像数。对照运行时固定制品、resize 模式、Top-K、板卡镜像和调度设置。测量延迟或 FPS 时，在匹配板卡上计时推理阶段，并记录线程数和工作模式。
