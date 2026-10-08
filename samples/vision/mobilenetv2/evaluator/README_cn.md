# MobileNetV2 评估
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

主机检查需要 sample 根目录 `requirements-host.txt` 的用户态 Python 依赖，
不需要板卡 SDK。功能板卡检查需要目标板卡及其 `hbm_runtime` 镜像、已准备
制品与标签文件。数据集级评估还需随结果声明 OE/板卡工具链。

<a id="command"></a>
## 命令

主机检查（cwd：仓库根目录；成功判据：全部 OK，退出码 0）：

```bash
python3 -m unittest discover -s samples/vision/mobilenetv2/tests -v
```

X5 功能板卡检查（前置：`bash samples/vision/mobilenetv2/model/download.sh x5`；成功判据：退出码 0 且 Top-K 符合预期）：

```bash
python3 samples/vision/mobilenetv2/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv2:mobilenetv2_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv2/model/mobilenetv2_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv2/test_data/Scottish_deerhound.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

S100/S600 替换 `s:` 引用与 `s100/`/`s600/` 制品路径；标签文件共用。同板多次运行对照时，固定同一图像、制品字节、标签、resize 类型与
Top-K，在标签格式化之前比较类别 ID 与原始分数；预期类别 ID 相同、
分数差在 1e-5 内。X5 与 S 的结果互相对照不构成同板对照。

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

| 项目 | 数值 | 来源 |
| --- | --- | --- |

X5 发布（x5-v1.1.3）的已发布数值：

| 模型 | 尺寸 | 类别数 | 参数量 (M) | Float Top-1 | Quant Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV2 | 224x224 | 1000 | 3.4 | 72.0% | 68.17% | 1.42 | 1152.07 |

对 `s-v1.1.2` 的 S100/S600 制品，在匹配板卡上运行，并按[数据集级评估](#boundaries)
计算精度与计时。

<a id="boundaries"></a>
## 数据集级评估

计算数据集 Top-1 精度时，将每张验证图像通过 `--test-img` 传给运行时入口，把返回的 Top-1 类别 ID 与该图像的模型真值索引对照，再用正确预测数除以已评测的带标签图像数。对照运行时固定制品、resize 模式、Top-K、板卡镜像和调度设置。测量延迟或 FPS 时，在匹配板卡上计时推理阶段，并记录线程数和工作模式。
