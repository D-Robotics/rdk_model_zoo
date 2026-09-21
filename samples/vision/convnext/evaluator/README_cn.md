# ConvNeXt 评估

评估有两个独立目的：确认某块板卡按预期张量契约执行所选制品，以及在
声明的数据集和工具链下度量精度或延迟。本目录对两者做说明；自身不带
精度工具（见 [边界](#boundaries)）。

<a id="dataset"></a>
## 数据集

当前范围不适用：本 sample 做功能检查（随附测试图），不运行数据集级
精度评估。如需数据集级评估，需要用户自行准备 ImageNet 验证数据
（ILSVRC2012 val，50,000 张）；不提供数据集下载或准备脚本。

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
python3 -m unittest discover -s samples/vision/convnext/tests -v
```

X5 功能板测（前置：`bash samples/vision/convnext/model/download.sh x5`；
成功判据：退出码 0 且 Top-5 含挂钩相关类别）：

```bash
python3 samples/vision/convnext/runtime/python/main.py \
  --target x5 \
  --asset-id x5:convnext:ConvNeXt_atto_224x224_nv12.bin \
  --model-path samples/vision/convnext/model/ConvNeXt_atto_224x224_nv12.bin \
  --test-img samples/vision/convnext/test_data/cheetah.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

同板前后对照：用同一图像、模型字节、标签、resize 类型和 Top-K 运行
legacy 平台入口
（`platforms/x5/samples/vision/convnext/runtime/python/main.py`），
在标签格式化之前比较类别 ID 和 Top-K 分数（ids 相同；分数按声明的容差判定——判定标准 |分数差| < 1e-5；精确平局以记录的 top-8 逐 ID 证据裁定；不声明 raw tensor 逐位相等）。输出应为有限、非零，且同一
输入下重复运行保持稳定。

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| 契约通过 | 运行时接受制品，张量名/形状/dtype 与绑定一致，返回一个 F32 分数向量 | 任一制品在其匹配板卡上 |
| Top-K 一致 | 规范实现与旧实现 softmax 后 Top-K 类别 ID 相同，分数在容差内（判据 |分数差| < 1e-5；精确平局以 top-8 逐 ID 证据裁定；不声明 raw tensor 逐位相等） | 同板、同制品字节、同图、同 resize、同 Top-K |
| Top-1 精度 | argmax 正确的比例 | ImageNet val——本 sample 未评估 |
| 延迟 / FPS | 推理计时 | 本 sample 未评估；下方历史数值条件未声明 |

<a id="outputs"></a>
## 输出

主机检查打印 unittest 结果。功能板测在 stdout 打印 Top-K（类别 ID、
分数、标签），可用 `--img-save-path` 可选写标注图。取证时保存：板卡
身份、模型引用、运行时 metadata、原始 F32 分数张量、Top-K 输出、图像
路径、resize 类型和完整命令行。

<a id="reference-results"></a>
## 参考结果

| 项目 | 值 | 来源 |
| --- | --- | --- |
| 主机测试 | 28 OK（2026-09-21，作者自检） | 迁移证据 |
| 板上对照（规范化 vs legacy） | not-run（B3 板端冒烟待执行；执行后回填） | — |
| 数据集精度 / 延迟 | 本 sample not-run | — |

X5 源发布（rdk_x5 @ac11571，x5-v1.1.3）的已发布历史数值（源说明：
Float Top-1 为量化前 ONNX 结果，Quant Top-1 为部署模型结果；延迟为单帧
单线程单核，FPS 为 4 线程并发 100% BPU 利用率，CPU 8xA55@1.8GHz 性能
模式、BPU 1xBayes-e@1GHz）：

| 模型 | 尺寸 | 类别数 | 参数量 (M) | Float Top-1 | Quant Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ConvNeXt_nano | 224x224 | 1000 | 15.59 | 77.37% | 71.75% | 5.71 | 200+ |
| ConvNeXt_pico | 224x224 | 1000 | 9.04 | 77.25% | 71.03% | 3.37 | 364+ |
| ConvNeXt_femto | 224x224 | 1000 | 5.22 | 73.75% | 72.25% | 2.46 | 556+ |

已发布表覆盖 nano/pico/femto——**不含 atto**（唯一有已发布制品的变体）；
可下载模型没有基准行。按发布原样记录，此处不复测。

<a id="boundaries"></a>
## 边界

本 sample 不带数据集级精度或延迟工具：检入材料只覆盖主机契约测试和
功能板测。主机测试通过永远不能认证板卡。板卡不可达或制品不可得时，
对应条目记为 `not-run`，不是失败后遗忘。
