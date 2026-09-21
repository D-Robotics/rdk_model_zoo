# EfficientNet 评估

评估有两个独立目的：确认某块板卡以预期的张量契约执行所选制品，以及
在声明数据集与工具链的条件下测量精度或延迟。本目录对两者做出说明；
目录本身不含精度评估工具（见[边界](#boundaries)）。

<a id="dataset"></a>
## 数据集

当前范围不适用：本 sample 做功能检查（随附测试图），不运行数据集级
精度评估。数据集级评估需要用户自行准备 ImageNet 验证集（ILSVRC2012
val，50,000 张）；不提供数据集下载或准备脚本。

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
文件共用。同板迁移前后对照请用相同的图像、制品字节、标签、resize
类型和 Top-K 运行旧平台入口
（`platforms/x5/samples/vision/efficientnet/runtime/python/main.py` 或
`platforms/s/samples/vision/efficientnet/runtime/python/main.py`），
在标签格式化之前对比类别 ID 与原始分数。X5 与 S 互相对照不能替代
同板前后对照。相同输入重复运行时输出应当有限、非零且稳定。

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| 契约通过 | 运行时接受制品，张量名/形状/dtype 与绑定一致，返回一个 F32 分数向量 | 任何已准备制品在匹配板卡上 |
| Top-K 一致 | 规范实现与旧实现类别 ID 与原始分数完全一致 | 同板、同制品字节、同图像、同 resize、同 Top-K |
| Top-1 精度 | argmax 正确的样本比例 | ImageNet val —— 本 sample 未评估 |
| 延迟 / FPS | 推理耗时 | 本 sample 未评估；下方历史数值的条件未完整声明 |

<a id="outputs"></a>
## 输出

主机检查打印 unittest 结果。板上功能检查在 stdout 打印 Top-K（类别
ID、分数、标签），可选通过 `--img-save-path` 写标注图像。留存证据时
请保存：板卡身份、模型引用、运行时 metadata、原始 F32 分数张量、
Top-K 输出、图像路径、resize 类型与命令行。

<a id="reference-results"></a>
## 参考结果

| 项目 | 取值 | 来源 |
| --- | --- | --- |
| 主机测试 | 25 OK（2026-09-21，作者自检） | 迁移证据 |
| 板上对照（规范 vs 旧实现） | not-run（B2 板端冒烟未执行；执行后回填） | — |
| 数据集精度 / 延迟 | 本 sample not-run | — |

已发布的历史数值，未在本仓库重测。

X5 源发布（rdk_x5 @ac11571，x5-v1.1.3；源说明：Float Top-1 为量化前
ONNX 结果，Quant Top-1 为部署模型结果，延迟为单帧单线程单核，FPS 为
多线程；CPU 8xA55@1.8GHz 性能模式，BPU 1xBayes-e@1GHz）：

| 模型 | 尺寸 | 参数量 (M) | Float Top-1 | Quant Top-1 | 单线程延迟 (ms) | 多线程延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientNet-B4 | 224x224 | 19.27 | 74.25% | 71.75% | 5.44 | 18.63 | 212.75 |
| EfficientNet-B3 | 224x224 | 12.19 | 76.22% | 74.05% | 3.96 | 12.76 | 310.30 |
| EfficientNet-B2 | 224x224 | 9.07 | 76.50% | 73.25% | 3.31 | 10.51 | 376.77 |

S 源发布（rdk_s @380e1a2，s-v1.1.2；除表格外条件未声明）：

| 变体 | 单线程延迟 | 单线程 FPS | 多线程延迟 | 多线程 FPS |
| --- | --- | --- | --- | --- |
| Lite0 | 0.448 ms | 2107.815 | 0.591 ms | 4827.886 |
| Lite1 | 0.489 ms | 1948.957 | 0.708 ms | 4086.470 |
| Lite2 | 0.565 ms | 1702.519 | 0.935 ms | 3123.682 |
| Lite3 | 0.668 ms | 1451.031 | 1.249 ms | 2345.518 |
| Lite4 | 0.915 ms | 1064.339 | 1.979 ms | 1487.055 |

<a id="boundaries"></a>
## 边界

本 sample 不附带数据集级精度或延迟评估工具：检入材料只覆盖主机契约
测试与板上功能检查。主机测试通过绝不等于板卡认证。板卡不可达或制品
不可用时，对应项记为 `not-run`，而不是失败后略过。
