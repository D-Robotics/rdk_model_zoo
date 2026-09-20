# MobileNetV1 评估

评估有两个独立目的：确认板卡按声明的张量契约执行所选制品，以及以明确的
数据集与工具链度量精度或延迟。本目录记录两者；不含自己的精度 harness
（见[边界](#boundaries)）。

<a id="dataset"></a>
## 数据集

当前范围为不适用：本 sample 做功能检查（内置测试图），不运行数据集级精度
评估。数据集级评估需要用户另行准备的 ImageNet 验证集（ILSVRC2012 val，
50,000 张）；本仓库不提供数据集下载或准备脚本。

<a id="environment"></a>
## 环境

主机检查需要 sample 根目录 `requirements-host.txt` 的用户态 Python 依赖，
不需要板卡 SDK。功能板卡检查需要目标板卡及其 `hbm_runtime` 镜像、已准备
制品与标签文件。数据集级评估还需随结果声明 OE/板卡工具链。

<a id="command"></a>
## 命令

主机检查（cwd：仓库根目录；成功判据：全部 OK，退出码 0）：

```bash
python3 -m unittest discover -s samples/vision/mobilenetv1/tests -v
```

X5 功能板卡检查（前置：`bash samples/vision/mobilenetv1/model/download.sh x5`；成功判据：退出码 0 且 Top-K 符合预期）：

```bash
python3 samples/vision/mobilenetv1/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv1:mobilenetv1_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv1/model/mobilenetv1_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv1/test_data/bulbul.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

S100/S600 替换 `s:` 引用与 `s100/`/`s600/` 制品路径；标签文件共用。同板
前后对照：用相同图片、模型字节、标签、缩放方式与 Top-K 运行旧平台入口
（`platforms/x5/samples/vision/mobilenetv1/runtime/python/main.py` 或
`platforms/s/samples/vision/mobilenetv1/runtime/python/main.py`），先比较类别 ID
与 raw 分数，再比较标签排版。X5 与 S 相互比较不能替代同板前后比较。

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| 契约通过 | 运行时接受制品，张量名称/shape/dtype 与绑定一致，返回一个 F32 分数向量 | 匹配板卡上的任一已准备制品 |
| Top-K 一致 | canonical 与旧入口运行之间类别 ID 与 raw 分数一致 | 同板、同制品字节、同图、同缩放、同 Top-K |
| Top-1 精度 | argmax 正确比例 | ImageNet val——本 sample 未评估 |
| 延迟 / FPS | 推理耗时 | 本 sample 未评估；下方历史数值条件未注明 |

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
| 主机测试 | 每 sample 14 OK（2026-09-21，B1 主机证据 JSON） | 迁移证据 |
| 板卡对照（canonical vs 旧入口） | not-run（B1 板端冒烟待用户执行） | — |
| 数据集精度 / 延迟 | 本 sample 未评估 | — |

X5 侧源发布（rdk_x5 @ac11571 (x5-v1.1.3)）的公开历史数据（条件未注明，不作为 canonical sample
的结果呈现）：

| 模型 | 尺寸 | 类别数 | 参数量 (M) | Float Top-1 | Quant Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV1 | 224x224 | 1000 | 4.2 | 71.7% | 65.4% | 0.58 | 2800+ |

S 侧源发布（rdk_s @380e1a2 (s-v1.1.2)）未公布该模型的精度/延迟数据，此处不推断。

<a id="boundaries"></a>
## 边界

本 sample 不随附数据集级精度或延迟 harness：仓库内材料只覆盖主机契约测试
与功能板卡检查。主机测试通过永远不等于板卡认证。板卡不可达或制品不可得
时对应项记为 `not-run`，而不是失败后遗忘。S600 复测在板卡访问恢复前保持
`not-run`。
