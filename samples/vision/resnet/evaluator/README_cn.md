# ResNet18 评估

评估有两个目的：确认目标板卡按预期张量契约执行所选制品，以及在明确数据集
与工具链后测量精度或延迟。本目录对两者做文档记录，但不含自有精度工具
（见[边界](#boundaries)）。

<a id="dataset"></a>
## 数据集

对当前范围不适用：本 sample 执行功能性检查（随仓测试图），不跑数据集级
精度评估。数据集级评估需要用户自行准备的 ImageNet 验证集（50,000 张，
ILSVRC2012 val）；本 sample 不提供数据集下载或准备脚本。

<a id="environment"></a>
## 环境

主机检查需要 sample 根目录 `requirements-host.txt` 的用户态 Python 依赖，
不需要板端 SDK。功能性板端检查需要带 `hbm_runtime` 镜像的目标板、已准备
的制品与标签文件。数据集级评估还需随结果一并说明的 OE/板端工具链。

<a id="command"></a>
## 评估命令

主机检查（cwd：仓库根目录；成功判据：全部 OK，退出码 0）：

```bash
python3 -m unittest discover -s samples/vision/resnet/tests -v
```

X5 功能性板端检查（前置：`bash samples/vision/resnet/model/download.sh x5`；
成功判据：退出码 0 且 Top-K 符合预期）：

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

S100/S600 替换 `s:resnet18:<target>/...` 引用与制品路径；标签文件两侧
共用。同板前后对照：用相同图片、模型字节、标签、缩放方式与 Top-K 运行旧
入口（`platforms/x5/samples/vision/resnet/runtime/python/main.py` 或
`platforms/s/samples/vision/resnet18/runtime/python/main.py`），先比较类别
ID 与 raw 分数，再比较标签排版。S 系列 C++ 检查执行
`bash samples/vision/resnet/runtime/cpp/run.sh`。

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| 契约通过 | 运行时接受制品，张量名/形状/类型与绑定一致，返回单一 F32 分数向量 | 任意已准备制品在匹配板卡上 |
| Top-K 一致性 | canonical 与旧运行结果类别 ID 与 raw 分数完全一致 | 同板、同制品字节、同图、同缩放、同 Top-K |
| Top-1 精度 | argmax 正确预测占比 | ImageNet val——本 sample 未评估 |
| 延迟 / FPS | 推理耗时 | 本 sample 未评估；下方历史数值条件未注明 |

<a id="outputs"></a>
## 输出

主机检查打印 unittest 结果。功能性板端检查在 stdout 打印 Top-K（类别 ID、
分数、标签），可用 `--img-save-path` 另存标注图。留证时保存板卡身份、
模型引用、运行时元数据、raw F32 分数张量、Top-K 输出、图片路径、缩放方式
与命令行。

<a id="reference-results"></a>
## 参考结果

| 项目 | 数值 | 来源 |
| --- | --- | --- |
| 主机测试 | 46 OK（2026-09-21，B1 主机 evidence JSON） | 迁移证据 |
| 板端对照 | X5 双板与 S100 上 canonical == 旧入口（类别 ID 与 raw 分数） | 2026-09-17 集成评审 |
| S100 C++ | Top-5 文本与源基线一致 | 2026-09-17 集成评审 |
| 数据集精度 / 延迟 | 本 sample not-run | — |

历史旧版数值（X5 评估记录）：Top-1 71.5%（float）/ 70.5%（量化），延迟
2.95 ms，449+ FPS。源记录未说明延迟/FPS 采用单次调用、批量还是多线程，
此处不做换算，也不将其呈现为 canonical sample 的结果。

<a id="boundaries"></a>
## 边界

本 sample 不附带数据集级精度或延迟工具：入库材料仅覆盖主机契约测试与
功能性板端检查。主机测试通过不能证明板端可用。板卡不可达或制品缺失时，
对应项记为 `not-run` 而不是忽略。S600 复验在板卡连接恢复前保持
`not-run`。
