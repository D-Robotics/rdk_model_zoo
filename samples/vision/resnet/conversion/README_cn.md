# ResNet 模型转换（ResNet18 / ResNet50 / ResNet152）

转换在 x86 Linux 主机的 RDK OpenExplore（OE）环境完成，不是板端操作。本目录
覆盖该 sample 发布的三个变体，各变体可复现范围不同——按下表如实声明，绝不
夸大：

| 变体 | 本目录内容 | 可复现范围 |
| --- | --- | --- |
| ResNet18（x5+s） | `export_resnet18_onnx.py` | 主机可重放 ONNX 导出；校准/YAML 由 OE `13_resnet18` 示例承担（声明的缺口） |
| ResNet50（仅 s） | — | 源头无配方（rdk_s @380e1a2 只有 README 指引）；以 OE `13_resnet50` 示例为准 |
| ResNet152（仅 s） | `resnet152_config.yaml`、`get_calibration_data.py`、`x86_inference.py` | 源分支 OE 配方原样保留；在 OE 内给定已发布 ONNX 与用户自备校准图即可重放——重放复现的是配方步骤，不等于与已发布制品数值等价（保留的 scale 差异见"校准"一节） |

ResNet152 三个文件逐字节来自
`rdk_s@380e1a2:samples/vision/resnet152/conversion/`（本批审计源；
SHA-256 由 `tests/test_conversion_layout.py` 固定）。调整它们属于源分支
变更，不是本地编辑。

<a id="source-model"></a>
## 源模型

三个变体均为 TorchVision ResNet 系列 ImageNet-1k 分类器：
[ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)、
[ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)、
[ResNet152](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html)。

- ResNet18：本地导出，固定 NCHW 输入 `[1,3,224,224]`、输出名 `output`、
  ImageNet-1k 类别数。canonical 导出器使用 TorchScript ONNX 导出
  （`dynamo=False`）以保持图稳定。`--weights IMAGENET1K_V1` 选择官方
  预训练权重；`--weights none` 生成随机权重图，仅用于离线结构检查。
- ResNet50：审计源只发布了 HBM 制品与 OE 分类示例指引；未发布 ONNX
  URL、YAML 或导出脚本。本目录不虚构。
- ResNet152：有已发布 ONNX——在 OE 容器内（或任何可访问该链接的 x86
  主机）：

```bash
# cwd：本 conversion 目录 — 输出：./resnet152.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet152.onnx
```

<a id="toolchain-targets"></a>
## 工具链与目标

使用与目标板卡匹配的 OE Docker/工具链版本，并记录镜像 tag、OE 版本与
主机日期。权威参考：
[RDK S 工具链概览](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)、
[D-Robotics 工具链下载](https://toolchain.d-robotics.cc/)。
通用的加载与挂载流程：

```bash
# 输入：OE 镜像压缩包 — 其余命令在容器内 /workspace 执行
export OE_IMAGE_TAR=/absolute/path/to/oe-image.tar
test -s "$OE_IMAGE_TAR"
docker load -i "$OE_IMAGE_TAR"
docker images
: "${OE_IMAGE:?Set OE_IMAGE to the loaded OE image:tag}"
export REPOSITORY_ROOT="${REPOSITORY_ROOT:-$PWD}"
docker run --rm -it --network host --shm-size=15g \
  -v "$REPOSITORY_ROOT":/workspace --workdir /workspace \
  "$OE_IMAGE" /bin/bash
```

目标：X5 用 `hb_mapper` 按 march `bayes-e` 编译（以所选 OE 版本为准）；
S100 用 `hb_compile`（`nash-e`），S600 用 `nash-p`。审计源引用的 OE
分类示例：
`samples/ai_toolchain/horizon_model_convert_sample/03_classification/`
下的 `13_resnet18` 与 `13_resnet50`。

<a id="export"></a>
## 导出

ResNet18——导出前置：PyTorch、TorchVision、ONNX（可选 ONNX Runtime 做
主机图冒烟）。在 OE 容器内 `/workspace` 执行：

```bash
# 输入：TorchVision 权重（未缓存时下载）
# 输出：$WORK/resnet18.onnx — 成功判据：onnx.checker 通过，退出码 0
export WORK="${WORK:-/workspace/resnet18-conversion}"
mkdir -p "$WORK"
python3 samples/vision/resnet/conversion/export_resnet18_onnx.py \
  --output "$WORK/resnet18.onnx" \
  --opset 11 \
  --weights IMAGENET1K_V1
```

离线冒烟可用 `--weights none`（随机权重，无精度含义）。仅当由其他记录
过的工具校验时才用 `--no-check` 跳过 `onnx.checker`。导出冒烟已在
Torch 2.7.1、TorchVision 0.22.1、ONNX 1.19、ONNX Runtime 1.23.2 下
执行：输出 `[1,1000]`，与同种子 PyTorch 图在 `1e-4` 内一致。这只是
ONNX 结构检查，不是 BPU 编译或精度结果。

ResNet50——审计制品没有已发布的导出步骤；从 OE `13_resnet50` 示例
开始。ResNet152——用[源模型](#source-model)一节的已发布 ONNX；没有要
运行的导出器。

<a id="calibration"></a>
## 校准

ResNet18——本仓库不可重放（已知缺口）：审计源未发布校准图片列表、
mean/scale 值或 ResNet18 YAML。从 OE 示例复制对应的 `13_resnet18`
文件并按其文档执行预处理与校准，同时记录确切校准数据。本目录的
ResNet152 配置是 ResNet152 的配方——不是 ResNet18 的配方，不得混用。

ResNet152——`get_calibration_data.py`（在 OE 容器内、cwd 为本
conversion 目录）把 100 张 ImageNet 验证集图片转成 float32 RGB 校准
数据。两个输入需要用户自备并在脚本中先行修改（脚本按源分支原样保留，
其默认路径属于旧分支目录树）：

```python
# get_calibration_data.py — 用户需修改的输入
src_image_dir = '../../../open_explorer/samples/ai_toolchain/horizon_model_convert_sample/01_common/calibration_data/imagenet/'  # 改为你的 ILSVRC2012_val_*.JPEG 目录
output_calib_dir = './calibration_data_rgb/'  # 与 resnet152_config.yaml 的 cal_data_dir 一致
```

```bash
# cwd：本 conversion 目录（OE 容器内）
# 输入：src_image_dir 内 >=100 张 ILSVRC2012_val_*.JPEG
# 输出：./calibration_data_rgb/*.npy（float32、RGB、NCHW）— 成功判据：打印 100 个文件
python3 get_calibration_data.py
```

相对路径在本 cwd 内闭环衔接：脚本写出 `./calibration_data_rgb/`，
即 `resnet152_config.yaml` 的 `cal_data_dir`；YAML 的 `onnx_model`
`./resnet152.onnx` 是[源模型](#source-model)一步下载到本目录的文
件；其 `working_dir`/`output_model_file_prefix` 产出
`./model_output/resnet152_224x224_nv12.hbm`，与[编译](#compile)一节
的声明一致。

两个保留源文件的归一化常量**并不一致**：脚本与 YAML 的 mean 相同
（`123.675 116.28 103.53`），但脚本统一乘 `0.017`，YAML 声明逐通道
`scale_value: 0.01712475 0.017507 0.01742919`。这是按源分支原样保留
的差异；已发布制品到底用哪一组系数校准、编译，本仓库未确认（未做
OE 重建或数值对照），也不认定哪一组是"正确值"。发布记录的
Mean/Scale 行与 YAML 值相同（见[转换后验证](#validation)）。

ResNet50——校准由 OE `13_resnet50` 示例承担；本目录无可执行内容。

<a id="compile"></a>
## 编译

ResNet18——在 OE 容器内，备好 `$WORK/resnet18.onnx` 与 `$OE_CONFIG`
（OE 示例的 YAML）后执行。X5 块：

```bash
# 输出：$WORK 下的 *.bin — 成功判据：hb_mapper 退出码 0
export X5_MARCH="${X5_MARCH:-bayes-e}"
hb_mapper --version
hb_mapper checker --model-type onnx --march "$X5_MARCH" --model "$WORK/resnet18.onnx"
hb_mapper makertbin --model-type onnx --config "$OE_CONFIG"
find "$WORK" -type f -name '*.bin' -print
```

S100/S600 块（YAML 必须包含匹配的 Nash target——S100 为 `nash-e`，
S600 为 `nash-p`）：

```bash
# 输出：$WORK 下的 *.hbm — 成功判据：hb_compile 退出码 0
hb_compile --help
hb_compile --config "$OE_CONFIG"
find "$WORK" -type f -name '*.hbm' -print
```

ResNet152——备好 `./resnet152.onnx` 与 `./calibration_data_rgb/`
（cwd：本 conversion 目录，OE 容器内）：

```bash
# 输入：./resnet152.onnx、./calibration_data_rgb（见校准一节）
# 输出：./model_output/resnet152_224x224_nv12.hbm — 成功判据：hb_compile 退出码 0
hb_compile --config resnet152_config.yaml
```

`resnet152_config.yaml` 默认 `march: nash-e`（S100）。为 S600 重编译时
把 `march` 改为 `nash-p`，其余字段不动。预期输出前缀
`resnet152_224x224_nv12`（与 Manifest 文件名
`resnet152_224x224_nv12.hbm` 一致）。

ResNet50——经 OE `13_resnet50` 示例编译；本目录未提供配置。只执行与
目标制品对应的块。若 OE 示例的命令拼写不同，按原样保留该命令与文件
并记录。

<a id="validation"></a>
## 转换后验证

用 OE 示例的 `hb_perf` 与 `hrt_model_exec` 对生成制品执行并完整记录
输出。ResNet152 可用 `x86_inference.py`（OE 容器内执行）在 x86 上以
相同的 padded-crop 预处理对照 ONNX / HBIR（`.bc`）/ HBM 输出。

再在匹配板卡上用 canonical 运行时确认：X5 元数据为单一 packed NV12
输入加名为 `prob` 的 F32 `[1,1000,1,1]` 输出；S100/S600 为 Y
`[1,224,224,1]`、UV `[1,112,112,2]` 与 F32 `[1,1000]` 输出；同图、
同缩放、同标签、同 Top-K 得到预期的类别 ID 与分数顺序。

源分支对 ResNet152 公开的转换记录（仅作背景——本仓库未复测）：

| 项目 | 数值（rdk_s @380e1a2 记录） |
| --- | --- |
| 运行时输入 / 训练输入 | NV12 / RGB（NCHW） |
| Mean / Scale | `123.675 116.28 103.53` / `0.01712475 0.017507 0.01742919` |
| March | `nash-e` |
| 校准相似度 | `0.994397` |
| 量化相似度 | `0.992285` |
| 工具链 FPS / 延迟 | `449.03` / `2.23 ms` |

状态：ResNet18 主机导出冒烟已完成；任何变体的真实 OE 编译与再生成
制品板端复验均为 **not-run**——本 sample 未重新构建已发布制品。

<a id="artifacts"></a>
## 产物

| 阶段 | 需保留的产物 |
| --- | --- |
| export | `resnet18.onnx`、导出参数、Torch/TorchVision/ONNX 版本；ResNet152：下载的 `resnet152.onnx` 及来源 URL |
| calibration | ResNet18：OE 示例的校准目录/清单与预处理记录；ResNet152：`calibration_data_rgb/` 与确切源图清单 |
| checker | checker 命令、目标配置、日志 |
| compile | 目标配置、OE 版本、生成输出目录 |
| deployment | X5 `resnet18_224x224_nv12.bin` 或 S `resnet{18,50,152}_224x224_nv12.hbm`（Manifest 文件名） |
| validation | `hb_perf`/`hrt_model_exec`/`x86_inference.py` 输出、raw 输出、延迟、精度输入 |

已发布文件名即本 sample 使用的 Manifest 行。导出器的 ONNX 输出名为
`output`；已发布 X5 运行时元数据的输出名为 `prob`、形状
`[1,1000,1,1]`，S 制品为 `output`、`[1,1000]`——这些名称与形状属于
目标制品契约。

<a id="known-gaps"></a>
## 缺失项

- ResNet18：已发布制品的校准集、YAML 或完整重放脚本未入库；这些步骤
  由 OE `13_resnet18` 示例承担。
- ResNet50：审计源未发布 ONNX URL、YAML、校准数据或脚本——只有指向
  OE `13_resnet50` 示例的 README 指引。重建该制品只能从该示例出发；
  本目录记录指引而非虚构配方。
- ResNet152：校准图片由用户自备（脚本默认源目录属于旧分支目录树，
  在本仓不存在）；脚本统一 `0.017` scale 与 YAML 逐通道 `scale_value`
  的差异按源分支原样保留——本仓库未做 OE 重建或数值对照来裁定，
  也不认定哪一组系数正确；S600（`nash-p`）构建与公开的 FPS/延迟记录
  未在本仓库重跑。
- 每个已发布制品实际使用的权重与 OE 配置未记录；同名的再生成文件在
  比较 target、输入元数据、输出 shape/dtype 与数值结果之前不等价。
- 本 sample 未执行真实 OE 编译（**not-run**）；仅记录了 ResNet18 的
  主机导出冒烟。
