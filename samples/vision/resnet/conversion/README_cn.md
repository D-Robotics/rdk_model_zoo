# ResNet 模型转换（ResNet18 / ResNet50 / ResNet152）

转换在 x86 Linux 主机的 RDK OpenExplore（OE）环境完成，不是板端操作。本目录
覆盖该 sample 发布的三个变体，各变体的转换流程不同：

| 变体 | 本目录内容 | 转换流程 |
| --- | --- | --- |
| ResNet18（x5+s） | `export_resnet18_onnx.py` | 先在主机运行本目录提供的 ONNX 导出器，校准数据与 YAML 取自 OE `13_resnet18` 示例 |
| ResNet50（仅 s） | — | 经 OE `13_resnet50` 示例的配方转换 |
| ResNet152（仅 s） | `resnet152_config.yaml`、`get_calibration_data.py`、`x86_inference.py` | 已发布 ONNX + 用户自备校准图，配合上述三个文件在 OE 内完成转换 |

<a id="source-model"></a>
## 源模型

三个变体均为 TorchVision ResNet 系列 ImageNet-1k 分类器：
[ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)、
[ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)、
[ResNet152](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html)。

- ResNet18：本地导出，固定 NCHW 输入 `[1,3,224,224]`、输出名 `output`、
  ImageNet-1k 类别数。导出器使用 TorchScript ONNX 导出
  （`dynamo=False`）以保持图稳定。`--weights IMAGENET1K_V1` 选择官方
  预训练权重；`--weights none` 生成随机权重图，仅用于离线结构检查。
- ResNet50：经 OE `13_resnet50` 分类示例转换——ONNX、校准和编译步骤从该示例
  开始。
- ResNet152：有已发布 ONNX——在 OE 容器内（或任何可访问该链接的 x86
  主机）：

```bash
# cwd：本 conversion 目录 — 输出：./resnet152.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet152.onnx
```

<a id="directory"></a>
## 目录结构

```text
conversion/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── export_resnet18_onnx.py  # Python 脚本
├── get_calibration_data.py  # Python 脚本
├── resnet152_config.yaml  # 配置
└── x86_inference.py  # Python 脚本
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
S100 用 `hb_compile`（`nash-e`），S600 用 `nash-p`。涉及的 OE
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
过的工具校验时才用 `--no-check` 跳过 `onnx.checker`。在 Torch 2.7.1、
TorchVision 0.22.1、ONNX 1.19、ONNX Runtime 1.23.2 下的主机导出检查
输出 `[1,1000]`，与同种子 PyTorch 图在 `1e-4` 内一致。

ResNet50——导出在 OE `13_resnet50` 示例内完成。ResNet152——用[源模型](#source-model)一节的已发布 ONNX；没有要运行的导出器。

<a id="calibration"></a>
## 校准

ResNet18——本目录提供导出器；校准数据与 YAML 取自 OE `13_resnet18`
示例。从 OE 示例复制对应的 `13_resnet18` 文件并按其文档执行预处理与
校准，同时记录确切校准数据。本目录的 ResNet152 配置是 ResNet152 的
配方——不是 ResNet18 的配方，不得混用。

ResNet152——`get_calibration_data.py`（在 OE 容器内、cwd 为本
conversion 目录）把 100 张 ImageNet 验证集图片转成 float32 RGB 校准
数据。两个输入需要用户自备并在脚本中先行修改；脚本默认路径指向
旧目录树，需要替换：

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

脚本与 YAML 的归一化常量**并不一致**：两者 mean 相同
（`123.675 116.28 103.53`），但脚本统一乘 `0.017`，YAML 声明逐通道
`scale_value: 0.01712475 0.017507 0.01742919`。再生成时让脚本与 YAML
使用同一组 scale；YAML 的逐通道值与发布记录的
Mean/Scale 行一致（见[转换后验证](#validation)）。

ResNet50——校准在 OE `13_resnet50` 示例内完成；本目录无对应步骤。

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
目标制品对应的块。若 OE 示例的命令拼写不同，按其确切命令执行并
随结果记录。

<a id="validation"></a>
## 转换后验证

用 OE 示例的 `hb_perf` 与 `hrt_model_exec` 对生成制品执行并完整记录
输出。ResNet152 可用 `x86_inference.py`（OE 容器内执行）在 x86 上以
相同的 padded-crop 预处理对照 ONNX / HBIR（`.bc`）/ HBM 输出。

再在匹配板卡上用 运行时确认：X5 元数据为单一 packed NV12
输入加名为 `prob` 的 F32 `[1,1000,1,1]` 输出；S100/S600 为 Y
`[1,224,224,1]`、UV `[1,112,112,2]` 与 F32 `[1,1000]` 输出；同图、
同缩放、同标签、同 Top-K 得到预期的类别 ID 与分数顺序。

源发布对 ResNet152 公开的转换记录：

| 项目 | 数值（记录） |
| --- | --- |
| 运行时输入 / 训练输入 | NV12 / RGB（NCHW） |
| Mean / Scale | `123.675 116.28 103.53` / `0.01712475 0.017507 0.01742919` |
| March | `nash-e` |
| 校准相似度 | `0.994397` |
| 量化相似度 | `0.992285` |
| 工具链 FPS / 延迟 | `449.03` / `2.23 ms` |

再生成制品需要执行 OE 编译，并按[验证](#validation)一节在板端复验。

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

已发布的 S 系列制品也可从模型服务器直接下载（S100 与 S600 文件名相同，
仅归档子目录不同；规范准备路径仍是 [模型下载器](../model/README_cn.md#preparation)）：

```bash
# ResNet18
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet18_224x224_nv12.hbm
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ResNet/resnet18_224x224_nv12.hbm
# ResNet50
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet50_224x224_nv12.hbm
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ResNet/resnet50_224x224_nv12.hbm
# ResNet152
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet152_224x224_nv12.hbm
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ResNet/resnet152_224x224_nv12.hbm
```

<a id="known-gaps"></a>
## 额外准备

- ResNet18：校准集、YAML 与预处理记录从 OE `13_resnet18` 示例获取。
- ResNet50：ONNX、校准数据与 YAML 从 OE `13_resnet50` 示例获取，并在该
  示例内完成转换。
- ResNet152：校准图片由用户自备——把 `get_calibration_data.py` 的
  `src_image_dir` 改为你的 ILSVRC2012 验证集目录；脚本统一 `0.017`
  scale 与 YAML 逐通道 `scale_value` 不同，再生成时让脚本与 YAML 使用
  同一组一致的 scale。
- 每个已发布制品实际使用的权重与 OE 配置未记录；同名的再生成文件在
  比较 target、输入元数据、输出 shape/dtype 与数值结果之前不等价。

<a id="self-trained"></a>
## 自训练 TorchVision ResNet18 checkpoint

默认导出脚本面向官方 `IMAGENET1K_V1` 权重；自训练 `state_dict` 走同一固定
合同导出：

```bash
# conversion 环境（已装 PyTorch + TorchVision）；cwd：本目录
python3 export_resnet18_onnx.py \
  --checkpoint /path/to/my_resnet18.pth \
  --num-classes 4 \
  --output my_resnet18_4class.onnx
```

规则与保证：

- CLI 约定：`--checkpoint` 与 `--weights` 互斥（argparse 直接拒绝组合）；
  `--num-classes` 仅随 `--checkpoint` 使用，缺失或单独给出都明确报错；
  两个权重选项都不给时保持官方 `IMAGENET1K_V1` 默认（1000 类输出）不变。
- 结构：仅支持 TorchVision `resnet18`。分类头重建为
  `nn.Linear(512, num_classes)`，checkpoint 必须以 `strict=True` 加载——
  来自其他 ResNet 变体或类别数不符的 checkpoint 会直接失败，而不是导出
  半随机的图。不承诺任意 ResNet 结构或自定义输出头兼容。
- `--num-classes`（>= 2）决定 ONNX `output` 宽度；导出会打印所用
  torch/torchvision 版本，请随训练记录一并留存。
- 图合同与官方导出完全一致：NCHW 输入 `data` `[1, 3, 224, 224]`
  float32、输出 `output` `[1, num_classes]`、静态形状、TorchScript
  导出器（`dynamo=False`）。
- 前处理：训练变换必须与编译所用的 OE 配置兼容（与官方制品相同：
  `--resize-type` 对应的 letterbox/直接缩放、OE 配置内置的归一化）；
  导出脚本不改动前处理。
- 产物用 [工具链与目标](#toolchain-targets) 下的既有 OE 配置编译（X5
  `hb_mapper` → `.bin`，S100/S600 `hb_compile` → `.hbm`）；不暗示任何
  已发布制品配方适用于自训练输出。
- 运行时合同：用 `classify.ResNetClassifier(model_path, target=target,
  input_size=(224, 224), class_count=<num-classes>)` 加载
  编译产物，标签可选（见 [自定义模型
  章节](../runtime/python/README_cn.md#custom-model)）；绑定会把实际输出
  宽度与 `class_count` 核对。
