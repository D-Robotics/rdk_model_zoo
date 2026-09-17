# ResNet18 图像分类

本 sample 可在 RDK X5、S100 或 S600 上运行 ResNet18 ImageNet 分类模型。
维护代码只有一条 Python 流程和一条 S 系列 C++ 流程。原 X5 和 S18 的
Python 入口仍然可用，但已经变为兼容 wrapper，并调用同一份 canonical
实现。

[English README](README.md) · [Python 运行时](runtime/python/README_cn.md) ·
[模型准备](model/README_cn.md) · [模型转换](conversion/README_cn.md) ·
[评估](evaluator/README_cn.md)

## 选择板卡和制品

选择与实际运行板卡对应的一行。下面的限定引用直接对应平台发布
Manifest 中已有的行；指定自定义模型路径时也必须提供该引用。

| 板卡 | 引用 | 下载器生成的本地制品 | 传给运行时的输入 |
| --- | --- | --- | --- |
| X5 | `x5:resnet:resnet18_224x224_nv12.bin` | `model/resnet18_224x224_nv12.bin` | 一个 packed NV12 张量 |
| S100 | `s:resnet18:s100/resnet18_224x224_nv12.hbm` | `model/s100/resnet18_224x224_nv12.hbm` | 独立 Y 和 UV 张量 |
| S600 | `s:resnet18:s600/resnet18_224x224_nv12.hbm` | `model/s600/resnet18_224x224_nv12.hbm` | 独立 Y 和 UV 张量 |

Manifest 是 URL、格式以及可选发布者 hash 的权威来源。Manifest 没有
ResNet18 的 S100P 制品行，因此选择逻辑会拒绝 S100P，不猜测兼容性。
ResNet50 和 ResNet152 仍是旧资料，不属于本可运行的 ResNet18 sample。

## 准备环境

板端 Python 运行需要镜像中与板卡匹配的 `hbm_runtime`、NumPy 和
OpenCV-Python。只有真正执行模型时才导入板端 SDK；`--help`、列出制品、
dry-run 和主机测试不需要 SDK。不要通过安装另一套 SDK 来绕过板卡不匹配。

Manifest 读取还需要 PyYAML。开发主机可使用独立环境安装
`requirements-host.txt` 中的用户态依赖：

```bash
python3 -m venv .venv-resnet
source .venv-resnet/bin/activate
python3 -m pip install -r samples/vision/resnet/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

板端预装的 `hbm_runtime` 由 RDK 镜像提供，不包含在主机依赖文件中。

C++ 示例是从 S18 源码合并到本 sample 的实现。板端编译需要 CMake、C++17
编译器、OpenCV 开发头文件/库、`gflags`，以及已安装的 Horizon DNN
头文件/库（`hbDNN`、`hbucp` 和 `fmt`）。构建会读取
`/sys/class/boardinfo/soc_name`，面向 S100 或 S600 板卡。启动脚本不会执行
`apt`、`pip` 或下载模型。

模型转换在 x86 Linux 的 OpenExplore 环境完成，不在板端完成。请看
[conversion/README_cn.md](conversion/README_cn.md) 中的源模型、明确的 ONNX
导出命令、OE 示例路径和需要记录的制品。

## 显式准备模型

在仓库根目录执行下面的命令，将一个发布制品下载到 canonical model 目录：

```bash
bash samples/vision/resnet/model/download.sh x5
bash samples/vision/resnet/model/download.sh s100
bash samples/vision/resnet/model/download.sh s600
```

也可以执行 `python3 samples/vision/resnet/model/download.py --target s100`。
下载器从 `platforms/x5/docs/release/models.yaml` 或
`platforms/s/docs/release/models.yaml` 读取对应行，先写临时文件，在存在
发布者 hash 时校验，然后原子完成写入。当前行没有发布者 SHA-256，因此
会提示观察到的 digest 不能证明来源。下载是显式操作，推理过程不会触发下载。

如果制品放在其他位置，必须同时传入上表中的完整引用和文件路径。单凭
文件名不能决定 packed 或 split NV12 协议。

## 运行 Python sample

先用 `--help` 查看参数。模型已在对应板卡上准备好后，以下是完整命令：

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file platforms/x5/datasets/imagenet/imagenet_classes.names

python3 samples/vision/resnet/runtime/python/main.py \
  --target s100 \
  --asset-id s:resnet18:s100/resnet18_224x224_nv12.hbm \
  --model-path samples/vision/resnet/model/s100/resnet18_224x224_nv12.hbm \
  --test-img samples/vision/resnet/test_data/zebra_cls.jpg \
  --label-file platforms/s/datasets/imagenet/imagenet_classes.names

python3 samples/vision/resnet/runtime/python/main.py \
  --target s600 \
  --asset-id s:resnet18:s600/resnet18_224x224_nv12.hbm \
  --model-path samples/vision/resnet/model/s600/resnet18_224x224_nv12.hbm \
  --test-img samples/vision/resnet/test_data/zebra_cls.jpg \
  --label-file platforms/s/datasets/imagenet/imagenet_classes.names
```

对于板端其他路径，只有在板卡身份可读时才能使用 `--target auto`。
`--list-models` 和 `--dry-run` 可在运行前检查选择：

```bash
python3 samples/vision/resnet/runtime/python/main.py --list-models --target auto
python3 samples/vision/resnet/runtime/python/main.py --dry-run --target x5
python3 samples/vision/resnet/runtime/python/main.py --dry-run \
  --asset-id s:resnet18:s600/resnet18_224x224_nv12.hbm
```

命令输出稳定的 Top-K 类别 ID、分数和标签。`--top-k` 与旧拼写 `--topk`
默认值都是 5。`--resize-type 0` 将图片拉伸为 224x224；`1` 保持宽高比并
使用 BGR 127 填充，是旧源码默认值。X5 直接缩放使用 linear，S100/S600
保留旧源码的 nearest 直接缩放。

## 运行合并后的 C++ sample

在准备好 S100 或 S600 制品及 C++ 板端依赖后执行：

```bash
bash samples/vision/resnet/runtime/cpp/run.sh
```

默认启动器使用 `model/s100` 下的制品。S600 要显式选择制品；同一个检出
目录在两块板上使用时，建议使用独立 build 目录：

```bash
MODEL_PATH="$PWD/samples/vision/resnet/model/s600/resnet18_224x224_nv12.hbm" \
BUILD_DIR="$PWD/samples/vision/resnet/runtime/cpp/build-s600" \
bash samples/vision/resnet/runtime/cpp/run.sh
```

启动器调用 CMake，构建 `resnet18`，并把模型、sample 图片和 S 系列
ImageNet 标签文件传给二进制程序。它会先检查这些文件，不修改系统。
C++ 代码使用 `platforms/s/utils/c_utils` 中已有的工具源；数值预处理、
BPU 调用和 Top-K 后处理都作为源代码保留，而不是只放链接或再建一套平台
目录。审计基线没有 X5 对应的 C++ 源码。

## 库流程和输出契约

canonical Python 流程如下：

```text
resolve_selection -> require_execution_target -> RuntimeModelRunner.load
  -> bind_model -> ClassificationTask.predict
       -> tensor_io.prepare_nv12 -> runtime call
       -> classification.topk_from_scores
```

`ClassificationTask` 接收 BGR `uint8` 数组，返回
`ClassificationResult(class_ids, scores, labels)`。`RuntimeModelRunner` 是
唯一 SDK 边界，会在返回前校验运行时名称、形状、类型和单一 F32 分数输出。
X5 接收 packed `(1,336,224,1)` uint8 NV12；S100/S600 接收 Y
`(1,224,224,1)` 和 UV `(1,112,112,2)` uint8 数组。

审计的两个 Python wrapper 都对返回的 score 向量调用 softmax。canonical
契约把它记录为 `unverified_score_vector` 上的 `legacy_softmax`：X5 张量
名为 `prob` 且看起来已经归一化，但现有转换资料不能证明归一化是在图内还是
wrapper 中完成。因此本 sample 保留源代码行为，不宣称新的输出语义。

兼容导入路径仍然有效：

```python
from platforms.x5.samples.vision.resnet.runtime.python.resnet import ResNet, ResNetConfig
from platforms.s.samples.vision.resnet18.runtime.python.resnet18 import Resnet18, Resnet18Config
```

`ResNet.pre_process`/`forward`/`post_process` 保留 X5 的嵌套输入和
`(topk_idx, topk_prob, topk_labels)` 元组；`Resnet18` 保留嵌套 split 输入和
`(class_id, probability)` 列表。两个类都实例化 canonical binding，并委托
预处理和解码；可选的 `runtime=`/`runtime_factory=` 只用于主机 fixture 测试。

## 源码映射和限制

| 审计源码 | Canonical 符号或文件 | 保留行为 |
| --- | --- | --- |
| X5 `ResNet.pre_process` | `ClassificationTask.pre_process` → `tensor_io.prepare_nv12` | packed NV12、linear 直接缩放 |
| S18 `Resnet18.pre_process` | `ClassificationTask.pre_process` → `tensor_io.prepare_nv12` | Y/UV 分离、nearest 直接缩放 |
| X5/S18 `forward` | `RuntimeModelRunner.__call__` | 兼容层保留嵌套，canonical runner 使用扁平张量 |
| X5/S18 `post_process` | `classification.topk_from_scores` | legacy softmax 和稳定降序 Top-K |
| X5/S18 Python `main.py` | canonical `runtime/python/main.py` | 旧路径提供默认值和完整 Manifest ID |
| S18 `runtime/cpp/{inc,src}` | canonical `runtime/cpp/{inc,src}` | 保留 hbDNN、图像工具和输出流程 |
| X5/S18 model 脚本 | canonical `model/download.py` | Manifest URL/格式/hash 和原子写入 |

旧 ResNet50/152 的转换和运行时文件不会被当作 ResNet18 实现。审计的
ResNet18 源码指向官方 OE 分类示例，但没有随 sample 提供 YAML、校准集或
可重放的板端转换配方。因此 canonical exporter 只生成固定形状 ONNX 源图；
在声称与已发布制品等价前，仍需记录每个制品实际使用的权重和 OE 配置。
限制和阶段说明见[转换说明](conversion/README_cn.md)。

## 故障排查

| 现象 | 检查 |
| --- | --- |
| `Cannot identify this board` | dry-run 可使用显式目标；真实执行只能在对应板卡上进行，显式目标本身不是硬件证据。 |
| `model_path requires --asset-id` | 从 `--list-models` 复制完整限定引用，不要只传文件名。 |
| S100P 没有发布制品 | Manifest 没有 ResNet18 S100P 行，只能使用与板卡匹配的 S100/S600 制品。 |
| 输入形状或类型不匹配 | 确认制品引用和运行时元数据，不要交叉使用 X5 packed 与 S split 制品。 |
| 下载出现 hash 警告 | Manifest 没有发布者 SHA-256；在本地证据中保留观察 digest，并另行核实来源。 |
| C++ 配置或编译失败 | 检查 OpenCV/gflags/Horizon DNN 开发包、C++17 和板端 SoC 文件；启动器不会安装依赖。 |
| 输出与旧运行不同 | 先比较相同制品、图片、缩放模式、Top-K 和 raw output，再改变 score 语义。 |

在仓库根目录运行主机检查：

```bash
python3 -m unittest discover -s samples/vision/resnet/tests -v
```

主机检查覆盖选择、元数据拒绝、图像几何、packed/split 张量布局、兼容
wrapper API 和 Top-K 数值解码。板端 Python 与原生 C++ 结果应分别记录板卡
身份、制品引用、命令及 raw output。
