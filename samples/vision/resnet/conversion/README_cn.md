# ResNet18 模型转换

模型转换在 x86 Linux 的 RDK OpenExplore（OE）环境完成，不是板端运行时操作。
审计的 X5 和 S18 目录都把 TorchVision ResNet18 作为源模型，并指向 OE 分类
示例，但 ResNet18 目录没有原始 YAML、校准集或完整重放脚本。本目录补齐
源 ONNX 导出步骤，并记录 OE 负责的后续步骤，不凭空添加可能生成不同制品
的配置。

## 1. 准备转换主机

使用与目标板卡匹配的 OE Docker/工具链版本，并记录镜像标签、OE 版本和主机
日期。S18 源文档指向的 OE 示例路径为：

```text
samples/ai_toolchain/horizon_model_convert_sample/03_classification/13_resnet18
```

将本检出目录挂载到该环境。Docker 镜像名称和文件名随版本变化，应从 OE
发布文档获取，不要把过期名称写进 sample。模型转换不需要 `hbm_runtime`，但
下面的 OE 命令必须在 `PATH` 中。

导出器需要 PyTorch、TorchVision 和 ONNX。ONNX Runtime 可用于可选的主机图
冒烟检查，但不能替代 OE checker 或板端运行。导出前先检查版本，不要让缺少
依赖的情况触发隐式安装：

```bash
python3 - <<'PY'
import importlib

for name in ("torch", "torchvision", "onnx"):
    module = importlib.import_module(name)
    print(f"{name}={getattr(module, '__version__', 'unknown')}")
try:
    import onnxruntime
    print(f"onnxruntime={onnxruntime.__version__}")
except ImportError:
    print("onnxruntime=not-installed (optional for the host smoke test)")
PY
```

主机没有这些包时，请按 PyTorch 和 ONNX 官方说明在用户自己的虚拟环境中
安装兼容版本。仓库侧的小型辅助依赖列在
[`requirements-host.txt`](../requirements-host.txt)；编译器和运行库由 OE
镜像提供。

权威参考是 [RDK S 工具链总览](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)、
[D-Robotics 工具链下载页](https://toolchain.d-robotics.cc/) 和
[TorchVision ResNet18 页面](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)。
请从这些来源选择对应版本的工具包和镜像。通用的加载与挂载流程如下：

```bash
# 将 OE_IMAGE_TAR 设置为 OE 版本提供的镜像压缩包。
export OE_IMAGE_TAR=/absolute/path/to/oe-image.tar
test -s "$OE_IMAGE_TAR"
docker load -i "$OE_IMAGE_TAR"
docker images

# 将 OE_IMAGE 设置为 docker images 输出的准确 image:tag。
: "${OE_IMAGE:?Set OE_IMAGE to the loaded OE image:tag}"
export REPOSITORY_ROOT="${REPOSITORY_ROOT:-$PWD}"
docker run --rm -it --network host --shm-size=15g \
  -v "$REPOSITORY_ROOT":/workspace --workdir /workspace \
  "$OE_IMAGE" /bin/bash
```

进入容器后在 `/workspace` 中继续执行命令。容器镜像、`OE_CONFIG` 和目标
板卡必须属于同一版本系列。

## 2. 导出源 ONNX 图

统一导出器使用固定 NCHW 输入 `[1,3,224,224]`、输出名 `output` 和 1000 个
ImageNet 类别，并使用 TorchScript ONNX 导出器（`dynamo=False`），便于 OE
流程保持稳定：

```bash
export WORK="${WORK:-/workspace/resnet18-conversion}"
mkdir -p "$WORK"
python3 samples/vision/resnet/conversion/export_resnet18_onnx.py \
  --output "$WORK/resnet18.onnx" \
  --opset 11 \
  --weights IMAGENET1K_V1
```

`IMAGENET1K_V1` 是明确的权重选择；本地没有缓存时可能下载官方 TorchVision
权重。`--weights none` 适合离线检查图结构，但会生成随机权重，不能用于精度
结论，也不能据此证明与 RDK 已发布制品等价：

```bash
export WORK="${WORK:-/workspace/resnet18-conversion}"
mkdir -p "$WORK"
python3 samples/vision/resnet/conversion/export_resnet18_onnx.py \
  --output "$WORK/resnet18-random.onnx" \
  --opset 11 \
  --weights none
```

默认使用 `onnx.checker` 检查图。只有已由其他记录工具校验时才使用
`--no-check`。导出 smoke test 已在 Torch 2.7.1、TorchVision 0.22.1、ONNX
1.19 和 ONNX Runtime 1.23.2 上运行，得到 `[1,1000]` 输出，并与同 seed 的
PyTorch 图在 `1e-4` 容差内一致。这只是 ONNX 结构检查，不是 BPU 编译或精度
结果。

## 3. 获取 OE 转换输入

将匹配版本 `13_resnet18` 示例中的文件复制到转换工作目录，并按该示例的
预处理和校准流程执行。审计的 RDK sample 没有发布校准图片列表、mean/scale
数值或 ResNet18 YAML，这些值必须来自生成待重建制品的 OE 版本。应记录实际
校准数据和文件，不能把旧 ResNet152 目录中的配置套用给 ResNet18。

图侧输入是 RGB/NCHW，部署运行时输入是 224x224 NV12：X5 需要一个 packed
buffer，S100/S600 需要独立 Y 和 UV 输入。OE 配置必须完成目标所需的图转换；
Python/C++ 运行时代码只做图片缩放和 NV12 布局转换，不额外应用未记录的归一化。

## 4. 执行 OE 检查和编译

使用匹配 OE 示例提供的命令。审计的 X5 说明列出了
`hb_mapper checker`、`hb_mapper makertbin`，S18 说明列出了
`hb_compile`。仓库没有历史 YAML，因此要把 `OE_CONFIG` 设置为匹配 OE
`13_resnet18` 示例实际提供的配置文件。先用下面的检查明确所需输入：

```bash
export WORK="${WORK:-/workspace/resnet18-conversion}"
export ONNX="${ONNX:-$WORK/resnet18.onnx}"
: "${OE_CONFIG:?Set OE_CONFIG to the YAML from your OE 13_resnet18 sample}"
test -s "$ONNX"
test -s "$OE_CONFIG"

# 很多 OE YAML 会以当前目录解析相对路径。请保留 OE 示例要求的工作目录，
# 或把复制后的 YAML 路径改为绝对路径。执行下方目标代码块前，在复制的
# YAML 中把 `working_dir`/输出目录设为 "$WORK"，并把 `onnx_model` 和校准
# 路径指向上面的实际文件。
```

X5 版本先检查 mapper，再用同一配置编译。审计到的 X5 示例使用
`bayes-e`；运行前请根据所选 OE 版本确认 march：

```bash
export X5_MARCH="${X5_MARCH:-bayes-e}"
hb_mapper --version
hb_mapper checker --model-type onnx --march "$X5_MARCH" --model "$ONNX"
hb_mapper makertbin --model-type onnx --config "$OE_CONFIG"
find "$WORK" -type f -name '*.bin' -print
```

S100 或 S600 版本要求所选 YAML 包含对应 Nash 目标（S100 为 `nash-e`，S600
为 `nash-p`）。把目标选择写入配置和证据；不要从制品文件名或路径推断：

```bash
hb_compile --help
hb_compile --config "$OE_CONFIG"
find "$WORK" -type f -name '*.hbm' -print
```

只执行当前重建制品对应的目标代码块。如果 OE 示例使用不同的命令拼写，
或把模型/校准路径放在其他文件中，请在证据中保留原始完整命令和文件，
不要猜测替代配置。`find` 只用于定位输出；检查元数据后再把选定文件复制到
Manifest 目标路径。

请保留以下阶段和制品：

| 阶段 | 应保留的制品 |
| --- | --- |
| 导出 | `resnet18.onnx`、导出参数、Torch/TorchVision/ONNX 版本 |
| 校准 | OE 示例的校准目录/列表和预处理记录 |
| 检查 | checker 命令、目标配置和日志 |
| 编译 | 目标配置、OE 版本和生成目录 |
| 部署 | X5 `resnet18_224x224_nv12.bin` 或 S `resnet18_224x224_nv12.hbm` |
| 验证 | `hb_perf`/`hrt_model_exec` 命令、raw output、延迟和精度输入 |

已发布文件名对应本 sample 使用的 Manifest 行。文件名相同并不能证明等价，
必须比较目标、输入元数据、输出形状/类型和数值结果。

导出器将 ONNX 图输出命名为 `output`。已发布 X5 运行时元数据将输出命名为
`prob`，形状为 `[1,1000,1,1]`；S 制品通常使用 `output`，形状为 `[1,1000]`。
这些名称和形状属于目标制品契约。ONNX 导出或编译命令成功，并不能证明生成
制品会被运行时接受：应检查编译后元数据，在统一运行时绑定准确输出，并在
发布前完成对应板卡检查。

## 5. 发布前验证

使用 OE 示例中的 `hb_perf` 和 `hrt_model_exec` 命令运行生成制品，并保存完整
输出。然后在匹配板卡上用统一运行时契约确认：

* X5 暴露一个 packed NV12 输入及 F32 `[1,1000,1,1]` 输出。
* S100/S600 暴露 Y `[1,224,224,1]`、UV `[1,112,112,2]` 及 F32 `[1,1000]`
  输出。
* 使用相同图片、缩放方式、标签文件和 Top-K 时，类别 ID 及排序符合预期。

旧 X5 评估记录的历史参考为 float 模型 Top-1 71.5%、量化模型 Top-1 70.5%，
延迟 2.95 ms、FPS 449+。这些是已发布旧评估结果；源文档没有说明延迟或 FPS
是否采用单次调用、批处理或多线程，因此不要由一个数推导另一个数，也不要把
它们写成重新生成模型的新结果。
S50/S152 转换资料不是 ResNet18 配方，仍不属于本 sample。

## 已知限制

导出器是可重放的源图辅助工具，不能替代 OE 示例。当前源码没有证明已发布
`.bin`/`.hbm` 使用的确切权重、校准集、YAML、目标配置或输出语义归属。只有
记录这些文件和板测结果后，才能把重建模型标为等价或可用于生产。模型 URL
和可选 hash 继续保存在平台 Manifest；转换文档只引用制品，不创建第二份 URL
注册表。
