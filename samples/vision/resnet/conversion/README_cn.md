# ResNet18 模型转换

转换在 x86 Linux 主机的 RDK OpenExplore（OE）环境完成，不是板端操作。审计
的 X5 与 S18 源目录都指向 TorchVision ResNet18 和 OE 分类示例，但都没有
随附原始 YAML、校准集或完整重放脚本。本目录提供源图导出步骤，并如实记录
OE 负责的步骤，不虚构可能生成不同制品的配置。

<a id="source-model"></a>
## 源模型

TorchVision ResNet18（[上游](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)），
以固定 NCHW 输入 `[1,3,224,224]`、输出名 `output`、ImageNet-1k 类别数导出。
canonical 导出器使用 TorchScript ONNX 导出（`dynamo=False`）以保持图稳定。
`--weights IMAGENET1K_V1` 选择官方预训练权重（未缓存时从 TorchVision 权重库
下载）；`--weights none` 生成随机权重图，仅用于离线结构检查。

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
S100 用 `hb_compile`（`nash-e`），S600 用 `nash-p`。审计源引用的 OE 分类
示例位于
`samples/ai_toolchain/horizon_model_convert_sample/03_classification/13_resnet18`。

<a id="export"></a>
## 导出

导出前置：PyTorch、TorchVision、ONNX（可选 ONNX Runtime 做主机图冒烟）。
在 OE 容器内 `/workspace` 执行：

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

离线冒烟可用 `--weights none`（随机权重，无精度含义）。仅当由其他记录过的
工具校验时才用 `--no-check` 跳过 `onnx.checker`。导出冒烟已在 Torch 2.7.1、
TorchVision 0.22.1、ONNX 1.19、ONNX Runtime 1.23.2 下执行：输出 `[1,1000]`，
与同种子 PyTorch 图在 `1e-4` 内一致。这只是 ONNX 结构检查，不是 BPU 编译
或精度结果。

<a id="calibration"></a>
## 校准

本仓库不可重放（已知缺口）：审计源未发布校准图片列表、mean/scale 值或
ResNet18 YAML。从 OE 示例复制对应的 `13_resnet18` 文件并按其文档执行
预处理与校准，同时记录确切校准数据。其他旧目录下的 ResNet152 配置不是
ResNet18 的配方。校准输入未记录前，不得宣称再生成制品等价。

<a id="compile"></a>
## 编译

在 OE 容器内，备好 `$WORK/resnet18.onnx` 与 `$OE_CONFIG`（OE 示例的 YAML）
后执行。X5 块：

```bash
# 输出：$WORK 下的 *.bin — 成功判据：hb_mapper 退出码 0
export X5_MARCH="${X5_MARCH:-bayes-e}"
hb_mapper --version
hb_mapper checker --model-type onnx --march "$X5_MARCH" --model "$WORK/resnet18.onnx"
hb_mapper makertbin --model-type onnx --config "$OE_CONFIG"
find "$WORK" -type f -name '*.bin' -print
```

S100/S600 块（YAML 必须包含匹配的 Nash target——S100 为 `nash-e`，S600 为
`nash-p`）：

```bash
# 输出：$WORK 下的 *.hbm — 成功判据：hb_compile 退出码 0
hb_compile --help
hb_compile --config "$OE_CONFIG"
find "$WORK" -type f -name '*.hbm' -print
```

只执行与目标制品对应的块。若 OE 示例的命令拼写不同，按原样保留该命令与
文件并记录。

<a id="validation"></a>
## 转换后验证

用 OE 示例的 `hb_perf` 与 `hrt_model_exec` 对生成制品执行并完整记录输出；
再在匹配板卡上用 canonical 运行时确认：X5 元数据为单一 packed NV12 输入
加名为 `prob` 的 F32 `[1,1000,1,1]` 输出；S100/S600 为 Y `[1,224,224,1]`、
UV `[1,112,112,2]` 与 F32 `[1,1000]` 输出；同图、同缩放、同标签、同 Top-K
得到预期的类别 ID 与分数顺序。状态：主机导出冒烟已完成；真实 OE 编译与
再生成制品的板端复验为 **not-run**——本 sample 未重新构建已发布制品。

<a id="artifacts"></a>
## 产物

| 阶段 | 需保留的产物 |
| --- | --- |
| export | `resnet18.onnx`、导出参数、Torch/TorchVision/ONNX 版本 |
| calibration | OE 示例的校准目录/清单与预处理记录 |
| checker | checker 命令、目标配置、日志 |
| compile | 目标配置、OE 版本、生成输出目录 |
| deployment | X5 `resnet18_224x224_nv12.bin` 或 S `resnet18_224x224_nv12.hbm`（Manifest 文件名） |
| validation | `hb_perf`/`hrt_model_exec` 输出、raw 输出、延迟、精度输入 |

已发布文件名即本 sample 使用的 Manifest 行。导出器的 ONNX 输出名为
`output`；已发布 X5 运行时元数据的输出名为 `prob`、形状 `[1,1000,1,1]`，
S 制品为 `output`、`[1,1000]`——这些名称与形状属于目标制品契约。

<a id="known-gaps"></a>
## 缺失项

- 已发布制品的校准集、YAML 或完整重放脚本未入库；这些步骤由 OE
  `13_resnet18` 示例承担。
- 每个已发布制品实际使用的权重与 OE 配置未记录；同名的再生成文件在比较
  target、输入元数据、输出 shape/dtype 与数值结果之前不等价。
- 本导出器 ONNX 的真实 OE 编译未执行（**not-run**）；仅记录了主机导出
  冒烟。
