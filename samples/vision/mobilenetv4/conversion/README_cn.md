[English](README.md) | 简体中文

# MobileNetV4 模型转换

固定版本 timm checkpoint 流程使用 [`export.py`](export.py)，命令见[公共主机流程](../../../../utils/tools/mobilenet/README_cn.md)。该流程明确记录权重、中心裁剪预处理、batch=1 logits 合同和全量评测输入；下文既有制品命令按各自合同使用。

模型转换在 x86 Linux 主机上的 RDK OpenExplore (OE) 环境中执行，不是板卡
操作。使用下面的导出脚本、校准工具和目标板卡配置准备部署模型。

<a id="source-model"></a>
## 源模型

timm `mobilenetv4_conv_small` 与 `mobilenetv4_conv_medium` 预训练
权重，由 `get_mobilenetv4_onnx.py` 固定。脚本导出 small 为
`[1,3,224,224]`、medium 为 `[1,3,256,256]`——X5 medium 的几何差异
见[补充准备](#known-gaps)。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── MobileNetV4_medium.yaml  # 配置
├── MobileNetV4_small.yaml  # 配置
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── get_calibration_data.py  # Python 脚本
├── get_mobilenetv4_onnx.py  # Python 脚本
├── mobilenetv4_medium_config.yaml  # 配置
├── mobilenetv4_small_config.yaml  # 配置
├── timm2onnx_local.py  # Python 脚本
└── x86_medium_inference.py  # Python 脚本
```

<a id="toolchain-targets"></a>
## 工具链与目标

使用与目标板卡匹配的 OE Docker/工具链版本，并记录镜像 tag、OE 版本与
主机日期。权威入口：
[RDK S 工具链总览](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)、
[D-Robotics 工具链下载](https://toolchain.d-robotics.cc/)。
目标：X5 用 `hb_mapper` 编译，march `bayes-e`；S100 用 `hb_compile`，
march `nash-e`；S600 为 `nash-p`。容器挂载仓库到 `/workspace` 并给足共享
内存（`--shm-size=15g`）。

<a id="export"></a>
## 导出

在 OE 容器内（或任一装有 `torch`、`timm`、`onnx`、`onnxsim` 的主机），
cwd 为 `samples/vision/mobilenetv4/conversion`：

```bash
# 输入：timm 预训练权重（未缓存时下载）
# 输出：下列 .onnx 文件 — 成功判据：onnx 化简检查通过
python3 get_mobilenetv4_onnx.py    # -> mobilenetv4_conv_small.onnx + mobilenetv4_conv_medium.onnx
```

导出器使用 onnx-simplifier 并打印参数量（small 3,761,480 / medium
9,681,560）。

安装导出依赖：

```bash
pip install timm onnx onnxsim
```

### ONNX 导出示例

在 x86 导出环境中运行以下示例。需要 PyTorch；使用简化步骤时还需安装 `onnxsim`。导出文件名与所选 YAML 的 `onnx_model` 必须对应。

onnx 模型使用的是 timm 库 (PyTorch Image Models) 中的模型进行转换的，使用以下命令安装所需要的包：

```shell
pip install timm onnx
```

模型转换以 mobilenetv4_conv_small 为例：

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify
from timm.models import create_model

from timm.models.mobilenetv3 import mobilenetv4_conv_medium, mobilenetv4_conv_small

def count_parameters(onnx_model_path):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    # Get the initializers (weights in the model)
    initializer = model.graph.initializer

    # Calculate the total number of parameters
    total_params = 0
    for tensor in initializer:
        # Get the dimensions of each weight
        dims = tensor.dims
        # Calculate the number of parameters in this weight (product of all dimensions)
        params = 1
        for dim in dims:
            params *= dim
        total_params += params

    return total_params

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model('mobilenetv4_conv_small', pretrained=True)
    model.eval()

    # print the model structure

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "mobilenetv4_conv_small.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=11,
        verbose=True,
        input_names=["data"],  # Input name
        output_names=["output"],  # Output name
    )

    # Simplify the ONNX model
    model_simp, check = simplify(onnx_file_path)

    if check:
        print("Simplified model is valid.")
        simplified_onnx_file_path = "mobilenetv4_conv_small.onnx"
        onnx.save(model_simp, simplified_onnx_file_path)
        print(f"Simplified model saved to {simplified_onnx_file_path}")
    else:
        print("Simplified model is invalid!")

    onnx_model_path = simplified_onnx_file_path  # Replace with your ONNX model path
    total_params = count_parameters(onnx_model_path)
    print(f"Total number of parameters in the model: {total_params}")
```

<a id="calibration"></a>
## 校准

校准辅助脚本从 `src_image_dir` 读取 `ILSVRC2012_val_*.JPEG`。该参数默认值为
`../../../open_explorer/samples/ai_toolchain/horizon_model_convert_sample/01_common/calibration_data/imagenet/`；
运行前将 `src_image_dir` 改为本机 ImageNet 验证集目录。BGR 预处理依次执行 padded center
crop、resize、HWC→CHW、`RGB2BGRTransformer`、×255、mean
`103.94 116.78 123.68`、×0.017。两个注释开关选择图像尺寸。
选择并记录校准集使用的验证图像。

按各 YAML 准备对应输入：

| 配置（目标） | `cal_data_dir` | 布局与尺寸 | 准备步骤 |
| --- | --- | --- | --- |
| `mobilenetv4_small_config.yaml`（s100；s600 仅改 march） | `./calibration_data_bgr_224` | BGR，224 | 保持辅助脚本的 224 输出目录与 `data_transformer(224)`；设置本机图像目录，并在生成前将 mean 常量（`103.94 116.78 123.68`）与 S YAML 数值（`103.53 116.28 123.675`）对齐。 |
| `mobilenetv4_medium_config.yaml`（s100；s600 仅改 march） | `./calibration_data_bgr_256` | BGR，256 | 将两个注释开关切换到 `output_calib_dir = './calibration_data_bgr_256/'` 与 `active_transformers = data_transformer(256)`；设置图像目录，并将 mean 常量与 S YAML 数值对齐。 |
| `MobileNetV4_small.yaml`（x5） | `./calibration_data_rgb_f32` | RGB，224 | 准备 224x224 float32 RGB 校准数组，按 X5 YAML 的通道顺序和归一化数值处理。 |
| `MobileNetV4_medium.yaml`（x5） | `./calibration_data_rgb_f32` | RGB，224 | 准备 224x224 float32 RGB 校准数组，按 X5 YAML 的通道顺序和归一化数值处理。 |

<a id="compile"></a>
## 编译

构建配置：

| 配置 | 目标 | 配置引用的输入 | 命令（OE 容器内） |
| --- | --- | --- | --- |
| `MobileNetV4_small.yaml` | x5 | `./mobilenetv4_conv_small.onnx`、`./calibration_data_rgb_f32`（按[校准](#calibration)准备） | `hb_mapper makertbin --config MobileNetV4_small.yaml` |
| `MobileNetV4_medium.yaml` | x5 | `./mobilenetv4_conv_medium_deploy.onnx`（224x224 图）、`./calibration_data_rgb_f32`（按[校准](#calibration)准备） | `hb_mapper makertbin --config MobileNetV4_medium.yaml` |
| `mobilenetv4_small_config.yaml` | s100（s600：march `nash-p`） | `./mobilenetv4_conv_small.onnx`（一致）、`./calibration_data_bgr_224`（脚本产出） | `hb_compile --config mobilenetv4_small_config.yaml` |
| `mobilenetv4_medium_config.yaml` | s100（s600：march `nash-p`） | `./mobilenetv4_conv_medium.onnx`（与导出器的 256 导出一致）、`./calibration_data_bgr_256`（两行开关切换后由脚本产出） | `hb_compile --config mobilenetv4_medium_config.yaml` |

构建 X5 medium 时，准备 224x224 的 `mobilenetv4_conv_medium_deploy.onnx`
以及 RGB float32 校准数据 `./calibration_data_rgb_f32`。上文导出器写出
256x256 的 `mobilenetv4_conv_medium.onnx`；请另行导出或取得适用于此 YAML
的 224x224 图。仅改名不会改变图的输入几何尺寸。

S600 变体只需把 S 侧 YAML 中的 march 改为 `nash-p`。上表 march 取自 YAML
文件本身（X5 `bayes-e`，S100 `nash-e`）。每次重建部署前，对照目标、输入 metadata、输出 shape/dtype 与数值结果。
几何说明：S 侧 medium 配置按 `mobilenetv4_medium_config.yaml` 记录的
256x256 输入编译；X5 medium 配置构建已发布的 224x224 制品。两种
几何都是真实存在的，运行时契约表按目标分别记录。

S100 编译器也支持直接传入 ONNX。校准和 NV12 设置使用上面的 YAML 命令：

```bash
hb_compile --model mobilenetv4_conv_small.onnx --march nash-e
hb_compile --model mobilenetv4_conv_medium.onnx --march nash-e
hb_compile --config mobilenetv4_small_config.yaml
hb_compile --config mobilenetv4_medium_config.yaml
```

<a id="validation"></a>
## 验证

对再生成制品，按 OE 手册执行 `hb_perf` 与 `hrt_model_exec` 并保留完整
输出；随后在匹配板卡上用 运行时确认契约。输入 shape 按
目标×变体给出——与运行时契约表及已发布制品文件名一致：

| 目标 / 变体 | metadata 暴露的输入 | 输出 |
| --- | --- | --- |
| x5，small 与 medium | 单个 packed NV12 输入，224x224（`MobileNetV4_conv_{small,medium}_224x224_nv12.bin`） | F32 `[1,1000,1,1]` |
| s100/s600，small | Y `[1,224,224,1]`、UV `[1,112,112,2]`（`mobilenetv4_small_224x224_nv12.hbm`） | F32 `[1,1000]` |
| s100/s600，medium | **Y `[1,256,256,1]`、UV `[1,128,128,2]`**（`mobilenetv4_medium_256x256_nv12.hbm`） | F32 `[1,1000]` |

S medium 使用 256x256 输入。输出为原始 logits，由运行时任务施加 softmax。

原 S 构建的已发布量化记录（量化后余弦相似度）：

```text
mobilenetv4_medium:
Calibrated Cosine: 0.999759
Quantized Cosine: 0.999863

mobilenetv4_small:
Calibrated Cosine: 0.999892
Quantized Cosine: 0.99988
```

原 S 构建的工具链性能记录：

```text
mobilenetv4_medium:
FPS (1 core): 2468.07
latency: 0.41 ms (405.2 us)
BPU conv original OPs per run: 2,160,488,448

mobilenetv4_small:
FPS (1 core): 5698.18
latency: 0.18 ms (175.5 us)
BPU conv original OPs per run: 372,011,136
```

<a id="artifacts"></a>
## 转换文件

- `get_mobilenetv4_onnx.py`
- `timm2onnx_local.py`
- `get_calibration_data.py`
- `MobileNetV4_small.yaml`
- `MobileNetV4_medium.yaml`
- `mobilenetv4_small_config.yaml`
- `mobilenetv4_medium_config.yaml`
- `x86_medium_inference.py`

<a id="known-gaps"></a>
## 补充准备

- X5 medium YAML 期望名为 `mobilenetv4_conv_medium_deploy.onnx`、输入为 224x224 的模型图；现有 medium 导出辅助程序生成 256x256 图。编译 X5 medium 前，先使 ONNX 几何尺寸与 YAML 一致，并按相同尺寸准备校准数据。
- X5 small 与 medium 校准目录为 RGB `./calibration_data_rgb_f32`；现有校准器生成 BGR 数据。编译前转换通道顺序并使用 X5 YAML 归一化值。
- 将校准辅助程序的源图像目录设置为本地数据集路径。S medium 配置需同时切换两处注释掉的 256 几何开关。按所选 S YAML 的 `mean_value` 对齐辅助程序 mean 常量。
- `x86_medium_inference.py` 在主机运行 medium ONNX；部署时使用编译所得 HBM 与板端运行时。
