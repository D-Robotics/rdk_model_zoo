[English](README.md) | 简体中文

# MobileNetV3 模型转换

固定版本 timm checkpoint 流程使用 [`export.py`](export.py)，命令见[公共主机流程](../../../../utils/tools/mobilenet/README_cn.md)。该流程明确记录权重、中心裁剪预处理、batch=1 logits 合同和全量评测输入；下文既有制品命令按各自合同使用。

模型转换在 x86 Linux 主机上的 RDK OpenExplore (OE) 环境中执行，不是板卡
操作。使用下面的导出脚本、校准工具和目标板卡配置准备部署模型。

<a id="source-model"></a>
## 源模型

timm `mobilenetv3_large_100` 预训练权重（MobileNetV3-Large），由
`get_mobilenetv3_onnx.py` 固定；两个平台均为 NCHW 输入
`[1,3,224,224]`。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── MobileNetV3_config.yaml  # 配置
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── get_calibration_data.py  # Python 脚本
├── get_mobilenetv3_onnx.py  # Python 脚本
├── mobilenetv3_s_config.yaml  # 配置
└── timm2onnx_local.py  # Python 脚本
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
cwd 为 `samples/vision/mobilenetv3/conversion`：

```bash
# 输入：timm 预训练权重（未缓存时下载）
# 输出：下列 .onnx 文件 — 成功判据：onnx 化简检查通过
python3 get_mobilenetv3_onnx.py    # -> ./mobilenetv3_large_100.onnx
```

导出器使用 onnx-simplifier 并打印参数量（`mobilenetv3_large_100` 为
5,470,832）；预期 metadata 输出为 `mean (0.485, 0.456, 0.406)`、
`std (0.229, 0.224, 0.225)`、"Simplified model is valid."。

安装导出依赖：

```bash
pip install timm onnx onnxsim
huggingface-cli login
python3 get_mobilenetv3_onnx.py
```

通过 Hugging Face 获取受限权重时，使用上述登录命令完成认证。

### ONNX 导出示例

在 x86 导出环境中运行以下示例。需要 PyTorch；使用简化步骤时还需安装 `onnxsim`。导出文件名与所选 YAML 的 `onnx_model` 必须对应。

onnx 模型使用的是 timm 库 (PyTorch Image Models) 中的模型进行转换的，使用以下命令安装所需要的包：

```shell
pip install timm onnx
```

模型转换以 mobilenetv3_large_100 为例：

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify
from timm.models import create_model

from timm.models.mobilenetv3 import mobilenetv3_large_100

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
    model = create_model('mobilenetv3_large_100', pretrained=True)
    model.eval()

    # print the model structure

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "mobilenetv3_large_100.onnx"

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
        simplified_onnx_file_path = "mobilenetv3_large_100.onnx"
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
运行前将 `src_image_dir` 改为本机 ImageNet 验证集目录。预处理输出 BGR 数据：padded center
crop 224、resize、HWC→CHW、`RGB2BGRTransformer`、×255、mean
`103.94 116.78 123.68`、×0.017，写入 `./calibration_data_bgr/`。
选择并记录校准集使用的验证图像。

按各 YAML 准备对应输入：

| 配置（目标） | `cal_data_dir` | 布局与数值 | 准备步骤 |
| --- | --- | --- | --- |
| `mobilenetv3_s_config.yaml`（s100；s600 仅改 march） | `./calibration_data_bgr` | BGR，mean `103.53 116.28 123.675` | 将 `src_image_dir` 设为本机验证集目录；生成 BGR 数据前，将辅助脚本 mean 常量（`103.94 116.78 123.68`）与 YAML 数值对齐。 |
| `MobileNetV3_config.yaml`（x5） | `./calibration_data_rgb_f32` | RGB，mean `123.675 116.28 103.53` | 准备 224x224 float32 RGB 校准数组，按 X5 YAML 的通道顺序和归一化数值处理；使用独立 RGB 预处理链。 |

<a id="compile"></a>
## 编译

构建配置：

| 配置 | 目标 | 配置引用的输入 | 命令（OE 容器内） |
| --- | --- | --- | --- |
| `MobileNetV3_config.yaml` | x5 | `./mobilenetv3_large_100.onnx`（与导出器输出一致）、`./calibration_data_rgb_f32`（按[校准](#calibration)准备） | `hb_mapper makertbin --config MobileNetV3_config.yaml` |
| `mobilenetv3_s_config.yaml` | s100（s600：march 改 `nash-p`） | `./mobilenetv3_large_100.onnx`（一致）、`./calibration_data_bgr`（改源目录后由脚本产出） | `hb_compile --config mobilenetv3_s_config.yaml` |

上表 march 取自 YAML 文件本身（X5 `bayes-e`，S100 `nash-e`）。对比目标、输入 metadata、输出
shape/dtype 与数值结果之后，才能把再生成制品视为与已发布制品等价。
重命名说明：S 源文件名为 `mobilenetv3_config.yaml`；在不区分大小写
的文件系统上与 X5 的 `MobileNetV3_config.yaml` 冲突，因此此处 S 副本
更名为 `mobilenetv3_s_config.yaml`——仅文件名变化，内容原样。

S100 编译器也支持直接传入 ONNX。生成经过校准的 NV12 部署模型时，使用上面的目标 YAML：

```bash
hb_compile --model mobilenetv3_large_100.onnx --march nash-e
hb_compile --config mobilenetv3_s_config.yaml
```

<a id="validation"></a>
## 验证

对再生成制品，按 OE 手册执行 `hb_perf` 与 `hrt_model_exec` 并保留完整
输出；随后在匹配板卡上用 运行时确认契约：X5 暴露一个 packed
NV12 输入与 F32 `[1,1000,1,1]` 输出；S100/S600 暴露 Y `[1,224,224,1]`、
UV `[1,112,112,2]` 与 F32 `[1,1000]` 输出；输出语义为原始 logits
（softmax 由运行时任务施加）。

原 S 构建的已发布量化记录（量化后余弦相似度）：

```text
TensorName: output
Calibrated Cosine: 0.911233
Quantized Cosine: 0.909042
```

原 S 构建的工具链性能参考：

```text
FPS (1 core): 2616.81
latency: 0.38 ms (382.1 us)
BPU conv original OPs per run: 433,179,520
```

<a id="artifacts"></a>
## 转换文件

- `get_mobilenetv3_onnx.py`
- `timm2onnx_local.py`
- `get_calibration_data.py`
- `MobileNetV3_config.yaml`
- `mobilenetv3_s_config.yaml`

<a id="known-gaps"></a>
## 补充准备

- 在不区分大小写的文件系统上执行 S 侧导出时，使用上文的 YAML 文件名 `mobilenetv3_s_config.yaml`。
- X5 需按 `MobileNetV3_config.yaml` 准备 RGB 顺序的 `./calibration_data_rgb_f32`。现有校准辅助程序生成 BGR 数据，因此应转换通道顺序并使用 X5 YAML 的 RGB 归一化，不要仅重命名 BGR 目录。
- 运行校准前，将辅助程序的源图像目录设置为本地 ImageNet 路径。辅助程序 mean 常量与 S YAML 的 `mean_value` 略有差异；按所选目标对齐校准值与配置。
- 编译后使用板端运行时检查产物契约。
