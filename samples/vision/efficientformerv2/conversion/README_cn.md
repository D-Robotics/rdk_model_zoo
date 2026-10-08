# EfficientFormerV2 转换

本目录提供转换资产：三份参考 PTQ YAML
（`EfficientFormerv2_s0_config.yaml`、`EfficientFormerv2_s1_config.yaml`、
`EfficientFormerv2_s2_config.yaml`）。目录**不带导出脚本，也不带校准
数据生产脚本**，因此这是参考配置集而非完全可复现的流程；可复现边界见
[补充准备](#known-gaps)。

<a id="source-model"></a>
## 源模型

EfficientFormerV2-S0/S1/S2（论文 [EfficientFormerV2: Rethinking Vision
Transformers for MobileNet Size and
Speed](https://arxiv.org/abs/2212.08059)）。三份 YAML 期望
`./efficientformerv2_s0.onnx`、`./efficientformerv2_s1.onnx`、
`./efficientformerv2_s2.onnx`，但没有记录导出配方，也未固定权重
——ONNX 出处未经验证。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── EfficientFormerv2_s0_config.yaml  # 配置
├── EfficientFormerv2_s1_config.yaml  # 配置
├── EfficientFormerv2_s2_config.yaml  # 配置
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>
## 工具链与目标

模型转换在 x86 Linux 主机上的 RDK X5 OpenExplorer Docker（march
`bayes-e`）中执行，绝不在板卡上运行。请准备含 `hb_mapper`、`hb_perf`、
`hrt_model_exec` 的工具链；离线 Docker 镜像可从 D-Robotics 开发者论坛
（[topic 35229](https://forum.d-robotics.cc/t/topic/35229)）获取。

<a id="export"></a>
## 导出

目录不含导出脚本。上游 EfficientFormerV2 流程使用 `timm` 导出 ONNX：

1. 安装所需 Python 包，如 `timm`、`onnx`、`onnxsim`。
2. 用 `timm.models.create_model` 创建目标 EfficientFormerV2 模型，如
   `efficientformerv2_s0`、`efficientformerv2_s1`、`efficientformerv2_s2`。
3. 用 `1x3x224x224` 虚拟输入，通过 `torch.onnx.export` 导出模型。
4. 用 `onnxsim.simplify` 化简 ONNX 模型。
5. 在 OE 环境中编译化简后的 ONNX 模型（见"编译"）。

产物必须命名为 `efficientformerv2_s0.onnx`、
`efficientformerv2_s1.onnx`、`efficientformerv2_s2.onnx` 并放在本目录
（或修改 YAML 的 `onnx_model`）。

### ONNX 导出示例

在 x86 导出环境中运行以下示例。需要 PyTorch；使用简化步骤时还需安装 `onnxsim`。导出文件名与所选 YAML 的 `onnx_model` 必须对应。

onnx 模型使用的是 timm 库 (PyTorch Image Models) 中的模型进行转换的，使用以下命令安装所需要的包：

```shell
pip install timm onnx
```

模型转换以 efficientformerv2_s0 为例，其余两个模型同理：

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify
from timm.models import create_model

from timm.models.efficientformer_v2 import efficientformerv2_s0, efficientformerv2_s1, efficientformerv2_s2

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
    model = create_model('efficientformerv2_s0', pretrained=True)
    model.eval()

    # print the model structure

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "efficientformerv2_s0.onnx"

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
        simplified_onnx_file_path = "efficientformerv2_s0.onnx"
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

三份 YAML 均使用
`./calibration_data_rgb_f32`（float32 RGB `.npy`），使用
`calibration_type: 'max'` 且逐变体 `max_percentile`：S0 与 S1 为
`0.999`，S2 为 `0.9995`。等价数据必须遵循 YAML 数值（mean
`123.675 116.28 103.53`，scale `0.01712475 0.017507 0.01742919`，
224x224）。

<a id="compile"></a>
## 编译

在 OE 环境中，两个输入就绪后编译对应变体：

```bash
# cwd：本 conversion 目录
# 输入：./efficientformerv2_s0.onnx + ./calibration_data_rgb_f32
# 输出：working_dir 'EfficientFormerv2_s0_int16_model_output'，
#       产出的 .bin 按 output_model_file_prefix 命名（见下文）
hb_mapper checker --config EfficientFormerv2_s0_config.yaml
hb_mapper makertbin --config EfficientFormerv2_s0_config.yaml
```

这里每份 YAML 都携带变体身份：`output_model_file_prefix` 为
`EfficientFormerv2_s{0,1,2}_224x224_nv12`，产出的 `.bin` 名称直接复现
Manifest 基名（`EfficientFormerv2_s0_224x224_nv12.bin`……），无需改名，
且每个变体编译进各自的 `working_dir`，无冲突。三份 YAML 均设置
`compile_mode: 'latency'` / `optimize_level: 'O3'`，并携带 `node_info`
Softmax int16 摆放（S0/S1 为 5 个节点，S2 为 10 个）。S0 额外设置
`debug_mode: "dump_calibration_data"` 与
`optimization: "set_all_nodes_int16"`；S1/S2 未设置这两项。

<a id="validation"></a>
## 转换后验证

使用 OE 包中的以下工具进行主机模型检查： `hb_perf` 与
`hrt_model_exec`。板端功能检查走样例运行时：
`python3 samples/vision/efficientformerv2/runtime/python/main.py --target x5 --asset-id x5:efficientformerv2:EfficientFormerv2_s0_224x224_nv12.bin...`
（见 [runtime/python/README_cn.md](../runtime/python/README_cn.md)）。
运行时期望的输入张量为 NV12 打包前的 `1x3x224x224`，输出为
ImageNet-1k 分类 logits。

<a id="artifacts"></a>
## 配方文件

三份参考 YAML 即本目录的转换资产；其 SHA-256 如下，可用于核对本地副本：

| 文件 | SHA-256 |
| --- | --- |
| `EfficientFormerv2_s0_config.yaml` | `a0415f8a4a3f75976be1c8a5a0bf30aa9874b95ee5f61ec10d6ee7147c5d4351` |
| `EfficientFormerv2_s1_config.yaml` | `530d78e7d2e28eb57832b2f8d48d7d1d8f4de5359b127eac915922be6353fcc9` |
| `EfficientFormerv2_s2_config.yaml` | `b35d73d6059f5e7765f415eac1863a72792d0e090d6c04640e1164b4814ad699` |

<a id="known-gaps"></a>
## 补充准备

为每个 `efficientformerv2_s*.onnx` 输入准备对应模型图，并在 `./calibration_data_rgb_f32` 准备 float32 RGB `.npy` 数据，按所选 YAML 的 224x224 归一化处理（mean `123.675 116.28 103.53`、scale `0.01712475 0.017507 0.01742919`）。S0 设置 `debug_mode: "dump_calibration_data"` 与 `optimization: "set_all_nodes_int16"`；S1/S2 未设置这两项。S0 使用 `working_dir: 'EfficientFormerv2_s0_int16_model_output'`，S1/S2 使用 `EfficientFormerv2_s{1,2}_224x224_nv12`。按各 YAML 的输出目录分别构建，并运行上文命令。
