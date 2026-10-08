[English](README.md) | 简体中文

# MobileOne 转换

<a id="source-model"></a>
## 源模型

使用 apple/ml-mobileone 上游流程：加载匹配的未融合权重，执行 `reparameterize_model(model)`，再导出并简化 ONNX。构建时记录上游修订、包版本与权重摘要。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── MobileOne_S0_config.yaml  # 配置
├── MobileOne_S1_config.yaml  # 配置
├── MobileOne_S2_config.yaml  # 配置
├── MobileOne_S3_config.yaml  # 配置
├── MobileOne_S4_config.yaml  # 配置
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>
## 工具链与目标

5 份 YAML 面向 X5 `march: bayes-e`。构建时记录使用的 OE 版本。

| Config | ONNX input path | Working directory | Compiled basename |
| --- | --- | --- | --- |
| `MobileOne_S0_config.yaml` | `./mobileone_s0.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S1_config.yaml` | `./mobileone_s1.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S2_config.yaml` | `./mobileone_s2.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S3_config.yaml` | `./mobileone_s3.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S4_config.yaml` | `./mobileone_s4.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |


工具链资源:

- [OE Docker environment](https://forum.d-robotics.cc/t/topic/35229)

<a id="export"></a>
## ONNX 导出

使用 apple/ml-mobileone 上游流程：加载匹配的未融合权重，执行 `reparameterize_model(model)`，再导出并简化 ONNX。构建时记录上游修订、包版本与权重摘要。

使用下面的 ONNX 导出示例，在上表路径准备匹配图，名义输入 RGB NCHW 1×3×224×224、输出 ImageNet-1k。YAML input_shape/input_name 为空，维度与名称从图读取，必须核对。

### ONNX 导出示例

在 x86 导出环境中运行以下示例。需要 PyTorch；使用简化步骤时还需安装 `onnxsim`。导出文件名与所选 YAML 的 `onnx_model` 必须对应。

onnx 模型使用的是 MobileOne 模型源码进行转换的，使用以下命令安装所需要的包：

```shell
pip install timm onnx
```

模型源码下载使用以下命令：

```shell
git clone https://github.com/apple/ml-mobileone.git
```

模型转换以 mobileone_s0 为例，其余四个模型同理：

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify

from mobileone import *

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
    model_path = "mobileone_s0_unfused.pth.tar"
    model = mobileone(variant='s0')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = reparameterize_model(model)

    # print(model)

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "mobileone_s0.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=11,
        verbose=True,
        input_names=["data"],  # 输入名
        output_names=["output"],  # 输出名
        keep_initializers_as_inputs=True
    )

    # Simplify the ONNX model
    model_simp, check = simplify(onnx_file_path)

    if check:
        print("Simplified model is valid.")
        simplified_onnx_file_path = "mobileone_s0.onnx"
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

所有 YAML 需要 `./calibration_data_rgb_f32`（float32、校准 default），训练输入 RGB/NCHW、运行输入 NV12。mean 为 123.675/116.28/103.53，scale 为 0.01712475/0.017507/0.01742919。缺少数据集选择/数量、准备脚本及实际校准文件。须核对浮点语义与图和归一化，不能给 NV12 目录改名冒充 RGB 校准数据。

<a id="compile"></a>
## 编译

在 OE 环境内、补齐 ONNX 图与校准数据前提后执行：

```bash
# cwd: repository root, then conversion directory
cd samples/vision/mobileone/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./mobileone_s0.onnx
hb_mapper makertbin --model-type onnx --config MobileOne_S0_config.yaml
```

该配置预期输出为 `MobileOne_224x224_nv12_int8/MobileOne_224x224_nv12.bin`，各配置均使用 latency/O3。所有变体共用 `MobileOne_224x224_nv12_int8` 目录及 `MobileOne_224x224_nv12` 前缀，必须隔离构建并保留变体身份。YAML 设置 32 个编译 jobs，需要匹配构建主机资源；未设置删节点覆盖项或 debug_mode。

<a id="validation"></a>
## 转换后验证

推理前核对 packed NV12 几何 224×224、squeeze 后为 (1000,) 的 F32
分数输出。首个变体示例：

```bash
# cwd: repository root on X5
python3 samples/vision/mobileone/runtime/python/main.py --target x5 \
  --asset-id x5:mobileone:MobileOne_S0_224x224_nv12.bin \
  --model-path samples/vision/mobileone/conversion/MobileOne_224x224_nv12_int8/MobileOne_224x224_nv12.bin \
  --test-img samples/vision/mobileone/test_data/tiger_beetle.JPEG
```

准确引用用于选择契约，不证明重建字节等于发布字节。记录新哈希和图来源，对照源/统一输出，交付前评估精度。

<a id="artifacts"></a>
## 产物

编译路径见上表，逐变体发布文件与目标见[模型准备](../model/README_cn.md#artifacts)，下载落在 sample model 目录。移动已验证构建时保留变体与来源，改名本身不是修复。

<a id="known-gaps"></a>
## 补充准备

重新构建 MobileOne 时，加载匹配的未融合权重（例如 `mobileone_s0_unfused.pth.tar`），执行 `reparameterize_model(model)`，再导出并简化为所选 YAML 路径下的 RGB/NCHW 1×3×224×224 模型图。按 YAML 的 mean `123.675/116.28/103.53`、scale `0.01712475/0.017507/0.01742919` 与 `default` 校准准备 `./calibration_data_rgb_f32` float32 RGB 数据。使用 X5 `bayes-e` 配置构建，并记录实际框架与 OE 版本。
