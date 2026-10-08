# ResNeXt 转换

<a id="source-model"></a>
## 源模型

ResNeXt 源引用 timm resnext50_32x4d 与 ONNX 简化，导出示例见下文；构建时记录包版本、权重版本与摘要。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── ResNeXt50_32x4d_config.yaml  # 配置
```

<a id="toolchain-targets"></a>
## 工具链与目标

使用 X5 OpenExplore 工具链，march 为 `bayes-e`。

| YAML | ONNX path (conversion cwd) | Output path |
| --- | --- | --- |
| `ResNeXt50_32x4d_config.yaml` | `./ResNeXt50_32x4d.onnx` | `ResNeXt50_32x4d_224x224_nv12/ResNeXt50_32x4d_224x224_nv12.bin` |


工具链资源:

- [OE Docker environment](https://forum.d-robotics.cc/t/topic/35229)

<a id="export"></a>
## ONNX 导出

使用下面的导出示例，在 YAML 旁准备 `ResNeXt50_32x4d.onnx`，名义输入 RGB NCHW 1×3×224×224；input_shape/name 为空，从图读取。

### ONNX 导出示例

在 x86 导出环境中运行以下示例。需要 PyTorch；使用简化步骤时还需安装 `onnxsim`。导出文件名与所选 YAML 的 `onnx_model` 必须对应。

onnx 模型使用的是 timm 库 (PyTorch Image Models) 中的模型进行转换的，使用以下命令安装所需要的包：

```shell
pip install timm onnx
```

模型转换以 resnext50_32x4d 为例：

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify
from timm.models import create_model

from timm.models.resnet import resnext50_32x4d

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
    model = create_model('resnext50_32x4d', pretrained=True)
    model.eval()

    # print the model structure

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "resnext50_32x4d.onnx"

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
        simplified_onnx_file_path = "resnext50_32x4d.onnx"
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

YAML 训练输入 RGB/NCHW、运行输入 NV12，mean 123.675/116.28/103.53，scale 0.01712475/0.017507/0.01742919。ResNeXt 使用./calibration_data_rgb_f32、float32、校准 default，但没有准备脚本/数量/数据。

<a id="compile"></a>
## 编译

在 OE 环境内、ONNX 图与校准数据就绪后执行：

```bash
# cwd: repository root
cd samples/vision/resnext/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./ResNeXt50_32x4d.onnx
hb_mapper makertbin --model-type onnx --config ResNeXt50_32x4d_config.yaml
```

保留 latency/O3。输出基名与发布文件名一致，但名称一致不能证明图、字节或精度相同。

<a id="validation"></a>
## 转换后验证

核对实际 metadata、224×224 packed NV12 与 squeeze 后 1000 分数的单 F32 输出。重建模型可用准确契约引用和外部路径选择，其哈希/来源必须与发布制品分开记录。

```bash
# cwd: repository root on X5; published-artifact smoke
bash samples/vision/resnext/model/download.sh x5 50_32x4d
python3 samples/vision/resnext/runtime/python/main.py --target x5 --variant 50_32x4d
```

<a id="artifacts"></a>
## 产物

发布制品及落地路径见[模型准备](../model/README_cn.md#artifacts)。

<a id="known-gaps"></a>
## 补充准备

推理请使用 [model/README_cn.md](../model/README_cn.md) 中的 X5 Manifest 制品。重新构建需准备 `ResNeXt50_32x4d.onnx`（RGB/NCHW 1×3×224×224 输入）、`./calibration_data_rgb_f32` float32 RGB 校准数据、mean `123.675/116.28/103.53`、scale `0.01712475/0.017507/0.01742919`，并使用上文 X5 `bayes-e` 配置。构建前选择匹配的上游权重与 OE 工具链。
