# EfficientFormer 转换

本目录提供转换资产：两份参考 PTQ YAML
（`EfficientFormer_l1_config.yaml`、`EfficientFormer_l3_config.yaml`）。
编译前，按 YAML 指定路径准备模型 ONNX 图与校准数据，再使用下文 OE 命令。

<a id="source-model"></a>
## 源模型

EfficientFormer-L1 与 EfficientFormer-L3（论文 [EfficientFormer:
Vision Transformers at MobileNet
Speed](https://arxiv.org/abs/2206.01191)）。YAML 期望
`./efficientformer_l1.onnx` / `./efficientformer_l3.onnx`；导出时按模型变体准备匹配权重，并将图保存至对应 YAML 路径。

其他参考：[arXiv:2206.00171](https://arxiv.org/abs/2206.00171).

<a id="directory"></a>
## 目录结构

```text
conversion/
├── EfficientFormer_l1_config.yaml  # 配置
├── EfficientFormer_l3_config.yaml  # 配置
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>
## 工具链与目标

模型转换在 x86 Linux 主机上的 RDK X5 OpenExplorer Docker 内执行
（march `bayes-e`），从不在板卡上运行。请准备含 `hb_mapper`、
`hb_perf`、`hrt_model_exec` 的工具链；离线
Docker 镜像可从 D-Robotics 开发者论坛
（[topic 35229](https://forum.d-robotics.cc/t/topic/35229)）获取。

<a id="export"></a>
## 导出

使用上游 EfficientFormer 流程和 `timm` 导出 ONNX：

1. 用 `timm.models.create_model` 创建目标 EfficientFormer 模型，如
   `efficientformer_l1` 或 `efficientformer_l3`。
2. 用 `torch.onnx.export` 导出模型。
3. 用 `onnxsim.simplify` 化简 ONNX 模型。
4. 在 OE 环境中编译化简后的 ONNX 模型（见"编译"）。

产物必须命名为 `efficientformer_l1.onnx` /
`efficientformer_l3.onnx` 并放在本目录（或修改 YAML 的 `onnx_model`）。

### ONNX 导出示例

在 x86 导出环境中运行以下示例。需要 PyTorch；使用简化步骤时还需安装 `onnxsim`。导出文件名与所选 YAML 的 `onnx_model` 必须对应。

**ONNX文件下载**：

onnx 模型使用的是 timm 库 (PyTorch Image Models) 中的模型进行转换的，使用以下命令安装所需要的包：

```shell
pip install timm onnx
```

模型转换以 efficientformer_l1 为例：

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify
from timm.models import create_model

from timm.models.efficientformer import efficientformer_l1, efficientformer_l3

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
    model = create_model('efficientformer_l1', pretrained=True)
    model.eval()

    # print the model structure

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "efficientformer_l1.onnx"

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
        simplified_onnx_file_path = "efficientformer_l1.onnx"
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

按 YAML 准备校准数据： `./calibration_data_rgb_f32`
（float32 RGB `.npy`）。在 224x224 尺寸下按 YAML 数值处理数据：mean
`123.675 116.28 103.53`、scale `0.01712475 0.017507 0.01742919`。

<a id="compile"></a>
## 编译

在 OE 环境内、两个输入就绪后，编译对应变体：

```bash
# 输入：./efficientformer_l1.onnx + ./calibration_data_rgb_f32
# 输出前缀：EfficientFormer_224x224_nv12
hb_mapper checker --config EfficientFormer_l1_config.yaml
hb_mapper makertbin --config EfficientFormer_l1_config.yaml
```

两份 YAML 均使用 `calibration_type: 'default'` 加
`optimization: "set_all_nodes_int16"`、`compile_mode: 'latency'` /
`optimize_level: 'O3'`；L3 另设 `jobs: 64`，且比 L1 携带更多
`node_info` Softmax int16 配置。

<a id="validation"></a>
## 验证

使用 OE 包中的以下工具进行主机模型检查： `hb_perf` 与
`hrt_model_exec`。板端功能检查走样例运行时：
`python3 samples/vision/efficientformer/runtime/python/main.py --target x5 --asset-id x5:efficientformer:EfficientFormer_l1_224x224_nv12.bin...`
（见 [runtime/python/README_cn.md](../runtime/python/README_cn.md)）。
运行时期望的输入张量为 NV12 打包前的 `1x3x224x224`，输出为
ImageNet-1k 分类 logits。

<a id="artifacts"></a>
## 产物

使用 `EfficientFormer_l1_config.yaml` 或 `EfficientFormer_l3_config.yaml`
及其匹配的 ONNX 图和校准目录。

<a id="known-gaps"></a>
## 补充准备

从匹配的上游权重与导出流程准备 `efficientformer_l1.onnx` 或 `efficientformer_l3.onnx`。按所选 YAML 的 RGB/NCHW 归一化准备 `./calibration_data_rgb_f32`（mean `123.675 116.28 103.53`、scale `0.01712475 0.017507 0.01742919`）。两份 YAML 共用 `working_dir: 'EfficientFormer_224x224_nv12_int16'` 与 `output_model_file_prefix: 'EfficientFormer_224x224_nv12'`；每次在独立工作目录构建一个变体，并在部署前使用对应 Manifest 文件名。使用匹配 YAML 执行上文 OE 命令。
