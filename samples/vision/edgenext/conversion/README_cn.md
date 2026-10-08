# EdgeNeXt 转换

本目录提供四份参考 PTQ YAML
（`EdgeNeXt_{base,small,x_small,xx_small}_config.yaml`）。按所选变体准备
YAML 指定的 ONNX 图与校准数据，再使用对应配置编译；每份 YAML 都指定
该变体的 ONNX 输入名和输出前缀。


<a id="source-model"></a>
## 源模型

EdgeNeXt base/small/x-small/xx-small（论文 [EdgeNeXt: Efficiently
Amalgamated CNN-Transformer Architecture for Mobile Vision
Applications](https://arxiv.org/abs/2206.10589)，参考实现
[mmaaz60/EdgeNeXt](https://github.com/mmaaz60/EdgeNeXt)）。各 YAML 要求
`./edgenext_{base,small,x_small,xx_small}.onnx`；按所选变体导出匹配
模型，并记录所用权重修订。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── EdgeNeXt_base_config.yaml  # 配置
├── EdgeNeXt_small_config.yaml  # 配置
├── EdgeNeXt_x_small_config.yaml  # 配置
├── EdgeNeXt_xx_small_config.yaml  # 配置
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>
## 工具链与目标

模型转换在 x86 Linux 主机的 RDK X5 OpenExplorer Docker 内执行
（march `bayes-e`），从不在板上运行。请准备含 `hb_mapper`、`hb_perf`、`hrt_model_exec` 的工具链；离线 Docker 镜像可从地瓜机器人开发者论坛（[topic 35229](https://forum.d-robotics.cc/t/topic/35229)）获取。

<a id="export"></a>
## 导出

使用上游 EdgeNeXt 流程和 `timm` 导出 ONNX：

1. 用 `timm.models.create_model` 创建目标 EdgeNeXt 模型，如
   `edgenext_base`、`edgenext_small`、`edgenext_x_small`、
   `edgenext_xx_small`。
2. 用 `torch.onnx.export` 导出模型。
3. 用 `onnxsim.simplify` 化简 ONNX 模型。
4. 在 OE 环境中编译化简后的 ONNX 模型（见"编译"）。

产物须命名为 `edgenext_<variant>.onnx` 并放在本目录（或调整 YAML 的
`onnx_model`）。

### ONNX 导出示例

在 x86 导出环境中运行以下示例。需要 PyTorch；使用简化步骤时还需安装 `onnxsim`。导出文件名与所选 YAML 的 `onnx_model` 必须对应。

onnx 模型使用的是 timm 库 (PyTorch Image Models) 中的模型进行转换的，使用以下命令安装所需要的包：

```shell
pip install timm onnx
```

模型转换以 edgenext_small 为例，其余三个模型同理：

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify
from timm.models import create_model

from timm.models.edgenext import edgenext_small, edgenext_base, edgenext_x_small, edgenext_xx_small

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
    model = create_model('edgenext_xx_small', pretrained=True)
    model.eval()

    # print the model structure

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "edgenext_xx_small.onnx"

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
        simplified_onnx_file_path = "edgenext_xx_small.onnx"
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

四份 YAML 均要求 `./calibration_data_rgb_f32`
（float32 RGB `.npy`），`calibration_type: 'max'`、
`max_percentile: 0.999`。准备 224x224 数据时按 YAML 数值处理：mean
`123.675 116.28 103.53`，scale `0.01712475 0.017507 0.01742919`。

<a id="compile"></a>
## 编译

在 OE 环境内、两个输入就绪后（以 base 为例；其余变体换用各自配置）：

```bash
# cwd：本转换目录
# 输入：./edgenext_base.onnx + ./calibration_data_rgb_f32
# 输出：working_dir 'EdgeNeXt_base_224x224_nv12'，产出
#       EdgeNeXt_base_224x224_nv12.bin——与清单名一致，无需重命名
hb_mapper checker --config EdgeNeXt_base_config.yaml
hb_mapper makertbin --config EdgeNeXt_base_config.yaml
```

四份 YAML 均设 `compile_mode: 'latency'` / `optimize_level: 'O3'`，并通过
`node_info` 将交叉协方差注意力（xca）Softmax 节点以 int16 I/O 摆上 BPU
——每份 3 处（stages 1/2/3）；xx-small 另加 13 处（共 16 处）。均不带
`debug_mode` 与 `set_all_nodes_int16`。

<a id="validation"></a>
## 验证

使用 OE 包中的 `hb_perf` 与 `hrt_model_exec` 检查主机模型。板端功能检查即样例运行时：
`python3 samples/vision/edgenext/runtime/python/main.py --target x5 --asset-id x5:edgenext:EdgeNeXt_base_224x224_nv12.bin...`
（见 [runtime/python/README_cn.md](../runtime/python/README_cn.md)）。运行
时期望的输入张量为 NV12 打包前的 `1x3x224x224`，输出为 ImageNet-1k
分类 logits。

<a id="artifacts"></a>
## 产物

按变体使用对应的 YAML、ONNX 图与校准目录。

<a id="known-gaps"></a>
## 补充准备

为所选变体准备 YAML 指定的 ONNX 文件（`edgenext_<variant>.onnx`），输入为 RGB/NCHW 224x224。准备 `./calibration_data_rgb_f32` float32 RGB `.npy` 数据，使用 YAML mean `123.675 116.28 103.53` 与 scale `0.01712475 0.017507 0.01742919`。输出前缀携带变体名，生成的 `.bin` 使用已发布文件名。使用对应变体的 YAML 执行上文 checker 与编译命令。
