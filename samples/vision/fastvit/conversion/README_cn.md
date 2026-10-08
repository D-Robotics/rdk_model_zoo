# FastViT 转换

本目录提供四份参考 PTQ YAML
（`FastViT_{S12,SA12,T12,T8}_config.yaml`）。按变体准备 YAML 指定的
ONNX 图与 RGB float32 校准数据，再执行对应的 OE 编译命令。

各 YAML 的 `onnx_model` 指向本样例树之外的公共 model-zoo 路径，且四份配置
共用**无变体**输出前缀 `FastViT_224x224_nv12`（复现已发布基名需重命名）。
按配置路径准备各图，并为所选变体命名编译产物。

<a id="source-model"></a>
## 源模型

FastViT S12/SA12/T12/T8（论文 [FastViT: A Fast Hybrid Vision Transformer
using Structural
Reparameterization](https://arxiv.org/abs/2303.14189)）。各 YAML 从共享 `01_common` 模型库消费 ONNX
（见缺口 1），未记录导出配方、未固定权重——ONNX 来源未核实。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── FastViT_S12_config.yaml  # 配置
├── FastViT_SA12_config.yaml  # 配置
├── FastViT_T12_config.yaml  # 配置
├── FastViT_T8_config.yaml  # 配置
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>
## 工具链与目标

模型转换在 x86 Linux 主机的 RDK X5 OpenExplorer Docker 内执行
（march `bayes-e`），从不在板上运行。请准备含 `hb_mapper`、`hb_perf`、`hrt_model_exec` 的工具链；离线 Docker 镜像可从地瓜机器人开发者论坛（[topic 35229](https://forum.d-robotics.cc/t/topic/35229)）获取。

<a id="export"></a>
## 导出

按上游 FastViT 流程使用 `timm` 导出 ONNX：

1. 用 `timm.models.create_model` 创建目标 FastViT 模型，如
   `fastvit_t8`、`fastvit_t12`、`fastvit_s12`、`fastvit_sa12`。
2. 用 `torch.onnx.export` 导出模型。
3. 用 `onnxsim.simplify` 化简 ONNX 模型。
4. 在 OE 环境中编译化简后的 ONNX 模型（见"编译"）。

各配置要求 ONNX 输入位于
`../../../01_common/model_zoo/mapper/classification/FastViT/fastvit_<variant>.onnx`
——本仓库不携带该目录。重新生成输入时需恢复该布局或调整
`onnx_model`。

### ONNX 导出示例

在 x86 导出环境中运行以下示例。需要 PyTorch；使用简化步骤时还需安装 `onnxsim`。导出文件名与所选 YAML 的 `onnx_model` 必须对应。

onnx 模型使用的是 timm 库 (PyTorch Image Models) 中的模型进行转换的，使用以下命令安装所需要的包：

```shell
pip install timm onnx
```

模型转换以 fastvit_t8 为例，其余三个模型同理：

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify
from timm.models import create_model

from timm.models.fastvit import fastvit_s12, fastvit_sa12, fastvit_t8, fastvit_t12

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
    model = create_model('fastvit_t8', pretrained=True)
    model.eval()

    # print the model structure

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "fastvit_t8.onnx"

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
        simplified_onnx_file_path = "fastvit_t8.onnx"
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
（float32 RGB `.npy`）并使用 `calibration_type: 'default'`。等价数据必须
遵循 YAML 数值（mean `123.675 116.28 103.53`，scale `0.01712475
0.017507 0.01742919`，224x224）。

<a id="compile"></a>
## 编译

在 OE 环境内、两个输入就绪后（以 S12 为例；其余变体换用各自配置）：

```bash
# cwd：本转换目录
# 输入：外部 01_common ONNX（见缺口 1）+ ./calibration_data_rgb_f32
# 输出：working_dir 'FastViT_224x224_nv12_mix'，产出
#       FastViT_224x224_nv12.bin——需重命名，见缺口
hb_mapper checker --config FastViT_S12_config.yaml
hb_mapper makertbin --config FastViT_S12_config.yaml
```

四份 YAML 均设 `compile_mode: 'latency'` / `optimize_level: 'O3'`，并通过
`node_info` 将重参数化注意力/MLP 节点以 int16 I/O 摆上 BPU
（S12/SA12/T12/T8 各 5/6/4/10 处）。均不带 `debug_mode` 与
`set_all_nodes_int16`。

<a id="validation"></a>
## 验证

使用 OE 包中的 `hb_perf` 与 `hrt_model_exec` 检查主机模型。板端功能检查即样例运行时：
`python3 samples/vision/fastvit/runtime/python/main.py --target x5 --asset-id x5:fastvit:FastViT_S12_224x224_nv12.bin...`
（见 [runtime/python/README_cn.md](../runtime/python/README_cn.md)）。
运行时期望的输入张量为 NV12 打包前的 `1x3x224x224`，输出为
ImageNet-1k 分类 logits。

<a id="artifacts"></a>
## 配方文件

四份参考 YAML 即本目录的转换资产；其 SHA-256 如下，可用于核对本地副本：

| 文件 | SHA-256 |
| --- | --- |
| `FastViT_S12_config.yaml` | `50c5b40ab3d801ad72eae45a6927dcce4074cf48af90d62d0b974236d46eb8d2` |
| `FastViT_SA12_config.yaml` | `612f9e668d2a30549c33d72595bc84846d05c2404960b31276f174b8c6ddc8fe` |
| `FastViT_T12_config.yaml` | `17b23a8dc23423e499d68d0f9ec3cacf5b2184148fdaa710f20a125c7509a5e2` |
| `FastViT_T8_config.yaml` | `79ab7b5478b3978af871feb81c90e70b87838fa65d5d922cc8d14becbf7d6a0f` |

<a id="known-gaps"></a>
## 补充准备

YAML 的 ONNX 输入引用 `../../../01_common/model_zoo/mapper/classification/FastViT/...`。为所选 S12/SA12/T12/T8 变体在该路径准备对应 ONNX 图，或将 `onnx_model` 改为本地图路径。按 YAML 归一化准备 `./calibration_data_rgb_f32` RGB float32 校准数据。四份 YAML 共用 `output_model_file_prefix: 'FastViT_224x224_nv12'`；各变体在独立工作目录构建，部署时使用对应 Manifest 文件名。运行上文匹配的 checker/编译命令。
