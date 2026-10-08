[English](README.md) | 简体中文

# RepVGG 转换

<a id="source-model"></a>
## 源模型

按官方 RepVGG 流程加载训练权重，以 `create_RepVGG_B1g2(deploy=False)` 创建模型，并在 ONNX 导出前运行 `repvgg_model_convert`。构建时记录源码修订、PyTorch 版本与权重摘要。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── RepVGG_A0_config.yaml  # 配置
├── RepVGG_A1_config.yaml  # 配置
├── RepVGG_A2_config.yaml  # 配置
├── RepVGG_B0_config.yaml  # 配置
├── RepVGG_B1g2_config.yaml  # 配置
└── RepVGG_B1g4_config.yaml  # 配置
```

<a id="toolchain-targets"></a>
## 工具链与目标

6 份 YAML 面向 X5 `march: bayes-e`。构建时记录使用的 OE 版本。

| Config | ONNX input path | Working directory | Compiled basename |
| --- | --- | --- | --- |
| `RepVGG_A0_config.yaml` | `./RepVGG-A0.onnx` | `RepVGG-A0_224x224_nv12` | `RepVGG-A0_224x224_nv12.bin` |
| `RepVGG_A1_config.yaml` | `./RepVGG-A1.onnx` | `RepVGG-A1_224x224_nv12` | `RepVGG-A1_224x224_nv12.bin` |
| `RepVGG_A2_config.yaml` | `./RepVGG-A2.onnx` | `RepVGG-A2_224x224_nv12` | `RepVGG-A2_224x224_nv12.bin` |
| `RepVGG_B0_config.yaml` | `./RepVGG-B0.onnx` | `RepVGG-B0_224x224_nv12` | `RepVGG-B0_224x224_nv12.bin` |
| `RepVGG_B1g2_config.yaml` | `./RepVGG-B1g2.onnx` | `RepVGG-B1g2_224x224_nv12` | `RepVGG-B1g2_224x224_nv12.bin` |
| `RepVGG_B1g4_config.yaml` | `./RepVGG-B1g4.onnx` | `RepVGG-B1g4_224x224_nv12` | `RepVGG-B1g4_224x224_nv12.bin` |


工具链资源:

- [OE Docker environment](https://forum.d-robotics.cc/t/topic/35229)

<a id="export"></a>
## ONNX 导出

按官方 RepVGG 流程加载训练权重，以 `create_RepVGG_B1g2(deploy=False)` 创建模型，并在 ONNX 导出前运行 `repvgg_model_convert`。构建时记录源码修订、PyTorch 版本与权重摘要。

使用下面的 ONNX 导出示例，在上表路径准备匹配图，名义输入 RGB NCHW 1×3×224×224、输出 ImageNet-1k。YAML input_shape/input_name 为空，维度与名称从图读取，必须核对。

### ONNX 导出示例

在 x86 导出环境中运行以下示例。需要 PyTorch；使用简化步骤时还需安装 `onnxsim`。导出文件名与所选 YAML 的 `onnx_model` 必须对应。

onnx 模型使用的是 RepVGG 模型源码进行转换的，使用以下命令安装所需要的包：

```shell
pip install timm onnx
```

在 Model Zoo 仓库根目录下载 RepVGG 源码：

```shell
git clone https://github.com/DingXiaoH/RepVGG.git
```

从 [RepVGG 官方仓库](https://github.com/DingXiaoH/RepVGG)的预训练模型列表获取 `RepVGG-B1g2-train.pth`，放入刚克隆的 `RepVGG/` 目录；也可将下面的 `model_path` 改为权重绝对路径。将以下代码保存为 `RepVGG/export_onnx.py`。

模型转换以 RepVGG-B1g2 为例，其余五个模型需选择对应的构造函数和权重：

```Python
from repvgg import *
import torch
import onnx
import torch.onnx

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
    model_path = "RepVGG-B1g2-train.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_RepVGG_B1g2(deploy=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = repvgg_model_convert(model)

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "RepVGG-B1g2.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=11,
        verbose=True,
        input_names=["data"],  # input name
        output_names=["output"],  # output name
    )

    param_count = count_parameters(onnx_file_path)
    print(f'Total number of parameters: {param_count}')
```

在 Model Zoo 仓库根目录执行导出，再将生成的 ONNX 放入转换目录：

```bash
cd RepVGG
python3 export_onnx.py
cp RepVGG-B1g2.onnx ../samples/vision/repvgg/conversion/
cd ..
```

<a id="calibration"></a>
## 校准

所有 YAML 需要 `./calibration_data_rgb_f32`（float32、校准 default），训练输入 RGB/NCHW、运行输入 NV12。mean 为 123.675/116.28/103.53，scale 为 0.01712475/0.017507/0.01742919。缺少数据集选择/数量、准备脚本及实际校准文件。须核对浮点语义与图和归一化，不能给 NV12 目录改名冒充 RGB 校准数据。

<a id="compile"></a>
## 编译

在 OE 环境内准备对应变体的 ONNX 图与校准数据。以下 A0 示例使用 `RepVGG-A0.onnx`；编译上面的 B1g2 导出文件时，将 checker 的模型路径设为 `./RepVGG-B1g2.onnx`，makertbin 的配置设为 `RepVGG_B1g2_config.yaml`：

```bash
# cwd: repository root, then conversion directory
cd samples/vision/repvgg/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./RepVGG-A0.onnx
hb_mapper makertbin --model-type onnx --config RepVGG_A0_config.yaml
```

该配置预期输出为 `RepVGG-A0_224x224_nv12/RepVGG-A0_224x224_nv12.bin`，各配置均使用 latency/O3。各变体目录/前缀不同，形式如 `RepVGG-A0`。编译文件名用连字符，而发布文件名用下划线。每份 YAML 还配置删除 Quantize/Dequantize/Transpose/Cast/Reshape 节点并启用 dump_calibration_data；这些变换后须验证图语义，改文件名不能证明等价。

<a id="validation"></a>
## 转换后验证

推理前核对 packed NV12 几何 224×224、squeeze 后为 (1000,) 的 F32
分数输出。首个变体示例：

```bash
# cwd: repository root on X5
python3 samples/vision/repvgg/runtime/python/main.py --target x5 \
  --asset-id x5:repvgg:RepVGG_A0_224x224_nv12.bin \
  --model-path samples/vision/repvgg/conversion/RepVGG-A0_224x224_nv12/RepVGG-A0_224x224_nv12.bin \
  --test-img samples/vision/repvgg/test_data/gooze.JPEG
```

准确引用用于选择契约，不证明重建字节等于发布字节。记录新哈希和图来源，对照源/统一输出，交付前评估精度。

<a id="artifacts"></a>
## 产物

编译路径见上表，逐变体发布文件与目标见[模型准备](../model/README_cn.md#artifacts)，下载落在 sample model 目录。移动已验证构建时保留变体与来源，改名本身不是修复。

<a id="known-gaps"></a>
## 补充准备

加载匹配的 RepVGG 训练权重，以 `deploy=False` 创建所选模型，并在 ONNX 导出前运行 `repvgg_model_convert`。将模型图导出至所选 YAML 路径，输入 RGB/NCHW 1×3×224×224、输出 ImageNet-1k；编译前检查输入名称与形状。按 mean `123.675/116.28/103.53` 和 scale `0.01712475/0.017507/0.01742919` 准备 `./calibration_data_rgb_f32` float32 RGB 数据。各 YAML 使用变体专属目录；部署文件按 Manifest 文件名保存。
