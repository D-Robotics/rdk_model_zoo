# ConvNeXt 转换

本目录提供重建 ConvNeXt 样例 RDK X5 部署模型的转换资产：三份参考
PTQ YAML（`ConvNeXt_atto.yaml`、`ConvNeXt_femto.yaml`、
`ConvNeXt_nano.yaml`）。
Manifest 对应 `atto` 配置；`femto` 与 `nano` 是额外构建变体。编译前按各 YAML 路径准备匹配的 ONNX 图与校准数据，再运行下文 OE 命令。

<a id="source-model"></a>
## 源模型

ConvNeXt（论文 [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)，
参考实现 [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)）。
atto/femto/nano 尺寸来自官方 ConvNeXt 尺寸阶梯。编译前检查所选 YAML 的 `onnx_model` 与模型变体是否一致；当前文件值列在[补充准备](#known-gaps)。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── ConvNeXt_atto.yaml  # 配置
├── ConvNeXt_femto.yaml  # 配置
├── ConvNeXt_nano.yaml  # 配置
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>
## 工具链与目标

模型转换在 x86 Linux 主机的 RDK X5 OpenExplorer Docker 内执行
（march `bayes-e`），从不在板上运行。请准备含 `hb_mapper`、`hb_perf`、
`hrt_model_exec` 的工具链；离线 Docker 镜像可从地瓜机器人开发者论坛
（[topic 35229](https://forum.d-robotics.cc/t/topic/35229)）获取。

<a id="export"></a>
## 导出

目录不含 ONNX 导出脚本。重新生成 ONNX 输入需自行复现上游 ConvNeXt
导出（官方仓库或自有管线），再在运行 `hb_mapper` 前更新所选 YAML 的
`onnx_model` 字段。
注意三份 YAML 指向**互为错位**的文件：atto 配置消费
`./convnext_femto.onnx`，femto 配置指向
`../../../01_common/model_zoo/mapper/classification/ConvNeXt/convnext_atto.onnx`
（本 sample 目录之外的路径），nano 配置指向 `./convnext_pico.onnx`
（目录中并无该尺寸的其它文件）。

### ONNX 导出示例

在 x86 导出环境中运行以下示例。需要 PyTorch；使用简化步骤时还需安装 `onnxsim`。导出文件名与所选 YAML 的 `onnx_model` 必须对应。

onnx 模型使用的是 timm 库 (PyTorch Image Models) 中的模型进行转换的，使用以下命令安装所需要的包：

```shell
pip install timm onnx
```

模型转换以 convnext_atto 为例，其余三个模型同理：

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify
from timm.models import create_model

from timm.models.convnext import convnext_atto, convnext_femto, convnext_pico, convnext_tiny

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
    model = create_model('convnext_atto', pretrained=True)
    model.eval()

    # print the model structure

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "convnext_atto.onnx"

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
        simplified_onnx_file_path = "convnext_atto.onnx"
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

三份 YAML 均要求 `./calibration_data_rgb_f32`
（float32 RGB `.npy`）并使用 `calibration_type: 'default'`。等价数据必须
遵循 YAML 数值（mean `123.675 116.28 103.53`，scale `0.01712475
0.017507 0.01742919`，224x224）；这是声明的要求，不是已验证的管线。

<a id="compile"></a>
## 编译

在 OE 环境内、两个输入就绪后（以 atto 为例），先校验 ONNX 模型，再构建
部署模型：

```bash
# cwd：本转换目录
# 输入：该 YAML 的 onnx_model 目标（见缺口 1——atto 配置按交付原样指向
#       ./convnext_femto.onnx）+ ./calibration_data_rgb_f32
# 输出：working_dir 'ConvNeXt-deploy_224x224_nv12'，产出
#       ConvNeXt-deploy_224x224_nv12.bin——需重命名，见缺口 3
hb_mapper checker --config ConvNeXt_atto.yaml
hb_mapper makertbin --config ConvNeXt_atto.yaml
```

转换 femto/nano 时，对 `ConvNeXt_femto.yaml`、`ConvNeXt_nano.yaml`
重复同一流程。

三份 YAML 共享部署协议：`march: bayes-e`、运行时输入 `nv12`、训练输入
`rgb`（`NCHW` 布局）、归一化 `data_mean_and_scale`、前缀
`ConvNeXt-deploy_224x224_nv12`；均设 `compile_mode: 'latency'` /
`optimize_level: 'O3'`，并通过 `node_info` 将深度卷积/归一化节点以
int16 I/O 摆上 BPU（atto 8 处、femto 5 处、nano 8 处；无 Softmax 摆放
——ConvNeXt 没有注意力 softmax），均不带 `debug_mode` 与
`set_all_nodes_int16`。转换前请确认所选 YAML 的 `onnx_model`、
`cal_data_dir`、`working_dir`、`output_model_file_prefix`。

<a id="validation"></a>
## 验证

在 x86 主机上检查编译产物：

```bash
# cwd：本转换目录；输入：产出的 .bin（重命名为清单名见缺口 3）
hb_perf model_perf \
    --model ./ConvNeXt-deploy_224x224_nv12.bin \
    --input-shape data 1x3x224x224
```

```bash
# cwd：本转换目录
hrt_model_exec perf \
    --model_file ./ConvNeXt-deploy_224x224_nv12.bin \
    --thread_num 1
```

板端功能检查即样例运行时
（`python3 samples/vision/convnext/runtime/python/main.py --target x5
--asset-id x5:convnext:ConvNeXt_atto_224x224_nv12.bin ...`，见
[runtime/python/README_cn.md](../runtime/python/README_cn.md)）。期望的
运行时协议为 224x224 NV12 输入、F32 `1x1000x1x1` 输出。

<a id="artifacts"></a>
## 转换文件

三份 YAML 定义了上文列出的变体、输入路径与 X5 编译配置。

<a id="known-gaps"></a>
## 补充准备

编译前，将模型图与所选 YAML、变体一一对应：

- 当前 `onnx_model` 值为 atto → `./convnext_femto.onnx`、femto → `../../../01_common/model_zoo/mapper/classification/ConvNeXt/convnext_atto.onnx`、nano → `./convnext_pico.onnx`。导出或选择相应的 atto/femto/nano 模型图，并将所选 YAML 的 `onnx_model` 设置为该文件。
- 准备 `./calibration_data_rgb_f32` float32 RGB `.npy` 数据，按 `calibration_type: 'default'`、mean `123.675 116.28 103.53`、scale `0.01712475 0.017507 0.01742919` 和 224x224 输入处理。
- 每份 YAML 都使用 `output_model_file_prefix: 'ConvNeXt-deploy_224x224_nv12'`，产物为 `ConvNeXt-deploy_224x224_nv12.bin`。各变体使用独立输出目录，并在部署前将所选制品命名为 Manifest 文件名。Manifest 列出可下载的 atto 变体。
- 使用上文对应的 YAML、模型图和校准数据运行 OE checker 与编译命令。
