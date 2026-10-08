[English](README.md) | 简体中文

# EfficientNet 转换

S 配方可直接使用下文的导出脚本、校准脚本与逐变体 YAML。
X5 B2/B3/B4 需先准备匹配的 ONNX 图与[补充准备](#known-gaps)
列出的校准输入，再用对应 X5 YAML 编译。

<a id="source-model"></a>
## 源模型

- S（lite0..lite4）：经 `timm` 导出的 EfficientNet-Lite 权重
  （`tf_efficientnet_lite0.in1k`.. `tf_efficientnet_lite4.in1k`），即
  TensorFlow TPU EfficientNet-Lite 系列
  （<https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>）。
- X5（b2/b3/b4）：EfficientNet B2/B3/B4。使用 timm 的
  `create_model → torch.onnx.export → onnxsim.simplify` 导出所选变体，
  并记录 ONNX 图使用的权重修订。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── EfficientNet_B2_config.yaml  # 配置
├── EfficientNet_B3_config.yaml  # 配置
├── EfficientNet_B4_config.yaml  # 配置
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── efficientnet_lite0_config.yaml  # 配置
├── efficientnet_lite1_config.yaml  # 配置
├── efficientnet_lite2_config.yaml  # 配置
├── efficientnet_lite3_config.yaml  # 配置
├── efficientnet_lite4_config.yaml  # 配置
├── get_calibration_data.py  # Python 脚本
├── get_efficientnet_lite0_onnx.py  # Python 脚本
├── get_efficientnet_lite1_onnx.py  # Python 脚本
├── get_efficientnet_lite2_onnx.py  # Python 脚本
├── get_efficientnet_lite3_onnx.py  # Python 脚本
├── get_efficientnet_lite4_onnx.py  # Python 脚本
├── timm2onnx_local.py  # Python 脚本
└── x86_inference.py  # Python 脚本
```

<a id="toolchain-targets"></a>
## 工具链与目标

在 x86 Linux 主机上的目标平台 OpenExplorer Docker 内执行模型转换。

- S100：march `nash-e`（交付 YAML 中的取值）。
- S600：同一配置将 march 改为 `nash-p`（改 YAML 或传工具链的 march
  覆盖参数）；量化配置其余不变。
- X5：march `bayes-e`。

- OE 资源入口（Docker + 开发包）：
  <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE 工具链手册：<https://toolchain.d-robotics.cc/>


工具链资源:

- [OE Docker environment](https://forum.d-robotics.cc/t/topic/35229)

<a id="export"></a>
## 导出（S 配方）

cwd：本 `conversion/` 目录。先安装导出依赖（在合适的 Python 3 环境中
`pip install timm onnx onnxsim`），再运行对应变体的导出脚本，例如：

```bash
# 输入：timm 权重 tf_efficientnet_lite0.in1k（由 timm 下载）
# 输出：./tf_efficientnet_lite0.onnx（opset 11，经 onnxsim 化简）
# 成功判据：脚本打印参数量并输出 "Simplified model is valid."
python3 get_efficientnet_lite0_onnx.py
```

| 变体 | 导出脚本 | ONNX 文件 | 前缀（== Manifest 文件名） | 输入 |
| --- | --- | --- | --- | --- |
| lite0 | `get_efficientnet_lite0_onnx.py` | `tf_efficientnet_lite0.onnx` | `efficientnet_lite0_224x224_nv12` | 224x224 |
| lite1 | `get_efficientnet_lite1_onnx.py` | `tf_efficientnet_lite1.onnx` | `efficientnet_lite1_240x240_nv12` | 240x240 |
| lite2 | `get_efficientnet_lite2_onnx.py` | `tf_efficientnet_lite2.onnx` | `efficientnet_lite2_260x260_nv12` | 260x260 |
| lite3 | `get_efficientnet_lite3_onnx.py` | `tf_efficientnet_lite3.onnx` | `efficientnet_lite3_300x300_nv12` | 300x300 |
| lite4 | `get_efficientnet_lite4_onnx.py` | `tf_efficientnet_lite4.onnx` | `efficientnet_lite4_380x380_nv12` | 380x380 |

`timm2onnx_local.py` 是替代辅助脚本：从本地下载的权重文件
（而非 timm hub）导出；需按目标变体修改其中的 `model_name`。

### ONNX 导出示例

在 x86 导出环境中运行以下示例。需要 PyTorch；使用简化步骤时还需安装 `onnxsim`。导出文件名与所选 YAML 的 `onnx_model` 必须对应。

onnx 模型使用的是 timm 库 (PyTorch Image Models) 中的模型进行转换的，使用以下命令安装所需要的包：

```shell
pip install timm onnx
```

模型转换以 efficientnet_b2 为例，其余两个模型同理：

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify
from timm.models import create_model

from timm.models.efficientnet import efficientnet_b2, efficientnet_b3, efficientnet_b4

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
    model = create_model('efficientnet_b2', pretrained=True)
    model.eval()

    # print the model structure

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "efficientnet_b2.onnx"

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
        simplified_onnx_file_path = "efficientnet_b2.onnx"
        onnx.save(model_simp, simplified_onnx_file_path)
        print(f"Simplified model saved to {simplified_onnx_file_path}")
    else:
        print("Simplified model is invalid!")

    onnx_model_path = simplified_onnx_file_path  # Replace with your ONNX model path
    total_params = count_parameters(onnx_model_path)
    print(f"Total number of parameters in the model: {total_params}")
```

<a id="calibration"></a>
## 校准（S 配方）

cwd：本 `conversion/` 目录。一个脚本服务全部五个变体：

```bash
# 输入：ILSVRC2012_val_*.JPEG 图像（脚本使用排序后的前 100 张）
# 输出：./calibration_data_rgb/*.npy（float32），与 YAML 的 cal_data_dir 一致
# 成功判据：打印 "成功生成 ... 个校准数据文件"（列出 100 个文件）
python3 get_calibration_data.py
```

预处理链为 `ShortSideResize(224) → CenterCrop(224) → HWC2CHW →
Scale(255.0) → Mean([127,127,127]) → Scale(0.007843)`，已与每份 YAML 的
`mean_value: 127 127 127` / `scale_value: 0.007843 0.007843 0.007843`
一致——脚本与 YAML 数值吻合。

运行前需要知道两个配方事实：

1. 脚本默认的 `src_image_dir` 指向原 OpenExplorer 校准目录
   （`../../../open_explorer/samples/ai_toolchain/.../calibration_data/imagenet/`）。
   该相对路径在本仓库中**无法**解析——目录不存在。运行前请把
   `src_image_dir` 改为你自己的 ILSVRC2012 验证集 JPEG 目录。
2. resize/crop 对**所有**变体固定为 224（包括 240/260/300/380 的
   lite1..lite4）；配方对全部变体复用同一套校准数据，按交付原样。

<a id="compile"></a>
## 编译（S 配方）

cwd：本 `conversion/` 目录，位于 OE Docker 内，且已完成导出与校准：

```bash
# S100 构建（输入：./tf_efficientnet_lite0.onnx + ./calibration_data_rgb）
# 输出：./model_output/efficientnet_lite0_224x224_nv12.hbm
hb_compile --config efficientnet_lite0_config.yaml
```

S600 需先把 YAML 中 `march` 从 `nash-e` 改为 `nash-p`（或传工具链的
march 覆盖参数）。输出前缀与 Manifest 文件名完全一致，产出的 `.hbm`
无需改名即可复制到 `model/s100/` 或 `model/s600/`。只有修改源模型或
转换配置时才需要重新构建——运行时 sample 下载的是已发布制品。

X5 YAML 用对应的 X5 OE 流程编译（`hb_mapper`/`hb_compile` + 配置
文件）；未固定的部分见[补充准备](#known-gaps)第 5 条。

<a id="validation"></a>
## 验证

- `x86_inference.py`（S 配方）在 OE 环境内的 x86 主机上运行
  ONNX/HBIR/HBM 参考推理（导入 `horizon_tc_ui`）；用于把编译产物与
  ONNX 浮点模型对照。
- 板上功能检查走统一运行时：
  `python3 samples/vision/efficientnet/runtime/python/main.py --target s100 --asset-id s:efficientnet:s100/efficientnet_lite0_224x224_nv12.hbm...`
  （见 [runtime/python/README_cn.md](../runtime/python/README_cn.md)）。

<a id="artifacts"></a>
## 配方文件

本目录提供三份 X5 YAML、五份 S YAML、五个逐变体导出脚本、
校准脚本、`timm2onnx_local.py` 和 `x86_inference.py`，共 16 个配方文件。
S YAML 使用 `working_dir: './model_output'`、
`calibration_type: 'max'`、`optimize_level: 'O2'`；X5 YAML 使用
`calibration_type: 'default'`、`compile_mode: 'latency'`、
`optimize_level: 'O3'`，且 B2/B4 携带 B3 没有的 `node_info` int16
配置。

<a id="known-gaps"></a>
## 补充准备

X5 构建需准备 B2/B3/B4 ONNX 图及 `./calibration_data_rgb_f32` float32 RGB `.npy` 输入。按 YAML 数值处理：mean `123.675 116.28 103.53`、scale `0.01712475 0.017507 0.01742919`、输入 224x224。三份 X5 YAML 均使用 `output_model_file_prefix: 'EfficientNet_224x224_nv12'` 与 `debug_mode: 'dump_calibration_data'`；B3 使用 `working_dir: 'model_output'`，B2/B4 使用 `'EfficientNet_224x224_nv12'`。各变体在独立目录构建，并用对应 Manifest 文件名保存产物。使用上文 checker/编译命令和 X5 OE 工具链。S100/S600 按本指南前文完整 `hb_compile` 配方操作。
