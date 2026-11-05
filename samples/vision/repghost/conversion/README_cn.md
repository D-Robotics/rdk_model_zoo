[English](README.md) | 简体中文

# RepGhost 转换

<a id="source-model"></a>
## 源模型

源说明基于 PyTorch/timm RepGhost 变体，但未固定 timm/torch 版本、权重修订或权重摘要。按下面的 ONNX 示例导出匹配权重，编译前核对所选 YAML。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── RepGhost_100.yaml  # 配置
├── RepGhost_111.yaml  # 配置
├── RepGhost_130.yaml  # 配置
├── RepGhost_150.yaml  # 配置
└── RepGhost_200.yaml  # 配置
```

<a id="toolchain-targets"></a>
## 工具链与目标

五份 YAML 均面向 X5，使用 `march: bayes-e`。构建时记录使用的 OE 版本。

| Variant | Config | Required ONNX | Published filename |
| --- | --- | --- | --- |
| `100` | `RepGhost_100.yaml` | `./repghostnet_100.onnx` | `RepGhost_100_224x224_nv12.bin` |
| `111` | `RepGhost_111.yaml` | `./repghostnet_111.onnx` | `RepGhost_111_224x224_nv12.bin` |
| `130` | `RepGhost_130.yaml` | `./repghostnet_130.onnx` | `RepGhost_130_224x224_nv12.bin` |
| `150` | `RepGhost_150.yaml` | `./repghostnet_150.onnx` | `RepGhost_150_224x224_nv12.bin` |
| `200` | `RepGhost_200.yaml` | `./repghostnet_200.onnx` | `RepGhost_200_224x224_nv12.bin` |


工具链资源:

- [OE Docker environment](https://forum.d-robotics.cc/t/topic/35229)

<a id="export"></a>
## ONNX 导出

检入材料不足以给出经过验证的导出命令。需准备匹配变体的 RGB NCHW ONNX（名义输入 1×3×224×224），先核对真实输入输出。YAML 的 `input_shape` 和 `input_name` 为空，实际从图读取，并未强制限定 224。按表格把图放在对应 YAML 旁。

### ONNX 导出示例

在 x86 导出环境中运行以下示例。需要 PyTorch；使用简化步骤时还需安装 `onnxsim`。导出文件名与所选 YAML 的 `onnx_model` 必须对应。

onnx 模型使用的是 timm 库 (PyTorch Image Models) 中的模型进行转换的，使用以下命令安装所需要的包：

```shell
pip install timm onnx
```

模型转换以 repghostnet_100 为例，其余四个模型同理：

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify
from timm.models import create_model

from timm.models.repghost import repghostnet_100, repghostnet_111, repghostnet_130, repghostnet_150, repghostnet_200

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
    model = create_model('repghostnet_100', pretrained=True)
    model.eval()

    # print the model structure

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "repghostnet_100.onnx"

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
        simplified_onnx_file_path = "repghostnet_100.onnx"
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

各配置读取 `./calibration_data_rgb_f32`，数据类型为 float32，校准算法为 default，训练输入为 RGB/NCHW，运行输入为 NV12。mean 为 123.675/116.28/103.53，scale 为 0.01712475/0.017507/0.01742919。需自行选择校准图片并生成符合图和 YAML 归一化要求的 RGB 浮点数据；本目录未提供校准预处理脚本或数据。

<a id="compile"></a>
## 编译

在 OE 环境内、补齐 ONNX 与 RGB float32 校准数据后执行（cwd 为本
conversion 目录）。以 100 为例：

```bash
cd samples/vision/repghost/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./repghostnet_100.onnx
hb_mapper makertbin --model-type onnx --config RepGhost_100.yaml
```

所有配置的 working_dir 为 `RepGhost_224x224_nv12`，输出前缀也为 `RepGhost_224x224_nv12`，均不含变体。预期 bin 为 `RepGhost_224x224_nv12/RepGhost_224x224_nv12.bin`。各变体必须隔离工作目录或在下一次构建前保留输出。编译选项为 latency/O3 并启用 dump_calibration_data，不能让多个变体并发写同一目录。

<a id="validation"></a>
## 转换后验证

先核对 224 几何、packed NV12、squeeze 后 (1000,) 的单 F32 输出及分数语义，随后可用准确引用与独立路径运行重建 100 制品：

```bash
# cwd: repository root on X5
python3 samples/vision/repghost/runtime/python/main.py --target x5 \
  --asset-id x5:repghost:RepGhost_100_224x224_nv12.bin \
  --model-path samples/vision/repghost/conversion/RepGhost_224x224_nv12/RepGhost_224x224_nv12.bin \
  --test-img samples/vision/repghost/test_data/ibex.JPEG
```

选择与模型匹配的目标和引用，再将其输出和数据集精度与浮点模型比较。

<a id="artifacts"></a>
## 产物

各变体的部署文件名见上表，下载目录为 `samples/vision/repghost/model/`。编译结果使用相同基名；为各变体使用独立工作目录，并按对应的部署文件名保存。

<a id="known-gaps"></a>
## 补充准备

每个变体选择对应的 PyTorch/timm 权重并记录其修订与摘要。在对应 YAML 旁导出 RGB/NCHW 模型图；由于 `input_shape` 和 `input_name` 为空，应检查实际输入输出 元数据。按 float32 RGB、`cal_data_type: float32`、`calibration: default`、mean `123.675/116.28/103.53` 与 scale `0.01712475/0.017507/0.01742919` 准备 `./calibration_data_rgb_f32`。每份 YAML 都写入 `RepGhost_224x224_nv12/RepGhost_224x224_nv12.bin`，各变体应使用独立工作区构建。

<!-- converted-batch-20261102 -->
## 固定版本转换配方

[Shared export/calibration instructions](../../../../utils/tools/classification_conversion/README.md).

```bash
python3 utils/tools/classification_conversion/export.py --model repghostnet_100.in1k --checkpoint /path/to/pinned/model.safetensors --calibration-sample /path/to/run/calibration_rgb_f32/0000.rgb --output /path/to/run/model.onnx
cp samples/vision/repghost/conversion/recipes/timm-100-in1k_x5.yaml /path/to/run/compile.yaml
# In the X5 OE container, cwd /path/to/run:
hb_mapper checker --model model.onnx --model-type onnx --march bayes-e
hb_mapper makertbin --model-type onnx --config compile.yaml
```

各规格使用 converted.json 中对应的模型、固定权重及 recipes 配方，输出到全新目录。校准为原始 RGB float32 二进制张量，不重复归一化，不套用历史板端数据。
<!-- /converted-batch-20261102 -->

<!-- converted-all-20261103 -->

## 本轮 S 系列转换规格（2026-11-03）

每个目标使用自己的 PTQ、BC、完整 NodeInfo 和最终 OE。内部 INT16 配置不能替代实际公共 dtype；本轮按已读取的 float32 输出绑定。具体配方及 SHA 见 [converted_targets_20261103.json](converted_targets_20261103.json)。

- `s100 / timm-100-in1k-s-int16-o1-20261103 / model`: [timm_100_in1k_s_int16_o1_20261103_s100_model.yaml](recipes/timm_100_in1k_s_int16_o1_20261103_s100_model.yaml)
- `s100p / timm-100-in1k-s-int16-o1-20261103 / model`: [timm_100_in1k_s_int16_o1_20261103_s100p_model.yaml](recipes/timm_100_in1k_s_int16_o1_20261103_s100p_model.yaml)
- `s600 / timm-100-in1k-s-int16-o1-20261103 / model`: [timm_100_in1k_s_int16_o1_20261103_s600_model.yaml](recipes/timm_100_in1k_s_int16_o1_20261103_s600_model.yaml)
- `s100 / timm-111-in1k-s-int16-o1-20261103 / model`: [timm_111_in1k_s_int16_o1_20261103_s100_model.yaml](recipes/timm_111_in1k_s_int16_o1_20261103_s100_model.yaml)
- `s100p / timm-111-in1k-s-int16-o1-20261103 / model`: [timm_111_in1k_s_int16_o1_20261103_s100p_model.yaml](recipes/timm_111_in1k_s_int16_o1_20261103_s100p_model.yaml)
- `s600 / timm-111-in1k-s-int16-o1-20261103 / model`: [timm_111_in1k_s_int16_o1_20261103_s600_model.yaml](recipes/timm_111_in1k_s_int16_o1_20261103_s600_model.yaml)
- `s100 / timm-130-in1k-s-int16-o1-20261103 / model`: [timm_130_in1k_s_int16_o1_20261103_s100_model.yaml](recipes/timm_130_in1k_s_int16_o1_20261103_s100_model.yaml)
- `s100p / timm-130-in1k-s-int16-o1-20261103 / model`: [timm_130_in1k_s_int16_o1_20261103_s100p_model.yaml](recipes/timm_130_in1k_s_int16_o1_20261103_s100p_model.yaml)
- `s600 / timm-130-in1k-s-int16-o1-20261103 / model`: [timm_130_in1k_s_int16_o1_20261103_s600_model.yaml](recipes/timm_130_in1k_s_int16_o1_20261103_s600_model.yaml)
- `s100 / timm-150-in1k-s-int16-o1-20261103 / model`: [timm_150_in1k_s_int16_o1_20261103_s100_model.yaml](recipes/timm_150_in1k_s_int16_o1_20261103_s100_model.yaml)
- `s100p / timm-150-in1k-s-int16-o1-20261103 / model`: [timm_150_in1k_s_int16_o1_20261103_s100p_model.yaml](recipes/timm_150_in1k_s_int16_o1_20261103_s100p_model.yaml)
- `s600 / timm-150-in1k-s-int16-o1-20261103 / model`: [timm_150_in1k_s_int16_o1_20261103_s600_model.yaml](recipes/timm_150_in1k_s_int16_o1_20261103_s600_model.yaml)
- `s100 / timm-200-in1k-s-int16-o1-20261103 / model`: [timm_200_in1k_s_int16_o1_20261103_s100_model.yaml](recipes/timm_200_in1k_s_int16_o1_20261103_s100_model.yaml)
- `s100p / timm-200-in1k-s-int16-o1-20261103 / model`: [timm_200_in1k_s_int16_o1_20261103_s100p_model.yaml](recipes/timm_200_in1k_s_int16_o1_20261103_s100p_model.yaml)
- `s600 / timm-200-in1k-s-int16-o1-20261103 / model`: [timm_200_in1k_s_int16_o1_20261103_s600_model.yaml](recipes/timm_200_in1k_s_int16_o1_20261103_s600_model.yaml)

S 校准必须使用 ONNX 输入域的归一化 `.npy`；NV12 Runtime 的 mean/scale 仍写在配方中。公共工具的 `--target` 和 `--normalization-json` 从同一 checkpoint 导出记录读取实际归一化。

[Shared classification workflow](../../../../utils/tools/classification_conversion/README.md).

原始浮点门禁的阈值、真实输入和 checkpoint 绑定保持不变。编译估算及首条校准输入的 NodeInfo 诊断不作为数据集精度或板端性能。
