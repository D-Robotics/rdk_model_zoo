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

五份原样源 YAML 均面向 X5 `march: bayes-e`。构建时记录使用的 OE 版本。

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

所有配置读取 `./calibration_data_rgb_f32`，类型 float32，校准算法 default，训练输入 RGB/NCHW、运行输入 NV12。mean 为 123.675/116.28/103.53，scale 为 0.01712475/0.017507/0.01742919。缺少校准图选择、数量、预处理脚本和产物；这属于前提缺口，不是可执行准备配方。须确认浮点数据与图和 YAML 归一化一致，不能把 NV12 目录改名冒充 RGB 校准数据。

<a id="compile"></a>
## 编译

在 OE 环境内、补齐 ONNX 与 RGB float32 校准数据后执行（cwd 为本
conversion 目录）。以 100 为例：

```bash
cd samples/vision/repghost/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./repghostnet_100.onnx
hb_mapper makertbin --model-type onnx --config RepGhost_100.yaml
```

所有配置的 working_dir 为 `RepGhost_224x224_nv12`，输出前缀也为 `RepGhost_224x224_nv12`，均不含变体。预期 bin 为 `RepGhost_224x224_nv12/RepGhost_224x224_nv12.bin`。各变体必须隔离工作目录或在下一次构建前保留输出。源编译选项为 latency/O3 并启用 dump_calibration_data，不能让多个变体并发写同一目录。

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

引用只选择契约，不证明重建字节等于发布字节。单独记录重建哈希与来源，对照源实现并评估量化精度后再交付。

<a id="artifacts"></a>
## 产物

逐变体发布文件名见上表，下载落在 `samples/vision/repghost/model/`。编译在工作目录生成共同基名；移动已验证构建时须保留变体身份，单纯改名不能证明图等价或精度一致。

<a id="known-gaps"></a>
## 补充准备

每个变体选择对应的 PyTorch/timm 权重并记录其修订与摘要。在对应 YAML 旁导出 RGB/NCHW 模型图；由于 `input_shape` 和 `input_name` 为空，应检查实际输入输出 metadata。按 float32 RGB、`cal_data_type: float32`、`calibration: default`、mean `123.675/116.28/103.53` 与 scale `0.01712475/0.017507/0.01742919` 准备 `./calibration_data_rgb_f32`。每份 YAML 都写入 `RepGhost_224x224_nv12/RepGhost_224x224_nv12.bin`，各变体应使用独立工作区构建。
