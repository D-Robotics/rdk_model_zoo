[English](README.md) | 简体中文

# FasterNet 转换

本目录提供转换资产：四份参考 PTQ YAML
（`FasterNet_{S,T0,T1,T2}_config.yaml`）。目录**不带导出脚本和校准
数据产出脚本**；编译前按 YAML 路径准备 ONNX 图与校准数据，详细步骤见
[补充准备](#known-gaps)。

四份 YAML 共用输出前缀 `FasterNet_224x224_nv12`，而 `working_dir`
互不一致（S 用 `model_output`，T0 加 `_mix` 后缀，T1/T2 用裸前缀）；
各变体在独立目录构建。

<a id="source-model"></a>
## 源模型

FasterNet S/T0/T1/T2（论文 [Run, Don't Walk: Chasing Higher FLOPS for
Faster Neural Networks](https://arxiv.org/abs/2303.03667)）。各 YAML 要求
`./fasternet_{s,t0,t1,t2}.onnx`；导出时使用匹配的模型权重，并将图保存至对应 YAML 路径。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── FasterNet_S_config.yaml  # 配置
├── FasterNet_T0_config.yaml  # 配置
├── FasterNet_T1_config.yaml  # 配置
├── FasterNet_T2_config.yaml  # 配置
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>
## 工具链与目标

模型转换在 x86 Linux 主机的 RDK X5 OpenExplorer Docker 内执行
（march `bayes-e`）。请准备含 `hb_mapper`、`hb_perf`、`hrt_model_exec` 的工具链；离线 Docker 镜像可从地瓜机器人开发者论坛（[topic 35229](https://forum.d-robotics.cc/t/topic/35229)）获取。

<a id="export"></a>
## 导出

目录不含导出脚本。原始 FasterNet 流程使用官方源码导出 ONNX：

1. 从参考仓库获取官方 FasterNet 源码与预训练权重。
2. 创建目标 FasterNet 模型，如 `fasternet_t0`、`fasternet_t1`、
   `fasternet_t2`、`fasternet_s`。
3. 用 `1x3x224x224` 虚拟输入，通过 `torch.onnx.export` 导出模型。
4. 用 `onnxsim.simplify` 化简 ONNX 模型。
5. 在 OE 环境中编译化简后的 ONNX 模型（见"编译"）。

重新生成 ONNX 输入需按上述流程执行；
产物须命名为 `fasternet_<variant>.onnx`（小写，与 YAML 一致）并放在
本目录（或调整 YAML 的 `onnx_model`）。

### ONNX 导出示例

在 x86 导出环境中运行以下示例。需要 PyTorch；使用简化步骤时还需安装 `onnxsim`。导出文件名与所选 YAML 的 `onnx_model` 必须对应。

onnx 模型使用的是 FasterNet 模型源码进行转换的，使用以下命令安装所需要的包：

```shell
pip install timm onnx
```

模型源码下载使用以下命令：

```shell
git clone https://github.com/JierunChen/FasterNet.git
```

模型转换以 fasternet_t2 为例，其余三个模型同理：

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify

from models.fasternet import *

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
    model_path = "fasternet_t2-epoch.289-val_acc1.78.8860.pth"
    model = fasternet_t2()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # print(model)

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "fasternet_t2.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=11,
        verbose=True,
        input_names=["data"],  # 输入名
        output_names=["output"],  # 输出名
    )

    # Simplify the ONNX model
    model_simp, check = simplify(onnx_file_path)

    if check:
        print("Simplified model is valid.")
        simplified_onnx_file_path = "fasternet_t2.onnx"
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

在 OE 环境内、两个输入就绪后（以 S 为例；其余变体换用各自配置）：

```bash
# cwd：本转换目录
# 输入：./fasternet_s.onnx + ./calibration_data_rgb_f32
# 输出：working_dir 'model_output'（S）/ 'FasterNet_224x224_nv12_mix'（T0）
#       / 'FasterNet_224x224_nv12'（T1/T2），产出
#       FasterNet_224x224_nv12.bin——需重命名，见缺口
hb_mapper checker --config FasterNet_S_config.yaml
hb_mapper makertbin --config FasterNet_S_config.yaml
```

四份 YAML 均设 `compile_mode: 'latency'` / `optimize_level: 'O3'`。仅
T0 配置通过 `node_info` 将节点以 int16 I/O 部署到 BPU（2 处 partial-conv
相关摆放）；S/T1/T2 无摆放。均不带 `debug_mode` 与
`set_all_nodes_int16`。

<a id="validation"></a>
## 验证

目录不含 x86 参考脚本。模型检查请按 OE 包使用 `hb_perf` 与
`hrt_model_exec`。板端功能检查即样例运行时：
`python3 samples/vision/fasternet/runtime/python/main.py --target x5 --asset-id x5:fasternet:FasterNet_S_224x224_nv12.bin...`
（见 [runtime/python/README_cn.md](../runtime/python/README_cn.md)）。
运行时期望的输入张量为 NV12 打包前的 `1x3x224x224`，输出为
ImageNet-1k 分类 logits。

<a id="artifacts"></a>
## 配方文件

四份参考 YAML 即本目录的转换资产；其 SHA-256 如下，可用于核对本地副本：

| 文件 | SHA-256 |
| --- | --- |
| `FasterNet_S_config.yaml` | `f0455d5ec5b1c2b4d63f5c153b14060a1b3c17d9b02f3fbfab848239a040e867` |
| `FasterNet_T0_config.yaml` | `c62dd1daedf245e826dcea215ac7adec4dddc5b654b3092c6c44be67d93b371a` |
| `FasterNet_T1_config.yaml` | `e4a123c23edeb38e6835215ea814a1997082bcc0889ea96dfd93fe8b703a0f7a` |
| `FasterNet_T2_config.yaml` | `ad65f79a6e74191d17da60b727416f9c824e8597cfa4fc4d0112597aae1dd944` |

<a id="known-gaps"></a>
## 补充准备

为所选 `fasternet_<variant>.onnx` 准备对应模型图，并在 `./calibration_data_rgb_f32` 准备 RGB float32 校准数据，使用 YAML 的 224x224 输入与归一化值。四份 YAML 共用 `output_model_file_prefix: 'FasterNet_224x224_nv12'`；`working_dir` 分别为 S 的 `model_output`、T0 的 `FasterNet_224x224_nv12_mix`、T1/T2 的 `FasterNet_224x224_nv12`。各变体在独立目录构建，并以对应 Manifest 文件名保存产物。使用匹配 YAML 运行上文 OE 命令。

<!-- converted-batch-20261102 -->
## 固定版本转换配方

[Shared export/calibration instructions](../../../../utils/tools/classification_conversion/README.md).

```bash
python3 utils/tools/classification_conversion/export.py --model fasternet_t0.in1k --checkpoint /path/to/pinned/model.safetensors --calibration-sample /path/to/run/calibration_rgb_f32/0000.rgb --output /path/to/run/model.onnx
cp samples/vision/fasternet/conversion/recipes/timm-t0-in1k_x5.yaml /path/to/run/compile.yaml
# In the X5 OE container, cwd /path/to/run:
hb_mapper checker --model model.onnx --model-type onnx --march bayes-e
hb_mapper makertbin --model-type onnx --config compile.yaml
```

各规格使用 converted.json 中对应的模型、固定权重及 recipes 配方，输出到全新目录。校准为原始 RGB float32 二进制张量，不重复归一化，不套用历史板端数据。
<!-- /converted-batch-20261102 -->

<!-- converted-all-20261103 -->

## 本轮 S 系列转换规格（2026-11-03）

每个目标使用自己的 PTQ、BC、完整 NodeInfo 和最终 OE。内部 INT16 配置不能替代实际公共 dtype；本轮按已读取的 float32 输出绑定。具体配方及 SHA 见 [converted_targets_20261103.json](converted_targets_20261103.json)。

- `s100 / timm-s-in1k-s-int16-o1-20261103 / model`: [timm_s_in1k_s_int16_o1_20261103_s100_model.yaml](recipes/timm_s_in1k_s_int16_o1_20261103_s100_model.yaml)
- `s100p / timm-s-in1k-s-int16-o1-20261103 / model`: [timm_s_in1k_s_int16_o1_20261103_s100p_model.yaml](recipes/timm_s_in1k_s_int16_o1_20261103_s100p_model.yaml)
- `s600 / timm-s-in1k-s-int16-o1-20261103 / model`: [timm_s_in1k_s_int16_o1_20261103_s600_model.yaml](recipes/timm_s_in1k_s_int16_o1_20261103_s600_model.yaml)
- `s100 / timm-t0-in1k-s-int16-o1-20261103 / model`: [timm_t0_in1k_s_int16_o1_20261103_s100_model.yaml](recipes/timm_t0_in1k_s_int16_o1_20261103_s100_model.yaml)
- `s100p / timm-t0-in1k-s-int16-o1-20261103 / model`: [timm_t0_in1k_s_int16_o1_20261103_s100p_model.yaml](recipes/timm_t0_in1k_s_int16_o1_20261103_s100p_model.yaml)
- `s600 / timm-t0-in1k-s-int16-o1-20261103 / model`: [timm_t0_in1k_s_int16_o1_20261103_s600_model.yaml](recipes/timm_t0_in1k_s_int16_o1_20261103_s600_model.yaml)
- `s100 / timm-t1-in1k-s-int16-o1-20261103 / model`: [timm_t1_in1k_s_int16_o1_20261103_s100_model.yaml](recipes/timm_t1_in1k_s_int16_o1_20261103_s100_model.yaml)
- `s100p / timm-t1-in1k-s-int16-o1-20261103 / model`: [timm_t1_in1k_s_int16_o1_20261103_s100p_model.yaml](recipes/timm_t1_in1k_s_int16_o1_20261103_s100p_model.yaml)
- `s600 / timm-t1-in1k-s-int16-o1-20261103 / model`: [timm_t1_in1k_s_int16_o1_20261103_s600_model.yaml](recipes/timm_t1_in1k_s_int16_o1_20261103_s600_model.yaml)
- `s100 / timm-t2-in1k-s-int16-o1-20261103 / model`: [timm_t2_in1k_s_int16_o1_20261103_s100_model.yaml](recipes/timm_t2_in1k_s_int16_o1_20261103_s100_model.yaml)
- `s100p / timm-t2-in1k-s-int16-o1-20261103 / model`: [timm_t2_in1k_s_int16_o1_20261103_s100p_model.yaml](recipes/timm_t2_in1k_s_int16_o1_20261103_s100p_model.yaml)
- `s600 / timm-t2-in1k-s-int16-o1-20261103 / model`: [timm_t2_in1k_s_int16_o1_20261103_s600_model.yaml](recipes/timm_t2_in1k_s_int16_o1_20261103_s600_model.yaml)

S 校准必须使用 ONNX 输入域的归一化 `.npy`；NV12 Runtime 的 mean/scale 仍写在配方中。公共工具的 `--target` 和 `--normalization-json` 从同一 checkpoint 导出记录读取实际归一化。

[Shared classification workflow](../../../../utils/tools/classification_conversion/README.md).

原始浮点门禁的阈值、真实输入和 checkpoint 绑定保持不变。编译估算及首条校准输入的 NodeInfo 诊断不作为数据集精度或板端性能。
