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
（march `bayes-e`），从不在板上运行。请准备含 `hb_mapper`、`hb_perf`、`hrt_model_exec` 的工具链；离线 Docker 镜像可从地瓜机器人开发者论坛（[topic 35229](https://forum.d-robotics.cc/t/topic/35229)）获取。

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
T0 配置通过 `node_info` 将节点以 int16 I/O 摆上 BPU（2 处 partial-conv
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
