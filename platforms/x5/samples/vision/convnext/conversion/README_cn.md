# 模型转换

本目录提供 ConvNeXt 图像分类样例在 `RDK X5` 平台上的模型转换资产。

当前样例运行时使用 `.bin` 模型，并通过 `hbm_runtime` 执行推理。如果只需要直接运行推理，请参考 [`../model/README_cn.md`](../model/README_cn.md) 下载预编译模型。本说明仅面向需要从 ONNX 模型重新生成部署模型的场景。

## 目录结构

```text
.
├── ConvNeXt_atto.yaml     # ConvNeXt Atto 的 PTQ 配置
├── ConvNeXt_femto.yaml    # ConvNeXt Femto 的 PTQ 配置
├── ConvNeXt_nano.yaml     # ConvNeXt Nano 的 PTQ 配置
├── README.md
└── README_cn.md
```

## 当前支持的模型变体

当前 conversion 资产覆盖以下 ConvNeXt 变体：

| YAML 文件 | 模型变体 | 运行时输入 | 目标平台 |
| --- | --- | --- | --- |
| `ConvNeXt_atto.yaml` | ConvNeXt Atto | `224x224 NV12` | `RDK X5 / bayes-e` |
| `ConvNeXt_femto.yaml` | ConvNeXt Femto | `224x224 NV12` | `RDK X5 / bayes-e` |
| `ConvNeXt_nano.yaml` | ConvNeXt Nano | `224x224 NV12` | `RDK X5 / bayes-e` |

## 环境准备

开始转换前请准备：

1. 安装带有 `hb_mapper`、`hb_perf`、`hrt_model_exec` 的 RDK X5 OpenExplorer 工具链。
2. 准备与目标变体对应的 ConvNeXt ONNX 模型。
3. 准备 PTQ 校准数据。当前 YAML 默认使用 `./calibration_data_rgb_f32` 下的 RGB float 校准数据。

## 准备 ONNX

ConvNeXt 的原始模型来源：

- 论文: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- 官方实现: [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

需要先准备目标变体对应的 ONNX 模型。当前 sample 仓库中没有提供 ONNX 导出脚本，请使用官方 ConvNeXt 工程或用户自有导出流程生成对应 ONNX，然后在执行 `hb_mapper` 之前更新目标 YAML 中的 `onnx_model` 字段。

## PTQ 转换

建议先使用 `hb_mapper checker` 检查 ONNX：

```bash
hb_mapper checker --config ConvNeXt_atto.yaml
```

检查通过后执行模型编译：

```bash
hb_mapper makertbin --config ConvNeXt_atto.yaml
```

转换 `ConvNeXt_femto` 或 `ConvNeXt_nano` 时，流程相同，只需切换为对应 YAML。

## YAML 配置说明

本目录下的 YAML 共享同一套部署协议：

- `march: bayes-e`
- 运行时输入类型: `nv12`
- 训练输入类型: `rgb`
- 训练布局: `NCHW`
- 归一化方式: `data_mean_and_scale`
- 输出模型前缀: `ConvNeXt-deploy_224x224_nv12`

正式转换前，建议确认所选 YAML 中的以下字段：

- `onnx_model`
- `cal_data_dir`
- `working_dir`
- `output_model_file_prefix`

## 转换后验证

模型转换完成后，可使用 `hb_perf` 查看性能：

```bash
hb_perf model_perf \
    --model ./ConvNeXt-deploy_224x224_nv12.bin \
    --input-shape data 1x3x224x224
```

也可以使用 `hrt_model_exec` 做基础运行验证：

```bash
hrt_model_exec perf \
    --model_file ./ConvNeXt-deploy_224x224_nv12.bin \
    --thread_num 1
```

## 运行时协议

生成的部署模型默认遵循以下协议：

- 输入 tensor 类型: `NV12`
- 输入分辨率: `224x224`
- 输出 tensor 形状: `1x1000x1x1`
- 输出 tensor 类型: `F32`

该协议与 [`../runtime/python/README_cn.md`](../runtime/python/README_cn.md) 中的 Python 推理接口保持一致。
