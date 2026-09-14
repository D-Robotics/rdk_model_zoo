# 模型转换

本目录提供 MODNet 人像抠图样例在 `RDK X5` 平台上的模型转换资产。

当前样例运行时使用 `.bin` 模型，并通过 `hbm_runtime` 执行推理。如果只需要直接运行推理，请参考 [`../model/README_cn.md`](../model/README_cn.md) 下载预编译模型。本说明仅面向需要从 ONNX 模型重新生成部署模型的场景。

## 目录结构

```text
.
├── onnx_export                     # ONNX 导出脚本
├── ptq_yamls                       # PTQ 配置 YAML 文件
├── README.md
└── README_cn.md
```

## 环境准备

开始转换前请准备：

1. 安装带有 `hb_mapper`、`hb_perf`、`hrt_model_exec` 的 RDK X5 OpenExplorer 工具链。
2. 准备 MODNet ONNX 模型。
3. 准备 PTQ 校准数据。

## 准备 ONNX

MODNet 的原始模型来源：

- 论文: [Is a Green Screen Really Necessary for Real-Time Portrait Matting?](https://arxiv.org/abs/2011.11961)
- 官方实现: [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet)

需要先准备对应的 ONNX 模型。请使用官方 MODNet 工程或用户自有导出流程生成 ONNX，然后在执行 `hb_mapper` 之前更新目标 YAML 中的 `onnx_model` 字段。

## PTQ 转换

建议先使用 `hb_mapper checker` 检查 ONNX：

```bash
hb_mapper checker --config modnet.yaml
```

检查通过后执行模型编译：

```bash
hb_mapper makertbin --config modnet.yaml
```

## 转换后验证

模型转换完成后，可使用 `hb_perf` 查看性能：

```bash
hb_perf model_perf \
    --model ./modnet_512x512_rgb.bin \
    --input-shape input 1x3x512x512
```

也可以使用 `hrt_model_exec` 做基础运行验证：

```bash
hrt_model_exec perf \
    --model_file ./modnet_512x512_rgb.bin \
    --thread_num 1
```

## 运行时协议

生成的部署模型默认遵循以下协议：

- 输入 tensor 类型: `Float32 NCHW RGB`
- 输入分辨率: `512x512`
- 输入归一化方式: `(pixel - 127.5) / 127.5`（范围 [-1, 1]）
- 输出 tensor 形状: `1x1x512x512`
- 输出 tensor 类型: `F32`（alpha matte，范围 [0, 1]）

该协议与 [`../runtime/python/README_cn.md`](../runtime/python/README_cn.md) 中的 Python 推理接口保持一致。
