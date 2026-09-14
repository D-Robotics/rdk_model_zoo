[English](./README.md) | 简体中文

# SigLIP 模型转换说明

本 sample 使用预编译 SigLIP HBM 模型。SigLIP 视觉编码器的 LayerNorm 等结构对量化精度较敏感，当前公开内容以可直接部署的 RDK S100/S100P HBM 产物为主。

## 转换说明

SigLIP 视觉编码器基于 Google 在 HuggingFace 发布的权重进行量化和编译，模型面向 Nash 架构 BPU。当前 sample 不提供可复现的通用转换脚本；如需直接部署，请使用 [model/README_cn.md](../model/README_cn.md) 中列出的 `.hbm` 模型。

## OE 资源

- OE 资源入口（docker+OE开发包）：<https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE 工具链在线手册：<https://toolchain.d-robotics.cc/>

## 输入协议

- 输入名称：`_input_0`
- 输入格式：float32 NCHW RGB
- 输入范围：`[-1, 1]`
- 预处理：等比例缩放，使用 RGB `(127, 127, 127)` padding，再执行 `/127.5 - 1.0`

## 子模型协议

每个 HBM 文件包含两个固定子模型：

| 子模型 | 输出名称 | 说明 |
|---|---|---|
| `pooler_output` | `_output_0` | 图像全局嵌入向量 |
| `last_hidden_state` | `_output_0` | patch 级视觉特征 |

## 编译结果检查

```bash
hrt_model_exec model_info --model_file bpu-siglip-base-patch16-224.hbm
hrt_model_exec perf --thread_num 1 --model_name pooler_output --model_file bpu-siglip-base-patch16-224.hbm
hrt_model_exec perf --thread_num 1 --model_name last_hidden_state --model_file bpu-siglip-base-patch16-224.hbm
```

## License

本目录遵循 [Apache 2.0 License](../../../../LICENSE)。
