# LPRNet 模型转换

本目录记录 LPRNet 在 RDK X5 上的转换侧说明。

LPRNet 模型的完整转换流程请直接参考 OE 包。

## 当前提供的资产

仓库当前提供：

- 本示例使用的已发布 `.bin` 模型
- 输入输出张量协议说明

仓库当前不提供完整的站内转换工具链。

## 支持的 X5 模型

| 模型 | 输入尺寸 | 运行格式 |
| --- | --- | --- |
| `lpr.bin` | `1x3x24x94` | `.bin` |

## `hb_mapper checker`

按 OE 包准备好转换资产后，可使用：

```bash
hb_mapper checker --model-type onnx --config your_lprnet_config.yaml
```

## `hb_mapper makertbin`

生成可部署的 `.bin` 模型：

```bash
hb_mapper makertbin --model-type onnx --config your_lprnet_config.yaml
```

## `hrt_model_exec`

板端可使用下面命令查看模型输入输出：

```bash
hrt_model_exec model_info --model_file lpr.bin
```

## 输出协议

本示例使用的 X5 LPRNet 模型沿用原始 demo 的输入输出协议：

- 输入张量：`1 x 3 x 24 x 94`，`float32`，`NCHW`
- 输出张量：`1 x 68 x 18`，`float32`，`NCHW`

Python runtime 会使用 CTC 风格的去重规则解码最终字符序列。
