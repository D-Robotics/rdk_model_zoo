[English](./README.md) | 简体中文

# 模型转换

本目录给出 PaddleOCR sample 的模型转换说明，包括 PTQ 配置文件、OpenExplorer 编译环境要求、模型编译命令和结果验证步骤。

## 目录结构

```text
conversion/
|-- README.md
|-- README_cn.md
`-- ptq_yamls/
    |-- paddleocr_det_config.yaml
    `-- paddleocr_rec_config.yaml
```

## 转换流程

```text
PaddleOCR 模型 -> ONNX 导出 -> 校准数据准备 -> hb_mapper checker -> hb_mapper makertbin -> .bin 模型
```

## 编译环境

模型检查和编译请使用官方 OpenExplorer Docker 镜像，或等价的 OE 包编译环境。

```bash
docker pull openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
docker run -it --rm -v /path/to/rdk_model_zoo:/data openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

或者前往地瓜开发者社区获取离线版本的 Docker 镜像: [https://forum.d-robotics.cc/t/topic/28035](https://forum.d-robotics.cc/t/topic/28035)

以下转换命令应在 Docker 容器或 OE 编译环境中执行，不应在 RDK 板端执行。

## 现有 PTQ 配置

| 文件 | 模型 | 运行时输入 | 输出模型前缀 |
| --- | --- | --- | --- |
| `ptq_yamls/paddleocr_det_config.yaml` | OCR 检测模型 | `nv12` | `en_PP-OCRv3_det_infer-deploy_640x640_nv12` |
| `ptq_yamls/paddleocr_rec_config.yaml` | OCR 识别模型 | `featuremap` | `en_PP-OCRv3_rec_infer-deploy_48x320_rgb` |

这些 YAML 文件是 `hb_mapper makertbin` 使用的 PTQ 参考配置。实际转换时需要根据本地模型文件更新 `onnx_model`、校准数据路径和输出前缀。

## 校准数据

编译前需要准备具有代表性的校准数据。

- 检测模型：图片需匹配 `paddleocr_det_config.yaml` 中定义的预处理协议。
- 识别模型：featuremap 校准数据需匹配 `paddleocr_rec_config.yaml` 中定义的识别模型输入协议。
- 校准样本数量需结合数据集和模型稳定性确定，建议使用训练域内的代表性样本。

## 模型检查

编译前使用 `hb_mapper checker` 检查算子支持情况。

```bash
hb_mapper checker --model-type onnx --march bayes-e --model en_PP-OCRv3_det_infer.onnx
hb_mapper checker --model-type onnx --march bayes-e --model en_PP-OCRv3_rec_infer.onnx
```

## 模型编译

使用现有 PTQ YAML 作为编译入口。

```bash
hb_mapper makertbin --model-type onnx --config ptq_yamls/paddleocr_det_config.yaml
hb_mapper makertbin --model-type onnx --config ptq_yamls/paddleocr_rec_config.yaml
```

编译完成后，生成的 `.bin` 文件位于 YAML 中配置的 `working_dir` 目录。

## 结果验证

查看模型信息：

```bash
hrt_model_exec model_info --model_file model_output/en_PP-OCRv3_det_infer-deploy_640x640_nv12.bin
hrt_model_exec model_info --model_file model_output/en_PP-OCRv3_rec_infer-deploy_48x320_rgb.bin
```

检查性能信息：

```bash
hb_perf model_output/en_PP-OCRv3_det_infer-deploy_640x640_nv12.bin
hb_perf model_output/en_PP-OCRv3_rec_infer-deploy_48x320_rgb.bin
```

## 运行模型格式

本 sample 在 `RDK X5` 上使用 `.bin` 模型。

预编译模型可通过 [model](../model/README_cn.md) 目录下载。只有在需要基于自定义 ONNX 或校准数据重新生成模型时，才需要执行转换编译流程。

## License

本目录中的工具和文档遵循仓库顶层 License。
