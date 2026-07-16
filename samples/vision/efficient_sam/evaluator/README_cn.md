[English](./README.md) | 简体中文

# EfficientSAM-Tiny 评测说明

本 sample 不包含数据集级 mask 精度评测。量化精度记录在 `../conversion/VALIDATION.md`。

## 性能评测

性能已在 RDK X5 板端使用 `hrt_model_exec perf` 验证。每条命令使用 `hrt_model_exec` 默认 200 帧，并分别测试 encoder 与 decoder `.bin` 模型。

### 命令

```bash
cd samples/vision/efficient_sam
hrt_model_exec perf --model_file ./model/efficient_sam_vitt_encoder_512x512_default_none.bin --thread_num 1
hrt_model_exec perf --model_file ./model/efficient_sam_vitt_encoder_512x512_default_none.bin --thread_num 8
hrt_model_exec perf --model_file ./model/efficient_sam_vitt_decoder_fixedprompt_512_default.bin --thread_num 1
hrt_model_exec perf --model_file ./model/efficient_sam_vitt_decoder_fixedprompt_512_default.bin --thread_num 8
```

### 结果

| 模型 | 线程数 | 平均延迟 | FPS | 备注 |
| --- | ---: | ---: | ---: | --- |
| Encoder .bin | 1 | 1451.073 ms | 0.689135 | `hrt_model_exec perf` |
| Encoder .bin | 8 | 1974.671 ms | 3.965380 | `hrt_model_exec perf` |
| Decoder .bin | 1 | 86.532 ms | 11.553175 | `hrt_model_exec perf` |
| Decoder .bin | 8 | 155.994 ms | 50.565231 | `hrt_model_exec perf` |

## 功能冒烟验证

功能 demo 已在 RDK X5 板端的 `runtime/python` 目录下通过 `bash run.sh` 验证，并生成：

- `test_data/efficient_sam_full_mask_result.jpg`
- `test_data/efficient_sam_binary_mask.png`