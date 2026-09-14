[English](./README.md) | 简体中文

# MobileSAM 评测说明

## 性能评测

性能已在 RDK X5 板端使用 `hrt_model_exec perf` 验证。每条命令使用 `hrt_model_exec` 默认 200 帧，并分别测试 encoder 与 decoder `.bin` 模型。

### 命令

```bash
cd samples/vision/mobile_sam
hrt_model_exec perf --model_file ./model/mobile_sam_image_encoder_norm_512x512_allint16.bin --thread_num 1
hrt_model_exec perf --model_file ./model/mobile_sam_image_encoder_norm_512x512_allint16.bin --thread_num 8
hrt_model_exec perf --model_file ./model/mobile_sam_decoder_512_box_default.bin --thread_num 1
hrt_model_exec perf --model_file ./model/mobile_sam_decoder_512_box_default.bin --thread_num 8
```

### 结果

| 模型 | 线程数 | 平均延迟 | FPS | 备注 |
| --- | ---: | ---: | ---: | --- |
| Encoder .bin | 1 | 1402.542 ms | 0.712979 | `hrt_model_exec perf` |
| Encoder .bin | 8 | 2091.229 ms | 3.772934 | `hrt_model_exec perf` |
| Decoder .bin | 1 | 96.198 ms | 10.393262 | `hrt_model_exec perf` |
| Decoder .bin | 8 | 171.752 ms | 45.783301 | `hrt_model_exec perf` |

## 功能冒烟验证

功能 demo 已在 RDK X5 板端的 `runtime/python` 目录下通过 `bash run.sh` 验证，并生成：

- `test_data/mobile_sam_full_mask_result.jpg`
- `test_data/mobile_sam_binary_mask.png`