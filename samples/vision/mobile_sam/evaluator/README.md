English | [简体中文](./README_cn.md)

# MobileSAM Evaluator

## Performance Evaluation

Performance was measured on an RDK X5 board with `hrt_model_exec perf`. Each command uses the default `hrt_model_exec` frame count of 200 and validates encoder and decoder `.bin` files separately.

### Commands

```bash
cd samples/vision/mobile_sam
hrt_model_exec perf --model_file ./model/mobile_sam_image_encoder_norm_512x512_allint16.bin --thread_num 1
hrt_model_exec perf --model_file ./model/mobile_sam_image_encoder_norm_512x512_allint16.bin --thread_num 8
hrt_model_exec perf --model_file ./model/mobile_sam_decoder_512_box_default.bin --thread_num 1
hrt_model_exec perf --model_file ./model/mobile_sam_decoder_512_box_default.bin --thread_num 8
```

### Results

| Model | Threads | Average Latency | FPS | Notes |
| --- | ---: | ---: | ---: | --- |
| Encoder .bin | 1 | 1402.542 ms | 0.712979 | `hrt_model_exec perf` |
| Encoder .bin | 8 | 2091.229 ms | 3.772934 | `hrt_model_exec perf` |
| Decoder .bin | 1 | 96.198 ms | 10.393262 | `hrt_model_exec perf` |
| Decoder .bin | 8 | 171.752 ms | 45.783301 | `hrt_model_exec perf` |

## Runtime Smoke Test

The functional demo was validated on the RDK X5 board from `runtime/python` with `bash run.sh`. It generated:

- `test_data/mobile_sam_full_mask_result.jpg`
- `test_data/mobile_sam_binary_mask.png`