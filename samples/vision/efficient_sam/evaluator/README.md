English | [简体中文](./README_cn.md)

# EfficientSAM-Tiny Evaluator

This sample does not include dataset-level mask accuracy evaluation. Quantization accuracy is recorded in `../conversion/VALIDATION.md`.

## Performance Evaluation

Performance was measured on an RDK X5 board with `hrt_model_exec perf`. Each command uses the default `hrt_model_exec` frame count of 200 and validates encoder and decoder `.bin` files separately.

### Commands

```bash
cd samples/vision/efficient_sam
hrt_model_exec perf --model_file ./model/efficient_sam_vitt_encoder_512x512_default_none.bin --thread_num 1
hrt_model_exec perf --model_file ./model/efficient_sam_vitt_encoder_512x512_default_none.bin --thread_num 8
hrt_model_exec perf --model_file ./model/efficient_sam_vitt_decoder_fixedprompt_512_default.bin --thread_num 1
hrt_model_exec perf --model_file ./model/efficient_sam_vitt_decoder_fixedprompt_512_default.bin --thread_num 8
```

### Results

| Model | Threads | Average Latency | FPS | Notes |
| --- | ---: | ---: | ---: | --- |
| Encoder .bin | 1 | 1451.073 ms | 0.689135 | `hrt_model_exec perf` |
| Encoder .bin | 8 | 1974.671 ms | 3.965380 | `hrt_model_exec perf` |
| Decoder .bin | 1 | 86.532 ms | 11.553175 | `hrt_model_exec perf` |
| Decoder .bin | 8 | 155.994 ms | 50.565231 | `hrt_model_exec perf` |

## Runtime Smoke Test

The functional demo was validated on the RDK X5 board from `runtime/python` with `bash run.sh`. It generated:

- `test_data/efficient_sam_full_mask_result.jpg`
- `test_data/efficient_sam_binary_mask.png`