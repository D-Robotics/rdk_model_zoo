# Evaluation

This directory documents the accuracy and performance validation steps for PP-LiteSeg-STDC1 on RDK X5.

## Numerical Consistency Verification

Use `hb_mapper infer` or `hb_verifier` to compare the quantized ONNX output with the generated `.bin`.
For semantic segmentation models, use cosine similarity >= `0.95` on the logits tensor (before argmax) as the baseline threshold.

> Note: Cosine similarity on the argmax output (integer class map) is not meaningful. Always compare at the logit layer.

## Dataset-Level Evaluation

Full accuracy validation can be performed by:
1. Running PaddleSeg's evaluation pipeline on the floating-point model to obtain the baseline mIoU.
2. Running the board inference pipeline on the same validation set and comparing mIoU.

## Performance Validation

**In OpenExplorer Docker:**

```bash
hb_perf conversion/ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12_output/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin
```

Expected result: FPS ≈ 95, latency ≈ 10.5 ms (1024×512, single core).

**On-board:**

```bash
hrt_model_exec model_info \
    --model_file pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin

hrt_model_exec perf \
    --model_file pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin \
    --core_id=0 \
    --thread_num=1 \
    --profile_path="."
```
