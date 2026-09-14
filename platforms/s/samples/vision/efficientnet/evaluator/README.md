English | [简体中文](./README_cn.md)

# EfficientNet-Lite Evaluation

This directory provides lightweight functional evaluation material for this sample. Full ImageNet accuracy evaluation is not included in this sample directory.

## Functional Check

Run the default demo:

```bash
cd ../runtime/python
bash run.sh
```

Run the entry script directly:

```bash
python3 main.py \
  --model-path /opt/hobot/model/s100/basic/efficientnet_lite0_224x224_nv12.hbm \
  --test-img ../../test_data/Scottish_deerhound.JPEG \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

For `test_data/Scottish_deerhound.JPEG`, the Top-K output should contain a dog-related ImageNet class with a high confidence score. The output should be finite, non-zero, and stable across repeated runs with the same input.

## Performance Records

| Variant | Single-thread latency | Single-thread FPS | Multi-thread latency | Multi-thread FPS |
| --- | --- | --- | --- | --- |
| Lite0 | `0.448 ms` | `2107.815` | `0.591 ms` | `4827.886` |
| Lite1 | `0.489 ms` | `1948.957` | `0.708 ms` | `4086.470` |
| Lite2 | `0.565 ms` | `1702.519` | `0.935 ms` | `3123.682` |
| Lite3 | `0.668 ms` | `1451.031` | `1.249 ms` | `2345.518` |
| Lite4 | `0.915 ms` | `1064.339` | `1.979 ms` | `1487.055` |

The reference result image is `test_data/result.png`.
