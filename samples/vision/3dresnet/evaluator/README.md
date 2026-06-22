English | [简体中文](./README_cn.md)

# Model Evaluation

This directory records the evaluation information from the original 3DResNet demo.

## Functional Check

This sample uses an archery video clip. The video was preprocessed into `video0.npy`, and the runtime sample prints Top-5 Kinetics-400 predictions. A correct functional result should include `archery` as the Top-1 class for the provided clip.

Reference images used by this sample:

![Archery frame](../test_data/readme_img/image-4.png)
![Top-5 result](../test_data/readme_img/image-5.png)

## Performance Record

This sample uses `hrt_model_exec` for performance measurement. The record is:

| Threads | Frames | Total Latency (ms) | Average Latency (ms) | FPS |
| ------- | ------ | ------------------ | -------------------- | --- |
| 1 | 100 | 18267.76 | 182.68 | 5.47 |
| 2 | 100 | 18291.76 | 182.93 | 10.82 |
| 4 | 100 | 18501.06 | 185.03 | 21.07 |
| 8 | 100 | 24743.56 | 249.19 | 30.74 |

The original additional metrics screenshot is preserved here:

![Additional metrics](../test_data/readme_img/image-6.png)

The original notes report BPU occupancy around 5.2%, ION memory around 91.9 MB, read bandwidth around 533, and write bandwidth around 304.
