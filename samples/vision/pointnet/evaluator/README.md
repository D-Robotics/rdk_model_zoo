English | [简体中文](./README_cn.md)

# PointNet Model Evaluation Guide

This directory records functional checks, performance data, and result validation notes for the PointNet chair part segmentation model.

## Functional Check

This sample uses `chair.pts` as the test point cloud. The runtime saves:

- `result_orig.png`: normalized input point cloud visualization
- `result.png`: predicted chair part segmentation visualization

The expected output should contain four chair part labels: `back`, `seat`, `leg`, and `arm`. The prediction should not collapse to a single label, and the saved segmentation image should show distinct chair regions.

Reference images provided with this sample:

![Original chair point cloud](../test_data/readme_img/chair.png)
![PointNet chair segmentation result](../test_data/readme_img/chair_res.png)

## Performance Record

Performance is measured with `hrt_model_exec`. The record is:

| Threads | Frames | Total Latency | Average Latency | FPS |
| ------- | ------ | ------------- | --------------- | --- |
| 1 | 100 | 143.63 | 1.43 | 689.61 |
| 2 | 100 | 216.20 | 2.16 | 914.32 |
| 4 | 100 | 429.76 | 4.30 | 910.70 |
| 8 | 100 | 839.86 | 8.35 | 910.84 |

## Accuracy Record

See the int16 quantization notes in [conversion/README.md](../conversion/README.md). The `trans` accuracy is greater than 0.9999, and the `pred` accuracy is greater than 0.98.

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).
