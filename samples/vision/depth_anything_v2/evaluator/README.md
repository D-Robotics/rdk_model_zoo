English | [简体中文](./README_cn.md)

# Depth Anything V2 Model Evaluation Guide

This directory records performance data, depth-map result checks, and board-side monitoring metrics for Depth Anything V2.

## Performance Data

Use `hrt_model_exec` to test HBM model performance:

```bash
hrt_model_exec perf --model_file depth_any.hbm --frame_count 100 --thread_num 1
```

| Threads | Frames | Total Latency (ms) | Average Latency (ms) | FPS |
| --- | --- | --- | --- | --- |
| 1 | 100 | 13738.43 | 137.38 | 7.27 |
| 2 | 100 | 26375.53 | 263.74 | 7.54 |
| 4 | 100 | 52214.07 | 521.90 | 7.54 |
| 8 | 100 | 102309.64 | 1020.35 | 7.54 |

## Result Check

Test image:

![Depth Anything V2 input](../test_data/furseal.jpg)

Reference depth map produced by the HBM model:

![Depth Anything V2 depth result](../test_data/readme_img/depth_color.png)

Result validation should confirm:

- The output depth map matches the spatial structure of the input image.
- The depth map is not all black, all white, or a fixed single color.
- The saved result clearly shows depth differences between foreground and background.

## Board Monitoring Metrics

Use the following command to inspect board-side runtime metrics:

```bash
hrt_ucp_monitor
```

Reference record:

![Depth Anything V2 monitor](../test_data/readme_img/image.png)

- BPU occupancy: 95.4%
- ION memory usage: about 300 MB
- Read bandwidth: about 15920
- Write bandwidth: about 11650

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).
