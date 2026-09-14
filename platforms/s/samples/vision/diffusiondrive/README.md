English | [简体中文](./README_cn.md)

# DiffusionDrive Model Sample

DiffusionDrive is a truncated diffusion policy for real-time end-to-end autonomous driving. This sample runs a deterministic NAVSIM planning graph on RDK S100P and S600 and visualizes the planned trajectory, predicted agents, and seven-class BEV semantics.

## Algorithm Overview

The model fuses a three-camera panorama, LiDAR BEV histogram, ego status, and explicit diffusion noise. A two-step truncated diffusion decoder produces eight future ego poses, while auxiliary heads predict agents and a semantic BEV map.

- Official project: <https://github.com/hustvl/DiffusionDrive>
- Paper: <https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html>
- Dataset and benchmark: <https://github.com/autonomousvision/navsim>

## Directory Structure

```text
.
|-- conversion/                 # OpenExplorer v3.7.0 INT16-first BPU PTQ configuration
|-- evaluator/                  # Float-versus-board output comparison
|-- model/                      # S100P/S600 HBM placement and download helper
|-- runtime/python/             # hbm_runtime inference and visualization
|-- test_data/                  # Four-input NAVSIM sample and reference result
|-- README.md                   # English overview
`-- README_cn.md                # Chinese overview
```

## QuickStart

The launcher detects S100P or S600 from `/sys/class/boardinfo/soc_name`, downloads the matching HBM when necessary, and then runs inference:

```bash
cd runtime/python
bash run.sh
```

The script loads the model once, quantizes four float32 inputs using HBM metadata, performs BPU inference, and writes:

```text
runtime/python/diffusiondrive_outputs.npz
runtime/python/diffusiondrive_result.png
```

See [runtime/python/README.md](runtime/python/README.md) for parameters and integration details.

Five deterministic NAVSIM examples are packaged in `test_data/case_*`. Run all of them with:

```bash
cd runtime/python
bash run_all_cases.sh
```

| `case_017` | `case_042` |
| --- | --- |
| ![Intersection traffic](test_data/case_017/result.png) | ![Dense multi-lane traffic](test_data/case_042/result.png) |
| `case_073` | `case_099` |
| ![Open boulevard](test_data/case_073/result.png) | ![Wide intersection](test_data/case_099/result.png) |

## Model Conversion

Converted HBM files are provided for `nash-m`/S100P and `nash-p`/S600. To regenerate them, follow [conversion/README.md](conversion/README.md). Both request graph-wide INT16 activation PTQ with max calibration; HMCT keeps GridSample at INT8 because this toolchain does not support INT16 GridSample. Every resulting model segment remains on the BPU, with no CPU fallback.

## Evaluation

Use [evaluator/README.md](evaluator/README.md) to compare decoded board outputs against float reference tensors.

The accuracy columns below use `case_000` for a direct platform comparison. Performance uses valid INT16 inputs generated from the real `case_017`, a fixed BPU core, and 200 frames. Single-thread results characterize latency; two concurrent submission threads characterize aggregate BPU throughput following the Model Zoo convention.

| Metric | S100P | S600 |
| --- | ---: | ---: |
| Trajectory cosine similarity | 0.999857 | 0.999833 |
| Agent-state cosine similarity | 0.996879 | 0.997052 |
| BEV cosine similarity | 0.998913 | 0.998918 |
| BEV pixel agreement | 0.943726 | 0.944061 |
| BEV mean IoU | 0.865501 | 0.868425 |
| Single-thread latency | 14.370 ms | 7.215 ms |
| Single-thread throughput | 69.375 FPS | 138.247 FPS |
| Two-thread average task latency | 28.024 ms | 13.856 ms |
| Two-thread aggregate throughput | 71.109 FPS | 143.767 FPS |
| CPU inference time | 0.0 ms | 0.0 ms |

`--thread_num` controls the number of concurrent host threads submitting BPU tasks; it is not the CPU-core count. There is no separate "all threads" mode. Two threads are reported because both platforms are already close to BPU saturation at that point.

Across all five packaged cases, the S100P mean trajectory, agent-state, and BEV cosine similarities are `0.999785`, `0.997986`, and `0.998799`; mean BEV pixel agreement is `0.955664` and mean IoU is `0.819837`.

## Inference Result

The result image contains the stitched camera input, seven-class semantic BEV, LiDAR histogram, planned trajectory, and predicted agent boxes.

| Class ID | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Meaning | background | road | walkway | centerline | static object | vehicle | pedestrian |

Road is rendered gray. A nearly all-gray panel therefore indicates that the model predicted road for nearly every pixel; it is not a missing color-map entry.

![DiffusionDrive S600 result](test_data/reference_result.png)

## License

This sample follows the repository-level Apache License 2.0. DiffusionDrive and NAVSIM assets remain subject to their original licenses.
