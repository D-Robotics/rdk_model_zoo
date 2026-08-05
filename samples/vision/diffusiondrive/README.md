English | [简体中文](./README_cn.md)

# DiffusionDrive Model Sample

DiffusionDrive is a truncated diffusion policy for real-time end-to-end autonomous driving. This sample runs a deterministic NAVSIM planning graph on RDK S600 and visualizes the planned trajectory, predicted agents, and seven-class BEV semantics.

## Algorithm Overview

The model fuses a three-camera panorama, LiDAR BEV histogram, ego status, and explicit diffusion noise. A two-step truncated diffusion decoder produces eight future ego poses, while auxiliary heads predict agents and a semantic BEV map.

- Official project: <https://github.com/hustvl/DiffusionDrive>
- Paper: <https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html>
- Dataset and benchmark: <https://github.com/autonomousvision/navsim>

## Directory Structure

```text
.
|-- conversion/                 # OpenExplorer v3.7.0 full-INT16 PTQ configuration
|-- evaluator/                  # Float-versus-board output comparison
|-- model/                      # S600 HBM placement and download helper
|-- runtime/python/             # hbm_runtime inference and visualization
|-- test_data/                  # Four-input NAVSIM sample and reference result
|-- README.md                   # English overview
`-- README_cn.md                # Chinese overview
```

## QuickStart

The S600 HBM model must be present at `model/s600/diffusiondrive_r34_256x1024_s600.hbm`. Then run:

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

A converted HBM is provided on the target development board. To regenerate it, follow [conversion/README.md](conversion/README.md). The accepted configuration uses INT16 activations throughout the graph with max calibration. Every model segment remains on the BPU; no CPU fallback is introduced.

## Evaluation

Use [evaluator/README.md](evaluator/README.md) to compare decoded board outputs against float reference tensors.

The checked-in sample was validated on an S600 board with the following results:

| Metric | Result |
| --- | ---: |
| Trajectory cosine similarity | 0.999833 |
| Agent-state cosine similarity | 0.997052 |
| BEV cosine similarity | 0.998918 |
| BEV pixel agreement | 0.944061 |
| BEV mean IoU | 0.868425 |
| Single-thread latency / throughput | 7.229 ms / 138.060 FPS |
| CPU inference time | 0.0 ms |

## Inference Result

The result image contains the stitched camera input, seven-class semantic BEV, LiDAR histogram, planned trajectory, and predicted agent boxes.

| Class ID | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Meaning | background | road | walkway | centerline | static object | vehicle | pedestrian |

Road is rendered gray. A nearly all-gray panel therefore indicates that the model predicted road for nearly every pixel; it is not a missing color-map entry.

![DiffusionDrive S600 result](test_data/reference_result.png)

## License

This sample follows the repository-level Apache License 2.0. DiffusionDrive and NAVSIM assets remain subject to their original licenses.
