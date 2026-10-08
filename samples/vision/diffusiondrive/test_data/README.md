[English](README.md) | [简体中文](README_cn.md)

# DiffusionDrive deterministic examples

This directory provides six prepared input/reference pairs and six S600 reference visualizations. Each NPZ input contains camera, LiDAR, status and diffusion-noise features.


## Directory structure

```text
test_data/
├── case_000/  # Files for case_000
├── case_017/  # Files for case_017
├── case_042/  # Files for case_042
├── case_073/  # Files for case_073
├── case_099/  # Files for case_099
├── reference_inputs.npz  # Default prepared input
├── reference_outputs.npz  # Default float reference outputs
├── reference_result.png  # Default reference rendering
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

## Input contract

| Tensor | Shape | Type | Interpretation |
| --- | --- | --- | --- |
| `camera` | `[1,3,256,1024]` | float32 | Prepared left/front/right RGB panorama |
| `lidar` | `[1,1,256,256]` | float32 | Prepared LiDAR BEV histogram |
| `status` | `[1,8]` | float32 | Prepared ego status and driving command |
| `noise` | `[1,20,8,2]` | float32 | Fixed truncated-diffusion noise |

These are logical float features, not physical quantized HBM buffers. The sample quantizes them using actual runtime metadata; it does not reconstruct NAVSIM features from raw sensors. The original dataset sample IDs, preparation script and full sensor sources are not included. Do not guess status component meanings beyond the supplied source description. Keep noise unchanged for a deterministic comparison; per-SDK determinism is confirmed by running on that SDK.

## Files and reference outputs

The default pair is `reference_inputs.npz` and `reference_outputs.npz`; `reference_result.png` is the source-recorded display. Each case directory contains `inputs.npz`, `reference_outputs.npz` and `result.png`.

| Float reference tensor | Shape | Meaning |
| --- | --- | --- |
| `trajectory` | `[1,8,3]` | Eight ego poses `[x,y,heading]` |
| `agent_states` | `[1,30,5]` | `[x,y,heading,length,width]` for thirty candidates |
| `agent_labels` | `[1,30]` | Agent logits, not probabilities |
| `bev_semantic_map` | `[1,7,128,256]` | Seven-class BEV logits, not labels |

The source identifies these as PyTorch float outputs, not annotation truth. Source metadata and known HBM hashes do not pin the full upstream checkpoint/export history. All packaged arrays are finite float32; runtime archives use separate raw and decoded schemas.

## Five S600 examples

The following S600 results compare quantized model outputs with the prepared float references.

| Case | Scene | Predicted agents | BEV pixel agreement | BEV mean IoU |
| --- | --- | ---: | ---: | ---: |
| `case_000` | Wide signalized intersection | 7 | 0.944061 | 0.868425 |
| `case_017` | Signalized intersection with nearby traffic | 7 | 0.944000 | 0.761217 |
| `case_042` | Dense multi-lane urban traffic | 13 | 0.966736 | 0.728740 |
| `case_073` | Open straight boulevard | 6 | 0.966156 | 0.876931 |
| `case_099` | Wide intersection with many detected agents | 14 | 0.958862 | 0.899669 |

Mean IoU includes classes present in either prediction. A small number of class-4 pixels in case_017/case_042 has a disproportionate effect on macro IoU, so retain pixel agreement for context. The [evaluator](../evaluator/README.md) defines the current validation and metrics explicitly.

| case_017 | case_042 |
| --- | --- |
| ![Reference intersection result](case_017/result.png) | ![Reference dense traffic result](case_042/result.png) |
| case_073 | case_099 |
| ![Reference boulevard result](case_073/result.png) | ![Reference wide intersection result](case_099/result.png) |

See the [default visualization](reference_result.png) and [case_000 visualization](case_000/result.png).

## Running examples

From the repository root on a prepared S600:

```bash
python3 -m samples.vision.diffusiondrive.runtime.python.main --target s600 --input-npz samples/vision/diffusiondrive/test_data/case_017/inputs.npz --output outputs/diffusiondrive_case017
```

Run the five cases with explicit target and a new batch directory:

```bash
bash samples/vision/diffusiondrive/runtime/python/run_all_cases.sh --target s600 --output outputs/diffusiondrive_cases
```

For command inspection, add `--dry-run` to the batch command. It validates all five input archives and prints commands without executing the SDK or creating output directories. Batch runs stop on the first failed case and keep completed run records; no later case is reported as passed. Use `--target s100p` with the distinct S100P model when that board is available. No auto-download or board fallback is performed.

## Coordinate and display interpretation

Trajectory/agent x is forward and y is left in ego-local meters. Rendering uses 0.25 m raster pixels and the source rotations/crop so forward appears upward. The blue box is ego, orange is the planned trajectory, red boxes are candidates above the agent threshold. The display has no control output or trajectory execution.

| Class ID | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Meaning | background | road | walkway | centerline | static object | vehicle | pedestrian |

Road is gray; a nearly gray BEV panel can mean road dominates the predictions, not a missing palette entry. Images are visualization aids, not a substitute for raw-array comparisons. DiffusionDrive/NAVSIM assets remain subject to their original terms; this directory does not supply a complete licensed evaluation dataset.
