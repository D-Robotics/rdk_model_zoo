[English](README.md) | [简体中文](README_cn.md)

# DiffusionDrive conversion

The two source PTQ YAMLs are preserved byte-for-byte. They document compiler settings; running them requires the export and calibration inputs prepared below.

<a id="source-model"></a>
## Source model

The source refers to the official [DiffusionDrive project](https://github.com/hustvl/DiffusionDrive) and its NAVSIM checkpoint, without pinning the checkpoint URL/hash, upstream revision or exporter. The [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html) describes the algorithm; it does not identify the bytes behind this release. Obtain the matching model code, weights and export procedure before claiming a reproducible rebuild.

For inference with the published assets, use [model preparation](../model/README.md). Published HBM checksums authenticate those published files; an independently exported model needs its own digest record.

<a id="toolchain-targets"></a>
## Toolchain and targets

The recorded x86 Linux toolchain image is:

```text
registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
```

| Target | Configuration | March | Output under this directory |
| --- | --- | --- | --- |
| S100P | `configs/diffusiondrive_r34_256x1024_s100p.yaml` | `nash-m` | `build/s100p/hbm/diffusiondrive_r34_256x1024_s100p.hbm` |
| S600 | `configs/diffusiondrive_r34_256x1024_s600.yaml` | `nash-p` | `build/s600/hbm/diffusiondrive_r34_256x1024_s600.hbm` |

Both request graph-wide INT16 activations/max calibration, O2/latency, core count 1, 32 compiler jobs, cache enabled and no input/output padding. The source records GridSample remaining INT8 because v3.7.0 does not support INT16 GridSample. The graph is INT16-first; each operator's and public IO tensor's actual type comes from the compiled artifact's metadata. Configurations cover S100P/S600 only — there is no S100/X5 configuration.

<a id="export"></a>
## Export contract and missing code

The required deterministic ONNX has these four float feature inputs:

| Name | Shape |
| --- | --- |
| `camera` | `[1,3,256,1024]` |
| `lidar` | `[1,1,256,256]` |
| `status` | `[1,8]` |
| `noise` | `[1,20,8,2]` |

Outputs are trajectory `[1,8,3]`, agent states `[1,30,5]`, agent logits `[1,30]` and BEV logits `[1,7,128,256]`, under the names in the [runtime contract](../runtime/python/README.md#stage-io). Noise stays an explicit input. The source requires replacing ScatterND-style in-place writes with concatenation and fixed adaptive average pooling with static depthwise convolution.

The export must apply those rewrites and produce the caller-supplied file `build/diffusiondrive_navsim_bpu_clean_float.onnx`, relative to this directory, matching the input/output contract above. Validate the exported float outputs against the intended checkpoint before PTQ.

<a id="calibration"></a>
## Calibration preparation

The source requires at least 100 real NAVSIM mini samples. For each sample, provide corresponding finite float32 `.npy` tensors in `calibration_data/camera`, `calibration_data/lidar`, `calibration_data/status` and `calibration_data/noise`. Pair the same sample across all four directories; use the input order `camera;lidar;status;noise` and the shapes above. YAML declares featuremap/NCHW inputs and `separate_batch: false`.

The source does not include the original calibration set, a feature-preparation/export script or a sample manifest. The six demonstration archives are insufficient for the stated >=100-sample recipe; duplicating them does not supply the missing representative data. Do not feed already quantized runtime buffers into a calibration directory expecting logical float features. Keep dataset identity, tensor hashes and preparation revision when building a new calibration set.

<a id="compile"></a>
## Compile after prerequisites exist

Inside the recorded toolchain environment, start from the repository root, then change to this directory so all YAML relative paths resolve consistently:

```bash
cd samples/vision/diffusiondrive/conversion
hb_compile -c configs/diffusiondrive_r34_256x1024_s600.yaml
hb_compile -c configs/diffusiondrive_r34_256x1024_s100p.yaml
```

These are the source compiler commands. They require the ONNX and four complete calibration directories first. Preserve the compiler version, logs, generated reports, graph-placement report and resulting HBM digest, and verify numerical correctness and target execution after compilation. The toolchain and export/calibration inputs are user-provided; no script here downloads a toolchain or fabricates inputs.

<a id="validation"></a>
## Validation

On each matching board, inspect the generated model's actual IO metadata and then run valid prepared inputs through the runtime. HRT invocations use the following forms from this directory; model-info/performance tools are board-side dependencies:

```bash
hrt_model_exec model_info --model_file build/s600/hbm/diffusiondrive_r34_256x1024_s600.hbm
hrt_model_exec model_info --model_file build/s100p/hbm/diffusiondrive_r34_256x1024_s100p.hbm
```

For performance, the source requires valid quantized `camera.bin,lidar.bin,status.bin,noise.bin`; GridSample may reject or behave outside its supported range with arbitrary random data. The files must reflect the model's actual dtype/quantization, not raw float NPZ bytes. After explicitly preparing those files, one/two-thread forms are:

```bash
hrt_model_exec perf --model_file build/s600/hbm/diffusiondrive_r34_256x1024_s600.hbm --thread_num 1 --core_id 1 --input_file camera.bin,lidar.bin,status.bin,noise.bin
hrt_model_exec perf --model_file build/s600/hbm/diffusiondrive_r34_256x1024_s600.hbm --thread_num 2 --core_id 1 --input_file camera.bin,lidar.bin,status.bin,noise.bin
hrt_model_exec perf --model_file build/s100p/hbm/diffusiondrive_r34_256x1024_s100p.hbm --thread_num 1 --core_id 1 --input_file camera.bin,lidar.bin,status.bin,noise.bin
hrt_model_exec perf --model_file build/s100p/hbm/diffusiondrive_r34_256x1024_s100p.hbm --thread_num 2 --core_id 1 --input_file camera.bin,lidar.bin,status.bin,noise.bin
```

The source reports all-INT8 BEV cosine 0.370948; changing only four BEV-head nodes to INT16 produced cosine 0.371840 and mean IoU 0.143013. It attributes the remaining degradation to upstream fused features at `/_backbone/Add_6`. INT16-first/max produced S600 case_000 BEV cosine 0.998918, agreement 0.944061, mean IoU 0.868425; S100P values were 0.998913, 0.943726, 0.865501, with five-case means 0.998799, 0.955664, 0.819837.

These are source records. The source also records BPU-only segments/0.0 ms CPU inference, and case_017 S100P 14.370 ms /69.375 FPS (one thread),71.109 FPS aggregate (two threads); S600 7.215 ms /138.247 FPS and143.767 FPS aggregate. Full accuracy/performance tables and the distinction between case_000 comparison and case_017 profiling are retained in [evaluation](../evaluator/README.md#reference-results).

<a id="artifacts"></a>
## Artifacts and identity

Keep each generated HBM under its target output path alongside compiler logs and a manifest of ONNX, calibration and configuration hashes. A custom-converted model gets its own asset identity and binding: register the new asset, record its digest, and validate it explicitly. The runtime verifies the published SHA-256 for published paths, so custom artifacts run through their own registered binding with its recorded digest; the published checksums stay unchanged.

Inspect actual names, shapes, dtypes, scales, zero points and axes before testing; physical HBM metadata is authoritative over the logical float reference. Compare decoded outputs with the [offline evaluator](../evaluator/README.md), keeping raw tensors and provenance. The evaluator reports descriptive metrics; release thresholds are defined by your release process.

<a id="known-gaps"></a>
## Additional preparation

Missing checkpoint/export revision, rewrite code, original calibration data and exact profiling input binaries are the preparation inputs this recipe expects from the operator. Toolchain availability, complete graph placement, real HBM metadata, OE compilation, board output parity and performance are verified when the recipe is executed. The retained configurations and documentation list these dependencies so they can be prepared and checked before execution.
