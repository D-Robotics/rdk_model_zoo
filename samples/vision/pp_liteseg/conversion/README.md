English | [简体中文](README_cn.md)

# PP-LiteSeg conversion recipe

<a id="source-model"></a>
## Source model

Use PaddleSeg PP-LiteSeg-STDC1 with config `configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml` and static RGB NCHW input `(1,3,512,1024)`. Obtain a trained checkpoint from [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) or your own training, and record the selected repository revision and checkpoint before export.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── onnx_export/  # Files for onnx_export
├── ptq_yamls/  # Files for ptq_yamls
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── build_bin.sh  # Shell command
└── prepare_calibration.py  # Python script
```

<a id="toolchain-targets"></a>
## Toolchain and targets

Use the X5 `bayes-e` toolchain: PaddlePaddle 3.0.0, Python 3.8–3.10, and OE 1.2.8. Install paddle2onnx, onnx, and onnxsim versions compatible with the selected PaddleSeg revision using the commands below.

```bash
# Export environment, separate from the board SDK environment
python3 -m pip install paddlepaddle==3.0.0 paddle2onnx onnx onnxsim
# Obtain and select the PaddleSeg revision required by your checkpoint first.
git clone https://gitee.com/paddlepaddle/PaddleSeg.git /data/PaddleSeg
python3 -m pip install -r /data/PaddleSeg/requirements.txt
python3 -m pip install -e /data/PaddleSeg
# cwd: repository root; source OE image location (availability not rechecked)
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker run -it --rm -v "$(pwd):/open_explorer" -w /open_explorer openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
# Inside OE container:
hb_mapper --version
hb_perf --version
```

Optional source packages: [OE SDK](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/horizon_x5_open_explorer_v1.2.8-py310_20240926.tar.gz), [Chinese manual](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/x5_doc-v1.2.8-py310-cn.zip), [English manual](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/x5_doc-v1.2.8-py310-en.zip). Use the matching OE 1.2.8 package and manual.

Download the OE 1.2.8 SDK and manuals when using the package installation:

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/horizon_x5_open_explorer_v1.2.8-py310_20240926.tar.gz
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/x5_doc-v1.2.8-py310-cn.zip
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/x5_doc-v1.2.8-py310-en.zip
```

<a id="export"></a>
## Export

```bash
# cwd: repository root; export environment, trained weights supplied by user
cd samples/vision/pp_liteseg/conversion
PADDLESEG_DIR=/data/PaddleSeg CHECKPOINT=/data/checkpoints/pp_liteseg_stdc1_cityscapes.pdparams EXPORT_DIR="$PWD/inference_model/pp_liteseg_stdc1_cityscapes_1024x512" ONNX_DIR="$PWD/onnx" bash onnx_export/export_pp_liteseg_stdc1_onnx.sh
```

The checkpoint path is resolved before entering PaddleSeg; CONFIG is relative to that external checkout unless absolute. EXPORT_DIR and ONNX_DIR are absolute above to avoid cwd ambiguity. tools/export.py produces model.json/model.pdiparams as assumed by the source Paddle 3 recipe, then paddle2onnx uses opset 11 and onnxsim fixes the shape. Confirm actual filenames with your chosen version. Missing checkpoint/config fails before export. Expected ONNX: `onnx/pp_liteseg_stdc1_cityscapes_1024x512_sim.onnx`.

Inspect output names, shape and type before compilation. The runtime requires `(1,512,1024,1)` int32 class IDs. The external export recipe has not been proved to produce this deployment boundary. A logits output needs an explicitly validated graph adaptation or a separately supported runtime contract; renaming the file or applying argmax twice is not a fix.

<a id="calibration"></a>
## Calibration

Use representative road scenes; the source suggests 20–50 images. Preparation converts decoded BGR to RGB, INTER_LINEAR-resizes to 1024×512 and writes little-endian float32 NCHW `(1,3,512,1024)` with raw 0..255 values. Do not normalize twice: YAML applies means 123.675/116.28/103.53 and scales 1/58.395, 1/57.12, 1/57.375.

```bash
# cwd: repository root; Python + OpenCV + NumPy, no OE/board required
cd samples/vision/pp_liteseg/conversion
python3 prepare_calibration.py --src /data/cityscapes/calibration_images --out calibration_data_rgb_f32_1024x512 --width 1024 --height 512 --num 50 --seed 0
```

Each tensor is 6,291,456 bytes. --num defaults to 50, --seed to 0; sorted recursive jpg/jpeg/png/bmp inputs are sampled deterministically without replacement when needed. Unique filenames preserve colliding basenames. Existing output directory or sibling manifest is rejected; choose a new directory for another run. Unreadable selected images return an error. The sibling `calibration_data_rgb_f32_1024x512.manifest.json` records shape, seed and input/output hashes; only raw tensors enter the compiler directory. Select calibration images representative of the deployment inputs.

<a id="compile"></a>
## Compile

```bash
# cwd: repository root, inside OE container; ONNX and calibration already prepared
cd samples/vision/pp_liteseg/conversion
hb_mapper checker --model-type onnx --march bayes-e --model onnx/pp_liteseg_stdc1_cityscapes_1024x512_sim.onnx
hb_mapper makertbin --config ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12.yaml --model-type onnx
# Alternative orchestration of checker + makertbin + hb_perf:
bash build_bin.sh
```

Run direct commands OR build_bin.sh, not both for a normal build. Read checker logs before accepting unsupported operators. The YAML retains source-relative ONNX/calibration paths, output prefix and working_dir. The script expects the output under ptq_yamls/..._output; confirm OE path resolution in your actual container. Missing expected BIN now fails even if makertbin returned zero. CAL_SRC optionally prepares data first, and therefore requires a fresh calibration directory. If overriding MODEL or CONFIG, keep the YAML ONNX/calibration paths consistent: MODEL alone only changes the checker input. <a id="validation"></a>
## Validation

Inspect runtime metadata and compare exact per-pixel class IDs for matched inputs. The source logit-cosine threshold ≥0.95 is meaningful only if both comparison graphs expose matching pre-argmax logits; it cannot be applied to integer class IDs. Dataset mIoU requires labeled validation data and an implemented dataset runner, which this sample does not ship.

```bash
# cwd: sample directory inside OE; source expected compiler output location
hb_perf conversion/ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12_output/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin
# cwd: sample directory on X5, published model explicitly prepared
hrt_model_exec model_info --model_file model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin
hrt_model_exec perf --model_file model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin --core_id=0 --thread_num=1 --profile_path="."
```

<a id="artifacts"></a>
## Artifacts

Expected sequence: trained.pdparams → inference model.json/model.pdiparams → original/simplified ONNX → OE logs and *_output BIN → local validation report. Preserve checkpoint and PaddleSeg revision, environment versions, calibration manifest, model hashes and logs alongside a locally compiled model. Do not overwrite the published model merely because the filename matches. Custom files may use explicit runtime asset-id to request the same tensor contract; their identity is recorded by the observed digest and the compilation logs.

<a id="known-gaps"></a>
## Additional preparation

Prepare the checkpoint, PaddleSeg and export packages, then record their versions with the exported graph and compiled artifact. The source documentation's expectation of ≈95 FPS / ≈10.5 ms is not a measured value in this repository. Unsupported graph operators require inspection, not blindly deleting argmax, because the runtime boundary is already a class map. Changing calibration_type to mix or using more representative data is an experiment requiring revalidation, not a guaranteed repair.

## Reproduce the S float-logits export

Keep the checkpoint, calibration tensors, ONNX and HBM on dev-server. Use an isolated environment from `requirements-export.txt` and PaddleSeg commit `3c4db66de1d9d59d0628ed87590b6308a2f4aa2a`. The official checkpoint URL is recorded in `source_export_20261104.json`; its observed SHA-256 is `86543ad69437495f8e7688d497a52d2472ee098efa918dcfa551803570f5edf3` (not a separately verified publisher digest).

```bash
python onnx_export/export_logits.py --source /server/PaddleSeg --checkpoint /server/model.pdparams --output /server/new-export --samples /server/a.jpg /server/b.jpg /server/c.jpg
python prepare_calibration.py --src /server/representative-road-images --out /server/calibration/normalized --num 33 --seed 42 --source-logits
```

The exporter strictly loads every trained tensor. Three non-trainable attention scales introduced by the pinned upstream source are explicitly reconstructed as its `Constant(1)`, preserving the original `(1 - attention)` formula. Four full-logit source/ONNX checks passed, and the portable exporter reproduced the compiled ONNX byte-for-byte. Do not use the legacy raw 0..255 calibration tensors or ImageNet normalization for this S variant. The recorded run used 33 existing COCO train2017 images with traffic-category annotations, its own normalized-input manifest and a full-output source CPU/SDK CPU parity gate at atol=1e-4, rtol=1e-3 before independent target PTQ. Runtime preprocessing uses the same RGB [-1,1] tensor domain.

<!-- BEGIN CPU S CONVERSIONS 20261104 -->

## Additional CPU PTQ S profiles (2026-11-04)

These exact trained-checkpoint exports passed the source host gate and the full-output CPU source/CPU SDK gate (three frozen images plus deterministic random input, atol=1e-4, rtol=1e-3). Each target has its own INT16 PTQ, compiler metadata, OE report and checksum in `converted_targets_20261103.json`. Compiler estimates describe conversion; board accuracy, Runtime 1/2 concurrency and C++ 1/2 streams remain not run.

Models and intermediate outputs stay on dev-server. The Mac holds code and metadata only; Archive uploads stream from server files without a local model copy.

| Target | Native variant | Manifest filename | SHA-256 | Bytes |
|---|---|---|---|---:|
| s100 | `stdc1-cityscapes-1024x512-logits-int16-cpuptq-o1-20261104` | `s100/pp_liteseg_stdc1_cityscapes_nashe_1024x512_featuremap_int16_cpuptq_20261104_20261103_o1.hbm` | `6cee553d56e66ab2008aac8a7cf982f48a5ef75f4ddcada1b9dae547a1212491` | 227956272 |
| s100p | `stdc1-cityscapes-1024x512-logits-int16-cpuptq-o1-20261104` | `s100p/pp_liteseg_stdc1_cityscapes_nashm_1024x512_featuremap_int16_cpuptq_20261104_20261103_o1.hbm` | `b704d7922bfbd9a5903ae00376875b123fb27118c238dc5964b565387290dff1` | 227767152 |
| s600 | `stdc1-cityscapes-1024x512-logits-int16-cpuptq-o1-20261104` | `s600/pp_liteseg_stdc1_cityscapes_nashp_1024x512_featuremap_int16_cpuptq_20261104_20261103_o1.hbm` | `f443ec10a16a71bbae53062d0a992e4fcd656dae7fe01764d4c1d5b96909f611` | 13808424 |

<!-- END CPU S CONVERSIONS 20261104 -->
