English | [简体中文](README_cn.md)

# PP-LiteSeg conversion recipe

<a id="source-model"></a>
## Source model

Recipe inherited from X5 ac11571: PaddleSeg PP-LiteSeg-STDC1, config `configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml`, static NCHW RGB `(1,3,512,1024)`. Obtain a compatible trained checkpoint from [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) or your own training. Neither checkpoint bytes nor an exact PaddleSeg revision is bundled or pinned. Record both before export; this recipe is not evidence that the published BIN can be reproduced byte-for-byte.

<a id="toolchain-targets"></a>
## Toolchain and targets

X5 only, march bayes-e; no S-series recipe. The source names PaddlePaddle 3.0.0 (Python 3.8–3.10 suggested there) and OE 1.2.8. Export package compatibility must be resolved for the selected PaddleSeg revision; paddle2onnx/onnx/onnxsim versions were not pinned. The script no longer installs packages implicitly. Commands below retain the source setup detail but have not been executed in this host migration.

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

Optional source packages: [OE SDK](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/horizon_x5_open_explorer_v1.2.8-py310_20240926.tar.gz), [Chinese manual](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/x5_doc-v1.2.8-py310-cn.zip), [English manual](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/x5_doc-v1.2.8-py310-en.zip). Archive URLs are source references, not newly verified downloads.

<a id="export"></a>
## Export

```bash
# cwd: repository root; export environment, trained weights supplied by user
cd samples/vision/pp_liteseg/conversion
PADDLESEG_DIR=/data/PaddleSeg CHECKPOINT=/data/checkpoints/pp_liteseg_stdc1_cityscapes.pdparams EXPORT_DIR="$PWD/inference_model/pp_liteseg_stdc1_cityscapes_1024x512" ONNX_DIR="$PWD/onnx" bash onnx_export/export_pp_liteseg_stdc1_onnx.sh
```

The checkpoint path is resolved before entering PaddleSeg; CONFIG is relative to that external checkout unless absolute. EXPORT_DIR and ONNX_DIR are absolute above to avoid cwd ambiguity. tools/export.py produces model.json/model.pdiparams as assumed by the source Paddle 3 recipe, then paddle2onnx uses opset 11 and onnxsim fixes the shape. Confirm actual filenames with your chosen version. Missing checkpoint/config fails before export. Expected ONNX: `onnx/pp_liteseg_stdc1_cityscapes_1024x512_sim.onnx`.

Inspect output names, shape and type before compilation. The canonical runtime requires `(1,512,1024,1)` int32 class IDs. The external export recipe has not been proved to produce this deployment boundary. A logits output needs an explicitly validated graph adaptation or a separately supported runtime contract; renaming the file or applying argmax twice is not a fix.

<a id="calibration"></a>
## Calibration

Use representative road scenes; the source suggests 20–50 images. Preparation converts decoded BGR to RGB, INTER_LINEAR-resizes to 1024×512 and writes little-endian float32 NCHW `(1,3,512,1024)` with raw 0..255 values. Do not normalize twice: YAML applies means 123.675/116.28/103.53 and scales 1/58.395, 1/57.12, 1/57.375.

```bash
# cwd: repository root; Python + OpenCV + NumPy, no OE/board required
cd samples/vision/pp_liteseg/conversion
python3 prepare_calibration.py --src /data/cityscapes/calibration_images --out calibration_data_rgb_f32_1024x512 --width 1024 --height 512 --num 50 --seed 0
```

Each tensor is 6,291,456 bytes. --num defaults to 50, --seed to 0; sorted recursive jpg/jpeg/png/bmp inputs are sampled deterministically without replacement when needed. Unique filenames preserve colliding basenames. Existing output directory or sibling manifest is rejected; choose a new directory for another run. Unreadable selected images fail rather than silently reducing the dataset. The sibling `calibration_data_rgb_f32_1024x512.manifest.json` records shape, seed and input/output hashes; only raw tensors enter the compiler directory. This validates data preparation, not calibration quality.

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

Run direct commands OR build_bin.sh, not both for a normal build. Read checker logs before accepting unsupported operators. The YAML retains source-relative ONNX/calibration paths, output prefix and working_dir. The script expects the output under ptq_yamls/..._output; confirm OE path resolution in your actual container. Missing expected BIN now fails even if makertbin returned zero. CAL_SRC optionally prepares data first, and therefore requires a fresh calibration directory. If overriding MODEL or CONFIG, keep the YAML ONNX/calibration paths consistent: MODEL alone only changes the checker input.

<a id="validation"></a>
## Validation

Host tests cover calibration bytes/naming and shell failure paths using fake tools; real export/compiler/performance runs remain not-run. Inspect runtime metadata and compare exact per-pixel class IDs for matched inputs. The source logit-cosine threshold ≥0.95 is meaningful only if both comparison graphs expose matching pre-argmax logits; it cannot be applied to integer class IDs. Dataset mIoU requires labeled validation data and an implemented dataset runner, which this sample does not ship.

```bash
# cwd: sample directory inside OE; source expected compiler output location
hb_perf conversion/ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12_output/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin
# cwd: sample directory on X5, published model explicitly prepared
hrt_model_exec model_info --model_file model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin
hrt_model_exec perf --model_file model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin --core_id=0 --thread_num=1 --profile_path="."
```

<a id="artifacts"></a>
## Artifacts

Expected sequence: trained .pdparams → inference model.json/model.pdiparams → original/simplified ONNX → OE logs and *_output BIN → local validation report. Preserve checkpoint and PaddleSeg revision, environment versions, calibration manifest, model hashes and logs alongside a locally compiled model. Do not overwrite the published model merely because the filename matches. Custom files may use explicit runtime asset-id to request the same tensor contract, but the unknown publisher digest cannot authenticate them.

<a id="known-gaps"></a>
## Known gaps

No pinned checkpoint/PaddleSeg/export-package combination; no observed exported graph or compiler artifact; no verification of actual output metadata, numerical accuracy or performance. Source expectations of ≈95 FPS / ≈10.5 ms are not measurements from this migration. Unsupported graph operators require inspection, not blindly deleting argmax, because the runtime boundary is already a class map. Changing calibration_type to mix or using more representative data is an experiment requiring revalidation, not a guaranteed repair.
