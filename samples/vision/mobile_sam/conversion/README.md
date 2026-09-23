English | [简体中文](./README_cn.md)

# MobileSAM conversion

This directory records the source conversion capability for the MobileSAM image encoder and box-prompt decoder. The unified scripts accept `--target x5`, `--target s100`, `--target s100p`, or `--target s600`; conversion has not been run in this migration.

<a id="source-model"></a>
## Source model

The upstream source is [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM), with checkpoint `weights/mobile_sam.pt`. The fixed source does not pin an upstream commit or checkpoint digest. The encoder consumes `normalized_images` `(1,3,512,512)` float32 RGB NCHW and emits `image_embeddings` `(1,256,32,32)` float32. The decoder consumes `image_embeddings` `(1,256,32,32)` and runtime `boxes` `(1,4)` float32, then emits `low_res_masks` `(1,3,128,128)` and `iou_predictions` (decoder metadata must confirm its final rank).

Preprocessing is RGB resize to 512×512 followed by ImageNet normalization in channel order: mean `[123.675,116.28,103.53]`, standard deviation `[58.395,57.12,57.375]`. The source default box is `[185,120,380,445]` in the 512-pixel coordinate system. The decoder box remains a runtime input; it is not baked into the ONNX graph.

<a id="toolchain-targets"></a>
## Toolchain and targets

| Target | config directory | march | compiler |
|---|---|---|---|
| `x5` | `configs/x5/` | `bayes-e` | `hb_mapper makertbin --model-type onnx` |
| `s100` | `configs/s100/` | `nash-e` | `hb_compile` |
| `s100p` | `configs/s100p/` | `nash-m` | `hb_compile` |
| `s600` | `configs/s600/` | `nash-p` | `hb_compile` |

X5 source configs use the OE X5 environment and default calibration. S configs use the OE S-series 3.7.0 environment, `set_all_nodes_int16`, max calibration, and `max_percentile: 0.9999`. Published asset names and SHA fields are authoritative in `docs/release/x5/models.yaml` and `docs/release/s/models.yaml`; the source manifests mark SHA values as unknown (`null`).

Export and float embedding generation require host PyTorch, ONNX, ONNX Runtime, NumPy and OpenCV. The fixed sources do not pin their package versions or the upstream repository revision; this remains a reproducibility prerequisite, not a tested environment specification. Check these imports inside the chosen export environment, separate from board inference:

```bash
# cwd: this conversion directory, inside the chosen host export environment
python3 -c "import torch, onnx, onnxruntime, numpy, cv2; print(torch.__version__, onnx.__version__, onnxruntime.__version__)"
```

The exporter actually imports `ultralytics.models.sam.build.build_mobile_sam`; cloning the MobileSAM repository alone does not provide that package. The source's setup step is `python3 -m pip install ultralytics` in the export environment. Its version is unpinned in the source, so record the installed version and verify the `build_mobile_sam`/`set_imgsz` APIs before export. This command has not been executed here.

<a id="export"></a>
## Export ONNX

The X5 source has a helper for the upstream checkout. The fixed S source has no equivalent helper, so prepare the checkout and checkpoint manually for S targets.

```bash
python3 scripts/download_assets.py --target x5 --workspace ./workspace
# S targets: place ./workspace/MobileSAM and
# ./workspace/MobileSAM/weights/mobile_sam.pt manually.
```

The X5 helper downloads `mobile_sam.pt` when it is absent or smaller than 1,000,000 bytes, after cloning or reusing the checkout. The S source has no equivalent helper, so its source and checkpoint remain manual prerequisites.

Export both graphs from this directory:

```bash
python3 scripts/export_encoder_onnx.py --target x5 \
  --repo ./workspace/MobileSAM \
  --weights ./workspace/MobileSAM/weights/mobile_sam.pt \
  --output ./mobile_sam_image_encoder_norm_512_op11.onnx
python3 scripts/export_decoder_onnx.py --target x5 \
  --repo ./workspace/MobileSAM \
  --checkpoint ./workspace/MobileSAM/weights/mobile_sam.pt \
  --output ./mobile_sam_decoder_512_box_op11.onnx \
  --box 185 120 380 445
```

Change `--target` for S100/S100P/S600. The exporter defaults to size 512 and opset 11; `--size`, `--opset`, and decoder `--box` are explicit overrides. The scripts do not download the checkpoint themselves.

<a id="calibration"></a>
## Calibration

Encoder calibration reads representative RGB images, resizes them to 512×512, converts to NCHW, and applies the exact ImageNet mean/std above. X5 writes raw `.rgbchw` tensors; S writes `.npy` tensors. The producer count is an output-file count; it does not establish that the inputs are independent or representative.

```bash
python3 scripts/prepare_calibration.py --target x5 \
  --src ./calibration_images --out . --num 30 --size 512
# ./calibration_data_norm_512/*.rgbchw
python3 scripts/dump_encoder_embedding.py --image ../test_data/dogs.jpg \
  --onnx ./mobile_sam_image_encoder_norm_512_op11.onnx --output ./encoder_embedding.bin

python3 scripts/prepare_decoder_calibration.py --target x5 \
  --embedding ./encoder_embedding.bin --out ./decoder_calibration \
  --num 30 --box 185 120 380 445
# ./decoder_calibration/calibration_embeddings/*.bin
# ./decoder_calibration/calibration_boxes/*.bin
```

For S targets, use an explicit calibration root so paths match the YAMLs:

```bash
python3 scripts/prepare_calibration.py --target s100 \
  --src ./calibration_images --out ./calibration_data_norm_512 --num 30 --size 512
python3 scripts/dump_encoder_embedding.py --image ../test_data/dogs.jpg \
  --onnx ./mobile_sam_image_encoder_norm_512_op11.onnx --output ./encoder_embedding.bin
python3 scripts/prepare_decoder_calibration.py --target s100 \
  --embedding ./encoder_embedding.bin --out ./decoder_calibration \
  --num 30 --box 185 120 380 445
```

This writes `./calibration_data_norm_512/normalized_images/*.npy`, `./decoder_calibration/image_embeddings/*.npy`, and `./decoder_calibration/boxes/*.npy`. The dump helper runs the float encoder ONNX with the source ImageNet transform and requires host `onnxruntime`; it was not run here. No embedding was generated in this migration.

The dump example uses the committed `../test_data/dogs.jpg` after ONNX export, so its input exists. Replace it with a representative calibration photograph for actual quantization. The helper was derived from the S source and can consume either target's float encoder ONNX with the same tensor protocol; this migration only exercised an injected ORT fixture. The `calibration_images/` set is user-supplied, not bundled. Scaling one embedding and jittering boxes preserves the source demonstration recipe; it does not establish a representative calibration dataset. Only size 512 is covered by the committed configurations and runtime; changing exporter `--size` also requires corresponding new configurations and runtime bindings.

<a id="compile"></a>
## Compile

With the target ONNX and calibration paths present, run the committed dispatcher:

```bash
python3 scripts/quantize.py --target x5
python3 scripts/quantize.py --target s100
python3 scripts/quantize.py --target s100p
python3 scripts/quantize.py --target s600
```

Use `--target <target> --config <path>` to compile one YAML. X5 invokes `hb_mapper makertbin --model-type onnx`; S invokes `hb_compile --config`. X5 outputs are under `bpu_model_output_norm_512_allint16/` and `bpu_model_output_decoder_default/`. S outputs are under target-specific encoder/decoder working directories and use the manifest names in the YAMLs. Compilation was not run here. The committed S paths are:

| target | role | ONNX | calibration | working directory | output prefix |
|---|---|---|---|---|---|
| S100 | encoder | `mobile_sam_image_encoder_norm_512_op11.onnx` | `./calibration_data_norm_512/normalized_images` | `bpu_model_output_encoder_nashe` | `mobile_sam_image_encoder_norm_512x512_nashe` |
| S100 | decoder | `mobile_sam_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings;./decoder_calibration/boxes` | `bpu_model_output_decoder_nashe` | `mobile_sam_decoder_512_nashe` |
| S100P | encoder | `mobile_sam_image_encoder_norm_512_op11.onnx` | `./calibration_data_norm_512/normalized_images` | `bpu_model_output_encoder_nashm` | `mobile_sam_image_encoder_norm_512x512_nashm` |
| S100P | decoder | `mobile_sam_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings;./decoder_calibration/boxes` | `bpu_model_output_decoder_nashm` | `mobile_sam_decoder_512_nashm` |
| S600 | encoder | `mobile_sam_image_encoder_norm_512_op11.onnx` | `./calibration_data_norm_512/normalized_images` | `bpu_model_output_encoder_nashp` | `mobile_sam_image_encoder_norm_512x512_nashp` |
| S600 | decoder | `mobile_sam_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings;./decoder_calibration/boxes` | `bpu_model_output_decoder_nashp` | `mobile_sam_decoder_512_nashp` |

Every config and expected compiler output (relative to this conversion cwd):

| Config | ONNX | Calibration | Compiler output |
| --- | --- | --- | --- |
| `configs/s100/mobile_sam_decoder_512_nashe_config.yaml` | `./mobile_sam_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings;./decoder_calibration/boxes` | `bpu_model_output_decoder_nashe/mobile_sam_decoder_512_nashe.hbm` |
| `configs/s100/mobile_sam_encoder_nashe_config.yaml` | `./mobile_sam_image_encoder_norm_512_op11.onnx` | `./calibration_data_norm_512/normalized_images` | `bpu_model_output_encoder_nashe/mobile_sam_image_encoder_norm_512x512_nashe.hbm` |
| `configs/s100p/mobile_sam_decoder_512_nashm_config.yaml` | `./mobile_sam_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings;./decoder_calibration/boxes` | `bpu_model_output_decoder_nashm/mobile_sam_decoder_512_nashm.hbm` |
| `configs/s100p/mobile_sam_encoder_nashm_config.yaml` | `./mobile_sam_image_encoder_norm_512_op11.onnx` | `./calibration_data_norm_512/normalized_images` | `bpu_model_output_encoder_nashm/mobile_sam_image_encoder_norm_512x512_nashm.hbm` |
| `configs/s600/mobile_sam_decoder_512_nashp_config.yaml` | `./mobile_sam_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings;./decoder_calibration/boxes` | `bpu_model_output_decoder_nashp/mobile_sam_decoder_512_nashp.hbm` |
| `configs/s600/mobile_sam_encoder_nashp_config.yaml` | `./mobile_sam_image_encoder_norm_512_op11.onnx` | `./calibration_data_norm_512/normalized_images` | `bpu_model_output_encoder_nashp/mobile_sam_image_encoder_norm_512x512_nashp.hbm` |
| `configs/x5/mobile_sam_decoder_512_box_default_config.yaml` | `./mobile_sam_decoder_512_box_op11.onnx` | `./decoder_calibration/calibration_embeddings; ./decoder_calibration/calibration_boxes` | `bpu_model_output_decoder_default/mobile_sam_decoder_512_box_default.bin` |
| `configs/x5/mobile_sam_image_encoder_norm_512x512_config.yaml` | `./mobile_sam_image_encoder_norm_512_op11.onnx` | `./calibration_data_norm_512` | `bpu_model_output_norm_512_allint16/mobile_sam_image_encoder_norm_512x512_allint16.bin` |

A single-config command must include its matching `--target`: the script does not infer the compiler from YAML and otherwise defaults to X5. For example: `python3 scripts/quantize.py --target s100 --config configs/s100/mobile_sam_decoder_512_nashe_config.yaml`.

These input fields are copied from each compilation config, not observed SDK metadata; inspect the compiled model afterwards. For X5, type lists runtime / train; for S it is input_type.

| Config | input_name | input_shape | type |
| --- | --- | --- | --- |
| `configs/s100/mobile_sam_decoder_512_nashe_config.yaml` | `image_embeddings;boxes` | `1x256x32x32;1x4` | `featuremap;featuremap / featuremap;featuremap` |
| `configs/s100/mobile_sam_encoder_nashe_config.yaml` | `normalized_images` | `1x3x512x512` | `featuremap / featuremap` |
| `configs/s100p/mobile_sam_decoder_512_nashm_config.yaml` | `image_embeddings;boxes` | `1x256x32x32;1x4` | `featuremap;featuremap / featuremap;featuremap` |
| `configs/s100p/mobile_sam_encoder_nashm_config.yaml` | `normalized_images` | `1x3x512x512` | `featuremap / featuremap` |
| `configs/s600/mobile_sam_decoder_512_nashp_config.yaml` | `image_embeddings;boxes` | `1x256x32x32;1x4` | `featuremap;featuremap / featuremap;featuremap` |
| `configs/s600/mobile_sam_encoder_nashp_config.yaml` | `normalized_images` | `1x3x512x512` | `featuremap / featuremap` |
| `configs/x5/mobile_sam_decoder_512_box_default_config.yaml` | `image_embeddings; boxes` | `1x256x32x32; 1x4` | `featuremap; featuremap / featuremap; featuremap` |
| `configs/x5/mobile_sam_image_encoder_norm_512x512_config.yaml` | `normalized_images` | `1x3x512x512` | `featuremap / featuremap` |

<a id="validation"></a>
## Post-conversion validation

No export, calibration, compile, model download, or board validation was run in this migration. Before accepting an artifact, inspect actual SDK metadata and require matching input/output names, ranks, shapes, and native dtypes for both submodels. Confirm the decoder box input remains `(1,4)` and is in the same 512-pixel coordinate system as preprocessing. Runtime casts cannot prove native quantization metadata. The source contains no dataset-level accuracy harness; any board result must record its own image, box, model, SDK, and resource conditions.

<a id="artifacts"></a>
## Artifacts

| Target | encoder | decoder | expected output |
|---|---|---|---|
| X5 | `mobile_sam_image_encoder_norm_512x512_allint16.bin` | `mobile_sam_decoder_512_box_default.bin` | `../model/` after explicit preparation or local copy |
| S100 | `mobile_sam_image_encoder_norm_512x512_nashe.hbm` | `mobile_sam_decoder_512_nashe.hbm` | `../model/nash-e/` |
| S100P | `mobile_sam_image_encoder_norm_512x512_nashm.hbm` | `mobile_sam_decoder_512_nashm.hbm` | `../model/nash-m/` |
| S600 | `mobile_sam_image_encoder_norm_512x512_nashp.hbm` | `mobile_sam_decoder_512_nashp.hbm` | `../model/nash-p/` |

Generated ONNX, calibration tensors, quantizer metadata, and compiled models remain outside version control. The active manifests and target YAMLs determine exact filenames.

<a id="known-gaps"></a>
## Known gaps

- Upstream source and checkpoint versions/digests are not pinned by the fixed source.
- S has no source `download_assets.py`; acquiring its upstream checkout and checkpoint is a manual prerequisite.
- X5 and S use different calibration file formats and target config layouts, so the unified producer branches on target while retaining the source recipes.
- Decoder output rank and all compiled native dtypes require real SDK metadata inspection.
- Conversion, downloads, and board validation are not-run in this migration.

## Provenance

The source-to-unified mapping and SHA-256 records are in [`SOURCE_MAP.json`](./SOURCE_MAP.json). Existing Apache-2.0 source notices remain represented by the source-derived files; wrapper code follows the repository license.
