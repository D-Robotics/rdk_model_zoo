English | [简体中文](./README_cn.md)

# EfficientSAM conversion

This directory records the source conversion capability for the EfficientSAM ViT-Tiny encoder and fixed two-positive-point decoder. The unified scripts select X5 or RDK-S with `--target`; the conversion flow has not been run in this migration.

<a id="source-model"></a>
## Source model

The upstream source is [yformer/EfficientSAM](https://github.com/yformer/EfficientSAM), using the source checkpoint `weights/efficient_sam_vitt.pt`. The fixed source does not provide a checkpoint version or SHA-256. X5 and S export wrappers differ because the source branches expose different EfficientSAM builder APIs; `scripts/export_encoder_onnx.py` and `scripts/export_decoder_onnx.py` retain those branches behind `--target`.

The decoder bakes two positive points `(248, 210)` and `(302, 315)` into ONNX. It has no runtime prompt tensor. The encoder contract is `batched_images` `(1,3,512,512)` float32 RGB NCHW → `image_embeddings` `(1,256,32,32)` float32. The decoder consumes that embedding and emits `low_res_masks` plus `iou_predictions`.

<a id="toolchain-targets"></a>
## Toolchain and targets

Run conversion on an x86 host in the toolchain matching the target. These commands are documented source commands and were not run here.

| Target | march/config directory | source toolchain | quantizer |
|---|---|---|---|
| `x5` | `configs/x5/` | `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310` | `hb_mapper makertbin` |
| `s100` | `configs/s100/` | OE S-series 3.7.0 environment | `hb_compile` / `nash-e` |
| `s100p` | `configs/s100p/` | OE S-series 3.7.0 environment | `hb_compile` / `nash-m` |
| `s600` | `configs/s600/` | OE S-series 3.7.0 environment | `hb_compile` / `nash-p` |

The active published model manifest is `docs/release/x5/models.yaml` or `docs/release/s/models.yaml`; all published model SHA fields are `null` (unknown). S has separate encoder and decoder YAML files for all three marches. X5 has two bayes-e YAML files.

Export and float embedding generation require host PyTorch, ONNX, ONNX Runtime, NumPy and OpenCV. The fixed sources do not pin their package versions or the upstream repository revision; this remains a reproducibility prerequisite, not a tested environment specification. Check these imports inside the chosen export environment, separate from board inference:

```bash
# cwd: this conversion directory, inside the chosen host export environment
python3 -c "import torch, onnx, onnxruntime, numpy, cv2; print(torch.__version__, onnx.__version__, onnxruntime.__version__)"
```

<a id="export"></a>
## Export ONNX

Set the working directory to `samples/vision/efficient_sam/conversion`. For X5, prepare the upstream tree and default checkpoint with the source helper; the X5 upstream builder reads only `<repo>/weights/efficient_sam_vitt.pt`, and the exporter rejects a different `--checkpoint` path. For S, the fixed source has no download helper, so place the upstream checkout and checkpoint manually at the paths below.

```bash
# X5 only; source helper, not run in this migration
python3 scripts/download_assets.py --target x5 --workspace ./workspace

# S targets: manually provide ./workspace/EfficientSAM and
# ./workspace/EfficientSAM/weights/efficient_sam_vitt.pt.
```

The X5 helper first tries `git clone`; if that fails it downloads the upstream `main` source ZIP and safely extracts only the `EfficientSAM-main/` tree in a private staging directory before moving it into the workspace. It then downloads `efficient_sam_vitt.pt` when the existing file is absent or smaller than 10,000,000 bytes. These are explicit user-invoked actions; the upstream `main` ZIP is not a pinned source revision. The S source has no equivalent helper and therefore has no automatic source or checkpoint download path.

Export the encoder and fixed-prompt decoder. Every command declares its target and output.

```bash
python3 scripts/export_encoder_onnx.py \
  --target x5 \
  --repo ./workspace/EfficientSAM \
  --checkpoint ./workspace/EfficientSAM/weights/efficient_sam_vitt.pt \
  --output ./efficient_sam_vitt_encoder_512_splitqkv_op11.onnx

python3 scripts/export_decoder_onnx.py \
  --target x5 \
  --repo ./workspace/EfficientSAM \
  --checkpoint ./workspace/EfficientSAM/weights/efficient_sam_vitt.pt \
  --output ./efficient_sam_vitt_decoder_fixedprompt_512_op11.onnx
```

For S100/S100P/S600, use `--target s100`, `--target s100p`, or `--target s600`; use `efficient_sam_vitt_encoder_512_op11.onnx` and `efficient_sam_vitt_decoder_512_op11.onnx` as the two output names. Success is an `Exported ... (N bytes)` line and two ONNX files in this conversion directory. The exporter uses opset 11 and fixed size 512 unless overridden.

<a id="calibration"></a>
## Calibration

Calibration must use real RGB photographs. The encoder tensors are float32 RGB CHW with `/255`; the source producer requires at least 20 output files. It repeats source paths when fewer files are available, so that count does not establish 20 independent representative images. The decoder calibration must start from a real float32 encoder embedding of exactly `(1,256,32,32)`, produced either by the exported float ONNX encoder or by the compiled encoder route. The fixed prompt is already in the decoder, so no decoder prompt tensor is generated.

```bash
# X5: cwd samples/vision/efficient_sam/conversion
python3 scripts/prepare_calibration.py \
  --target x5 --src ./calibration_images --out . --num 30 --size 512
# writes ./calibration_data_rgbchw_512/*.rgbchw

python3 scripts/dump_encoder_embedding.py --image ../test_data/dogs.jpg \
  --onnx ./efficient_sam_vitt_encoder_512_splitqkv_op11.onnx --output ./encoder_embedding.bin

python3 scripts/prepare_efficient_decoder_calibration.py \
  --target x5 --embedding ./encoder_embedding.bin \
  --out ./decoder_calibration --num 30
# writes ./decoder_calibration/calibration_embeddings/*.bin
```

For S, use an explicit calibration root so the generated paths match the S YAMLs:

```bash
python3 scripts/prepare_calibration.py --target s100 \
  --src ./calibration_images --out ./calibration_data --num 30 --size 512
python3 scripts/dump_encoder_embedding.py --image ../test_data/dogs.jpg \
  --onnx ./efficient_sam_vitt_encoder_512_op11.onnx --output ./encoder_embedding.bin
python3 scripts/prepare_efficient_decoder_calibration.py --target s100 \
  --embedding ./encoder_embedding.bin --out ./decoder_calibration --num 30
```

This writes `./calibration_data/batched_images/*.npy` and `./decoder_calibration/image_embeddings/*.npy`, matching the S YAMLs. The dump helper runs the exported float ONNX encoder with the source RGB `/255` transform; it requires host `onnxruntime` and was not run here. No host or board inference was run in this migration.

The dump example uses the committed `../test_data/dogs.jpg` after ONNX export, so its input exists. Replace it with a representative calibration photograph for actual quantization. The helper was derived from the S source and can consume either target's float encoder ONNX with the same tensor protocol; this migration only exercised an injected ORT fixture. The `calibration_images/` set is user-supplied, not bundled. Scaling one embedding preserves the source demonstration recipe; it does not establish a representative calibration dataset. Only size 512 is covered by the committed configurations and runtime; changing exporter `--size` also requires corresponding new configurations and runtime bindings.

<a id="compile"></a>
## Compile

Run from this directory after the target-specific ONNX and calibration files exist. The command only dispatches the committed source recipe; it does not download inputs or choose a model automatically.

```bash
# X5: emits the two .bin files under the OE working directories
python3 scripts/quantize.py --target x5

# S100 / S100P / S600: emits the two .hbm files for the selected march
python3 scripts/quantize.py --target s100
python3 scripts/quantize.py --target s100p
python3 scripts/quantize.py --target s600
```

Use `--target <target> --config <path>` to compile one committed YAML. X5 invokes `hb_mapper makertbin --model-type onnx`; S invokes `hb_compile --config`. The X5 YAMLs use the source default calibration and do not explicitly declare all-node int16. The S YAMLs use `set_all_nodes_int16`, max calibration, percentile `0.9999`, and target-specific Nash marches. Their exact committed paths are:

| target | role | ONNX | calibration | working directory | output prefix |
|---|---|---|---|---|---|
| S100 | encoder | `efficient_sam_vitt_encoder_512_op11.onnx` | `./calibration_data/batched_images` | `bpu_model_output_encoder_nashe` | `efficient_sam_vitt_encoder_512x512_nashe` |
| S100 | decoder | `efficient_sam_vitt_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings` | `bpu_model_output_decoder_nashe` | `efficient_sam_vitt_decoder_512_nashe` |
| S100P | encoder | `efficient_sam_vitt_encoder_512_op11.onnx` | `./calibration_data/batched_images` | `bpu_model_output_encoder_nashm` | `efficient_sam_vitt_encoder_512x512_nashm` |
| S100P | decoder | `efficient_sam_vitt_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings` | `bpu_model_output_decoder_nashm` | `efficient_sam_vitt_decoder_512_nashm` |
| S600 | encoder | `efficient_sam_vitt_encoder_512_op11.onnx` | `./calibration_data/batched_images` | `bpu_model_output_encoder_nashp` | `efficient_sam_vitt_encoder_512x512_nashp` |
| S600 | decoder | `efficient_sam_vitt_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings` | `bpu_model_output_decoder_nashp` | `efficient_sam_vitt_decoder_512_nashp` |

Every config and expected compiler output (relative to this conversion cwd):

| Config | ONNX | Calibration | Compiler output |
| --- | --- | --- | --- |
| `configs/s100/efficient_sam_decoder_nashe_config.yaml` | `./efficient_sam_vitt_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings` | `bpu_model_output_decoder_nashe/efficient_sam_vitt_decoder_512_nashe.hbm` |
| `configs/s100/efficient_sam_encoder_nashe_config.yaml` | `./efficient_sam_vitt_encoder_512_op11.onnx` | `./calibration_data/batched_images` | `bpu_model_output_encoder_nashe/efficient_sam_vitt_encoder_512x512_nashe.hbm` |
| `configs/s100p/efficient_sam_decoder_nashm_config.yaml` | `./efficient_sam_vitt_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings` | `bpu_model_output_decoder_nashm/efficient_sam_vitt_decoder_512_nashm.hbm` |
| `configs/s100p/efficient_sam_encoder_nashm_config.yaml` | `./efficient_sam_vitt_encoder_512_op11.onnx` | `./calibration_data/batched_images` | `bpu_model_output_encoder_nashm/efficient_sam_vitt_encoder_512x512_nashm.hbm` |
| `configs/s600/efficient_sam_decoder_nashp_config.yaml` | `./efficient_sam_vitt_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings` | `bpu_model_output_decoder_nashp/efficient_sam_vitt_decoder_512_nashp.hbm` |
| `configs/s600/efficient_sam_encoder_nashp_config.yaml` | `./efficient_sam_vitt_encoder_512_op11.onnx` | `./calibration_data/batched_images` | `bpu_model_output_encoder_nashp/efficient_sam_vitt_encoder_512x512_nashp.hbm` |
| `configs/x5/efficient_sam_vitt_decoder_fixedprompt_512_default_config.yaml` | `./efficient_sam_vitt_decoder_fixedprompt_512_op11.onnx` | `./decoder_calibration/calibration_embeddings` | `bpu_model_output_decoder_fixedprompt/efficient_sam_vitt_decoder_fixedprompt_512_default.bin` |
| `configs/x5/efficient_sam_vitt_encoder_featuremap_config.yaml` | `./efficient_sam_vitt_encoder_512_splitqkv_op11.onnx` | `./calibration_data_rgbchw_512` | `bpu_model_output_512_default_none/efficient_sam_vitt_encoder_512x512_default_none.bin` |

A single-config command must include its matching `--target`: the script does not infer the compiler from YAML and otherwise defaults to X5. For example: `python3 scripts/quantize.py --target s100 --config configs/s100/efficient_sam_decoder_nashe_config.yaml`.

These input fields are copied from each compilation config, not observed SDK metadata; inspect the compiled model afterwards. For X5, type lists runtime / train; for S it is input_type.

| Config | input_name | input_shape | type |
| --- | --- | --- | --- |
| `configs/s100/efficient_sam_decoder_nashe_config.yaml` | `image_embeddings` | `1x256x32x32` | `featuremap / featuremap` |
| `configs/s100/efficient_sam_encoder_nashe_config.yaml` | `batched_images` | `1x3x512x512` | `featuremap / featuremap` |
| `configs/s100p/efficient_sam_decoder_nashm_config.yaml` | `image_embeddings` | `1x256x32x32` | `featuremap / featuremap` |
| `configs/s100p/efficient_sam_encoder_nashm_config.yaml` | `batched_images` | `1x3x512x512` | `featuremap / featuremap` |
| `configs/s600/efficient_sam_decoder_nashp_config.yaml` | `image_embeddings` | `1x256x32x32` | `featuremap / featuremap` |
| `configs/s600/efficient_sam_encoder_nashp_config.yaml` | `batched_images` | `1x3x512x512` | `featuremap / featuremap` |
| `configs/x5/efficient_sam_vitt_decoder_fixedprompt_512_default_config.yaml` | `image_embeddings` | `1x256x32x32` | `featuremap / featuremap` |
| `configs/x5/efficient_sam_vitt_encoder_featuremap_config.yaml` | `batched_images` | `1x3x512x512` | `featuremap / featuremap` |

<a id="validation"></a>
## Post-conversion validation

No export, calibration, compile, or board validation was run in this migration: status is `not-run`. The source historical X5 quantization records are preserved in [`QUANTIZATION_STATUS.md`](./QUANTIZATION_STATUS.md) and [`VALIDATION.md`](./VALIDATION.md), including encoder cosine `0.968013` and decoder cosines `low_res_masks=0.965641`, `iou_predictions=0.997313`. These are source historical values, not a new result. S source evaluator material is performance-only and has no dataset-level accuracy harness.

Before claiming a compiled artifact usable, inspect the actual SDK metadata for every target/march and verify input names, ranks, shapes, native dtypes, output names, ranks, shapes, and native dtypes. The source does not provide compiled HBM metadata; runtime float casts cannot establish native dtype.

<a id="artifacts"></a>
## Artifacts

| Target | Encoder | Decoder | expected output location |
|---|---|---|---|
| X5 | `efficient_sam_vitt_encoder_512x512_default_none.bin` | `efficient_sam_vitt_decoder_fixedprompt_512_default.bin` | `../model/` after explicit model preparation or local copy |
| S100 | `nash-e/efficient_sam_vitt_encoder_512x512_nashe.hbm` | `nash-e/efficient_sam_vitt_decoder_512_nashe.hbm` | `../model/nash-e/` |
| S100P | `nash-m/efficient_sam_vitt_encoder_512x512_nashm.hbm` | `nash-m/efficient_sam_vitt_decoder_512_nashm.hbm` | `../model/nash-m/` |
| S600 | `nash-p/efficient_sam_vitt_encoder_512x512_nashp.hbm` | `nash-p/efficient_sam_vitt_decoder_512_nashp.hbm` | `../model/nash-p/` |

The conversion output prefixes and runtime filenames are recorded in the target YAMLs and active manifests. Generated ONNX, calibration tensors, quantization metadata, and compiled models are not checked in here.

<a id="known-gaps"></a>
## Known gaps

- The upstream checkpoint has no pinned source version or digest in the fixed source.
- The X5 upstream builder reads only `<repo>/weights/efficient_sam_vitt.pt`; a different checkpoint path is rejected explicitly. S uses its explicit `--checkpoint` path.
- The S source has no `download_assets.py`; source acquisition is a manual prerequisite.
- X5 and S use different upstream builder APIs and different quantizer/config layouts; the unified scripts branch explicitly instead of pretending the recipes are byte-identical.
- Compiled artifact native dtype and S output spatial metadata remain unknown until binding reads real SDK metadata.
- Conversion, model download, and board validation are not-run in this migration.

## Provenance

The merge and source SHA-256 mapping are recorded in [`SOURCE_MAP.json`](./SOURCE_MAP.json). Source code and documentation retain the upstream Apache-2.0 provenance where present and the repository license applies to the unified wrapper code.
