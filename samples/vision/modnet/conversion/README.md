English | [简体中文](README_cn.md)

# MODNet conversion

<a id="source-model"></a>
## Source model

The public photographic model uses official [MODNet source](https://github.com/ZHKKKe/MODNet/tree/28165a451e4610c9d77cfdf925a94610bb2810fb), pinned to `28165a451e4610c9d77cfdf925a94610bb2810fb`. Its [official checkpoint folder](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR) identifies `modnet_photographic_portrait_matting.ckpt`, file ID `1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz`. The observed snapshot SHA-256 is `7c22235f0925deba15d4d63e53afcb654c47055bbcd98f56e393ab2584007ed8` (26,255,603 bytes). The author provides no publisher checksum; the observed digest fixes this conversion's checkpoint bytes.

The upstream [license statement](https://github.com/ZHKKKe/MODNet/blob/28165a451e4610c9d77cfdf925a94610bb2810fb/README.md#license) covers code, models and demos under Apache-2.0, excluding `doc/gif`. Preserve the upstream LICENSE and copyright notices.

<a id="directory"></a>
## Directory structure

`prepare.py` exports the official single-matte inference model, compares it with native PyTorch inference, validates calibration sources and writes an external X5 config. Existing README files retain the historical manual model context. Model binaries and calibration tensors are external.

<a id="toolchain-targets"></a>
## Toolchain and targets

X5 only: OpenExplorer image `registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8`, observed image ID `sha256:bb706c129bafb4d36d2c338c957369cc5fdb51e0eb65ad5c927b3fda4023ad3f`, Mapper 1.24.3. The export run used PyTorch 2.8.0, ONNX 1.17.0 and ONNX Runtime 1.19.2. This is PTQ conversion, with board performance and matting accuracy measured separately later.

<a id="export"></a>
## Export and calibration preparation

Clone the official repository into an external work directory, check out the exact commit, and download the official checkpoint. The source checkout must be clean. Prepare the fixed COCO person train2017 calibration set (200 images, seed 42) with a `manifest.json` containing `count`, `seed`, and `images` entries with `name`/`sha256`.

From this repository root, with the export dependencies already available:

```bash
python3 samples/vision/modnet/conversion/prepare.py \
  --source-dir /work/MODNet \
  --checkpoint /work/modnet_photographic_portrait_matting.ckpt \
  --checkpoint-sha256 7c22235f0925deba15d4d63e53afcb654c47055bbcd98f56e393ab2584007ed8 \
  --calibration-dir /work/coco-person-keypoints-train2017-seed42-200 \
  --output /work/modnet-run/prepared
```

The output directory must be new. Export is static opset 11, batch 1, with float32 input `input` `[1,3,512,512]` and matte `matte` `[1,1,512,512]`. The official ONNX module omits training-only semantic/detail outputs. Native `MODNet.forward(x, inference=True)[2]` is the reference. Three real calibration inputs and a deterministic random input must be finite and agree with ONNX Runtime (`atol=1e-4`, `rtol=1e-3`).

<a id="calibration"></a>
## Calibration contract

Calibration calls Runtime's `preprocess_image`: BGR→RGB, `(pixel-127.5)/127.5`, long-side resize with `INTER_AREA`, centered zero padding to 512, float32 NCHW. Each source hash is checked; every tensor gets a hash and statistics. Three written tensors are reread for the graph comparison. The YAML uses `featuremap`, NCHW and `no_preprocess`, so normalization is applied exactly once. The representative person images support initial conversion; matting accuracy needs a separate labeled evaluation set.

<a id="compile"></a>
## Compile

Inside the matching X5 environment, with all external paths mounted:

```bash
hb_mapper checker --model-type onnx --march bayes-e \
  --model /work/modnet-run/prepared/modnet_photographic_portrait_512x512_rgb_bayese_20261102.onnx
hb_mapper makertbin --model-type onnx --config /work/modnet-run/prepared/config.yaml
hb_model_info /work/modnet-run/output/modnet_photographic_portrait_512x512_rgb_bayese_20261102.bin
```

The config selects `bayes-e`, default calibration, O3 latency, one BPU core and two compiler jobs. Float32 `featuremap` input uses `input_source: {input: ddr}`; `pyramid` is incompatible with this tensor input. Keep each attempt's config/logs and use a fresh working directory.

<a id="validation"></a>
## Post-conversion validation

Verify checker/compiler completion, one BIN, `BPU march: bayes-e`, and actual float32 NCHW input and matte output metadata. Preserve the original OE output, node cosine and static compiler reports. Those reports are not board latency or matting accuracy. The repository Runtime binds the new artifact by exact asset ID and validates metadata at load.

<a id="artifacts"></a>
## Artifacts

The new profile is `modnet_photographic_portrait_512x512_rgb_bayese_20261102.bin`. Its identity is separate from historical manual `modnet_512x512_rgb.bin`. `prepare-report.json`, `calibration-manifest.json`, `preprocess-report.md`, YAML, ONNX, checkpoint identity, compiler logs, model-info and model digest accompany the conversion.

<a id="known-gaps"></a>
## Historical model and remaining measurements

The checkpoint/exporter/PTQ settings behind the historical manual `modnet_512x512_rgb.bin` remain unknown. Its historical performance table is preserved and does not describe the new photographic artifact. New board performance, board Runtime execution and labeled matting accuracy remain pending.
