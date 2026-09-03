English | [简体中文](./README_cn.md)

# DINOv2 Model Conversion

This directory provides the ONNX export and PTQ conversion pipeline for the
FAIR DINOv2 ViT-S/14 vision encoder on RDK S100/S100P/S600.

## Files

| File | Description |
|---|---|
| `mapper.py` | One-command conversion entry: ONNX export + calibration prep + hb_compile. |
| `onnx_export/export_dinov2.py` | PyTorch to ONNX export with BPU-friendly rewrites. |

## Quantization Recipe

The validated recipe is **featuremap float32 input + all-int16 + default
(KL) calibration**. All three ingredients are load-bearing and are fixed in
`mapper.py`:

| Ingredient | Why |
|---|---|
| `input_type_rt: featuremap` | The NV12 input chain collapses the executed cosine on embedding models (measured 0.999 simulated vs 0.01-0.12 executed through the YUV420 uint8 round-trip). |
| `all_node_type: int16` | int8 activations fail regardless of calibration (measured ceiling 0.91). Weight-only int8 is nearly free; activation int8 is not. |
| default calibration (`calibration_type` unset) | hmct's modelwise KL search tames the attention-logit outliers of the raw self-supervised backbone (logits-minus-max tensor range reaches -345). `max` + percentile calibration collapses to 0.18. |

## Measured Matrix (nash-e, OE 3.7.0, hmct 2.6.5 / hbdk 4.7.5)

| Config | cls cosine | patch cosine | Verdict |
|---|---|---|---|
| int8 + softmax-int32, featuremap, max | 0.081 | 0.803 | FAIL |
| int8 + softmax-int32, featuremap, KL | 0.892 | 0.894 | FAIL |
| int16, featuremap, max + 0.9999 | 0.184 | 0.840 | FAIL |
| int16, nv12, KL | 0.999 (sim) | 0.999 (sim) | FAIL (executed 0.01 / 0.12) |
| int16, featuremap, KL (this recipe) | **0.9989** | **0.9983** | **PASS** |

The register-token (`_reg4`) variant was measured and is intentionally not
shipped: its quantized cosine (0.80) is worse than the plain variant (0.999)
under per-tensor calibration, despite being the paper-recommended variant for
dense-feature quality.

## Usage

Run inside the OE docker image on an x86 host:

```bash
# 1. The OE 3.7 image provides Torch 2.6; preserve it. The exporter supports
# Torch >=2.4 and this setup was validated with these ONNX packages.
python3 -c "import torch; print(torch.__version__)"
pip install --upgrade onnx==1.19.0 onnxruntime==1.23.2
python3 -c "import onnx, onnxruntime; print(onnx.__version__, onnxruntime.__version__)"

# 2. Fetch the pinned source and official Apache-2.0 checkpoint.
git clone https://github.com/facebookresearch/dinov2.git
cd dinov2
git checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
cd ..
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
printf '%s  %s\n' \
  b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9 \
  dinov2_vits14_pretrain.pth | sha256sum -c -

# 3. Put 50 diverse real images (e.g. COCO val2017) into ./cal_images.

# 4. Convert.
python3 mapper.py \
    --weights ./dinov2_vits14_pretrain.pth \
    --weights-sha256 b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9 \
    --repo ./dinov2 \
    --repo-revision 7764ea0f912e53c92e82eb78a2a1631e92725fc8 \
    --cal-images ./cal_images \
    --march nash-e \
    --output-dir ./output
```

Do not run `pip install torch` or `pip install -r requirements.txt`. The
pinned upstream DINOv2 requirements install torch 2.0, which would downgrade
the OE Torch environment and is incompatible with this exporter's
`dynamo=False` path. Keep the OE 3.7 Torch 2.6 installation and install only
the validated `onnx==1.19.0` and `onnxruntime==1.23.2` additions above.

The exporter accepts either a local `--weights` path or an HTTP(S) URL. For an
HTTP(S) URL, it streams the download to a temporary file, verifies the complete
SHA-256 before loading, and removes the temporary file in `finally`; it does
not leave a final checkpoint on disk. Keep `--repo-revision` and
`--weights-sha256` pinned as above for reproducible conversion.

Calibration images must be real photographs. Random or synthetic data breaks
int16 calibration on this backbone. Their preprocessing matches evaluation:
OpenCV BGR to RGB, bicubic short-side resize to 256 while preserving aspect
ratio, center crop 224 by 224, `/255`, ImageNet mean/std normalization, then
contiguous float32 NCHW.

## OE Resources

Run model conversion on an x86 Linux host with the RDK S100/S600 OpenExplore
environment (OE 3.7.0, image `ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0`).
Model conversion is not intended to run on the board.

```bash
sudo docker run -it --rm --network host --shm-size=15g \
    -v "$(pwd)":/workspace -w /workspace \
    registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0 \
    /bin/bash
```

## License

The source weights are Apache-2.0 licensed
[DINOv2](https://github.com/facebookresearch/dinov2) artifacts published by
Meta AI. See [../../../../LICENSE](../../../../LICENSE).
