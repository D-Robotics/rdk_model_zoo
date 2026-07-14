[简体中文](./README_cn.md)

# Conversion

This directory contains every script and compiler configuration needed to reproduce the fixed-shape INT16 Paraformer deployment. The complete step-by-step explanation, graph-surgery rationale, calibration requirements, commands, troubleshooting notes, and deployment procedure are in `README_cn.md`.

Run the scripts from this directory in the numbered order: fixed-shape FunASR export, Decoder/Predictor/Encoder extraction, ONNX graph fixes, real-data calibration generation, and three HBM compilations. `11_eval_pipeline.py` evaluates FP32 or INT16 CER; `cif_numpy.py` is the CPU CIF reference used by the runtime.

```bash
cd conversion
hb_compile -c configs/encoder_int16.yaml
hb_compile -c configs/predictor_int16.yaml
hb_compile -c configs/decoder_int16.yaml
```

Use representative real data for calibration. Generated ONNX files, calibration tensors, logs, and HBM files are local build artifacts and must not be committed.
