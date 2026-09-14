English | [简体中文](./README_cn.md)

# KWS Model Evaluation

This directory records performance and functional validation notes for the KWS (Keyword Spotting) model on RDK S100.

## Performance Data

Use `hrt_model_exec` to test HBM model performance:

```bash
hrt_model_exec perf --model_file /root/kws/kws.hbm --frame_count 100
```

Reference result on RDK S100:

| Metric | Value |
|---|---|
| Frames | 100 |
| Average Latency | 1.176 ms |
| FPS | 830.875 |

## Functional Check

Run the Python runtime on the sample audio file `sample.wav`:

```bash
cd runtime/python
bash run.sh
```

The output prints the keyword confidence score. The provided `sample.wav` contains the wake-up keyword **"hey snips"**.

Expected output:

```text
Keyword confidence score: 0.985...
```

Under the sample audio, the confidence for the keyword "hey snips" should be approximately 98.5%.

## Notes

- The model uses MDTC (Multi-Scale Dynamic Temporal Convolution) architecture optimized for edge deployment.
- A confidence score above 0.5 generally indicates keyword detected. The threshold can be adjusted in `kws.py`.
