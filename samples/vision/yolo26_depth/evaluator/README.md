# Evaluator (YOLO26 Depth, RDK-S)

## Board validation status

The `hrt_model_exec perf` latency benchmark has been run on S100 (nash-e),
S100P (nash-m), and S600 (nash-p); results are filled into the table below.

Board-side numeric evaluation of the quantized YOLO26 Depth models against the
SUNRGBD benchmark.

## Scripts

- `prepare_sunrgbd.py` — prepare the SUNRGBD evaluation subset.
- `eval_sunrgbd.py` — run the quantized `.hbm` over the subset and compute depth
  metrics (e.g. AbsRel / RMSE) plus log-depth cosine vs the float reference.

## Validation results

Board inference was validated on all three boards × all five variants; log-depth
cosine vs float: 0.9985–0.9998.

## Latency

```bash
hrt_model_exec perf --model ../model/nash-e/yolo26n_depth_lite_nashe_768x768.hbm
```

### Measured latency

> To fill after board `hrt_model_exec perf`. Values in ms (BPU forward, single-core default).

| Variant | S100 (nash-e) | S100P (nash-m) | S600 (nash-p) |
|---|---|---|---|
| n | 3.165 | 2.254 | 1.760 |
| s | 4.490 | 3.244 | 2.363 |
| m | 8.246 | 5.986 | 4.062 |
| l | 9.790 | 7.090 | 4.881 |
| x | 19.059 | 12.853 | 9.097 |
