# Runtime (YOLO26 Depth, RDK-S)

Board-side monocular depth inference for YOLO26 Depth using `hbm_runtime`.
Python-only (the X5 sample also ships a C++ runtime; this RDK-S port keeps the
Python pipeline, which is sufficient for the demo and evaluation flow).

## Run

```bash
bash run.sh        # default variant `n`
bash run.sh m       # variant `m`
```

`run.sh` auto-detects the board, downloads the matching `.hbm` if missing, then
runs `python3 main.py --variant <v> --input ../../test_data/bus.jpg --output ./output`.

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | auto | Override the `.hbm` path. |
| `--variant` | `n` | Model variant `n`/`s`/`m`/`l`/`x`. |
| `--input` | (required) | Input image. |
| `--output` | (required) | Output directory. |
| `--warmup` | `3` | Unmeasured inference calls before timing. |
| `--priority` | `0` | Model scheduling priority. |
| `--bpu-cores` | `0` | BPU core indexes. |

## Output

- `depth.png` — Turbo-colorized depth.
- `overlay.png` — depth blended over the input image.
- `raw_logit.npy` (lite only) / `log_depth.npy` / `depth_native.npy` — raw and decoded tensors.
- `report.json` — model/input SHA-256, shapes, latency, and postprocess constants.

## Files

- `main.py` — entry; resolves board + variant, loads the `.hbm`, runs inference.
- `yolo26_depth.py` — `Yolo26Depth`, one class with two internal profiles selected
  per variant: NV12 (letterbox → NV12 → in-graph decode) for `n`/`s`/`m`, and
  lite (scale-fill → RGB `/255` float32 NCHW featuremap → `HB_HBMRuntime` →
  external decode) for `l`/`x`. Outputs are identical across profiles.
- `run.sh` — launcher with auto board detection + download-if-missing.
