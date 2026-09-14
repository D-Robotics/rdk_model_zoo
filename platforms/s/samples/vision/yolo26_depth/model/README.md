# Model (YOLO26 Depth, RDK-S)

Pre-quantized `.hbm` models for the five variants across three marches, under
the mixed release profile (see the sample README). Binaries are **not**
committed; `download_model.sh` fetches them from the model server.

Naming follows the profile split:

- `n` / `s` / `m` (NV12 profile): `yolo26{variant}_depth_{suffix}_768x768_nv12.hbm`
- `l` / `x` (lite profile): `yolo26{variant}_depth_lite_{suffix}_768x768.hbm`

where `{variant}` ∈ `n/s/m/l/x` and `{suffix}` ∈ `nashe/nashm/nashp`.

| Board | march | Dir |
|---|---|---|
| S100 | nash-e | `model/nash-e/` |
| S100P | nash-m | `model/nash-m/` |
| S600 | nash-p | `model/nash-p/` |

## Download

```bash
bash download_model.sh                 # auto-detect board, all 5 variants
bash download_model.sh nash-e          # one march, all variants
bash download_model.sh nash-e n        # one march, one variant
```
