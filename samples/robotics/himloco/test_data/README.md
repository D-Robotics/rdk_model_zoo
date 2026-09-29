# HIMLoco offline observations

[中文](README_cn.md)

The 21 `obs_history/*.bin` files and `runtime-input-manifest.json` are preserved
byte-for-byte from X5 commit `ac115717197920355fc390bb04299b20e6436864`.
Each file stores 270 little-endian float32 values (1080 bytes), current 45-value
observation first, followed by five previous observations. There is no header.

The manifest records source indices 0–20 from `rollout_evaluation.pt` (1504 samples),
source digest `49f5459a5ff4d8003d9ee9d95c1104d158688017408a74bcd1506ff171cc01ab`,
and per-file digests. The original rollout is not bundled here; its provenance is
inherited from the source record, not independently regenerated in this migration.
These are held-out offline runtime inputs, not a representative calibration set.

The Python CLI sorts numeric filenames, rejects duplicate indices and validates
the colocated manifest before loading each file. It hashes the same bytes it uses
for inference. To run one file on X5, use `--input-path
samples/robotics/himloco/test_data/obs_history/000000.bin`; see the
[complete runtime commands](../runtime/python/README.md#usage).

No reference action or robot-motion claim is attached to these inputs. Synthetic
SDK outputs in host tests are explicitly fixtures, not model predictions. Numerical
agreement alone cannot establish observation construction or closed-loop behavior.
