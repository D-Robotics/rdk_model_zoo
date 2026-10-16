English | [简体中文](README_cn.md)

# HIMLoco offline observations

[中文](README_cn.md)


## Directory structure

```text
test_data/
├── obs_history/  # Files for obs_history
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── runtime-input-manifest.json  # Structured data
```


The 21 `obs_history/*.bin` files and `runtime-input-manifest.json` contain the
offline example observations.
Each file stores 270 little-endian float32 values (1080 bytes), current 45-value
observation first, followed by five previous observations. There is no header.

The manifest records the per-file mapping to source indices 0–20 in
`rollout_evaluation.pt` (1504 samples); the original rollout is not bundled here.
These files are offline runtime inputs only; for calibration, use representative
observation data.

The Python CLI sorts numeric filenames, rejects duplicate indices and validates
each input against the colocated manifest before loading it. To run one file on
X5, use `--input-path
samples/robotics/himloco/test_data/obs_history/000000.bin`; see the
[complete runtime commands](../runtime/python/README.md#usage).

Construct the six-frame observation history using the policy’s training convention. The runtime returns twelve raw actions; controller scaling is described in the runtime guide.
