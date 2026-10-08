# HIMLoco model package

[中文](README_cn.md)

<a id="artifacts"></a>
## Artifacts

| File | Format | Target | Role |
| --- | --- | --- | --- |
| `bayes-e/himloco_go2_bayese_1x270.bin` | BIN | X5 only | Fused estimator and actor, obs_history float32 [1,270] → actions float32 [1,12] |

The [active X5 manifest](../../../../docs/release/x5/models.yaml) is authoritative
for the URL and hash. S100/S100P/S600 have no matching publication. The model is
already fused; a separate encoder or policy model cannot replace it by renaming.

<a id="directory"></a>
## Directory structure

```text
model/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── download.py  # Prepare model files
└── download_model.sh  # Shell command
```

<a id="preparation"></a>
## Preparation

From repository root, with Python, NumPy and PyYAML:

```bash
bash samples/robotics/himloco/model/download_model.sh --target x5 --dry-run
bash samples/robotics/himloco/model/download_model.sh --target x5
```

Preview prints identity, URL, destination and expected hash without downloading
or writing files. The second command explicitly prepares the model. Existing files
are checked and never overwritten. New downloads use a temporary file, check the
published digest and install atomically. A mismatch returns 2 and cannot leave a
new complete-looking BIN. Correct the input/path before retrying; do not rename an
incompatible model. `PYTHON` selects the shell helper's interpreter.

`--target` accepts only `x5` (default). `--output-dir` defaults to this model
directory and always receives a `bayes-e/` child. It does not change runtime defaults.
No toolchain or board SDK is required to prepare an existing published artifact.
Run the download explicitly when preparing the artifact; preparation is a host
step, separate from board inference.

<a id="accompanying-files"></a>
## Accompanying files

No vocabulary, label list or external normalization file is required. The caller
must supply correctly constructed six-frame observation history. The bundled
[test inputs](../test_data/README.md) include source indices and SHA-256 values;
they are not calibration data or a live robot state estimator.

<a id="local-paths"></a>
## Local paths

Runtime default: `samples/robotics/himloco/model/bayes-e/himloco_go2_bayese_1x270.bin`.
For an alternate location, provide both `--model-path /absolute/path/model.bin` and
`--asset-id x5:himloco:himloco_go2_bayese_1x270.bin` to the Python entry.
The published digest and board gate still apply; alternate paths do not authorize
a different policy.

<a id="formats-checksums"></a>
## Format and checksum

Published BIN SHA-256:
`7ce46ca2628f8bc236da0e8564180a1de92847bddf1ec00717ce7aa93e8c3e6a`.
Source: active manifest, inherited from the pinned X5 sample. Preparation success
proves file identity against this digest, not SDK compatibility, action accuracy
or closed-loop stability. The runtime additionally checks physical tensor metadata.
