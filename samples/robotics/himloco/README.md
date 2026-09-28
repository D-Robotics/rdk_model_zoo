# HIMLoco fused Go2 policy

[中文](README_cn.md)

<a id="overview"></a>
## Overview

HIMLoco estimates internal state from six 45-value observations and produces 12
policy actions. This sample uses the fused estimator/actor exported by
[himloco_lab](https://github.com/IsaacZH/himloco_lab), an Isaac Lab implementation of
[HIMLoco](https://github.com/OpenRobotLab/HIMLoco). Independently trained checkpoints
are not interchangeable. The migration source is X5 commit
`ac115717197920355fc390bb04299b20e6436864`.

The unified Python SDK entry, explicit model preparation and offline inputs are
implemented. Source conversion/evaluation tools and bilingual guides are now in
the unified directory; native C++ migration remains in progress. Quantization
instructions are inherited from the existing source scheme; no recipe rerun is required for this documentation work. This is not yet
whole-sample independent acceptance.

Input: `obs_history`, float32 `[1,270]`, current frame first. Output: `actions`,
float32 `[1,12]`. No additional normalization, history update, output scaling or
robot command is performed. The source controller applies
`default_joint_position + 0.25 * actions` outside this model boundary.

<a id="support-matrix"></a>
## Support matrix

| Target | Artifact | Unified Python | Unified C++ |
| --- | --- | --- | --- |
| X5 | Bayes-e BIN | Implemented; host SDK-double tests; board not-run | Migration pending; source implementation retained |
| S100 / S100P / S600 | No published matching artifact | Not supported | Not supported |

The source board environment was RDK OS 3.5.0-beta, DNN Runtime 1.24.5 and HBRT
3.15.55. That is historical source evidence, not a newly verified unified version.

<a id="prerequisites"></a>
## Prerequisites

Host preview and core tests require Python, NumPy and PyYAML. Real inference
requires an X5 with its matching BSP `hbm_runtime` and the exact published BIN.
Do not install the unrelated PyPI package with the same name. Conversion tools,
Torch and a training environment are not required to use the published model.
All commands below run from repository root. Set `PYTHON` for shell wrappers when
using a virtual environment.

<a id="quickstart"></a>
## Quick start

```bash
python samples/robotics/himloco/runtime/python/main.py --list-models
python samples/robotics/himloco/runtime/python/main.py --target x5 --dry-run
bash samples/robotics/himloco/model/download_model.sh --target x5 --dry-run
```

These previews do not load a board SDK, download, or write results. Prepare the
published model explicitly, then run offline observations on X5:

```bash
bash samples/robotics/himloco/model/download_model.sh --target x5
python samples/robotics/himloco/runtime/python/main.py --target x5 \
  --output-dir outputs/himloco
```

Each run needs a new output directory. No implicit model download occurs in
inference; mismatched targets or model hashes fail before SDK construction.
See the Python guide for single-file input, external model identity, scheduling,
report location, warmup and library integration.

<a id="expected-results"></a>
## Expected results and historical measurements

The default run processes 21 source-indexed observations after 10 warmups and
writes `000000.bin` through `000020.bin`, each 12 little-endian float32 actions,
plus `report.json`. A complete report has `status: completed`; a writable failure
report retains partial files and the failed index. Output files are numerical
evidence, not model assets or actuator commands. No live control loop is included.

The source recorded MIX PTQ output cosine `0.999606` and these measurements on the
same model, 100 inputs, 10 warmups:

| Source runtime | Timing scope | Mean | Sequential throughput |
| --- | --- | --- | --- |
| Python | `HB_HBMRuntime.run` | 0.885 ms | 1129.37 FPS |
| C++ | `hbDNNInfer` + `hbDNNWaitTaskDone` | 0.350 ms | 2853.09 FPS |

The compiler estimate was 0.063 ms. These are inherited source values, not this
migration's results. Timing scopes differ; the new Python task measures its bound
runner including adapter validation/copy and must not be compared as device-only
latency. Full source percentiles and evaluation semantics remain in the
[evaluator guide](evaluator/README.md).
Offline action agreement does not establish observation construction, joint
mapping, control-loop behavior or closed-loop stability.

<a id="directory"></a>
## Directory layout

- `model/`: explicit acquisition of the hash-pinned X5 BIN.
- `runtime/python/`: CLI/application, input provenance, binding/shared runner and pure policy stages.
- `test_data/`: 21 unchanged observation files and their source manifest.
- `tests/`: core, metadata and CLI checks with explicit model/SDK fixtures.
- `conversion/`: source fused export, calibration and Mapper recipe with bilingual guides.
- `evaluator/`: format/action comparisons, input preparation and historical measurements.
- Native C++: migration pending; source content retained.

<a id="entry-points"></a>
## Entry points

- [Model package](model/README.md): identity, paths, preparation and checksum.
- [Python runtime](runtime/python/README.md): commands, all options, outputs and public API.
- [Input provenance](test_data/README.md): source indices, byte layout and digest checks.
- [Source C++ guide](../../../platforms/x5/samples/robotics/himloco/runtime/cpp/README.md): historical native implementation, pending unified migration.
- [Model conversion](conversion/README.md): fused export/calibration/MIX recipe.
- [Model evaluation](evaluator/README.md): data, commands, metrics and source references.

<a id="license"></a>
## License

Sample code follows the repository [Apache-2.0 license](../../../LICENSE). Upstream
policies and rollout data retain their respective terms; the code license does
not establish rights to every external model or dataset.
