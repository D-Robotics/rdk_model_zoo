# good_sample (fixture)

A deliberately minimal, fully compliant sample used as the positive fixture
for `tools/sample_contract/check.py`.

<a id="overview"></a>
## Overview

Fixture classifier used only by the checker test-suite. Source: this
repository's test fixtures; there is no external upstream.

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Language | Status |
| --- | --- | --- | --- |
| x5 | fixture1 | python | supported-not-run |
| s100 | fixture1 | python | supported-not-run |

C++ is not provided; this sample does not claim dual-language support.

<a id="prerequisites"></a>
## Prerequisites

Host with Python 3.10+ for the unit tests; an RDK board with `hbm_runtime`
is required to run inference, which this fixture never executes.

<a id="quickstart"></a>
## Quick start

From the repository root, prepare the artifact, then run:

```bash
bash samples/tools/fixture/good_sample/model/download.sh --target x5
python3 samples/tools/fixture/good_sample/runtime/python/main.py \
  --target x5 --test-img samples/tools/fixture/good_sample/test_data/input.jpg
```

Success is a zero exit code and a printed top-5 list.

<a id="expected-results"></a>
## Expected results

The run prints the fixture top-5 list and exits 0; no output files are
written by the runtime itself.

<a id="directory"></a>
## Directory

- `model/` — artifact preparation ([README](model/README.md))
- `runtime/python/` — Python entrypoint ([README](runtime/python/README.md))
- `test_data/` — bundled input ([input.jpg](test_data/input.jpg))

<a id="entry-points"></a>
## Entry points

- Model preparation: [model/README.md](model/README.md)
- Python runtime: [runtime/python/README.md](runtime/python/README.md)

No conversion recipe or evaluator is provided for this fixture; see the
sample root README for the scope statement.

<a id="license"></a>
## License

Fixture content follows the repository top-level LICENSE; no additional
model license applies.
