# PointNet independent host review — changes required

Reviewer: Codex. Base `ba05780f`. B8/PointNet remains open. This review found a
blocking defect in the documented default CLI path; the author suite's 21 tests
passed on the current Python 3.14.7 host but do not exercise plotting.

## POINTNET-R1 — default plotting fails on supported Python versions (P1)

`runtime/python/visualization.py:13` retains an unrelated `parse_args` function
from the source CLI and annotates its return as `argparse.Namespace`, without
importing argparse. Python 3.13 raises NameError while importing the module.
`runtime/python/main.py:65` imports this module whenever `--no-plot` is absent,
which is the default quick-start path. The CLI does not catch NameError and
cannot write its final result report after this failure. README declares Python
3.10+ support. Python 3.14's deferred annotation behavior masks this import error;
passing the current host suite is therefore insufficient.

The source visualization module was imported with explicit NumPy/matplotlib
import doubles to isolate Python name resolution without installing plotting
dependencies. Actual Python 3.13.15 returned 1 with NameError; Python 3.14.7
imported successfully. This is not a plot-rendering test or board inference.
Exact fixture, source hashes, interpreter versions and full outputs are in
[evidence](evidence/2026-09-28-pointnet-independent-review/findings.json).

## Required Claude implementation and acceptance

Remove the unused copied CLI parser from the visualization helper, preserving
plot helpers and main.py as the sole CLI owner. Do not merely add argparse to
keep a second unused parser and contradictory defaults. Add meaningful coverage
of the default plotting path using injected runtime and an appropriate host
plotting backend/fixture, plus import coverage that catches supported-version
annotation failures. Verify requested original/segmented output and result report;
keep --no-plot behavior. No board or model run is required. Do not install a
quantization toolchain or change source recipe instructions.

The inspected task class already keeps pre_process, forward, post_process and
predict separate, with file IO in main and SDK access in its runner. Existing
normalization/parity and target/metadata tests pass. These observations do not
close the full sample or B8: the default-path defect must be fixed and independently
rechecked. Source diagram/benchmark preservation was documented in the author
record; whole-branch acceptance remains separate.
