# Depth Anything V2 independent host review — accepted within host scope

Reviewer: Codex. Candidate base `3278d649`; source S pin remains `380e1a2`.
No product files changed in this review. Accept this sample's current host
migration scope; whole B8/H4 and whole-branch acceptance remain open.

Code inspection covered task, selection/binding, per-frame geometry, shared
runner use, CLI persistence and separate visualization. The maintained README
explains the actual source transform: per-pixel RGB z-score after nearest-neighbor
stretch, not the misleading source docstring's ImageNet normalization. Frozen
context accompanies each frame. Forward keeps raw float depth; postprocess crops
letterbox padding and restores original dimensions. The interface returns owned
relative float depth, with display normalization outside the model stages.
Metadata checks require the declared float32 input/output shapes rather than
inferring public IO dtype from internal int16 quantization prose.

The explicit differences from the source are disclosed in both interface and
evaluator documentation: OpenCV half-pixel linear restoration replaces Torch,
letterbox cropping is corrected, constant maps produce defined zero grayscale,
and the new API returns float depth instead of a display image. No bit-exact
Torch/HBM equivalence or metric-distance interpretation is claimed. CLI retains
raw/restored arrays plus model/input hashes and separates historical performance
records from current execution. Unsupported targets fail; auto selects the sole
asset only before the real execution identity gate.

Independent verification: 15 host tests pass, including source preprocessing,
analytic geometry, explicit stages/predict, context and invalid tensor checks,
constant/full-range visualization, CLI and target refusal. Sample contract check
reports zero violations, one CLI policy skip, zero exemptions. Full command
output and accepted source hashes are in
[verification.json](evidence/2026-09-28-depth-anything-independent-review/verification.json).
The source-illustration audit also found all eight source image references
represented in the maintained guides; that lexical result alone was not used
as semantic acceptance.

The conversion guide preserves source diagrams and states which executable
assets were never supplied, without inventing an export/compile procedure.
Historical monitor/performance values retain their missing-condition caveats;
no new benchmark or dataset result is asserted. No board, real SDK, model download,
quantization recipe or toolchain installation was performed. Board remains
not-run; actual HBM/Torch output parity remains unmeasured, not a claimed pass.
These excluded runs do not block the user's current host-only delivery scope.

The ledger/plan is concurrently owned by the active MiniCPM package. Include
this disposition in the next coordinated ledger update; do not overwrite that
worker's file. Independent whole-B8 acceptance still needs remaining families.
