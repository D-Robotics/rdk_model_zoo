<!-- Template: runtime/python README (English). Contract: readme-contract.md §4.3.
     Keep anchors; replace ⟪…⟫; delete guidance when done. Parameter defaults are
     machine-checked against build_parser; the integration example must run as-is
     (verified by sample tests per inference-contract). -->

# Python Runtime — ⟪model name⟫

<a id="environment"></a>
## Environment

> **Must answer:** board/system requirement, Python version, dependencies (or
> “board image built-ins only”); explicit note that `hbm_runtime` exists only in
> the board image (host import fails by design).

- Board: ⟪targets⟫ with system image ≥ ⟪version⟫
- Python: ⟪version⟫; dependencies: ⟪list or none⟫
- `hbm_runtime` is provided by the board image only — this runtime does not run
  on a dev machine.

<a id="usage"></a>
## Usage

> **Must answer:** cwd; one default command (zero extra args) and one customized
> command; how success is judged (exit code / printed output / result files).

```bash
# cwd: repository root
python3 samples/⟪domain⟫/⟪name⟫/runtime/python/main.py --target ⟪target⟫ ⟪input⟫
# success: ⟪criterion⟫
```

<a id="parameters"></a>
## Parameters

> **Must answer:** EVERY CLI argument with its actual default from the parser
> (machine-checked). Kebab-case names. No parameter may be documented that the
> parser does not define, and none omitted.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | str | `auto` | ⟪…⟫ |
| ⟪arg⟫ | ⟪type⟫ | ⟪default⟫ | ⟪…⟫ |

<a id="results"></a>
## Results

> **Must answer:** output fields/files, their location, format and meaning
> (coordinate convention, class ids, confidence semantics) — field names must
> match what the code returns.

⟪e.g. prints Top-5 as (class_id, label, score); writes ⟪path⟫ as JPEG with
boxes in [x1,y1,x2,y2] pixel coords⟫

<a id="integration-example"></a>
## Integration Example

> **Must answer:** a COMPLETE runnable Python snippet. Every input/config
> variable is defined inside the example or points to a concrete local file
> prepared in a stated prerequisite step. No undefined references. The snippet
> is exercised by this sample's tests against a fixture.

Prerequisite: ⟪e.g. artifact downloaded per model/README.md; test image at
samples/⟪domain⟫/⟪name⟫/test_data/⟪image⟫⟫

```python
import ⟪module⟫

⟪binding = … (concrete construction with real paths)⟫
model = ⟪Model⟫(⟪args⟫)
result = model.predict(⟪defined input⟫)
print(⟪result field⟫)
```

<a id="stage-io"></a>
## Three-Stage I/O

> **Must answer:** the pre_process / forward / post_process contract of this
> sample as a readable summary consistent with the docstrings (see
> inference-contract). Multi-stage pipelines (e.g. OCR det→rec) get one
> subsection per stage plus the pipeline.predict composition.

- `pre_process`: ⟪Input⟫ → ⟪Tensors + Context⟫ (⟪shapes/dtypes/layout⟫)
- `forward`: ⟪tensor dict⟫ → ⟪RawOutputs⟫ (⟪output names/shapes/quantization
  semantics — raw logits vs dequantized⟫)
- `post_process`: ⟪RawOutputs + Context⟫ → ⟪Result⟫ (⟪result type/fields⟫)
- ⟪pipeline.predict composition for multi-stage samples⟫

<a id="troubleshooting"></a>
## Troubleshooting

> **Must answer:** real failure modes only (missing artifact, target mismatch,
> input size constraints) with the actual error text and the fix.

| Symptom | Cause | Fix |
| --- | --- | --- |
| ⟪error text⟫ | ⟪cause⟫ | ⟪fix⟫ |
