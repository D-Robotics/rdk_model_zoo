# B3 board validation tools

`b3_classification_compare.py` turns the B3 board-backlog recipe (the four
classification evaluator READMEs' "same-board before/after comparison" prose)
into one reusable, scope-fixed executable for
[the B3 board queue](../../docs/releases/unified-migration/2026-09-22-host-development-and-board-handoff.md):
convnext (atto), edgenext (base/small/x_small/xx_small), fasternet
(s/t0/t1/t2), fastvit (s12/sa12/t12/t8) — 13 variants on X5 8GB and X5 4GB.

It runs **one sample/target/variant per invocation**, on the board itself,
against the fixed X5 source pin `ac115717197920355fc390bb04299b20e6436864`.
The fixed source wrapper (`platforms/x5/samples/vision/<sample>/runtime/python/<sample>.py`)
and the unified entry (`resolve_selection` → `RuntimeModelRunner` →
`ClassificationTask`) each execute their own complete
pre_process → forward → post_process on the same image bytes, resize type,
Top-K and scheduling; neither side's result substitutes for the other's.

**Source closure pinning.** The `source_ref` constant alone pins nothing:
before the source module executes, every file in its closure — the legacy
entry plus the `utils.py_utils` modules it imports (`__init__`, `file_io`,
`preprocess`, `visualize`) — is byte-compared with the pin's git blob
(`git show <pin>:<path>`) against the `platforms/x5` snapshot copies, and the
pinned dependency modules are installed into `sys.modules` only for the
duration of the source module's execution (snapshot and restore; nothing
leaks into the process). The root `utils/py_utils/file_io.py` has drifted
from the pin (`load_imagenet_labels` rewritten as a proxy) and is never
executed by this tool; any closure mismatch or unavailable pin object
refuses the run (rc 2, `source_closure` evidence records every hash).

## Minimal dependencies

- On the **board**: the matching `hbm_runtime` Python environment (Python 3.10
  compatible — hashing reuses `samples/_shared/assets.py::sha256_file`),
  plus numpy, OpenCV and scipy (the fixed source imports
  `scipy.special.softmax`), and the full repository checkout deployed on the
  board. Run from the repository root.
- On a **host** (tests only): Python 3.10+, numpy, OpenCV, scipy, PyYAML.
  `hbm_runtime` is **not** required — the behavior tests inject a fake SDK
  runtime; nothing here claims board execution on a host.

## Usage

Prepare assets explicitly first (the tool never downloads); per sample:

```bash
bash samples/vision/convnext/model/download.sh x5          # atto
bash samples/vision/edgenext/model/download.sh x5 base      # or small/x_small/xx_small
bash samples/vision/fasternet/model/download.sh x5 s        # or t0/t1/t2
bash samples/vision/fastvit/model/download.sh x5 s12        # or sa12/t12/t8
```

Then, on the selected board, cwd = repository root, one cell of the matrix per
invocation with a fresh evidence directory:

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=/tmp/b3-convnext-atto-x5-8g-$STAMP
python3 tools/board_validation/b3_classification_compare.py \
  --sample convnext --target x5 --variant atto \
  --output-dir "$OUT" > "$OUT.stdout" 2> "$OUT.stderr"
echo $? > "$OUT.rc"
```

Omit `--variant` to follow the sample's own default selection semantics
(atto/base/s/s12); combine `--model-path` with an exact `--asset-id` to point
at a prepared artifact elsewhere. `--top-k` (default 5), `--resize-type`
(0/1, default 1), `--priority` and `--bpu-cores` (defaults 0 / `[0]`) apply
identically to both sides. Repeat per variant and per board (the 8GB/4GB
distinction is recorded via board identity plus `/proc/meminfo` MemTotal).

## Output

`comparison.json` plus one `.npy` per captured array (13 files in a
successful run) inside the new `--output-dir`:

- identity: UTC start/end, argv/cwd, git head/branch/dirty state, code
  SHA-256 for the tool, the executed fixed-source module, the pin-verified
  `utils.py_utils` dependency files it actually bound (observed + pin hashes,
  `matches_pin` per file), the `source_closure` verification record, the
  shared modules and the sample's runtime sources; board identity files,
  `/etc/os-release`, MemTotal; SDK module file/version; model
  publisher-vs-observed SHA-256 (`verify_asset_file`; publisher hashes are
  `null` in the manifests and stay `null`), image and label observed SHA-256.
- execution: per-side SDK metadata via
  `samples/_shared/runtime_meta.py::metadata_evidence` (never `asdict` — board
  `QuantParams` refuses copying), all pre inputs / raw outputs / Top-K and
  top-8 evidence arrays with shape/dtype/SHA-256, and the real exception and
  return code when anything fails.
- comparison: non-empty uint8 finite inputs with **exact bytes** (the legacy
  `(1, 3H/2, W, 1)` view and the unified flat buffer may differ in shape, not
  in bytes); non-empty raw outputs sharing shape/dtype, finite, with the
  complete difference (`raw_abs_diff_*.npy` plus max/mean/nonzero/argmax —
  reported, not asserted); Top-K with the requested count, **unique** IDs,
  identical ID sequences, per-ID abs score difference ≤ 1e-5 and **identical
  label strings** (the pinned `load_imagenet_labels` and the unified
  `load_labels` are separate implementations, so a silent loader change
  cannot hide); exact ties inside the top-(k+1) window are recorded with
  per-ID top-8 evidence from both sides and **never** relaxed into a pass.

Exit codes: `0` all checks passed; `1` comparison completed with failed checks
(arrays retained); `2` execution error — including a fixed-source closure that
does not match the pin — with error evidence retained. stdout carries one
machine-readable JSON summary (checks, tie flag, both sides' Top-K, evidence
path).

## Limitations

- This is a **fixed-image migration consistency check** — it is not a dataset
  accuracy, quantization-quality, or latency measurement, and passing it does
  not certify the model or the artifact beyond the compared bytes.
- Scope is the four B3 classification samples on X5-class boards; other
  targets fail selection with "no published asset" by design (the tool never
  defaults an unknown target or a lone asset into an implicit choice).
- The comparison result depends on the deployed checkout's fixed source; the
  recorded git/code hashes are the authority for which code produced it.
- Source-closure verification needs the pin commit object in the deployed
  checkout's git database; if it is missing, the tool **fails closed** (rc 2)
  instead of running on unverifiable sources.
- Conversion (OE export/calibration/compile) is out of scope here.

## Tests

Host-only fake-runtime behavior tests (no board SDK needed):

```bash
python3 -m unittest discover -s tools/board_validation/tests -v
```
