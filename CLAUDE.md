# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

RDK Model Zoo: D-Robotics' collection of BPU model samples and end-to-end deployment pipelines (export → PTQ quantization → inference → post-processing → evaluation) for RDK boards. It is sample/documentation content plus an Agent Skills pack (Skills source tree lives on `rdk_x5`) — there is no single build for the repo itself.

**Branches during the migration window (see `docs/releases/unified-migration/x5-s-migration-map.md`):** `develop` is the sample-centric integration line — X5 and S (S100/S100P/S600) samples are unified under `samples/` and hardware is selected per sample via `--target auto|x5|s100|s100p|s600`, resolved from board identity (`/sys/class/boardinfo/soc_name` → socinfo → device-tree) with no silent fallback. `rdk_x5` and `rdk_s` remain the customer delivery lines until the migration closes; `rdk_x3` is historical (archive-only, never sample-ized). Branch/model-filename suffixes are hints only — verify the actual platform from the target sample's README, code, and manifests. Per `AGENTS.md`: never switch branches or reset the worktree to resolve a conflict with user constraints; report conflicts instead. Read-only investigation is allowed by default; downloads, board runs, and publishing require explicit authorization.

## Commands

Model samples and conversion only run on RDK hardware / in the OpenExplorer Docker toolchain — not on a dev machine. On `develop`, host-side tests are the runnable commands: the unified samples carry unittest suites (e.g. `python3 -m unittest discover -s samples/vision/resnet/tests -v`) and `samples/_shared/tests`. Tooling added by the migration (contract checker `tools/sample_contract/`) documents its own commands in place.

**Entries that exist on `rdk_x5` but are NOT on `develop` yet** (do not assume them here; the web catalog stays on `rdk_x5` per ADR-0001 source/doc-site separation):

- `docs/catalog/` online catalog (Node.js ≥22.12, `npm run check` gate) and `docs/RELEASE.md` release dry-run.
- Repo-root `skills/` tree — lands on `develop` in Phase 1 (A3). Until then the execution-side installed copies live at `~/.claude/skills/` (installed from `rdk_x5` @ `ac11571`); the validation trio (`sync_references.py`, `validate_pack.py`, `unittest discover -s skills/tests`) runs from an `rdk_x5` checkout or the installed copy, not from this worktree.
- `docs/Model_Zoo_Repository_Guidelines.md` — the sample-standards baseline (Q1) is being authored on `develop` under `docs/sample-standards/`.

## Architecture

### Sample layout (the core unit)

Every sample under `samples/{vision,robotics,...}/<model>/` follows a fixed layout (`samples/vision/resnet` is the migration reference; `samples/vision/ultralytics_yolo` and `samples/vision/paddle_ocr` are the other pilots):

- `conversion/` — ONNX→HBM conversion configs/scripts for OpenExplorer
- `model/` — `download.sh`/`download_model.sh`; model binaries are **never committed** (`.gitignore` blocks `.onnx/.bin/.hbm/.pt`); downloaded from `archive.d-robotics.cc`
- `runtime/python/` (always) and `runtime/cpp/` (only where a source delivery provided one) — language coverage is declared per sample in its README support matrix; never claim dual-language support when C++ is absent
- `evaluator/`, `test_data/`, `tests/`, and bilingual `README.md` + `README_cn.md` at each level

File naming is mandated: model source named after the model (`yolov5.py` / `yolov5.hpp`+`yolov5.cc`), entry point always `main.py`/`main.cpp`, one-click script always `run.sh`. Model files: `<model_name>_<resolution>_<chip>.hbm`.

### Python model contract

Each model file defines a `XXXConfig` class (defaults must run out-of-the-box) and a `XXXModel` class implementing, in order: `__init__` (load model, extract metadata) → `set_scheduling_params` → `pre_process` (returns dict in `hbm_runtime` run format) → `forward` (input `{model_name: {input_name: tensor}}`) → `post_process` → `predict` (chains the three) → `__call__` (alias of predict). `main.py` uses `argparse` with kebab-case args (`--model-path`), every arg has `type`/`default`/`help`, and zero arguments must work. Shared helpers live in `utils/py_utils/` (samples append repo root to `sys.path`).

Detection tasks return `(boxes, scores, cls_ids)` as NumPy arrays — boxes `(N,4)` as `[x1,y1,x2,y2]`. Docstrings are Google-style; inline comments in English.

### C/C++ model contract

Config struct with defaults + model class where the constructor does no heavy work and `init()` loads the model/allocates tensors (returns 0 on success); free functions `pre_process`/`infer`/`post_process` pass tensors by reference/parameter (for future multithreading). `main.cpp` uses gflags with snake_case args (`--model_path`), defaults required. Comments are Doxygen. CMake detects the SoC at build time:

```cmake
file(READ "/sys/class/boardinfo/soc_name" SOC_NAME_RAW)
string(TOUPPER "${SOC_NAME_RAW}" SOC_NAME_UPPER)
add_definitions(-DSOC_${SOC_NAME_UPPER})
```
Python on `develop` resolves the target through `samples/_shared/platforms.py` from the same board-identity sources and **errors on unknown boards — no default fallback** (the legacy `get_soc_name()` default of `"s100"` exists only on the delivery branches). Never hardcode a platform.

### Manifests → catalog pipeline (release data flow)

Manifests are authoritative; the web catalog (on `rdk_x5`) is only their presentation layer. On `develop` during the migration window the manifests live at `platforms/{x5,s}/docs/release/*.yaml` and move to `docs/release/{x5,s}/` in Phase 1 (A4, same commit as the `samples/_shared/assets.py` repoint) — `docs/manifests/` is the `rdk_x5` layout, not present here. Rules: update manifests in the same commit that adds/removes/moves a model; unknown checksums are `sha256: null` (never guessed or copied), and release notes must disclose incomplete coverage; benchmarks only record published values with immutable evidence — never infer missing conditions.

### Skills pack (`skills/`)

Seven skills (`rdk-model-zoo*`) maintained from the `rdk_x5` default branch — but the branch is only the maintenance source and never implies a task targets X5. `_shared/` is the single edit source for shared rules; `tools/sync_references.py` regenerates per-skill copies (committed for flat installs; CI checks drift — never hand-edit copies). `skills/tools/` and `skills/tests/` are maintainer-only. Runtime helper scripts (e.g. `skills/rdk-model-zoo/scripts/read_catalog.py`) take explicit `--repo`/`--model` paths, never guess, never go online.

## Conventions that get reviewed

- **Bilingual docs**: user-facing READMEs exist as `README.md` + `README_cn.md`; keep them in sync.
- **README hierarchy**: when changing anything in a directory, check whether the parent README (up to the root model list) needs updating.
- **Releases** (procedure doc `docs/RELEASE.md` lives on `rdk_x5`): per-platform semver tags (`x5-v1.2.3`, `s-v*`, `x3-v*`); every release commit updates `VERSION`, `CHANGELOG.md` (the *sole* release-notes location — no separate release-note files), and both manifests; S/X3 release before X5; published tags are never moved (fixes go in a patch release). Unified-source release conventions follow ADR-0006 (`docs/adr/`).
- Catalog data boundaries (see `docs/catalog/README.md`): one row per actual model config; never merge across configs or sum stage timings into end-to-end numbers; missing values render as `—`, not guesses.
