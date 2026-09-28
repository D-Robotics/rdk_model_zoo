# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

RDK Model Zoo: D-Robotics' BPU model samples and end-to-end deployment pipelines (export → PTQ quantization → inference → post-processing → evaluation) for RDK boards (X5; S100/S100P/S600). Sample/documentation content plus an Agent Skills pack under `skills/`; there is no single build for the repo itself.

## Where you are (branches)

- **This checkout is a migration integration work branch** (`codex/b7-board-integration-*`). It carries in-flight X5/S unification work that is **not merged into `develop` and not released** — never describe local state here as shipped, merged, or independently accepted.
- `develop` is the sample-centric integration line: X5 and S (S100/S100P/S600) samples unified under `samples/`, hardware selected per sample via `--target auto|x5|s100|s100p|s600`, resolved from board identity (`/sys/class/boardinfo/soc_name` → socinfo → device-tree) with no silent fallback.
- `rdk_x5` and `rdk_s` are the customer delivery lines until the migration closes. Their tags, READMEs and release conventions govern their own refs; treat them as historical references here, not adaptation targets.
- `rdk_x3` is archive-only: historical material, never a new adaptation target (ADR-0007 records only a bounded native-API compatibility evaluation, `accepted-for-evaluation`).
- Branch/model-filename suffixes are hints only — verify the actual platform from the target sample's README, code, and manifests. Per `AGENTS.md`: never switch branches or reset the worktree to resolve a conflict with user constraints; report conflicts instead. Read-only investigation is allowed by default; downloads, board runs, and publishing require explicit authorization.
- People and Agents use the same native sample commands; no Skill, Node, or publisher is required for model inference (ADR-0003).

## Authoritative documents (read before sample work)

- `AGENTS.md` — contributor entry, host checks, evidence-separation rules.
- Active integration spec: `docs/superpowers/specs/2026-09-16-rdk-model-zoo-x5-s-agent-people-spec.md` (controls conflicts during the migration).
- Standards baseline: `docs/Model_Zoo_Repository_Guidelines.md` (develop unified-architecture edition; naming/layout conventions live here, not in this file).
- Contracts: `docs/sample-standards/readme-contract.md` (per-level README requirements, fixed anchor IDs, bilingual pairing) and `docs/sample-standards/inference-contract.md` (public interfaces, stage responsibilities, required tests); templates in `docs/sample-standards/templates/`.
- Decisions: `docs/adr/` — ADR-0001 source/doc-site split, ADR-0002 unified entry with transitional compatibility, ADR-0003 agent-assisted development with independently runnable samples, ADR-0005 samples run inside a full source checkout, ADR-0006 unified-source releases and platform matrix, ADR-0007 X3 native-API subset evaluation.
- Migration state and package records: `docs/releases/unified-migration/` (see `x5-s-migration-map.md`).

## Host commands (dev machine; no board or toolchain required)

From the repo root with `python3`:

```bash
python3 -m unittest discover -s samples/_shared/tests -v
python3 -m unittest discover -s samples/vision/resnet/tests -v
python3 -m unittest discover -s samples/vision/ultralytics_yolo/tests -v
python3 -m unittest discover -s samples/vision/paddle_ocr/tests -v
python3 -m unittest discover -s samples/_shared/tests -p test_vla_integration.py   # VLA pinned-submodule integration
python3 tools/sample_contract/check.py --sample samples/vision/resnet              # static contract checker (--scope migration for CI scope)
python3 -m unittest discover -s tools/sample_contract/tests -v                     # checker's own tests (fixtures)
python3 skills/tools/sync_references.py                                            # skills pack: reference-copy drift check (add --apply to write)
python3 skills/tools/validate_pack.py --pack-root skills
python3 -m unittest discover -s skills/tests -v
```

Model samples and conversion only run on RDK hardware / in the OpenExplorer Docker toolchain — not on a dev machine. Host checks green is not board verification: report host tests, board tests, artifact availability, and migration status separately; no result means `not-run`, not passed. The YOLO catalog comparison additionally needs a generated catalog: `npm --prefix tools/catalog-publisher run build`. Still absent from this tree (they remain `rdk_x5` concerns per ADR-0001 source/doc-site separation): `docs/catalog/` and `docs/RELEASE.md`.

## Sample layout (the core unit)

Every sample under `samples/{vision,llm,robotics,speech,vla}/<model>/` follows the unified layout (`samples/vision/resnet` is the single-model reference; `samples/vision/paddle_ocr` the multi-stage reference):

- `conversion/` — ONNX→HBM conversion configs/scripts for OpenExplorer; missing recipes must be declared in the level README (`known-gaps`), never disguised with generic commands
- `model/` — preparation/download scripts; model binaries are **never committed** (`.gitignore` blocks `.onnx/.bin/.hbm/.pt`); sourced from `archive.d-robotics.cc`; unknown checksums stay `sha256: null (unknown)` — never guessed or copied
- `runtime/python/` and/or `runtime/cpp/` (plus `legacy/` where a source delivery provided one) — **language coverage is declared per sample in its README support matrix**; never claim dual-language support when a language is absent (e.g. `samples/llm/gemma4-e2b` and `samples/llm/minicpm5-2b` ship C++/legacy runtimes, not Python)
- `evaluator/`, `test_data/`, `tests/`, and bilingual `README.md` + `README_cn.md` at each level required by the readme-contract
- VLA samples (ACT/Pi0) are pinned upstream git-submodule integrations — preserve their exact commits and layouts (`samples/vla/README.md`); anchor rules never require editing upstream gitlinks

## Interfaces — read the contract, not a summary

Python runtimes follow `docs/sample-standards/inference-contract.md`: a four-method public business interface (`pre_process` → `forward` → `post_process`, chained by `predict`), an explicit per-call context carried on the prepared input (never in fields a next call overwrites), validated raw outputs, and owned typed results — **not** a universal dict/tuple shape. Stage responsibilities are fixed (no download, NMS, task decoding, drawing, or file output inside `forward`); files separate task, binding, runner, and tensor-IO modules; multi-stage pipelines keep every stage's three steps public; LLM/streaming samples declare generate/stream/reset with written justification. Reference implementations: `samples/vision/resnet/runtime/python/classification.py` (single model), `samples/vision/paddle_ocr/runtime/python/pipeline.py` (multi-stage). Shared runtime components live under `samples/_shared/` (`platforms.py`, `assets.py`, `model_runner.py`, …); legacy compatibility helpers remain under `utils/py_utils`.

C++ runtimes follow the structure declared in each sample's runtime README (config with defaults, explicit init/lifecycle, pre/infer/post as separable units, Doxygen comments; SoC detection at build time from `/sys/class/boardinfo/soc_name` — see `samples/vision/resnet/runtime/cpp/CMakeLists.txt`). The contract deliberately does not force C++ to mirror the Python interface method-by-method.

`samples/_shared/platforms.py` resolves targets from board identity and **raises on unknown targets or unidentifiable boards — no default fallback**; never hardcode a platform.

## Manifests and release data flow

Manifests are authoritative; the web catalog (an `rdk_x5`/doc-site concern, ADR-0001) is only their presentation layer. **Active manifests in this tree: `docs/release/x5/models.yaml` and `docs/release/s/models.yaml`**, with target-identity aliases in `docs/release/platforms.json`. `platforms/{x5,s,x3}/docs/release/*.yaml` are archived frozen snapshots of the delivery branches (`platforms/README.md`, `platforms/registry.json`) — usable for historical facts only; their filename keys may differ from the active manifests. `docs/manifests/` is the `rdk_x5` layout, not present here. Rules: update manifests in the same commit that adds/removes/moves a model; benchmarks record only published values with immutable evidence — never infer missing conditions; release notes disclose incomplete coverage.

## Skills pack (`skills/`)

Seven skills (`rdk-model-zoo*`) live in this tree as an integration candidate — present and host-checkable here, which is **not a release or an independent acceptance statement**. The upstream maintenance source is the `rdk_x5` default branch, and a maintenance branch is only a document source, never a platform selector. `_shared/` is the single edit source for shared rules; `skills/tools/sync_references.py` regenerates the committed per-skill copies (never hand-edit copies; the read-only run doubles as the drift check). `skills/tools/` and `skills/tests/` are maintainer-only. Runtime helper scripts (e.g. `skills/rdk-model-zoo/scripts/read_catalog.py`) take explicit `--repo`/`--model` paths, never guess, never go online. Do not install skills/toolchains or edit installed copies under `~/.claude/` without explicit authorization.

## Conventions that get reviewed

- **Bilingual docs**: user-facing READMEs exist as `README.md` + `README_cn.md` with matching fixed anchor IDs (readme-contract §2–3); keep them in sync.
- **README hierarchy**: when changing anything in a directory, check whether the parent README (up to the root model list) needs updating.
- **Releases**: the target model is the unified-source scheme of ADR-0006 (unified version + per-platform support/verification matrix; Skills versioned independently). Delivery-line conventions (per-platform `x5-v*`/`s-v*` tags, per-ref `VERSION`, `CHANGELOG.md` as the sole release-notes location) govern the `rdk_x5`/`rdk_s` refs and are historical references on this branch; this work branch makes no release. Published tags are never moved — fixes go in a patch release.
- **Catalog data boundaries** (catalog tooling lives under `tools/catalog-publisher/`): one row per actual model config; never merge across configs or sum stage timings into end-to-end numbers; missing values render as `—`, not guesses.
