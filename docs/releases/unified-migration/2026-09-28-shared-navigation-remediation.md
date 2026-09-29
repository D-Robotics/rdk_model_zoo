# Shared and agent navigation remediation — 2026-09-28

Status: **author remediation only, applied by Claude Code + GLM against
H8-NAV-R1 in `2026-09-28-shared-navigation-independent-review.md`
(reviewer Codex). This document is not independent acceptance and does not
close H8. No board, SDK, OE or quantization claim is made or narrowed.**

Scope: documentation only — exactly three files:

- `samples/_shared/README.md`
- `skills/README.md`
- `CLAUDE.md`

No product code, skill instruction, shared-rule copy, pack version, manifest,
test, CLI flag, default or command block was changed. Existing README command
blocks are byte-identical (verified: the diff contains no command lines). No
board/SSH access, download, install, quantization, toolchain execution, commit
or push occurred. Other workers' in-flight files were not touched; this report
and its evidence directory are the only new files.

## 1. `samples/_shared/README.md`

**YOLOE-26 kernels section (EN + CN).** Replaced the stale "conversion and
native migration remain pending" / "转换和原生迁移仍待完成" statement. The
sections now state that the
[canonical conversion guide](../../../samples/vision/yoloe/conversion/README.md)
and [native C++ runtime](../../../samples/vision/yoloe/runtime/cpp/README.md)
(Chinese `README_cn.md` links) are implemented and host-reviewed — the
independent
YOLOE runtime/scorer review plus native host reviews
([2026-09-28-yoloe-independent-review.md](2026-09-28-yoloe-independent-review.md),
[2026-09-28-yoloe-native-cli-review.md](2026-09-28-yoloe-native-cli-review.md),
[2026-09-28-yoloe-native-preflight-review.md](2026-09-28-yoloe-native-preflight-review.md))
— while real SDK execution, board inference and a published S floating-output
asset remain not-run/not available. The pre-existing boundaries paragraphs
(EN "S public HBM artifacts declare quantized outputs …", CN "目前 S 公开 HBM
声明量化输出 …") and the `test_yoloe26_decode.py` command block are unchanged.
Links were verified to resolve to the actual canonical guides.

**Image bytes section (EN).** Removed the obsolete fixed count "used by all
three samples" / "from all three implementations" after checking actual
imports. The single `image.py:bgr_to_nv12_planes` implementation is now
described by its real consumers: `samples/_shared/tensor_io.py` wraps it (and
the classification samples reach it through that surface); the Ultralytics
YOLO, PaddleOCR, YOLOv5, YOLOE, FCOS, YOLO26 Depth, PP-LiteSeg, UNet and
UNetMobileNet runtimes import it directly — the nine direct sample importers
found by searching `from samples._shared.image import` in the current tree.
The extraction sentence no longer pins a source count. Note: the
`samples/_shared/image.py` module docstring still says "shared by YOLO, ResNet
and PaddleOCR"; that is a code file and is deliberately untouched per the
documentation-only scope.

## 2. `skills/README.md`

**目标仓库与平台 section.** The manifest paragraph previously said "当前目标 ref
优先查 `docs/manifests/`；历史 ref 可能保留 `docs/release/` 或根 `release/`",
which omitted the unified active manifests and implied a selection priority
the tools do not implement. The paragraph now describes the actual layouts and
tool behavior, read from the current scripts:

- Unified active manifests `docs/release/x5/models.yaml` and
  `docs/release/s/models.yaml` are the current maintenance locations, distinct
  from the frozen migration-window snapshots
  `platforms/{x5,s}/docs/release/` and `platforms/x3/release/`; delivery refs
  keep `docs/manifests/`, historical refs may keep root `release/`.
- `inspect_repo.py` (`manifest_candidates`, `PLATFORM_MANIFEST_PATTERNS`)
  reports only actually-present candidates tagged unified vs snapshot, prefers
  the unified manifest when both exist for the same platform, and does not
  infer target hardware from manifest presence.
- `read_catalog.py` (`MANIFEST_LAYOUTS`) implicitly selects only when exactly
  one flat-layout manifest exists; multiple layouts fail with
  `ambiguous-manifest`, no manifest with `manifest-not-found`, and unified
  per-platform manifests are read via explicit `--manifest`
  (e.g. `--manifest docs/release/s/models.yaml`).

The unknown/multiple-manifest behavior, the pack maintenance-branch statement,
candidate versions, the historical executed-case count and the
"local tool tests do not replace Agent/board/Hub acceptance" boundary are all
unchanged.

## 3. `CLAUDE.md`

**Manifests and release data flow section.** Corrected the grouped path
`platforms/{x5,s,x3}/docs/release/*.yaml` to the actual layouts: X5/S
snapshots live under `platforms/{x5,s}/docs/release/`, and X3 is the layout
exception — its historical manifests live at `platforms/x3/release/`
(delivery-line root `release/` layout; `manifest_directory` in
`platforms/registry.json`), which remains a current catalog input
(`inspect_repo` lists it as a manifest candidate; the catalog publisher's
`sources.json` X3 source resolves `platforms/x3` + `release`). X3's
historical/no-new-adaptation framing and the archived X5/S snapshot identity
are unchanged.

## Verification (host only; not board or behavioral evidence)

All commands run from the repository root with the coordination venv Python
(`../rdk_model_zoo/.venv`, Python 3.14.7); full outputs, command lists, UTC
timestamp, final file SHA-256 values and the diff are in
[evidence/2026-09-28-shared-navigation-remediation](evidence/2026-09-28-shared-navigation-remediation/):

- `git status --porcelain` filtered to the three paths: only the three
  intended files modified.
- Local markdown link check over the three edited files: 0 broken links.
- `python skills/tools/sync_references.py`: valid, applied=false, no drift.
- `python skills/tools/validate_pack.py`: valid, 7 skills, 83 eval cases,
  behavior_evaluated=false.
- `python -m unittest discover -s skills/tests`: 57 tests, OK.
- `python -m unittest discover -s samples/_shared/tests`: 158 tests, OK.
- `python tools/sample_contract/check.py --scope migration`: 51 samples,
  0 violations, 51 skips, 0 exemptions applied.

These deterministic checks certify documentation/tool structure only. Current
Agent behavior remains a separate H8 acceptance requirement, which stays open.
