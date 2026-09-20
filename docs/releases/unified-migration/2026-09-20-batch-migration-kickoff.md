# Batch migration kickoff — 2026-09-20

This checkpoint starts the full batch migration series (B1–B11) agreed in the
active planning session. It records the source baselines, the batch order, and
the per-batch record convention. Nothing has been migrated by this document
alone; per-sample status remains tracked in the
[migration map](x5-s-migration-map.md).

## Source baselines

| Branch | Commit | Note |
| --- | --- | --- |
| `develop` | `9f17f2a` | Integration line; carries the 3 pilot samples and `platforms/` snapshots |
| `rdk_x5` | `ac11571` | Customer delivery branch; 5 commits ahead of the `platforms/x5` snapshot |
| `rdk_s` | `380e1a2` | Customer delivery branch; ~75 files ahead of the `platforms/s` snapshot (e.g. `yoloe26_seg`) |
| `rdk_x3` | `0eb344b` | Historical archive target; not part of X5/S batch migration |

## Batch series

Extraction always uses branch tips, never the frozen `platforms/` snapshots.
Each batch closes with host verification, the Q1–Q5 delivery review, and user
board smoke on X5 (8GB/4GB) and S100 before the next batch starts.

| Batch | Scope (target paths under `samples/`) | Sources |
| --- | --- | --- |
| B1 | `vision/mobilenetv1..4`; resnet50/resnet152 variants into `vision/resnet`; lands platform Profile contract and sample↔manifest coverage check | x5+s |
| B2 | `vision/efficientnet`, `vision/efficientformer(v2)`, `vision/efficientvit` | x5 (efficientvit x5-only) + s |
| B3 | `vision/convnext`, `vision/edgenext`, `vision/fasternet`, `vision/fastvit` | x5 |
| B4 | `vision/repghost`, `vision/repvgg`, `vision/repvit`, `vision/mobileone`, `vision/resnext`, `vision/vargconvnet`, `vision/googlenet`, `vision/hgnetv2` | x5 |
| B5 | `vision/clip`, `vision/siglip`, `vision/dinov2`, `vision/vit`, `vision/3dresnet` (each independent) | x5+s |
| B6 | `vision/efficient_sam`, `vision/mobile_sam` | x5+s |
| B7 | `vision/yolov5`, `vision/fcos`, `vision/yoloworld`, `vision/lprnet`, `vision/modnet`, `vision/bytetrack` | x5+s |
| B8 | `vision/unet`, `vision/unetmobilenet`, `vision/pp_liteseg`, `vision/yolo26_depth`, `vision/depth_anything_v2`, `vision/lanenet`, `vision/pointnet`, `vision/diffusiondrive` | x5+s |
| B9 | yolo11/yolo11_pose/yolo11_seg/yolov13 lift into `vision/ultralytics_yolo`; `vision/yoloe` (yoloe + yoloe11_seg + yoloe26_seg) | x5+s |
| B10 | `robotics/himloco`; `speech/{asr,kws,paraformer}` | x5+s |
| B11 | `llm/{gemma4-e2b,minicpm5-2b}`; `vla/{act,pi0}` (gitlinks) | s |

## Per-batch record convention

- One review file per batch: `2026-09-XX-bN-<name>-review.md` in this
  directory, following the existing dated-review format (scope, old→new
  function map per source branch, verification with explicit `not-run`
  semantics, remaining work).
- Evidence artifacts go to `evidence/` as JSON with recorded cwd, full
  command, commit SHA, target/runtime and input/output hashes.
- The [migration map](x5-s-migration-map.md) status column flips per entry:
  `S` (static-checked) → `F` (mapping verified) → closed after the batch's
  board smoke passes. Board-unverified items stay `H` and are reported as
  `not-run`; they are never closed by host tests alone.

## Source-drift ledger

While migration runs, `rdk_x5` and `rdk_s` keep serving customers. Deltas on
those branches after the baselines above are recorded in three streams:

1. new samples added on a delivery branch;
2. fixes to samples already migrated into `develop`;
3. changes to shared tooling (`utils/`, manifests, download infrastructure).

Each stream records the source SHA and enters the incremental ledger section
of the affected batch review. Source changes are semantically adapted, never
copied over refactored implementations; both tips are re-checked before the
closure phase deletes `platforms/`.

## Prerequisites before B1

- Phase 0.5 baseline (Q1–Q5) passed: README contract and templates,
  inference responsibility contract, contract checker with negative fixtures,
  dual-perspective acceptance on the two reference samples, targeted Skills
  strengthening. See the active plan for the authoritative checklist.
- Phase 1 groundwork landed: `utils/`, `datasets/`, `skills/`, manifests at
  `docs/release/{x5,s}/` with `samples/_shared/assets.py` repointed, workflows
  reconciled, migration-map addenda for post-snapshot source additions.
- Phase 1.5 hardening (H1–H4) verified on the three pilot samples.
