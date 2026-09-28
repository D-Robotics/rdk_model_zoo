# B7 status-docs remediation (YOLOv5 + ByteTrack READMEs) — 2026-09-28

Status: **author remediation only, applied by Claude Code + GLM against
B7-DOC-R1 in `2026-09-28-b7-batch-independent-review.md` (reviewer Codex). This
document is not independent acceptance. B7 and H3 remain open; no board, SDK,
OE or quantization claim is made or narrowed, and no new board result is
requested or recorded.**

Scope: documentation only — the 11 README pairs under
`samples/vision/yolov5` (root, `model/`, `runtime/python/`, `runtime/cpp/`,
`conversion/`, `evaluator/`) and `samples/vision/bytetrack` (root, `model/`,
`runtime/python/`, `conversion/`, `evaluator/`). 20 of the 22 files changed;
the `bytetrack/runtime/python/` pair contained no current-state contradiction
and is untouched. No product code, test, manifest, conversion recipe, CLI
flag, default, path or command was changed. No board/SSH access, download,
quantization, toolchain execution, install, commit or push occurred. The
independent review/plan/ledger documents are Codex-owned and were not edited.

## Facts used (from the cited historical record)

Statements restored into the READMEs are taken from
`2026-09-24-b7-native-sdk-review.md` and its linked evidence, re-read before
editing: YOLOv5 X5 nine Python variants fully compared source/unified on one
X5 8GB and one 4GB board (`ae0f185`/`4d45f9a`,
[evidence](evidence/2026-09-24-b7-yolov5-x5-variants/),
[python comparison](evidence/2026-09-24-b7-python-comparison/),
[expanded boards](evidence/2026-09-24-b7-expanded-boards/)); S100/S600 `x-672`
`kite.jpg` Python comparisons; four-board native C++ build/smoke records
([initial](evidence/2026-09-24-b7-board-initial/),
[round 2](evidence/2026-09-24-b7-native-round2/)) with the source C++
numerical comparison still incomplete (checkpoint `3d6c14c`,
[connectivity record](evidence/2026-09-24-board-connectivity-interruption/));
S100P YOLOv5 rejection-only
([negatives](evidence/2026-09-24-b7-s100p-negative/)); ByteTrack S100
four-frame synthetic check
([evidence](evidence/2026-09-24-b7-bytetrack-s100/)) and S100/S600 first-30
real-video comparisons at `4d45f9a`
([evidence](evidence/2026-09-24-b7-bytetrack-realvideo30/)); ByteTrack S100P
asset URL HTTP 404 with no positive run
([record](evidence/2026-09-24-b7-s100p-negative/bytetrack-asset-download-404.json)).

## Findings addressed

| Review input | File(s) | Change |
| --- | --- | --- |
| B7-DOC-R1 — YOLOv5 root says "neither language was run on a board"; matrix rows `supported-not-run` / "board not-run" erase the nine-variant X5, S100/S600 and C++ smoke records | `yolov5/README.md`, `yolov5/README_cn.md` | Matrix rows became `supported-verified` (Python) / `supported-smoke` (C++), with a new note paragraph under the table that scopes the 2026-09-24 records to their exact artifact/image cases, links the evidence, and states three boundaries: historical records do not re-validate the current HEAD; `supported-smoke` is not the still-incomplete source C++ numerical comparison and carries no accuracy/performance claim; the S100P rejection paths are not positive support |
| B7-DOC-R1 — root quick start "it was not downloaded or run in this migration" contradicts the recorded X5 `n-v7.0` download + comparison | `yolov5/README.md`, `yolov5/README_cn.md` | The example is now identified as the default case downloaded and compared on real X5 8GB/4GB in the 2026-09-24 records (link), while this tree performs no new download or board run |
| B7-DOC-R1 — source overview reduced to one line | `yolov5/README.md`, `yolov5/README_cn.md` | Restored the concise architecture essentials from the S source description: CSPDarknet backbone, FPN+PAN feature fusion, three multi-scale detection heads (strides 8/16/32), and the `n/s/m/l/x` speed/accuracy trade-off |
| B7-DOC-R1 — model README "This migration did not run it" ignores recorded board preparation | `yolov5/model/README.md`, `_cn` | Preparation section now states the exact command ran on the 2026-09-24 board rounds (link) and this tree performs no new download; formats/checksums section notes the evidence archives record observed digests of the downloaded published artifacts (manifest publisher SHA-256 stays `null (unknown)`, table unchanged) |
| B7-DOC-R1 — Python runtime README "neither was used here" | `yolov5/runtime/python/README.md`, `_cn` | Sentence now records that the X5 example and the equivalent S100 `x-672` `kite.jpg` run were executed on real boards in the 2026-09-24 records (links), and that this tree performs no new board run |
| B7-DOC-R1 — native guide says X5/S100/S600 never built or ran | `yolov5/runtime/cpp/README.md`, `_cn` | Supported-boards table: X5/S100/S600 `supported-not-run` with false "no board/SDK/asset" claims became `supported-smoke` with per-board facts (X5 8GB+4GB; S100 first-round compile failure since fixed, round-2 real-SDK run; S600 `x-672`), plus a definition paragraph: real-board build + inference smoke at pinned commits with archived dumps, not the still-incomplete source numerical comparison, no accuracy/performance claim, no new board run. The stale "Board status" bullet was extended with round 2 (`4d45f9a`, both real SDKs, 14 detections on S100, X5 4GB/S600 builds) and the same boundaries |
| B7-DOC-R1 — conversion "All … board results are `not-run`" conflates OE not-run with published-model board history | `yolov5/conversion/README.md`, `_cn` | Post-conversion validation and the known-gaps bullet now say: export/calibration/compilation are `not-run` in this migration and no locally converted artifact exists; the 2026-09-24 board records concern the published runtime artifacts in `model/README.md`, which do not validate any local conversion. All commands remain untouched |
| B7-DOC-R1 — evaluator "This migration did not run it on a board" and "S100/S600 and current source/unified board comparison are `not-run`" contradict the Python comparison records | `yolov5/evaluator/README.md`, `_cn` | Command section and reference-results tail now state the tool ran on real boards at pinned commits (X5 nine variants 8GB/4GB; S100/S600 `x-672`), link the evidence, and keep the genuine gaps: no new board run from this tree, no re-validation of the current HEAD, MOT-style accuracy and the native C++ comparison remain `not-run` |
| B7-DOC-R1 — native comparison blanket "not-run in this migration" replaced an accurate boundary with a vague one | `yolov5/evaluator/README.md`, `_cn` | Native board status is now precise: the complete source/unified numerical comparison has not finished on any board (S100 fixed-source and unified real-SDK compiles completed at `3d6c14c`, X5 link interrupted — connectivity record linked); completed C++ board work is build/smoke only. Instructions, criteria and thresholds untouched |
| B7-DOC-R1 — ByteTrack root matrix `supported-not-run` on S100/S600 erases the tracker comparisons; S100P row implies pending verification | `bytetrack/README.md`, `bytetrack/README_cn.md` | S100/S600 became `supported-verified` scoped to the recorded cases (S100 four-frame synthetic; S100/S600 first 30 frames of `track_test.mp4`) with a note bounding them: exactly those frames/artifacts, not MOT accuracy, not the whole video, historical records only, no new board run. S100P stays `supported-not-run` but now states the recorded HTTP 404 for its published URL and that no positive run exists |
| B7-DOC-R1 — ByteTrack root "No model/video download was performed here" / "were not run" ignore recorded preparation | `bytetrack/README.md`, `bytetrack/README_cn.md` | Prerequisites and quick-start sentences now record that the 2026-09-24 rounds downloaded exactly this model and video for S100/S600 before their comparisons, while this tree performs no new download/preparation. Commands untouched |
| B7-DOC-R1 — ByteTrack model README "this migration did not run it" | `bytetrack/model/README.md`, `_cn` | Preparation records the S100/S600 board-round preparation (link); artifacts section states the S100P row is not positive availability (recorded 404, link; no S100P download or inference has ever run). Publisher checksums remain unknown |
| B7-DOC-R1 — ByteTrack conversion "No export, compile, board, or video validation was run" and matching known-gap bullet | `bytetrack/conversion/README.md`, `_cn` | Both sentences now separate the genuine gap (no export/compile run, no locally converted artifact) from the preserved history (published HBM rows and the first 30 frames of the public video ran on S100/S600 in the 2026-09-24 comparisons — linked to the evaluator README — which neither validates a local conversion nor extends beyond those frames) |
| B7-DOC-R1 — ByteTrack evaluator "this migration did not download it", "did not run it on a board", "Current board capture … `not-run`" | `bytetrack/evaluator/README.md`, `_cn` | Dataset section records the exact video was downloaded by the 2026-09-24 rounds (recorded SHA `4bbe5bf1…`, link); command section records the four-frame and first-30-frames board runs (links, all checks true) while keeping: no new board run, no re-validation of current HEAD, MOT-dataset accuracy `not-run`. Reference-results tail separates the recorded board captures from the still-not-run MOT benchmark. GIF figures and the threshold/association tuning bullets are byte-identical |

## Pairs checked and deliberately not changed

- `bytetrack/runtime/python/README.md` + `_cn`: no board-status sentence exists;
  "the input video is not bundled" remains true (the video stays external to
  the tree).
- `yolov5/prerequisites`, `expected-results`, `directory`, `entry-points`,
  license sections; all command blocks, parameter tables, integration
  examples, stage-I/O lists and troubleshooting lists in every edited file.
- `yolov5/conversion` source-model/export/calibration/compile/artifacts
  sections (their "not run" statements are true OE/conversion boundaries kept
  per the review).
- Historical performance tables in `yolov5` root/evaluator and `bytetrack`
  root/evaluator: byte-identical, still labelled historical and not re-run.
- ByteTrack source figures (`image1.png`, `image.png`, MOT17 GIFs) and the
  newly reviewed detection/association captions and threshold/association
  tuning paragraphs (root and evaluator, EN+CN): byte-identical.
- Skills pack: contains no mirrored YOLOv5/ByteTrack README content, so no
  `sync_references.py` run is required.

## New status tokens used (flagged for reviewer confirmation)

- `supported-verified` — follows the accepted FCOS/LPRNet/YOLOWorld README
  precedent: 2026-09-24 same-board source/unified comparisons at pinned
  board-test commits, always scope-bounded to the exact artifact/image cases
  in the adjacent note.
- `supported-smoke` (YOLOv5 C++ only) — new token, defined inline in both the
  root note and the C++ README: real-board compile + inference smoke at
  pinned commits with archived dump manifests; explicitly not the
  still-incomplete source C++ numerical comparison and no accuracy/performance
  claim. Chosen because neither `supported-verified` (numerical comparison)
  nor `supported-not-run` (erases the smoke records) is truthful for these
  four-board records.

## Verification (host, static only)

All commands run from repository root with `rdk_model_zoo/.venv/bin/python`;
raw outputs are committed in
[evidence/2026-09-28-b7-status-docs-remediation/](evidence/2026-09-28-b7-status-docs-remediation/).

- Scripted static checks: **all pass** (`check_links.py`, output
  `check_links.out`) — 11 pairs checked; every relative link and image link
  resolves; EN/CN pairs keep identical anchor sets, heading counts/level
  sequences, fence counts and image-link lists; 16 preserved-content probes
  (historical tables, source figures, tuning captions) all present; 42
  erase-history phrasings swept with zero remaining hits.
- Sample contract checker: `samples/vision/yolov5` and
  `samples/vision/bytetrack` each 0 violations, 1 pre-existing R-STAGE-PURITY
  policy skip, 0 exemptions (`contract-checks.out`).
- Host suites: YOLOv5 79 tests OK, ByteTrack 12 tests OK, shared 158 tests OK
  (`host-suites.out`) — same counts as the fresh B7 host results in the batch
  review.
- Scoped diff: `readme.diff` + `diffstat.txt` (20 files, +106/−71 lines). No
  file outside the two samples was modified by this remediation. Other working
  tree changes visible in `git status` (root/platforms/samples READMEs,
  gemma4-e2b sources, catalog tools) belong to concurrent sessions and are
  untouched here.

## Boundaries

This remediation changes descriptive prose and evidence links only. It does
not re-run or re-verify any download, conversion, compilation, inference or
comparison; it does not alter any CLI default, command, threshold, recipe or
code path; it does not strengthen any verification claim — every restored
historical statement carries its pinned-commit/evidence link plus the
"historical record, not a re-validation of the current HEAD" boundary, and the
genuine gaps (S100P unavailability, incomplete C++ source numerical
comparison, no OE conversion, no MOT/accuracy evaluation, MODNet manual asset
still unavailable) are restated rather than removed. Independent review of
this remediation is still required; B7 and H3 are not closed by this document.

---

# Append: B7-DOC-R2/R3 precision fixes — 2026-09-29

Applied by Claude Code + GLM against
`2026-09-28-b7-status-precision-review.md` (reviewer Codex). Same scope rules
as the main section above: the two samples' README package and author evidence
only; no code, recipe, checker-rule, reviewer/plan/ledger or board operation.
The R1 sections above are retained as history; where R2/R3 supersede their
wording, this append records the supersession.

## Facts re-read before editing

The exact retained records in
[evidence/2026-09-24-b7-bytetrack-realvideo30/](evidence/2026-09-24-b7-bytetrack-realvideo30/):
`b7-s100-realvideo-preparation.json` (`curl --fail --location --retry 2` into
`/tmp/rdk-b7-realvideo`, rc=127 `curl: command not found`),
`b7-s100-realvideo-python-download.json` (Python-stdlib fallback, rc=0, SHA
`4bbe5bf1…`), `b7-s600-64g-realvideo-preparation.json` (same curl form, rc=0,
same SHA), and `b7-s100-realvideo30-execution.json` (`compare.py` reusing
`--model-path samples/vision/yolov5/model/s100/yolov5x_672x672_nv12.hbm` and
`--input /tmp/rdk-b7-realvideo/track_test.mp4`). Also re-read
`docs/sample-standards/readme-contract.md` §4.1: matrix cells are the
three-state vocabulary `supported-verified` / `supported-not-run` /
`not-supported`.

## Findings addressed

| Review input | File(s) | Change |
| --- | --- | --- |
| **B7-DOC-R2** — root claimed both preparation commands were "exactly what" the S100/S600 rounds executed, but the records used different argv (curl flags into `/tmp`, model reused from the YOLOv5 sample directory) | `bytetrack/README.md`, `bytetrack/README_cn.md` (quick start, prerequisites) | Quick start now presents the two commands as the documented explicit preparation route for the same recorded resources (manifest HBM + `track_test.mp4`, SHA `4bbe5bf1…`), states the retained records' actual retrieval (curl form into a temporary directory, S100's Python-stdlib fallback after `curl` was missing, HBM reused from `samples/vision/yolov5/model/`), and says these exact argv were not themselves executed. Prerequisites now claims resource identity only and defers retrieval detail to Quick start. The two command blocks themselves are byte-identical |
| B7-DOC-R2 — model README "prepared this way on the 2026-09-24 board rounds" (same overclaim) | `bytetrack/model/README.md`, `_cn` | Now states the rounds compared HBM assets of the exact manifest identity but obtained them through the YOLOv5 sample downloader into `samples/vision/yolov5/model/`; the documented route was not itself executed. Command block untouched |
| B7-DOC-R2 — S100P universal "ever" phrasing ("no download or inference has ever run") exceeds the evidence: a recorded attempt did run and failed 404 | `bytetrack/README.md`, `bytetrack/README_cn.md` (matrix row + note), `bytetrack/model/README.md`, `_cn` | All four spots now say the cited 2026-09-24 round has no successful S100P download or positive inference record, instead of a universal "ever". Similar overclaims checked across both languages and root/model/evaluator: none remain (sweep below) |
| B7-DOC-R2 (similar overclaims found in the same sweep) | `yolov5/README.md`/`_cn` (quick start), `yolov5/model/README.md`/`_cn` (preparation) | Root quick start now says the recorded case ran with the same downloader but that "the argv below were not themselves the recorded board commands"; model README drops "This exact preparation ran" for "obtained the compared artifacts through this same downloader, executed there in its default-target form … the spelling with explicit `--variant`/`--output-dir` below was not itself the executed argv" |
| **B7-DOC-R3** — `supported-smoke` is outside the readme-contract §4.1 three-state vocabulary | `yolov5/README.md`, `yolov5/README_cn.md`, `yolov5/runtime/cpp/README.md`, `_cn` | Token removed everywhere. C++ matrix cells use `supported-not-run` with the smoke scope moved into the free-text status/note text: smoke records cover the C++ default variants only (X5 `s-v2.0` on 8GB/4GB; S100 `x-672` after the fixed first-round compile failure; S600 `x-672`), explicitly not the other X5 C++ variants; each note leads with "no source/unified numerical comparison" so historical execution is not relabelled as wholly not-run, and keeps the boundaries: smoke ≠ numerical verification, no accuracy/performance claim, no new board run, current HEAD not retested, source C++ comparison remains incomplete. The Python column keeps `supported-verified` with its exact-case scope. No fourth state was invented and no table column was added |

## Verification (host, static only — per the review, no runtime rerun)

Evidence refreshed in
[evidence/2026-09-28-b7-status-docs-remediation/](evidence/2026-09-28-b7-status-docs-remediation/):

- `check_links.py` / `check_links.out` (re-run): sweep extended to 55 patterns
  covering R1 erase-history phrasings, R2 exact-command/universal-"ever"
  overclaims and the R3 contract-external token — zero hits; all relative and
  image links resolve; EN/CN anchor/heading/fence/image parity holds for all
  11 pairs.
- `check_commands_unchanged.py` / `.out` (new): all 38 fenced code blocks
  across the 22 files are byte-identical to `HEAD` — commands preserved
  exactly.
- Sample contract checker re-run for both samples (`contract-checks.out`): 0
  violations, 1 pre-existing R-STAGE-PURITY skip each, 0 exemptions.
- `readme.diff` / `diffstat.txt` refreshed: still 20 files, +106/−71 net.
- Host unittest suites were **not** re-run for this prose-only delta (per the
  review's "no full runtime rerun needed"); the last recorded run remains the
  R1 round in `host-suites.out` (YOLOv5 79, ByteTrack 12, shared 158 — all
  OK), and no code file changed since.

## Boundaries (unchanged in substance)

No board/SSH/remote access, no model/video download, no
export/calibration/OE/HMCT/quantization, no installs, no git
add/commit/push/merge/reset/stash/switch, no subagents. Commands, figures,
historical performance tables, tuning captions, evidence archives and the
genuine gaps are preserved; the historical records still do not validate the
current HEAD, and the source C++ numerical comparison remains incomplete.
B7-DOC-R1/R2/R3 closure and H3 remain with the independent reviewer (Codex
reviews and syncs).
