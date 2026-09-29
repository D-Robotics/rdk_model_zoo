# B7 integrated host/doc review — 2026-09-28

Status: **documentation changes required**; H3 remains open. This follows H0 integration acceptance, not a new board run. Codex reviews; Claude Code + GLM implements.

## B7-DOC-R1 — Current customer guides erase existing historical board evidence

YOLOv5 root EN/CN says neither language was run on a board; its native guide says X5/S100/S600 never built or ran. Python evaluator and ByteTrack root/evaluator repeat migration-wide board-not-run statements. These contradict the preserved `2026-09-24-b7-native-sdk-review.md` and its exact linked raw evidence. Distinguish historical measured revision/case coverage from the current HEAD, which has no new board validation. Do not convert native smoke into native source/unified numerical acceptance, paper metrics into current results or negative tests into positive support.

Authoritative boundaries: YOLOv5 X5 nine Python variants on 8GB/4GB, S100/S600 x-672 Python source comparisons; four-board C++ build/smoke records with source C++ numerical comparison still incomplete; S100P YOLOv5 rejection-only (no asset). ByteTrack S100/S600 source-video first 30 frames comparison at 4d45f9a; S100 earlier four-frame check; S100P original asset URL returned 404 and positive inference was not run. MODNet manual asset remains unavailable. All historical claims must retain original commit/evidence identity; no new board result is requested or authorized.

Update affected bilingual root/model/runtime/conversion/evaluator status sentences, linking evidence where useful, while leaving valid instructions, source algorithm figures/tuning and recipe commands untouched. Conversion not-run is separate from previously tested published runtime models. Avoid blanket “not downloaded in this migration” language where actual prior preparation exists. Include source overview essentials rather than reducing source explanations to one line; YOLOv5 S source explains CSPDarknet and FPN/PAN with multi-scale heads. Preserve standalone YOLOv5 scope as previously authorized; no new sample retirement or feature changes here.

## Integration evidence

`evidence/2026-09-28-b7-batch-independent-review/prior-code-binding.json` verifies 117 B7 code/test files recorded by the accepted H0 integration review with **zero drift**. That prior review accepted the native audit/log archival repairs and all included branches. Fresh B7 host results are recorded separately in `host-suites.json`; neither hash stability nor host tests alone close the document finding. Historical first-round failure reports remain untouched.

Fresh integrated host suites pass: **184 tests** (YOLOv5 79, FCOS 38, YOLOWorld 19, LPRNet 23, MODNet 13, ByteTrack 12). Full commands/output are in `host-suites.json`. The code/test hash recheck remains unchanged; B7-DOC-R1 is queued to Claude Code + GLM after the existing B10 doc task. H3 is not closed until the corrected documentation is independently reviewed.
