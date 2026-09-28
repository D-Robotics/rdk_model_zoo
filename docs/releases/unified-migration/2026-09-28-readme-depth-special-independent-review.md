# Special sample README depth — independent review in progress

Reviewer: Codex. Baseline 8bd4ef4e. Initial status: changes-required. Current status: scoped README package
accepted after independent final recheck. Scope: Ultralytics YOLO, FCOS, ByteTrack and KWS source
README depth restoration by Claude Code + GLM.

## DOC-SPECIAL-R1 — ByteTrack threshold semantics

The added evaluator tuning description says increasing track-thresh filters out
more low-score boxes. Actual tracker_backend/byte_tracker.py partitions detections
into first association (score > track_thresh) and second association
(0.1 < score < track_thresh); new-track det_thresh is track_thresh + 0.1.
The CLI detector score threshold runs first and discarded detections cannot
be recovered by changing the tracker threshold. Correct bilingual root/evaluator
tuning prose to distinguish detector filtering, association grouping and new
track creation. Do not change algorithm behavior or invent tuning results.

Assigned to the original terminal Claude Code + GLM session for a bounded
documentation correction. Other images, source provenance and runtime descriptions
remain under review; this finding alone does not exhaust package verification.

## DOC-SPECIAL-R2 — Reversed ByteTrack figure identities

Codex opened both maintained images. The embedded image1.png is a single
horizontal strip of three street-scene frames, with colored boxes and yellow/red
triangle markers. It is not the three-row (a)/(b)/(c) motivation figure described
by the new caption. That figure is image.png: three frame columns and three
comparison rows; the low-score examples show 0.4 then 0.1. The author mapping
also incorrectly calls unused image.png the single strip, reversing both files.

Correct the bilingual caption and author provenance/mapping. Keep the image
actually embedded by fixed source; a second explanation figure is permitted
only with its distinct identity and explicit bundled-but-not-source-embedded
provenance. This is a visual-content defect missed by path/hash checks. A
sequential follow-up is queued for the original Claude session after R1.

## Partial inspection notes

Codex viewed both Ultralytics S result images: same bus scene, distinct boxes
and labels (names versus class IDs), consistent with the separate captions.
All three FCOS dataflow graphs were viewed: packed NV12 to NV12TOYUV444,
INT8 NHWC into the BPU subgraph, fifteen INT32 output heads. Visible pyramids
are 64..4, 96..6 and 112..7 with 80/4/1 channels, matching restored descriptions.
KWS pipeline values were checked against frontend.py/model_binding.py/main.py/
postprocess.py: 60000 samples, 25/10 ms, 80 mel, [1,373,80], max probability
and >= threshold 0.5. The algorithm description is attributed to fixed source
380e1a2; this review does not independently certify its training architecture.

## Final independent recheck — accepted (2026-09-28)

DOC-SPECIAL-R1 and R2 are closed. Codex read all four final ByteTrack bilingual
diffs and compared tuning semantics with byte_tracker.py: detector filtering
precedes score partitioning; second association uses 0.1..track_thresh; new
tracks require track_thresh+0.1; first-association score fusion is disabled by
mot20; buffer duration scales by frame_rate/30. The guide now preserves these
distinctions rather than treating every threshold as a detection filter.

The two PNG identities now match the actual images Codex viewed: image1.png is
the source-embedded horizontal strip; image.png is the additional bundled
three-row illustration. The latter tracked person has 0.8, 0.4, 0.1 scores,
not the foreground person's 0.9. Both languages and the author provenance map
now reflect this correction. MOT17 GIFs match fixed-source bytes and their
source evaluator references, and remain historical illustrations.

[Final checks](evidence/2026-09-28-readme-depth-special-independent-review/final-checks.json)
bind all ten changed README files: fenced command blocks unchanged, local file
links resolve, all twenty image references match their fixed X5/S source bytes.
Four fresh sample contract checks return zero violations and zero exemptions
with the existing CLI policy skips. The new YOLO26 result image is separately
identified, preserving both different S illustrations. FCOS graph descriptions
match the displayed shapes/types, and KWS restores source-attributed algorithm
context plus the actual 3.75-second/frontend/probability pipeline.

Accept this documentation package and close KWS-N1. No product runtime changed,
so unrelated runtime suites were not repeated; no board, dataset, export or
quantization workflow was executed. Together with the accepted nine B1/B2 and
twelve classifier packages, all 24 destinations from the source-image finding
have independent disposition. Repository-wide README and integration scope
remains separate; this is not automatic H1/H2/H9 closure.
