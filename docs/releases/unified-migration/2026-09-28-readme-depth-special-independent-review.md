# Special sample README depth — independent review in progress

Reviewer: Codex. Baseline 8bd4ef4e. Status: changes-required; full package
review in progress. Scope: Ultralytics YOLO, FCOS, ByteTrack and KWS source
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
