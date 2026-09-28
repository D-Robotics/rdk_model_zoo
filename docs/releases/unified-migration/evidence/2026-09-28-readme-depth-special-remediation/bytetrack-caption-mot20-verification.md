# Final narrow ByteTrack correction — score attribution and mot20 fusion (verification)

Reviewer's final narrow correction after DOC-SPECIAL-R2. Both items re-verified
before writing; prose-only, EN+CN, command blocks untouched, no code change.

## 1. image.png score attribution (root EN/CN caption)

Re-viewed `samples/vision/bytetrack/test_data/readme_img/image.png`
(sha256 `032728fb…`, unchanged). Row (a), frame t1 shows four yellow boxes:
0.9 (tall box, foreground person), **0.8** (smaller box — the tracked figure),
0.9 (man in black), 0.1 (right edge). The smaller box position matches the
0.4 box in t2 and the 0.1 box in t3, so the tracked person's visible trajectory
is **0.8 → 0.4 → 0.1**; the 0.9 boxes are the taller foreground person. The
previous sentence ("the walking woman's score falling 0.9 → 0.4 → 0.1")
conflated two boxes. Corrected in both languages to attribute 0.8/0.4/0.1 to
the smaller tracked person and 0.9 to the taller foreground person; only
visible numbers are used.

## 2. match-thresh fusion condition (root + evaluator EN/CN)

Code facts:

- `runtime/python/main.py:17` — `--mot20` is a maintained `store_true` flag
  (default `false`), passed into `TrackingConfig` (`main.py:31`) and listed in
  the runtime parameter table (`runtime/python/README*.md:40`,
  default `false`, "source matching mode").
- `tracker_backend/byte_tracker.py:202-203` (first association) and
  `:246-247` (unconfirmed-track stage): `if not self.args.mot20:
  dists = matching.fuse_score(dists, detections)` — fusion happens **only
  when `--mot20` is not set**; with `--mot20` the cost is plain 1 − IoU
  (`matching.py:87`). The second association (`:225-226`) never fuses and
  keeps its fixed 0.5 limit.

The previous prose ("cost = 1 − IoU, fused with detection score", stated
unconditionally) is corrected to: fused with detection score in the default
mode; `--mot20` (default `false`) disables the fusion.

## Files touched

- `samples/vision/bytetrack/README.md` / `README_cn.md` — overview caption
  score attribution; match-thresh fragment in the tuning paragraph.
- `samples/vision/bytetrack/evaluator/README.md` / `README_cn.md` —
  match-thresh bullet fusion condition.

R1 tuning semantics and R2 image identities are otherwise unchanged; no other
sample, no reviewer file, no code touched.
