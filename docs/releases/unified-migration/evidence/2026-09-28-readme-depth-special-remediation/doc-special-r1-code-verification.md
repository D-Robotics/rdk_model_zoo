# DOC-SPECIAL-R1 remediation — tracker-code verification for the tuning prose

Reviewer finding: DOC-SPECIAL-R1 (bytetrack evaluator/root tuning prose described
`--track-thresh` as "filters out more low-score boxes", mismatching the actual
tracker partitioning; `--match-thresh` direction and `--track-buffer` scaling
unspecified). Remediation verified every rewritten claim against the working-tree
tracker code before writing. No code or behavior change; prose-only, EN+CN.

## Code facts read (all paths relative to repo root)

| Claim in remediated prose | Code evidence |
| --- | --- |
| Detector `--score-thres` filters before the tracker; lowering `--track-thresh` cannot restore detector-discarded boxes | `runtime/python/main.py:16` (`--score-thres`, default `.25`) feeds the detector stage; tracker receives post-filter boxes only |
| First association = scores strictly above `track_thresh` | `tracker_backend/byte_tracker.py:171` `remain_inds = scores > self.args.track_thresh` |
| Second association = scores in (0.1, track_thresh), against still-tracked targets, fixed cost limit 0.5 | `byte_tracker.py:172-175` (`scores > 0.1` AND `scores < track_thresh`), `:224` (only `TrackState.Tracked` remainders), `:226` `linear_assignment(dists, thresh=0.5)` |
| New tracks initiate only from unmatched first-association boxes with score ≥ `track_thresh + 0.1` | `byte_tracker.py:148` `self.det_thresh = args.track_thresh + 0.1`, `:258-261` (`u_detection` loop, `if track.score < self.det_thresh: continue`) |
| `--match-thresh` = maximum accepted assignment cost; cost = 1 − IoU, fused with detection score in first association | `byte_tracker.py:204` → `tracker_backend/matching.py:37-48` `lap.lapjv(cost_limit=thresh)`; `matching.py:87` `cost_matrix = 1 - _ious`; `matching.py:171-179` `fuse_score` (`fuse_cost = 1 - (1-cost)*score`), applied at `byte_tracker.py:203` |
| Second association keeps its own fixed 0.5 limit (not `--match-thresh`) | `byte_tracker.py:226` |
| `--track-buffer` = lost-track window in 30 fps frames, scaled by `frame_rate / 30`; `--frame-rate` default 30 | `byte_tracker.py:149` `buffer_size = int(frame_rate / 30.0 * args.track_buffer)`, `:150` `max_time_lost = buffer_size`, `:265-268`; `main.py:17` `--frame-rate` default 30 |

Boundary note (implicit in the open-interval wording): a score exactly equal to
`track_thresh` matches neither `> track_thresh` nor `< track_thresh` and enters
neither association stage — left implicit in prose to stay concise.

## Files changed by this remediation (prose only, EN+CN)

- `samples/vision/bytetrack/README.md` / `README_cn.md` — `expected-results`
  tuning paragraph rewritten: score-thres vs track-thresh separation, tracker
  partition semantics, det_thresh, match-thresh as maximum accepted cost with
  direction, track-buffer frame_rate/30 scaling.
- `samples/vision/bytetrack/evaluator/README.md` / `README_cn.md` —
  "Tracker parameter tuning and applicability" bullets rewritten with the same
  code-verified semantics (incl. second-association fixed 0.5 limit).

Out-of-scope observation (not edited per instruction): the runtime parameter
table (`runtime/python/README*.md`) still carries the coarser one-line flag
descriptions ("BYTETracker 高分阈值" / "首次关联阈值"); flagged here for the
maintainer without expanding this remediation's write scope.

## Checks after remediation

- `tools/sample_contract/check.py --sample samples/vision/bytetrack` (and the
  other three in-scope samples): 0 violations, 0 exemptions —
  `checker-after-doc-special-r1.txt`.
- All fenced command blocks in the ten edited READMEs remain byte-identical to
  the pre-package snapshot (12-file comparison rerun).
- `git diff --check` rc=0.
- Bilingual parity of the rewritten facts: 380e1a2 pin/SHAs untouched; new prose
  tokens (`track_thresh + 0.1`, `0.5`, `frame_rate / 30`, `1 − IoU`, defaults
  0.25/0.3/0.8/60, `--frame-rate`) present in both languages.
