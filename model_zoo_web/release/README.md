# Model Zoo Web Release Inputs

This directory contains the reviewed, lightweight inputs required to reproduce
the published static site in a clean checkout.

- `inputs.json` maps catalog records to their release assets.
- `assets/` contains optimized website covers. Covers that show inference
  overlays must be produced from real model output. Sample-entry covers (board
  annotations or charts drawn from real tracker output) document their sources
  and hashes in `assets/sample-covers.json`.
- `sample-evidence/` contains small text summaries (JSON/log, each under
  64 KiB) backing sample-profile entries: board verification payloads and
  compile/build log excerpts. They are summaries, not full reports, and ship
  with the site under `sample-evidence/`.
- `reports/` contains compact structured OE data used by the website report
  view. Complete OE HTML reports and deployment models remain in OSS.

Paths in `inputs.json` are relative to this directory. The release build uses
this manifest by default; `MODEL_ZOO_INPUTS` may override it for local review.
