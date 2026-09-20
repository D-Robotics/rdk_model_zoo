# Model Zoo Web Release Inputs

This directory contains the reviewed, lightweight inputs required to reproduce
the published static site in a clean checkout.

- `inputs.json` maps catalog records to their release assets.
- `assets/` contains optimized website covers. Covers that show inference
  overlays must be produced from real model output.
- `reports/` contains compact structured OE data used by the website report
  view. Complete OE HTML reports and deployment models remain in OSS.

Paths in `inputs.json` are relative to this directory. The release build uses
this manifest by default; `MODEL_ZOO_INPUTS` may override it for local review.
