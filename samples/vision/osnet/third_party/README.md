# Upstream OSNet source

`osnet.py` is an unmodified copy of
[KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/osnet.py).
It is distributed under the adjacent `LICENSE.deep-person-reid` (MIT).

Source snapshot SHA-256:
`c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae`.
The file fetched from upstream on 2026-10-29 matches both the original
experiment's `export.json` and the independently checked source in its delivery
ZIP. The original build's upstream Git commit is unavailable; this record pins
the actual file bytes rather than inventing a commit.

Original checkpoint SHA-256:
`2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154`.
See `../evaluator/evidence/export.json` for the original source/weight/ONNX pair.
Inference never imports this PyTorch source; it is used only for conversion.
