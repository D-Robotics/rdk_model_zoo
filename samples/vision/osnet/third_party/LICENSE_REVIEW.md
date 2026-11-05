# OSNet source, weights and image permissions

Reviewed against the public upstream sources on 2026-11-02.

| Component | Evidence | Delivery scope |
| --- | --- | --- |
| D-Robotics integration code | Repository Apache-2.0 license | Source code, subject to that license |
| Pinned upstream `osnet.py` | Adjacent `LICENSE.deep-person-reid`, MIT | Keep the upstream copyright and permission notice |
| Current Market1501 checkpoint and its BIN/HBM derivatives | [Upstream owner discussion, issue 520](https://github.com/KaiyangZhou/deep-person-reid/issues/520) | Research use; unrestricted commercial redistribution is not approved by this review |
| Market1501 images, decoded fixtures and original retrieval cover | `readme.txt` in the [official dataset package](https://zheng-lab-anu.github.io/Project/project_reid.html) | Do not redistribute; prepare privately for research |

The dataset package states: “This dataset should be used for research only. Please DO NOT distribute or use it for commercial purpose.”

The exact checkpoint used by this sample has SHA-256
`2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154`.
It is the Market1501 x1.0 checkpoint linked by the
[official model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html),
not an ImageNet or MSMT17 checkpoint.

The author's [Hugging Face repository](https://huggingface.co/kaiyangzhou/osnet)
has an MIT model-card tag, but its inspected file list contains ImageNet and
MSMT17 weights, without this Market1501 checkpoint. That tag does not establish
a new permission grant for the exact bytes used here. The owner clarified the
research-only scope of pretrained models in the linked issue after an initial
reference to the code license.

No dataset images or model binaries accompany this source contribution. The
private acceptance measurements remain valid technical results; they do not
grant publication rights. Public model hosting requires permission covering
the exact checkpoint and converted derivatives, or replacement weights with
appropriate rights followed by conversion and evaluation of the new bytes.

On 2026-11-02 the user explicitly authorized Archive upload of the four accepted
BIN/HBM bundles with the research-only description retained. This hosting action
does not establish new upstream permission or approve unrestricted commercial
distribution. The user subsequently approved local source integration and formal catalog
registration for research use, stating that no additional authorization was
available. This decision is recorded in
[the release review](../evaluator/evidence/2026-11-02-release-review.json).
BIN/HBM redistribution terms remain separately unspecified by the cited
upstream statement; this record is not an additional upstream grant. Git push,
website deployment, Tags and GitHub Releases remain outside that approval.

The current Web cover is an AI-generated task illustration requested by the
user, without Market1501 pixels or a claim that it shows inference results.
Its generation provenance is recorded beside the Web asset. Any future cover
showing actual retrieval output must retain image permission and inference
provenance. Do not reuse the private Market1501 cover or fixture pixels.
