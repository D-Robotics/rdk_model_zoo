# Twelve classifier README depth — independent review

Reviewer: Codex. Baseline 416434ab. Status: changes-required. Scope: 24 root
READMEs for convnext, edgenext, fasternet, fastvit, googlenet, hgnetv2, mobileone,
repghost, repvgg, repvit, resnext and vargconvnet. No implementation edits by
reviewer; all three findings below are assigned to Claude Code + GLM.

## DOC-CLASS-R1 — FastViT attention depends on variant (P2)

The new overview describes stage 4 as attention for the whole delivered family.
The paper Figure 2 is a configuration diagram, not evidence that all four
variants use attention. The [official model definitions](https://github.com/apple/ml-fastvit/blob/main/models/fastvit.py)
(accessed 2026-09-28; fastvit_t8/t12/s12/sa12) use RepMixer at all four stages for
T8/T12/S12 and attention at the final stage for SA12. Correct the English/Chinese
overview and caption to retain this distinction. This source review is not an
inspection of compiled model bytes or a hardware validation claim.

## DOC-CLASS-R2 — RepViT block versus downsampling module (P2)

The new Figure 3 caption labels the stride-2 DW/1×1/FFN downsampling chain as a
normal RepViTBlock. The actual maintained figure shows that chain inside the
orange Downsample module, preceded by a RepViTBlock. Normal blocks use the
parallel depthwise/identity branches followed by FFN; the SE variant inserts SE.
The feature bullet also says token/channel mixers are in separate blocks, which
obscures their decoupling within a block. Correct both language descriptions.
Codex directly viewed both maintained RepViT figures and the FastViT figure.

## DOC-CLASS-R3 — ConvNeXt root/evaluator disagreement (P2)

Restoring the atto benchmark row in root READMEs is justified: the fixed source
README and archived benchmark entry convnext-atto-x5 both contain the row.
[Extracted fixed-source table](evidence/2026-09-28-readme-depth-classifiers-independent-review/convnext-source-performance.txt).
But both evaluator guides still state that atto has no published benchmark row.
Update those two subdirectory guides from the same source, preserving historical
conditions and no-new-measurement boundaries. The old missing-row assertion is
incorrect; do not remove the valid restoration to make the documents agree.

## Evidence and review boundaries

[Independent source/static check](evidence/2026-09-28-readme-depth-classifiers-independent-review/source-static-check.json):
all 24 root README command blocks are unchanged from the committed baseline,
all 48 image references resolve and match ac11571 byte-for-byte, local file links
have no missing targets, and all 12 sample checkers returned zero. These passing
checks do not detect the semantic defects above. Codex read all twelve English
diffs and relevant Chinese counterparts, source atto table, and directly viewed
representative ConvNeXt, FasterNet, FastViT, RepGhost, RepViT and ResNeXt images.
Historical screenshot provenance remains explicit. No real model, board,
quantization recipe, compiler or dataset run was performed. H1 and this package
remain open until the specific text corrections and final hashes are rechecked.
