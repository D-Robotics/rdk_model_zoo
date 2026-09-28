# Twelve classifier README depth — independent review

Reviewer: Codex. Baseline 416434ab. Initial status: changes-required.
Current status: scoped README package accepted after independent recheck. Scope: 24 root
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

## Final independent recheck (2026-09-28)

DOC-CLASS-R1, R2 and R3 are closed for this package. Codex read the revised
FastViT and RepViT descriptions in both languages and both ConvNeXt evaluator
diffs. FastViT now distinguishes T8/T12/S12 from SA12; RepViT separates the
normal block from the orange downsampling unit. ConvNeXt root and evaluator
now retain the same source atto row (73.25% float, 69.75% quant, 1.96 ms,
732+ FPS); the incorrect missing-row assertion and 72.50% prose are removed.
Historical measurements remain explicitly attributed and are not re-measured.

[Final independent evidence](evidence/2026-09-28-readme-depth-classifiers-independent-review/final-recheck.json)
binds all 26 reviewed documents to SHA-256. Only the four requested FastViT/RepViT
root documents changed from the initial 24-document snapshot; the additional
two documents are the ConvNeXt evaluator pair. All fenced command blocks remain
unchanged from the committed baseline. All 48 root image references still match
fixed source ac11571 byte-for-byte. Fresh ConvNeXt, FastViT and RepViT contract
checks each returned zero violations, one CLI policy skip, zero exemptions.
The other nine roots retain their initial reviewed hashes and checker evidence.

Acceptance covers these twelve root README pairs and the ConvNeXt evaluator
pair only. It closes this portion of source README depth restoration, not H1
or the full migration. No runtime change, board test, recipe execution, model
download, conversion or dataset measurement was performed by this review.
