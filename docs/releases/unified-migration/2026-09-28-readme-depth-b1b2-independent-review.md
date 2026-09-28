# Nine B1/B2 README depth — independent review

Reviewer: Codex. Scope: the English/Chinese root READMEs of resnet,
mobilenetv1/v2/v3/v4, efficientnet, efficientformer, efficientformerv2 and
efficientvit, plus five restored ResNet figures. Status: changes-required for
one caption; the bounded clarification below was returned to Claude Code + GLM.
This is part of DOC-DEPTH-R1, not whole H1 or subdirectory documentation closure.

## DOC-B1B2-R1 — MobileNetV3 SE position (P2)

Both new MobileNetV3 captions say SE gates the "1×1 output" / "门控1×1输出".
The actual maintained source figure shows pooling after depthwise convolution,
channel gating of the expanded feature map, then the final 1×1 projection.
This caption can teach the wrong block ordering. Correct both languages to
identify the feature map before the final projection. Codex directly viewed
`test_data/MobileNetV3_architecture.png`; this finding does not depend on a web
inference or a real model run. No product edits were made by the reviewer.

## DOC-B1B2-N1 — avoid an unsupported deployment generalization

EfficientFormerV2's new paragraph claims the costs kept earlier hybrid models
"off mobile devices" / "无法登上移动设备". This is stronger than the source
figure supports, and the adjacent EfficientFormer sample describes mobile
measurements. State the specific reductions in attention/downsampling overhead
without the blanket historical deployment claim. Requested in both languages.

## Other reviewed properties

The added text restores algorithm context and explains the source images inside
the relevant overview/result sections. EfficientFormer distinguishes iPhone
12/CoreML paper profiling from RDK performance. EfficientNet distinguishes the
X5 B2/B3/B4 family from S Lite geometry. Historical screenshots are marked as
source results and do not imply new board runs. ResNet S variant figures are
renamed clearly; shared architecture figures are deduplicated by content.

The reviewer inspected the English changes and corresponding Chinese captions,
and directly viewed representative architecture/result figures, including the
MobileNetV3, EfficientFormer, EfficientFormerV2, EfficientViT and ResNet figures.
Source-byte, local-link, command-block and checker results are recorded separately
in [source-static-check.json](evidence/2026-09-28-readme-depth-b1b2-independent-review/source-static-check.json).
A passing static check does not close the caption finding. Recheck final changed
text and hashes after Claude terminates before committing the customer package.
No weights, board, export/calibration, quantization recipe or toolchain execution
is part of this documentation review.

## Final independent recheck — accepted

After Claude's remediation process exited with rc=0, Codex read the revised
MobileNetV3 SE ordering and EfficientFormerV2 performance wording in both
languages. R1 and N1 are resolved. Only these four expected README files changed
since the source/static check. All 18 README command blocks remain identical to
the committed baseline and all referenced source-image hashes remain unchanged.
The two affected sample checkers again returned 0; the other seven earlier
checker results remain applicable to unchanged files.
[Final hashes and commands](evidence/2026-09-28-readme-depth-b1b2-independent-review/final-recheck.json).

Accept these nine samples' bounded root README depth restoration (18 bilingual
files, five newly restored ResNet images). All 50 image references have exact
fixed-source byte matches, all checked local links resolve, and the source
explanations/performance boundaries are preserved. This closes the first nine
sample portion of DOC-DEPTH-R1, not the remaining classifier/special packages,
all subdirectory reviews, H1, or whole migration. Earlier findings remain above
as history; current disposition for this package is accepted.
