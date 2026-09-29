# Source README illustration audit — changes required

Reviewer: Codex. Base `5311f9c4`. This is a source-depth audit for H1/H2/H9;
prior batch functional acceptance does not establish this broader requirement.
No recipe, model, board or SDK execution was performed.

## DOC-DEPTH-R1 — maintained documentation loses source illustrated explanations

The migration map yielded 72 source-to-destination mappings and 231 source image
references (English and Chinese occurrences count separately). The initial lexical
scan found 126 absent filename references. These are screening counts, not 126
independent bugs. Follow-up checked source-file existence, URL-decoded paths,
SHA-256-identical renamed images and the retired duplicate-YOLO user ruling.
Twenty-four active destinations still have at least one real, unreferenced source
illustration or historical result screenshot:

`bytetrack`, `convnext`, `edgenext`, `efficientformer`, `efficientformerv2`, `efficientnet`, `efficientvit`, `fasternet`, `fastvit`, `fcos`, `googlenet`, `hgnetv2`, `mobilenetv1`, `mobilenetv2`, `mobilenetv3`, `mobilenetv4`, `mobileone`, `repghost`, `repvgg`, `repvit`, `resnet`, `resnext`, `ultralytics_yolo`, `vargconvnet`.

Evidence: [full inventory with README/source/image hashes and rename checks](evidence/2026-09-28-source-readme-image-audit/inventory.json).
Retired yolov13_imoonlab source illustrations are explicitly excluded from
required restoration. Repeated source aliases mapping into ResNet or YOLO are
not requirements to duplicate the same diagram repeatedly.

Manual source/current inspection confirms the issue in EfficientViT: the source
explains attention data movement, cascaded group attention and batch normalization
with three illustrations, then shows its historical inference screenshot. The
current maintained root README gives the task/runtime overview but references
none of those four existing image files. FCOS conversion similarly retains all
three graph PNGs but does not link them; ByteTrack retains source overview and
MOT17 GIF assets without their source README references. Filename presence alone
would still be insufficient if pictures were pasted without explanation.

## Required restoration, assigned in bounded Claude packages

Start with B1/B2 roots: resnet, mobilenetv1/2/3/4, efficientnet,
efficientformer, efficientformerv2 and efficientvit (nine samples). Read the
fixed X5 ac11571 and applicable S 380e1a2 source README bodies, inspect existing
figures, and restore algorithm/deployment context plus historical example images
in both languages. Preserve the current quickstart/API/defaults, exact artifact
selection, support matrix and independent board evidence. Cite source pins and
clearly label historical screenshots; they are not new run evidence. Deduplicate
identical cross-source diagrams with an explicit disposition. Do not reduce this
to an image appendix or redirect readers to archived branches for core content.

Then cover the other fifteen destinations in bounded groups, including FCOS
conversion and ByteTrack evaluator. Existing conversion commands remain trusted
source recipes; do not rerun export/calibration/quantization to accept this work.
Do not automatically transplant stale captions if current semantics differ.
Source benchmarking numbers keep their conditions and provenance.

## Acceptance and limits

Review source explanation against maintained bilingual prose, image captions,
local paths and current interfaces. Run relevant static README checks and check
that existing command blocks and support claims were not accidentally changed.
Do not add tests that merely require a fixed word count or image filename list.
B8's eight sample families had no missing source filename references in this
scan, but that observation does not prove full source content or semantic parity.
This inventory does not yet cover root/platform README or external VLA gitlinks.
H1/H2/H9 remain open; this finding remains open until all affected active samples
have independent disposition.

## DOC-DEPTH-R1 closure (2026-09-28)

All 24 affected destinations now have independent disposition: nine B1/B2 roots
([review](2026-09-28-readme-depth-b1b2-independent-review.md)), twelve classifier
roots plus ConvNeXt evaluator ([review](2026-09-28-readme-depth-classifiers-independent-review.md)),
and Ultralytics/FCOS/ByteTrack ([review](2026-09-28-readme-depth-special-independent-review.md)).
Source images, context, current command invariance and corrected captions were
reviewed per package; non-restored assets have explicit provenance/disposition.
Close this source-image finding. It does not by itself certify every README
paragraph, repository/platform guide or H1/H2/H9 completion.
