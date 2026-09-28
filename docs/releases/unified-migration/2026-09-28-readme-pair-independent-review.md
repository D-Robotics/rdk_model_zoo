# README bilingual command review — open findings

Reviewer: Codex. Base `8318a83b8eb83fb308a2856037f586252149d0af`.
This is independent static review for H1/H9, not full README acceptance.
Implementation remains assigned to Claude Code + GLM; no customer documentation
or product code was changed by this review. No board, SDK, model download,
conversion, calibration or quantization command was executed.

## Coverage and limits

281 English/Chinese README pairs were found under native vision, speech,
robotics and LLM samples, excluding third_party/build/.git paths. No missing
Chinese sibling was found in that scope. Comparing shell-labelled code blocks
while ignoring blank and whole-comment lines produced 13 differences. Manual
inspection identified two actionable inconsistencies below; remaining differences
are line wrapping, translated comments/placeholders or translated directory trees.
This scan does not prove source-content preservation, parameter correctness,
all link validity or full bilingual semantic equivalence. VLA gitlinks, platform
and root documentation are outside this scan.

Evidence: [findings.json](evidence/2026-09-28-readme-pair-review/findings.json),
including base, file hashes, full difference inventory and actual image names.

## DOC-R1 — EdgeNeXt evaluator references missing bundled images (P2)

English evaluator README line 45 selects `test_data/zebra.JPEG`; Chinese line 40
selects `test_data/bittern.JPEG`. Neither exists. The tracked bundled image is
`test_data/Zebra.jpg`. Both commands therefore fail to load their image instead
of performing the described functional check. The Chinese prose also expects a
zebra result while selecting a bittern filename.

Claude follow-up: use the actual bundled image in both examples; keep the same
input and stated success criterion. Verify file existence and bilingual command
agreement statically. Do not run the board check or invent a new board result.

## DOC-R2 — OCR English inspection command uses a different output name (P2)

English conversion README line 197 adds an `en_` prefix to the S100 detector
artifact. Its own expected-output listing at line 176, the Chinese inspection
command at line 185, and `s100/paddleocr_det_configs.yaml` line 18 all specify
`PP-OCRv6_det_infer-deploy_640x640_nv12` without that prefix. Following the shown
recipe then copying the English inspection command addresses a different file.

Claude follow-up: align the English command with the documented/configured output
and Chinese counterpart. This is a README filename correction only; preserve the
source conversion recipe and do not execute it to verify this finding.

## Status

DOC-R1 and DOC-R2 remain open for the next Claude documentation package, after
its active MiniCPM package. Do not interrupt or expand the live implementation
package. H1/H9 remain open; this report does not change batch acceptance states.
