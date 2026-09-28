# README pair independent re-review — changes required

Reviewer: Codex. Base `2f128f2f`, reviewing the uncommitted author package
recorded in `2026-09-28-readme-pair-remediation.md`. Candidate hashes and independently
run checker output are in [evidence](evidence/2026-09-28-readme-pair-rereview/).
No model, board, export/calibration/quantization or toolchain provisioning was run.

DOC-R1 and DOC-R2 are accepted: the two EdgeNeXt examples now name the actual
`Zebra.jpg`; OCR EN now matches its configured S100 filename and CN. These are
static command/path corrections, not new board evidence. The three-sample checker
passes with 0 violations, 4 declared skips and 0 exemptions.

DOC-R3 restores all four images and substantial explanatory text, but cannot close
yet. New text must distinguish inherited source errors from maintained contracts.

## R3-A — pose shapes conflict with the maintained decoder (P2)

EN line 308 and the matching CN paragraph state 57 channels for 17 points with
three values each. `pose_decode.py:39–45` requires one class and `3 * nkpt` keypoint
channels; the published 17-point contract means 51. Visual inspection of the
historical PNG confirms it also prints 57 and 80 classes, while reshaping to 3×17.
Correct the prose and place a clear adjacent correction caption on the preserved
historical image. Retaining the original illustration is acceptable if its stale
shape labels are explicitly distinguished from today's binding. No raster edit,
model execution or recipe verification is needed.

## R3-B — axis names invert the displayed coordinate formula (P2)

EN lines 239–244 call x the row and y the column, but use x for horizontal box
coordinates and y for vertical coordinates. CN does the same. Define x as column
and y as row (or consistently rename the formula). The described grid must match
the decoder's horizontal/vertical convention.

## R3-C — blanket NMS claim loses the YOLOv10 exception (P2)

EN lines 254–256 / matching CN say the DFL stages are followed by class-wise NMS
while the section includes YOLOv10. The maintained decoder honors `nms='none'`,
and the Python README explicitly documents S YOLOv10 as NMS-free. Scope NMS to
bindings that require it and explicitly retain that exception. Do not change code.

## Additional clarity and reporting corrections

- EN lines 181–183 suggest a raw maximum equals the post-Sigmoid score. Say
  `sigmoid(max(logits)) = max(sigmoid(logits))`; the argmax ordering agrees but
  the raw value itself is not the probability. Make CN equally precise.
- Give the S source its actual pin `380e1a2...`, distinct from X5 `ac11571...`;
  the author record currently describes both as carried from the X5 pin.
- The author report says no checker ran while listing contract-checker output.
  Distinguish the executed README checker from unexecuted model/toolchain checks.
- Keep future edits within the assigned repository files. The worker log also
  records edits to local Claude memory, outside the stated file scope. Disclose
  those separately from repository changes; do not include or overwrite unrelated
  private memory in GitHub. No further memory edits are part of the revision task.

Revise the two Ultralytics READMEs and author record, retaining source depth,
existing recipes and bilingual parity. Re-run scoped static checks. Independent
acceptance remains pending these corrections; H1/H9 are not closed.

## Second pass — R3-A/B/C addressed; R3-D remains (P2)

The revised candidate corrects pose shape captions, axes, NMS scope, raw logits
and source pins. A further code-to-prose check found that the pose paragraph still
says multiplying keypoint coordinates by stride yields input-image coordinates.
For maintained DFL pose, `rdk_yolo_utils/postprocess.py:461` computes
`(raw_xy * 2 + anchor - 0.5) * stride`, with half-integer cell-center anchors;
`pose_decode.py:101–103` restores original-image geometry and applies sigmoid
to keypoint visibility logits. This differs from the existing YOLO26 direct
branch's `(raw_xy + anchor) * stride`. State the correct DFL formula and the
visibility/geometry steps instead of preserving an incomplete source description.
The published binding fixes `nkpt=17`, so avoid implying other counts are accepted
by changing the shape alone. This is static documentation correction only.
