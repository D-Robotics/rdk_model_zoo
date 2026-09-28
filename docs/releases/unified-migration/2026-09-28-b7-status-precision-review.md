# B7 status documentation — independent follow-up

Reviewer: Codex. B7-DOC-R1 remains changes-required pending these bounded fixes.

## B7-DOC-R2 — historical commands overstated

ByteTrack root EN/CN says both preparation commands are exactly what the historical
S100/S600 rounds executed. Current example uses the sample downloader into
samples/vision/bytetrack/model and curl -L into test_data. Actual retained
b7-bytetrack-realvideo30 preparation records use curl --fail --location --retry 2
into /tmp/rdk-b7-realvideo; the recorded inference command reuses a model at
samples/vision/yolov5/model. Thus asset/video identity can be inherited, but exact
command execution cannot. Describe the documented commands as the present explicit
preparation route for those same recorded resources, without claiming those argv
were executed. Preserve the commands themselves and all historical evidence.

S100P wording "no download or inference has ever run" also exceeds the evidence:
a recorded download attempt did run and failed 404. State that the cited round has
no successful download/positive inference record, rather than universal "ever".
Check both languages and root/model/evaluator for similar overclaims.

## B7-DOC-R3 — support status vocabulary

readme-contract section 4 support-matrix defines supported-verified /
supported-not-run / not-supported. The new supported-smoke value is outside this
contract. Use the established states with explicit smoke-only scope and exact
cases/variants in the status text, or separate smoke evidence into its own column
without inventing a fourth support state. Do not relabel known historical board
execution as wholly not-run, and do not imply all nine C++ variants were compared.
Numerical source C++ comparison remains incomplete. Current HEAD was not retested.

Scope remains the existing YOLOv5/ByteTrack README package and author evidence.
No code/recipes/commands/board operations or checker-rule changes are needed.
