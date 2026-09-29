# LaneNet independent host review — accepted within host scope

Reviewer: Codex. Base `56a45f3a`; source S `380e1a2` stays unchanged.
Accept the current sample's host migration scope. No blocking finding was found
in the reviewed responsibilities, tensor/ownership handling and customer guide.
The reviewer changed no sample implementation. Whole B8/H4 remains open.

Python task inspection confirms RGB/ImageNet preprocessing separated from raw
transport, named output validation and copied embedding/binary results. The model
file holds the three stages plus predict, while CLI, image conversion and rendering
live separately. Binary outputs must already contain 0/1; no invented clustering,
lane IDs or curve fitting is supplied. The intentional display rounding change
is documented, and raw float embeddings are kept separately from clipped pictures.

Native inspection covered input layout binding, dynamic-stride resolution without
out-of-rank access, checked size arithmetic, nonoverlapping/capacity-bound byte
strides, unique output-role binding, owned raw copies, scoped packed-model/buffer/
task cleanup, and exact typed output serialization. Auxiliary output names/types
are not invented; native float16 limitations are stated. Runtime identity rejects
S100P rather than silently reusing S100. Returned bytes outlive the SDK task;
cleanup failure and real SDK ABI are not certified by host fixtures.

Root/runtime/evaluator documents correctly distinguish the source embedding
visualization from lane-instance segmentation, preserve historical figures and
HRT numbers, describe current output filenames/dtypes and keep Python versus
native role-binding differences visible. C++ lifecycle text points to actual
headers and states the captured runner's ownership requirement. No dataset
accuracy workflow is fabricated from the source evaluator placeholder. Conversion
source content and its missing export prerequisites are retained as documentation;
no real recipe verification was required or performed.

## Independent verification

[verification.json](evidence/2026-09-28-lanenet-independent-review/verification.json)
contains the full commands, outputs, time, cwd and candidate file hashes.

- Twenty-two sample tests pass: stage/source arithmetic, labels and display,
  CLI/output boundaries, launcher behavior and native tensor/resource/NPY fixtures.
- Additional compilation with the existing real host OpenCV headers and
  `-std=c++17 -Wall -Wextra -Werror` succeeds for production `lanenet.cpp`,
  `visualization.cpp` and `main.cpp`. These were compile-only object checks.
- SDK-facing resource tests use clearly labeled fake headers and calls. The new
  OpenCV compilation does not establish a full linked vendor executable or SDK ABI.

Board inference, real SDK linking, performance and dataset accuracy remain
not-run. No model/dependency download, quantization, toolchain installation or
hardware access occurred. Ordinary host preparation fixtures are not execution
of the user's trusted conversion recipes. Earlier source-image audit confirms
three image references are represented, without substituting that lexical check
for the prose review above.

Roll this scoped disposition into the next coordinated ledger/plan update after
the active MiniCPM worker releases those files. It does not close other B8 samples
or claim whole-branch completion.
