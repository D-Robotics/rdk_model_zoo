# Native YOLOE preflight and shared identity/digest helpers

Base: `71293d9c4a58a98b2342284246b2129b285df986`. The executable/launcher is
still pending; this increment replaces an application-written preflight
placeholder with a concrete local identity/model/vocabulary policy. All H0–H9
work, other migrations and final independent review remain open. No board,
real SDK or OE validation was performed.

## Behavior and reuse

`make_preflight(expected_model_sha256, label_path)` creates the callback required
by SdkRunner. It reads fixed local boardinfo/socinfo/device-tree paths and rejects
unknown/mismatching hardware, missing/empty model files, malformed or mismatching
expected digests and vocabulary bytes other than the fixed ordered 4585-class
file. It never hashes the current model and silently adopts that result as the
expected digest. SdkRunner separately enforces target/variant/stack and physical
tensor contracts. An expected custom hash establishes byte identity, not a
publisher or conversion-toolchain certification. Publication selection and
recording those claims remain the launcher's responsibility.

The policy has no CLI/environment board-identity override. A pure observation
helper is exposed for host tests; the production callback always reads local
files. Shared native identity mirrors the active registry and Python fallback
precedence, including S100P refinement and refusal to fall back from unknown
higher-priority identity data. S600 may be recognized by the shared helper but
is still rejected by YOLOE's model binding/adapter.

Streaming SHA-256 was extracted from the existing YOLOv5 dump writer into
samples/_shared/cpp/sha256.h; YOLOv5 now delegates to it. No second hash algorithm
or crypto dependency was added. The common helper returns an empty string on
open/read/type failure, which differs from an empty regular file's valid digest.
The first cross-language test found a directory was incorrectly accepted
as a hashable file on this host; the helper now explicitly requires a regular
file. The initial failing log is preserved. Identity/model types are separate
from SDK/Runner headers, so the preflight library itself needs no OpenCV or SDK.

## Verification and documentation

The first preflight test compilation failed on the missing API header before
implementation. Native tests cover real model/vocabulary file hashing against
injected observations, mismatch/unknown rejection, uppercase expected digest,
known SHA vectors and one million `a` bytes. The production callback rejects
this unidentified host; no host fixture is reported as a board identity result.
Two Python tests compare all current registry aliases and negative/precedence
cases with actual shared Python detection over temporary files, and compare
streamed binary data at SHA padding/64 KiB read boundaries with hashlib. Missing
paths and directories must return a failure code, not an empty-content digest.
YOLOv5 regression protects the original dump and independent manifest verifier.

Bilingual native/shared README explains ownership, errors, policy limits and
building/linking. The complete SDK API example now supplies the real preflight
factory with explicit expected model digest and vocabulary, rather than requiring
a placeholder callback. Host commands and both API examples are verified; SDK
build remains unavailable and its missing-dependency gate stays explicit.

[Evidence directory](evidence/2026-09-28-yoloe-native-preflight/) contains logs,
commands, timestamps and final implementation hashes. This is implementation
self-check, not independent whole-branch acceptance.

Final recorded results: both documented native build modes passed all nine
YOLOE CTest cases; shared Ultralytics native tests passed 12/12. Both complete
README API examples compiled. Python regression passed 544 tests (31 YOLOE,
143 Ultralytics, 153 shared, 52 ResNet, 44 OCR, 27 checker, 5 export,
10 evaluator, 79 YOLOv5). Migration checker covered 45 samples with zero
violations, 47 declared policy skips and zero exemptions. Existing README
link/API checks and all eight new shared README links passed. The SDK dependency
gate returned the expected missing-header/library error, not SDK build success.
