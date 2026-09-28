# HIMLoco native SDK adapter

Base: `3982d34e`. Author implementation; whole-sample review remains open.

Native inference is separate from the four-stage policy. The adapter owns packed
model, tensors and task handles through RAII; cleanup also applies to partial
construction and SDK failures. Metadata requires one model/input/output, exact
names, finite float outputs with no manual dequantization, four-dimensional X5
shapes, batch one and 270/12 logical elements. Allocation capacity and checked
shape products are validated before copying. Input retains the archived source's
compact alignedShape submission convention; output extraction follows physical
aligned strides. Returned actions are owned independently of SDK buffers.

Production preflight reuses shared board aliases and streaming SHA-256. Actual
X5 identity and exact published local BIN are required before SDK initialization;
there is no download or runtime bypass. A host parity test pins the digest to the
active manifest. Priority is -1 or [0,255]. Timing retains source infer-plus-wait
scope, excluding cache work and input/output extraction.

Verification: SDK and preflight tests each failed before their implementations
existed, then passed. SDK doubles cover malformed capacity/alignment, wrong dtype
and quantization, shape overflow, first/second allocation failure, padded output,
priority, infer/wait/cache failures, nonfinite output, and cleanup. Preflight uses
production hashing with only board reading replaced, covering wrong/unknown
boards (including S100P alias), suffix, missing file and wrong digest rejection.
The SDK fixture explicitly replaces preflight for its synthetic model; this is a
link-time test double, not a production override. Both fixture executables passed
ASan/UBSan, and the complete HIMLoco host suite passed 19 tests.

[Full command evidence](evidence/2026-09-28-b10-himloco-sdk/host-checks.json).
[README contract](evidence/2026-09-28-b10-himloco-sdk/contract.json): zero violations,
one ordinary CLI policy skip, zero exemptions. Root/C++ bilingual guides describe
SDK ownership, integration, timing and current limitations.

Real SDK headers/linking/execution and board tests were not run. The native CLI,
build/launcher and file/report integration remain required migration work. No
quantization/export/calibration recipe was executed; recipe reruns are excluded
by the user's instruction. No new hardware compatibility claim is made.
