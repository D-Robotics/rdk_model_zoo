# HIMLoco source inventory and offline Python core

Status: partial author implementation, not whole-sample or independent acceptance.
Base: `882db9ea`; source X5: `ac115717197920355fc390bb04299b20e6436864`.
No SDK, board, robot control or quantization operation was executed.

## Source scope and remaining migration

All 52 tracked source files were byte-compared against the pinned X5 revision.
The set includes 21 binary observation files plus their manifest. The active X5
manifest publishes one fused Go2 BIN with SHA-256
`7ce46ca2628f8bc236da0e8564180a1de92847bddf1ec00717ce7aa93e8c3e6a`.
The policy consumes obs_history float32 [1,270] and emits actions float32 [1,12].
It is not the older separate encoder/policy deployment boundary.

| Source capability | Unified destination / status |
|---|---|
| Python pre/forward/post/predict | `runtime/python/policy.py` implemented and host tested |
| Python SDK metadata/loading/scheduling | Still to migrate using shared binding/runner helpers |
| File CLI, warmup, action dumps and provenance | Still to migrate outside policy core |
| C++ reusable model and offline application | Still to migrate with native lifetime and stride checks |
| Six-frame observation fixtures and digest manifest | Source retained; checked here, unified test_data packaging still pending |
| Export/calibration/Mapper workflow and README | Source scheme trusted; preserve and reorganize documentation, no real recipe verification required by latest user instruction |
| JIT/ONNX comparison, action-dump comparison, runtime input preparation and metrics | Still to migrate; separate ordinary evaluator code tests from prohibited recipe reruns |
| Root/model/native/conversion/evaluator README pairs | Full content migration still pending |

## Core correction

HimLocoTask keeps pre_process/forward/post_process/predict as the public numerical
interface. It receives one already bound runner. File I/O, asset acquisition,
SDK setup, CLI, warmup loops and evaluation do not belong in the task.
Preprocessing preserves the source 270-value flattening and float32 conversion,
while rejecting complex/string/object/bool data. Forward preserves exact raw
float32 actions without clipping, scaling, activation or dequantization. Strict
physical shape/dtype checks replace silent coercion of incompatible model output.

The source `_last_latency_ms` instance field makes delayed postprocessing refer to
the most recent call rather than the supplied output. RawOutputs now carries that
call's latency explicitly. Preprocessed inputs, raw outputs and final actions have
independent storage; source ascontiguousarray could reuse caller/SDK memory.
There is no implicit observation history or robot action application.

## Evidence

Six core tests pass: explicit/predict equality, unchanged action values, history
order/input ownership, per-call timing, invalid inputs and strict output contracts.
[Tests](evidence/2026-09-28-b10-himloco-core/tests.log),
[initial missing-module red](evidence/2026-09-28-b10-himloco-core/red.log).

All 21 real source observation files match their manifest digests, and produce
identical preprocessed tensors in source and unified implementations. The action
comparison uses a deterministic synthetic runner output, not actual model inference.
[Source comparison](evidence/2026-09-28-b10-himloco-core/source-comparison.json),
[reproducible script](evidence/2026-09-28-b10-himloco-core/compare_source.py).

Both runtime README examples were extracted and executed with host Python/NumPy.
[Example results](evidence/2026-09-28-b10-himloco-core/readme-examples.json).
They explicitly label fixture outputs and current partial implementation status.
No claim of policy accuracy, robot stability, SDK compatibility or board latency
follows from these tests. Whole-sample migration and all H0–H9 work continue.
