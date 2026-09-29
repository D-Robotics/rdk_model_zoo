# HIMLoco native policy stages

Base: `78e7894f`. Author implementation record; whole-sample review remains open.

The unified C++ inference file now contains only the four policy stages and
validation helpers. An injected Runner owns transport, SDK timing and resource
lifetime. Stage containers own their vectors; per-call timing travels with raw
outputs. Input and output dimensions/finiteness are checked, actions stay
unscaled, and runner exceptions propagate without cached successful results.
SDK loading, filesystem, logging and robot control are outside the policy file.

Ruling: align the unified native stage interface with the Python task using
value-returning pre_process/forward/post_process/predict. The archived C++ source
API remains untouched. Native SDK and CLI implementation is still required;
this isolated core is not declared a complete migration or board runtime.

Verification: a new native compilation test first failed because the core did
not exist, then passed. The host executable covers stage ownership, interleaved
timing, signed-zero preservation, size/finiteness/timing rejection, runner
exception propagation and byte-preserving preprocessing of 21 real source inputs.
An additional AddressSanitizer/UndefinedBehaviorSanitizer build and run passed.
Full HIMLoco suite: 16 tests pass. README contract: zero violations, one ordinary
CLI policy skip, zero exemptions. Both C++ guides and root navigation describe
actual implemented functionality and pending SDK/CLI work.

[Native command records](evidence/2026-09-28-b10-himloco-cpp-core/host-native.json)
and [contract report](evidence/2026-09-28-b10-himloco-cpp-core/contract.json).
No model inference, board connection, export, calibration or compilation recipe
was executed. Quantization recipe reruns remain excluded by user instruction.
