# B11 Gemma source import and launcher separation — in progress

Source: S `380e1a2bf42041af54be6f34935e50197cfadff9`; base `9cfb1036`.
Author implementation record, not independent acceptance. Closed=no.

## Implemented scope

All 67 source files are present at `samples/llm/gemma4-e2b`; 58 remain byte-identical.
No source functionality or conversion recipe was removed. Five native applications remain:
interactive chat, HTTP server, single-shot demo, text benchmark and golden verifier.

Preparation, build and execution are now separate. Launcher help and preview require only
Python 3 standard library. Real execution checks concrete board identity through the shared
registry, including S100P refinement, before building or starting a native process.
The build target macro is explicit rather than guessed from the unrefined SoC name.
CMake no longer invokes the dependency installer. Explicit dependency preparation and Rust
builds may still access the network; no claim of a self-contained offline build is made.

Root, native runtime and third-party bilingual guides now describe the separated workflow,
launcher argument placement, failure behavior and historical evidence. Existing architecture,
parameter tables, screenshots, context management and quantization tutorials are preserved.

Ruling: require an explicit application name before native flags (`run.sh main --max_tokens=512`)
so launcher flags and native gflags have unambiguous ownership. Bare `run.sh` still selects main;
existing named-app arguments are preserved. Cost: callers using source `run.sh --max_tokens=512`
must insert `main`. Python is orchestration only; inference remains native C++.

## Verification and limits

- Seven host orchestration tests pass, including an actual local shell fixture that checks exact
  argv/environment and nonzero exit propagation. Initial test discovery failed because the
  not-yet-implemented launcher was absent; six initial cases passed after implementation.
- `bash -n` build helper passes; real shell launcher preview produces `executed=false`.
- 42 local README file links resolve. Link check does not validate remote URLs or section anchors.
- Migration contract scope now includes this in-progress sample: 50 samples, **70 violations**,
  51 policy skips, zero exemptions. All 70 are Gemma's missing standardized README section
  anchors. This is a disclosed incomplete migration, not a green global gate. No checker rule
  or scope was weakened to hide it.
- No dependency installation, model download, conversion/export/calibration/quantization,
  native SDK build or board execution was performed. Quantization recipes are trusted material
  per the user's instruction and do not need rerunning for acceptance.

## Remaining work

Complete all README layers against the existing rich source content and normalize navigation;
refactor native preprocessing, SDK transport and application responsibilities; integrate explicit
artifact preparation and target identity without silently substituting same-named HBMs.
Published Gemma manifest hash fields and source README reference hashes need consistent
provenance handling. The current launcher does not assert model hash validation or prove SDK
ABI compatibility. MiniCPM, B11 independent acceptance and the full H0–H9 goal remain open.

Evidence: [host tests](evidence/2026-09-28-b11-gemma-launcher/host-tests.json),
[source bytes and local links](evidence/2026-09-28-b11-gemma-launcher/source-and-links.json),
[contract result including incomplete Gemma sections](evidence/2026-09-28-b11-gemma-launcher/migration-contract.json).
