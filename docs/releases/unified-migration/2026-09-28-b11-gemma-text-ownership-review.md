# B11 Gemma Text initialization ownership — author record

Base: `2c4da3fd`. This is implementation evidence, not independent acceptance.
B11 and H0–H9 remain open; board execution is not-run. Quantization recipes are
trusted source material and were not executed or revalidated.

## Problem and change

TextEngine previously owned raw tensor vectors and a raw packed-model handle.
An exception while loading or allocating escaped without releasing all acquired
resources. Its destructor assumed every KV alias had already been installed.
The original source/header at the base commit fails the new host constructor
failure test; the persisted baseline record contains the assertion and exit code.

`gemma4_model_io.hpp` now defines move-only tensor ownership separately from
inference. Each borrowed KV input is tracked explicitly, so partial initialization
and moves neither leak completed allocations nor free the cache owner's memory.
TextEngine clears both subgraphs before releasing the packed model on normal
teardown and constructor failure. Null handles, incompatible 35-input/31-output
counts and absent/nonpositive sequence dimensions fail before indexed access.

Ruling: preserve the fixed source Text export's 35/31 tensor layout rather than
accept arbitrary counts that the implementation cannot address. This is not a
claim of a complete tensor binding contract. A differently exported model will
need an explicit adapter; it must not silently enter this fixed-layout engine.

## Evidence

[Evidence directory](evidence/2026-09-28-b11-gemma-text-ownership/) contains baseline
failure, hashed code bindings, native output, Python output and contract output.

- 301 successive SDK acquisition/property/allocation failure points, normal
  teardown and six invalid descriptor cases pass. Allocations returning both a
  resource and an error are included. SDK and embedding loading are independent
  host doubles; any inference call aborts the test.
- ModelIo checks partial construction, moves/assignment, repeated clearing and
  borrowed cache survival. Eleven native CTests pass with ASan/UBSan.
- Fourteen Sample unittest tests pass. Migration contract remains 50 samples,
  zero violations, 51 policy skips and zero exemptions.
- C++ English/Chinese README explains ownership, errors, fixed layout and the
  limits of these host tests. Quantization instructions remain unchanged.

Commands: `python -m unittest discover -s samples/llm/gemma4-e2b/tests -v`,
`ctest --test-dir ../.coordination/gemma-vision-sanitized --output-on-failure`,
and `python tools/sample_contract/check.py --scope migration --format text`.
Python used `../rdk_model_zoo/.venv/bin/python`; CTest used the existing bundled
CMake runtime. The sanitizer build uses the existing host OpenCV installation.

## Remaining work

Full Text dtype/stride/capacity checks, session-stage organization, explicit model
preparation, MiniCPM and the remaining completion-plan tasks continue. The tests
are not proof of vendor SDK ABI, generation quality, SDK release-error behavior,
or injected `std::bad_alloc` behavior. Whole-branch independent review is pending.
