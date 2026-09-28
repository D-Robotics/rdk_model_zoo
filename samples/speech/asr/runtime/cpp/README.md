# ASR C++ contract core

English | [简体中文](README_cn.md)

<a id="supported-boards"></a>
## Supported boards
Canonical native integration is in progress. S100/S600 have published model identities, but this directory currently contains only portable decoding/normalization code and host tests. There is no board inference executable here yet. X5/S100P have no published ASR artifact. All board validation is not-run; host compilation is not SDK compatibility evidence. The original native capability remains in the [S source](../../../../../platforms/s/samples/speech/asr/runtime/cpp/) while migration continues.

<a id="dependencies"></a>
## Dependencies
The core requires C++17 and the standard library. Tests use CMake >=3.16, with Clang/GCC AddressSanitizer and UndefinedBehaviorSanitizer enabled by default. No OpenCV, board SDK or audio library is used in this core test. Future native integration must cover the source's libsndfile/libsamplerate audio path and UCP SDK resources; their runtime versions have not been certified.

<a id="build"></a>
## Build
```bash
# cwd: repository root; CMake and a C++17 compiler must be on PATH
cmake -S samples/speech/asr/runtime/cpp/tests -B /tmp/rdk-asr-contract -DASR_SANITIZERS=ON
cmake --build /tmp/rdk-asr-contract
ctest --test-dir /tmp/rdk-asr-contract --output-on-failure
```
This builds `test_contract`, not an ASR deployment binary. Sanitizers can be disabled with `-DASR_SANITIZERS=OFF` for unsupported host compilers, but such a run does not provide sanitizer evidence.

<a id="run"></a>
## Run
CTest executes `asr_contract`. It checks CTC/legacy decoding, negative logits, ties, nonfinite values, invalid vocabulary/IDs, source chunk geometry and normalization. It does not open a model or transcribe audio. Use the [Python workflow](../python/README.md) for the implemented inference entry; native audio/SDK/CLI work remains pending.

<a id="parameters"></a>
## Parameters
There is no inference CLI yet. The test has no runtime arguments. `ASR_SANITIZERS` is a CMake option, default `ON`. `normalize_probe.cc` is a test-only binary-file adapter for frontend comparisons, not a public audio frontend or runtime launcher.

<a id="interface-lifecycle"></a>
## Interface and resource lifecycle
`inc/contract.h` exposes `DecodeMode`, `decode_ids`, `decode_logits`, `source_chunk_size`, `PreparedChunk` and `normalize_and_pad` in namespace `asr`. All functions operate on caller-owned inputs and return value-owned results; there is no global decoder state, SDK handle or task allocation. Vocabulary strings must be unique and begin with `<pad>`. CTC collapses adjacent IDs before removing blank 0; legacy only removes blank. Each call is an independent sequence. Equal logits select the lowest index. Nonfinite logits and invalid shapes are rejected.

`normalize_and_pad` accepts already-mono, already-resampled finite samples. It uses variance plus 1e-5, then truncates/pads to 30000 floats and reports the valid length. Double accumulation may differ slightly from Python float32 accumulation. This helper does not read files or resample; agreement on its output does not establish native audio-path equivalence.

<a id="results-interpretation"></a>
## Interpreting results
CTest exit 0 means only the host contract assertions passed. Decode output concatenates vocabulary strings verbatim, preserving `|` and nonblank special tokens. No text cleanup, confidence, language model or cross-chunk stitching is performed. The historical native normalizer and decoder were different; their migration must use this explicit contract rather than silently preserving the old CTC bug.
