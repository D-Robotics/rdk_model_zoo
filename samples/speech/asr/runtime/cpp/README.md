# ASR C++ audio and contract core

English | [简体中文](README_cn.md)

<a id="supported-boards"></a>
## Supported boards
Canonical native integration is in progress. S100/S600 have published model identities, but this directory currently contains portable decoding, audio I/O, preprocessing and host tests. There is no board inference executable here yet. X5/S100P have no published ASR artifact. All board validation is not-run; host compilation is not SDK compatibility evidence. The original native capability remains in the [S source](../../../../../platforms/s/samples/speech/asr/runtime/cpp/) while migration continues.

<a id="dependencies"></a>
## Dependencies
The decoding/normalization core requires C++17 and the standard library. Tests use CMake >=3.18, with Clang/GCC AddressSanitizer and UndefinedBehaviorSanitizer enabled by default. The core-only test uses no OpenCV, SDK or audio library. Audio tests additionally require libsndfile and libsamplerate headers/libraries. Host evidence uses libsndfile 1.2.2 and libsamplerate 0.2.2 built in an isolated prefix; this does not certify board library versions. UCP SDK integration remains pending. The tested libsndfile build disables external codecs; WAV PCM/float is verified, while FLAC/other codec availability depends on your build.

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
CTest executes `asr_contract`. It checks CTC/legacy decoding, negative logits, ties, nonfinite values, invalid vocabulary/IDs, source chunk geometry and normalization. It does not open a model or transcribe audio. Use the [Python workflow](../python/README.md) for the implemented inference entry; native SDK/CLI work remains pending.

<a id="parameters"></a>
## Parameters
There is no inference CLI yet. The test has no runtime arguments. `ASR_SANITIZERS` is a CMake option, default `ON`. `ASR_AUDIO_TESTS` defaults to `OFF`; enabling it builds the real-library audio tests. `normalize_probe.cc` and `audio_probe.cc` are test-only output adapters, not deployment launchers.

<a id="interface-lifecycle"></a>
## Interface and resource lifecycle
`inc/contract.h` exposes `DecodeMode`, `decode_ids`, `decode_logits`, `source_chunk_size`, `PreparedChunk` and `normalize_and_pad` in namespace `asr`. All functions operate on caller-owned inputs and return value-owned results; there is no global decoder state, SDK handle or task allocation. Vocabulary strings must be unique and begin with `<pad>`. CTC collapses adjacent IDs before removing blank 0; legacy only removes blank. Each call is an independent sequence. Equal logits select the lowest index. Nonfinite logits and invalid shapes are rejected.

`normalize_and_pad` accepts already-mono, already-resampled finite samples. It uses variance plus 1e-5, then truncates/pads to 30000 floats and reports the valid length. Double accumulation may differ slightly from Python float32 accumulation. This helper does not read files or resample; agreement on its output does not establish native audio-path equivalence.

<a id="results-interpretation"></a>
## Interpreting results
CTest exit 0 means only the host contract assertions passed. Decode output concatenates vocabulary strings verbatim, preserving `|` and nonblank special tokens. No text cleanup, confidence, language model or cross-chunk stitching is performed. The historical native normalizer and decoder were different; their migration must use this explicit contract rather than silently preserving the old CTC bug.

## Audio I/O and preprocessing

`AudioReader(path)` owns a libsndfile handle and is noncopyable. It opens and
validates a nonempty file; failed construction and destruction close the handle.
`next(AudioChunk&)` returns an owned interleaved float buffer and source rate,
channels, frame offset and chunk index. It reads at most
`ceil(30000 × source_rate / 16000)` frames. Clean EOF returns false and clears
the destination; read errors throw. It does not perform numeric preprocessing.

`prepare_chunk(AudioChunk)` is independent of file I/O. It validates finite
interleaved data, averages channels, calls `SRC_SINC_BEST_QUALITY` when needed,
then normalizes and pads. Output is 30000 floats plus `valid_samples`.
Empty/malformed/overlong chunks and durations shorter than one target sample
are rejected. Resampling output length is taken from `output_frames_gen`.
Mean accumulation uses double precision to avoid multichannel float overflow.

The source uses independent windows and the libsamplerate simple API. We retain
that window contract; no resampler history or overlap crosses chunks. This is
not a continuous streaming resampler: the [upstream API guide](https://libsndfile.github.io/libsamplerate/api_simple.html)
requires a stateful API for continuous chunked audio. Python retains Fourier
resampling; results can differ at boundaries. Both use variance plus 1e-5 before
padding, correcting the source native standard-deviation-only normalization.

After explicitly installing audio development libraries, execute from the
repository root. `ASR_AUDIO_PREFIX` may point to their installation prefix;
omit it for the normal system search paths. No dependencies are downloaded by CMake.

```bash
cmake -S samples/speech/asr/runtime/cpp/tests -B /tmp/rdk-asr-audio -DASR_AUDIO_TESTS=ON -DASR_SANITIZERS=ON -DCMAKE_PREFIX_PATH="${ASR_AUDIO_PREFIX:-}"
cmake --build /tmp/rdk-asr-audio
ctest --test-dir /tmp/rdk-asr-audio --output-on-failure
```

Expected: `asr_contract`, `asr_audio` and `asr_task` pass. Audio tests write temporary WAV
fixtures at 8/16/44.1 kHz, check independent-window geometry, final padding,
constant input, ownership and malformed inputs. Assertions remain enabled in
Release builds. [Additional seven-chunk source comparison evidence](../../../../../docs/releases/unified-migration/evidence/2026-09-28-b10-asr-native-audio/)
uses actual libraries, bundled audio and generated inputs; the source resampler
is preserved and only its old normalization/padding is replaced for comparison.
It does not validate a real model or SDK.

## Three-stage task API

`asr.h` contains only task construction/configuration and `pre_process`,
`forward`, `post_process`, `predict`. Construction takes a `Runner`, observed
positive logit-step count, ordered 3503-token vocabulary and decoder mode
(default CTC). The caller must verify model identity, tensor metadata and the
vocabulary hash before constructing it; the task itself does not open files.
The SDK adapter that will supply those facts is still pending.

Preprocessing returns owned fixed-size waveform and valid length. Forward
validates that input and invokes the transport exactly once, returning owned
raw FLOAT32 logits. Postprocessing checks `[1,steps,3503]`, rejects nonfinite
values and decodes; no activation or file I/O occurs. Predict composes the
three. Native output retains the source FLOAT32 contract; Python's integer
SCALE support does not imply native integer support. The callable transport's
lifetime must cover the task, and concurrent calls require a thread-safe
transport; no thread-safety guarantee is made for an eventual SDK instance.

This complete host example uses synthetic input, tokens and transport; `AA`
is a fixed fixture result, not recognized speech. Compile with the include
path `runtime/cpp/inc`, `runtime/cpp/src/frontend.cc` and libsamplerate:

```cpp
#include "asr.h"
#include <iostream>
int main() {
  std::vector<std::string> vocabulary{"<pad>"};
  for (size_t i = 1; i < 3503; ++i)
    vocabulary.push_back("token" + std::to_string(i));
  vocabulary[5] = "A";
  asr::Runner fixture = [](const std::vector<float>&) {
    std::vector<float> logits(4 * 3503, 0.f);
    logits[5] = logits[3503 + 5] = logits[3 * 3503 + 5] = 1.f;
    return logits;
  };
  asr::ASR task(fixture, 4, vocabulary);
  asr::AudioChunk audio{{0.1f, 0.2f, 0.3f}, 16000, 1, 0, 0};
  auto prepared = task.pre_process(audio);
  auto raw = task.forward(prepared);
  std::cout << task.post_process(raw) << '\n';
}
```
