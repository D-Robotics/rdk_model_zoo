# ASR C++ audio and contract core

English | [简体中文](README_cn.md)

<a id="supported-boards"></a>
## Supported boards
Canonical native integration is in progress. S100/S600 have published model identities, but this directory currently contains portable decoding, audio I/O, preprocessing and host tests. There is no board inference executable here yet. X5/S100P have no published ASR artifact. All board validation is not-run; host compilation is not SDK compatibility evidence. The original native capability remains in the [S source](../../../../../platforms/s/samples/speech/asr/runtime/cpp/) while migration continues.

<a id="dependencies"></a>
## Dependencies
The decoding/normalization core requires C++17 and the standard library. Tests use CMake >=3.18, with Clang/GCC AddressSanitizer and UndefinedBehaviorSanitizer enabled by default. The core-only test uses no OpenCV, SDK or audio library. Audio tests additionally require libsndfile and libsamplerate headers/libraries. Host evidence uses libsndfile 1.2.2 and libsamplerate 0.2.2 built in an isolated prefix; this does not certify board library versions. The UCP adapter and preflight are implemented with host API doubles; real SDK build/ABI validation and the deployment CLI remain pending. The tested libsndfile build disables external codecs; WAV PCM/float is verified, while FLAC/other codec availability depends on your build.

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
CTest executes `asr_contract`. It checks CTC/legacy decoding, negative logits, ties, nonfinite values, invalid vocabulary/IDs, source chunk geometry and normalization. It does not open a model or transcribe audio. Use the [Python workflow](../python/README.md) for the implemented inference entry; the deployment CLI and real SDK validation remain pending.

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

Expected: `asr_contract`, `asr_audio`, `asr_task`, `asr_sdk_fixture` and `asr_preflight` pass. Audio tests write temporary WAV
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
`SdkRunner::metadata()` supplies observed steps/strides/allocation sizes after its binding checks; use the preflight factory described below before SDK calls.

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

## SDK adapter and identity gate

`SdkRunner(SdkModel{path, target}, preflight)` accepts only S100/S600 and requires
an explicit preflight callback before any SDK call. Production callers use
`make_preflight(expected_model_sha256, vocabulary_path)`: it reads the local
identity through the shared platform registry rules, rejects mismatches
(including S100 with an S100P board alias), checks the model digest and pins the
3503-token vocabulary digest. A locally recorded model digest identifies bytes;
when the manifest lacks a publisher digest, it does not prove publisher origin.

After preflight, the adapter requires exactly one named model, one FLOAT32
unquantized input `[1,30000]` and one FLOAT32 unquantized output `[1,T,3503]`.
T must be positive. Positive aligned allocation sizes and nonoverlapping,
float-aligned byte strides are checked before allocation; dynamic/missing
stride descriptors and integer outputs are rejected. Do not reinterpret raw
integer memory as float. Observed steps, byte strides, allocation sizes and
model name are available through `metadata()` for reports.

`infer(prepared)` validates the finite 30000-element input, zeroes allocation
padding, copies through the observed strides, cleans the input cache, calls the
shared synchronous UCP inference helper and invalidates the output cache.
The returned vector owns compact raw logits copied from observed output strides;
subsequent inference cannot overwrite it. Scheduling uses the shared UCP default
`HB_UCP_BPU_CORE_ANY`; this adapter does not expose a priority/core override.
No activation, CTC or audio I/O happens inside this transport. A task Runner
can capture a living SdkRunner and delegate to `infer`; keep it alive for all
calls and do not concurrently reuse a single SDK instance.

Model and tensor owners unwind partially initialized state. The shared tensor
owner now retains a nonnull allocation even when the SDK call fails, and rejects
success with a null address. Task creation/submit/wait/release and cache-flush
errors propagate to the caller. Destructors attempt resource release without
throwing; release failure itself cannot establish that the SDK freed a resource.

## Library build and verification boundaries

From the repository root, with the audio development libraries already prepared:

```bash
cmake -S samples/speech/asr/runtime/cpp -B /tmp/rdk-asr-library -DASR_BUILD_TESTS=ON -DASR_AUDIO_TESTS=ON -DCMAKE_PREFIX_PATH="${ASR_AUDIO_PREFIX:-}"
cmake --build /tmp/rdk-asr-library
ctest --test-dir /tmp/rdk-asr-library --output-on-failure
```

The default `ASR_BUILD_SDK=OFF` builds `asr_frontend` and `asr_preflight` static
libraries without vendor headers. `ASR_BUILD_TESTS` defaults OFF. Tests always
exercise the adapter against an explicitly named API double, not the vendor ABI.
To build `asr_sdk`, configure `-DASR_BUILD_SDK=ON` in an environment containing
`dnn/hb_dnn.h`, `hb_ucp.h`, `hb_ucp_sys.h`, `libdnn` and `libhbucp`. CMake requires
all of them and never silently substitutes the host double. Custom installations
can set `ASR_DNN_INCLUDE`, `ASR_UCP_INCLUDE`, `ASR_UCP_SYS_INCLUDE`,
`ASR_DNN_LIBRARY`, `ASR_UCP_LIBRARY`; use the proper target compiler/sysroot for
cross compilation. Both X5 and UCP headers visible together are rejected.

No supported minimum SDK/BSP version has yet been established. On this host,
SDK-enabled configuration was checked to fail because vendor headers are absent.
There is still no deployment CLI, automatic vocabulary JSON loader or complete
native transcript report. Use the implemented Python workflow while these are
integrated. [SDK implementation evidence](../../../../../docs/releases/unified-migration/evidence/2026-09-28-b10-asr-sdk/)
records five host tests, identity failures, strided tensors and partial-allocation
failure reproduction. Board/model inference remains not-run.
