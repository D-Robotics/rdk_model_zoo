# Paraformer native numerical library

[简体中文](README_cn.md) · [Sample overview](../../README.md)

This directory currently provides the native CPU CIF/text kernels and three-model
application composition. The SDK adapter, prepared-manifest reader and complete
native executable are still being migrated. The tests below are host checks, not
HBM inference. The source Python frontend → C++ inference capability remains in
scope; no alternate approximation of FunASR is introduced here.

<a id="environment"></a>
## Environment

The numerical library requires CMake 3.18 or newer and a C++17 compiler. It does
not include vendor SDK, JSON, NumPy, Torch or audio-library headers. Host builds
were checked on macOS arm64 with Apple Clang; exact compiler/platform versions are
in the [evidence](../../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-native-core-review.md).
No board SDK/ABI or S100 inference result is asserted. Real frontend generation
uses the separately documented [Python environment](../python/README.md#environment).

<a id="build"></a>
## Build and run host checks

From repository root, using a new build directory:

```bash
cmake -S samples/speech/paraformer/runtime/cpp -B /tmp/rdk-paraformer-core -DCMAKE_BUILD_TYPE=Release -DPARAFORMER_BUILD_TESTS=ON -DPARAFORMER_SANITIZERS=ON
cmake --build /tmp/rdk-paraformer-core -j 2
ctest --test-dir /tmp/rdk-paraformer-core --output-on-failure
```

Success means two CTest checks pass: numerical contract and synthetic three-model
composition. Address/undefined-behavior sanitizers are enabled for Clang/GNU in
this command. Release tests retain assertions. The production artifact is the
static `paraformer_contract` library; test executables are not an inference CLI.

<a id="parameters"></a>
## Build options and current entry points

| Option | Default | Effect |
| --- | --- | --- |
| `PARAFORMER_BUILD_TESTS` | `OFF` | Build/register the two host tests |
| `PARAFORMER_SANITIZERS` | `OFF` | Enable ASan/UBSan on Clang/GNU and propagate link flags |
| `CMAKE_BUILD_TYPE` | CMake default | Documentation checks use `Release` |

There are no board CLI flags yet. In an embedding CMake project, add this directory
with `add_subdirectory` and link `paraformer_contract`; its public include path
contains `contract.h` and `pipeline.h`. Do not compile this numerical library with
fast-math: float32 operation order is part of source parity. Clang/GNU builds
explicitly disable floating-point contraction for the library.

<a id="stage-io"></a>
## Numerical and stage contracts

All vectors represent contiguous flattened batch-one tensors. Float arrays are
owned; inputs are not modified. Shapes and finite values are checked before use.

| Interface | Inputs | Result |
| --- | --- | --- |
| `cif` | weights 401, hidden 401×512, explicit valid frames 0–400 | acoustic 100×512, int32 count capped at 100 |
| `decode` | logits 100×8404, count 0–100, 8,404 unique nonempty ordered strings | text and selected IDs |
| encoder callback | features 400×560 | context 400×512 |
| predictor callback | context 400×512 | weights 401 and hidden 401×512 |
| decoder callback | context, acoustic, int32 count, zero bias 512 | logits 100×8404 |

CIF masks weights at/after valid_frames before accumulation. No-fire input returns
zero acoustic values/count; the first 100 emissions are retained. It preserves
source float64 cumulative sums rounded to float32 and one fire per time step.
It is not a general multi-fire integrator for weights above one. Native CIF has
no implicit unmasked calibration mode. The old C++ source already handled no-fire
input; the old Python exception was repaired in the unified Python path.

Greedy text decoding uses the valid token prefix and first ID on equal scores,
filters tokens enclosed by `<...>`, removes every `@@` and concatenates without
separators. Repeated tokens remain repeated; it is not CTC. Returned IDs retain
special tokens that text rendering filters out. All logit rows are required to be
finite, even if the valid count uses only a prefix.

`Pipeline(encoder, predictor, decoder, vocabulary).predict(features, valid_frames)`
is application composition, not a model forward hiding CPU CIF between SDK calls.
It requires valid_frames 1–400, executes encoder → predictor → CIF → decoder, and
bypasses decoder when CIF count is zero. Callables must return owned raw arrays
synchronously. DecoderInput references are only valid during the callback; do not
retain them. The caller must keep any captured SDK resources alive and coordinate
access if they are not thread-safe. This library does not load models, select
boards, perform file I/O, set scheduling or compile a fake SDK fallback.

<a id="results"></a>
## Results and timings

`Prediction` contains text, token IDs/count, `decoder_executed` and `Timings`.
Encoder/predictor/CIF milliseconds are numeric; decoder milliseconds are an
`std::optional<double>` and remain absent on bypass. A failed stage throws and
does not return a successful prediction. Wrong frontend geometry/frame count
fails before model callbacks; wrong encoder output fails before predictor.

Timings cover runner calls and CPU CIF, excluding frontend, file I/O, validation
outside those calls and text rendering. They are not end-to-end latency and host
callbacks do not measure BPU performance. Pipeline outputs contain no CER or
accuracy estimate. The eventual native executable must add identity checks and
result/failure files before it can be called a complete deployment entry.

<a id="integration-example"></a>
## Runnable numerical API example

After the build above, still from repository root:

```bash
cat > /tmp/rdk-paraformer-core/example.cc <<'CPP'
#include "contract.h"
#include <iostream>
int main() {
    std::vector<float> weights(401, 0.f), hidden(401 * 512, 0.f);
    weights[0] = .75f; weights[1] = .75f; weights[2] = .5f;
    for (int h = 0; h < 512; ++h) {
        hidden[h] = 2.f; hidden[512+h] = 6.f; hidden[1024+h] = 10.f;
    }
    const auto result = paraformer::cif(weights, hidden, 3);
    std::cout << result.token_count << " " << result.acoustic[0] << " "
              << result.acoustic[512] << "\n";
}
CPP
c++ -std=c++17 -fsanitize=address,undefined -Isamples/speech/paraformer/runtime/cpp/inc /tmp/rdk-paraformer-core/example.cc /tmp/rdk-paraformer-core/libparaformer_contract.a -o /tmp/rdk-paraformer-core/example
/tmp/rdk-paraformer-core/example
```

Expected output is `2 3 8`. This is a complete synthetic numerical example,
not a recognized transcript. [test_pipeline.cc](tests/test_pipeline.cc) also gives
an executable callback-composition example with explicit synthetic model outputs.
The default two WAVs can already be converted to features through the Python
`--preprocess-only` command; consuming that prepared manifest in C++ is pending.

<a id="troubleshooting"></a>
## Verification and limits

Two native tests exercise fractional integration, padding, empty input, the cap,
invalid contracts, repeated/special/BPE tokens, equal-score ties, callback order,
zero-token bypass and malformed intermediate tensors. The comparison driver
checks 27 cases byte-for-byte against both the extracted pinned C++ CIF source
and unified Python, plus 20 native/Python text comparisons. See the linked report
for the exact compiler and reproducible script. The extracted source/driver is a
host evidence fixture, not a new maintained runtime copy.

A build failure for sanitizer runtime libraries requires matching compiler/linker
support; `PARAFORMER_SANITIZERS=OFF` builds without instrumentation but does not
reproduce the sanitizer check. Shape/type/name/identity checks for actual HBM
metadata belong to the upcoming SDK adapter; these host array checks cannot
replace them. Real SDK compilation, board inference, full native CLI, OE and CER
remain not-run or pending as appropriate.
