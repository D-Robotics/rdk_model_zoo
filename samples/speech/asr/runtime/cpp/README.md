# ASR native runtime

English | [简体中文](README_cn.md)

<a id="supported-boards"></a>
## Supported boards
The native workflow implements S100 and S600 selection, fixed-vocabulary ASR and complete-file chunk processing. X5/S100P have no ASR publication and are rejected. The SDK adapter is implemented and tested with API doubles; real vendor SDK compilation/ABI, model inference and board tests remain **not-run**. Host success is not a BSP certification. The original native source (historical `../../../../../platforms/s/samples/speech/asr/runtime/cpp/` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md) remains available for comparison.

<a id="dependencies"></a>
## Dependencies
C++17, CMake >=3.18, libsndfile and libsamplerate headers/libraries; the CLI additionally uses `nlohmann/json.hpp`, as did the source sample. Host evidence uses libsndfile 1.2.2, libsamplerate 0.2.2 and nlohmann JSON 3.11.3. The tested libsndfile disables external codecs: WAV PCM/float is verified; other format availability depends on the installed library. No OpenCV or gflags is required.

Real inference also requires the matching UCP SDK: `dnn/hb_dnn.h`, `hb_ucp.h`, `hb_ucp_sys.h`, `libdnn`, `libhbucp`. No minimum BSP version has been established. The launcher requires Python 3.10+, NumPy and PyYAML for shared selection utilities, but not Python audio/scipy or `hbm_runtime`. Dependencies and model preparation are explicit; no automatic download/install occurs.

<a id="build"></a>
## Build
On the matching board, `run.sh --build` verifies identity/model/input before configuring a Release build with `ASR_BUILD_SDK=ON`, `ASR_BUILD_CLI=ON`, tests OFF, then builds and runs `asr_demo`. It retains build logs. Model loading still checks observed metadata. An existing binary can be selected with `--binary`; it must be executable.

For library users, SDK and CLI builds default OFF. `asr_frontend` and `asr_preflight` build without vendor headers; `asr_sdk` requires SDK ON, and `asr_demo` requires both SDK and CLI ON. A CLI request without SDK is rejected. Missing vendor headers cause configuration failure, never fixture substitution. Cross compilation needs the appropriate compiler/sysroot. Custom installations may set `ASR_DNN_INCLUDE`, `ASR_UCP_INCLUDE`, `ASR_UCP_SYS_INCLUDE`, `ASR_DNN_LIBRARY`, `ASR_UCP_LIBRARY`, `ASR_JSON_INCLUDE`, or an installation prefix through CMake. Simultaneously visible X5/UCP headers are rejected.

Host-only validation, with dependencies installed and optional `ASR_AUDIO_PREFIX`/`ASR_JSON_INCLUDE` pointing to them:
```bash
# cwd: repository root; audio and JSON development dependencies already installed
cmake -S samples/speech/asr/runtime/cpp -B /tmp/rdk-asr-library -DASR_BUILD_TESTS=ON -DASR_AUDIO_TESTS=ON -DASR_CLI_TESTS=ON -DCMAKE_PREFIX_PATH="${ASR_AUDIO_PREFIX:-}" -DASR_JSON_INCLUDE="${ASR_JSON_INCLUDE:-/usr/include}"
cmake --build /tmp/rdk-asr-library
ctest --test-dir /tmp/rdk-asr-library --output-on-failure
```
Expected: seven tests pass. They use real audio libraries and an explicitly named `asr_cli_fixture`; `ASR_BUILD_TESTS`, `ASR_AUDIO_TESTS`, `ASR_CLI_TESTS` default OFF. `ASR_SANITIZERS` defaults ON for tests, with assertions retained in Release. The fixture emits `execution_backend: host-fixture`, which the public launcher refuses to accept as native SDK success. Host metadata/transport fixtures do not authenticate actual model properties.

<a id="run"></a>
## Run
From the repository root, inspect without a board or model:
```bash
python3 samples/speech/asr/runtime/cpp/launcher.py --list-models
python3 samples/speech/asr/runtime/cpp/launcher.py --target s100 --dry-run
python3 samples/speech/asr/runtime/cpp/launcher.py --target s600 --dry-run
```
On S100, after explicitly preparing dependencies:
```sh
# cwd: repository root, on the matching S100 board with its SDK installed
bash samples/speech/asr/model/download.sh --target s100
bash samples/speech/asr/runtime/cpp/run.sh --target s100 --build
# Later run: reuse the built binary and choose a new output directory
bash samples/speech/asr/runtime/cpp/run.sh --target s100 --decode-mode legacy --output-dir outputs/asr_cpp_legacy
```
For S600, download for s600 and replace the target; do not reuse S100's file. Default target auto reads the actual local identity. Use a new output directory for every launch. Shell `run.sh` resolves its own location and honors `PYTHON`; default input/vocabulary paths are sample-relative, while user relative paths/output paths use the caller's directory. Native binary defaults below assume repository-root cwd.

<a id="parameters"></a>
## Parameters
| Launcher option | Default | Meaning / 含义 |
| --- | --- | --- |
| `--target` | `auto` | Local identity; explicit s100/s600 for host dry-run / 本机识别，主机预检须指定 |
| `--asset-id` | `None` | Exact publication identity / 精确发布身份 |
| `--model-path` | `None` | External path requires exact asset-id / 外部路径须同时指定身份 |
| `--audio-file` | `samples/speech/asr/test_data/chi_sound.wav` | Sample-relative default / 默认相对 sample 定位 |
| `--vocab-file` | `samples/speech/asr/test_data/vocab.json` | Fixed source vocabulary / 固定源词表 |
| `--decode-mode` | `ctc` | ctc or legacy / CTC 或旧解码 |
| `--output-dir` | `outputs/asr_cpp` | New launch directory / 新启动记录目录 |
| `--build` | `false` | Explicit native build; conflicts with binary / 显式构建，与 binary 互斥 |
| `--binary` | `None` | Otherwise runtime/cpp/build/TARGET/asr_demo / 默认使用目标构建目录 |
| `--list-models` | `false` | List without SDK / 无 SDK 列表 |
| `--dry-run` | `false` | Prepare without building/inference / 只准备，不构建或推理 |
List/dry-run are mutually exclusive. No priority/core, sample-rate or window-size overrides are exposed: UCP uses `HB_UCP_BPU_CORE_ANY`, and published models use 30000 points at 16000 Hz. External model paths require the exact published asset ID. The launcher computes the actual digest after manifest verification; absent publisher SHA remains unknown provenance.

The native `asr_demo` accepts the following separate interface. Required means no default; unknown, duplicate and missing-value options fail. The launcher supplies all identities and absolute paths, so normal users do not need to assemble these flags.
| Binary option | Default |
| --- | --- |
| `--target` | Required: s100 or s600 |
| `--asset-id` | Required: s:asr:TARGET/asr.hbm |
| `--model-path` | Required |
| `--model-sha256` | Required: 64 hexadecimal digits |
| `--audio-file` | samples/speech/asr/test_data/chi_sound.wav |
| `--vocab-file` | samples/speech/asr/test_data/vocab.json |
| `--output-dir` | outputs/asr_cpp/result |
| `--decode-mode` | ctc |
| `--help` | false |

<a id="interface-lifecycle"></a>
## Interface and resource lifecycle
`AudioReader` exclusively owns its libsndfile handle. `next(AudioChunk&)` returns owned interleaved float data, source rate/channels, frame offset and index; clean EOF clears output and returns false, errors throw. Reads use `ceil(30000 × source_rate / 16000)` frames. It does not normalize or infer.

`ASR` contains construction/configuration and four stage methods. Construction takes a Runner, observed positive output-step count, ordered 3503-token vocabulary and decoder mode (default CTC).

- `pre_process(AudioChunk)` validates finite data, averages channels, independently resamples with `SRC_SINC_BEST_QUALITY`, normalizes with variance plus 1e-5, then pads to 30000. It returns owned floats and valid sample count. Empty, malformed, overlong and sub-target-sample chunks are rejected.
- `forward(PreparedChunk)` validates fixed finite input and invokes the runner once, returning owned raw logits.
- `post_process(raw)` checks `[1,T,3503]`, finite values and decodes. CTC collapses adjacent IDs before removing blank 0; legacy only removes blank. Nonblank token text, including `|`, is retained.
- `predict(AudioChunk)` composes the stages. Audio files, vocabulary parsing and report saving remain outside the task.

`SdkRunner` requires `make_preflight(model_digest, vocabulary_path)` before any SDK call. Preflight verifies exact local target (including S100P aliases), model bytes and the fixed vocabulary SHA. `load_vocabulary` parses the same hash-checked bytes into 3503 ordered tokens. The adapter binds one named model, unquantized FLOAT32 `[1,30000]` input and `[1,T,3503]` output. Positive allocation, nonoverlapping float-aligned strides and T are validated before allocation. Native integer/dynamic descriptors are rejected; Python's SCALE support does not imply native support.

Input padding is cleared and floats copied through observed strides. Cache clean, synchronous UCP task execution and output invalidation are checked; outputs are copied to owned compact vectors. Model/tensor owners unwind partial initialization, including error returns with a nonnull allocation. Success with a null allocation is rejected. Cleanup is nonthrowing and cannot guarantee reclamation after a real SDK release error. Do not concurrently reuse an SDK instance; a Runner capturing it must not outlive it.

The following complete host API example uses synthetic vocabulary and transport. `AA` is a fixture output, not speech recognition. Compile with `inc`, `src/frontend.cc` and libsamplerate; the evidence executes this exact bilingual example.
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

<a id="results-interpretation"></a>
## Results and interpretation
Launcher exit 0 requires the child to exit 0 **and** a completed `native-sdk` report whose identities, tensor metadata, chunk geometry and combined text validate. Its new directory contains `launch-report.json`, raw `configure/build/native.stdout.log` and `.stderr.log` for processes actually run, and native results in `result/`. The launch record includes exact argv/cwd, UTC process intervals/return codes, binary/model/audio/vocabulary/report hashes and publisher-checksum status. Preparation-only output explicitly says inference was not executed.

`result/result.json` (`rdk-model-zoo/asr-native-run/v1`) records backend, target/asset, hashes, decoder/frontend/configuration, observed model/tensor metadata and `chunks`. Each chunk has index, original frame offset/count/rate, valid target samples and decoded text; `text` concatenates the full file without extra separators. Failures after native output creation write `failed.json` with completed chunks and error; the launcher retains logs and marks failure. If report storage itself fails, stderr preserves the error. Failed results are not successful transcripts. Inputs are hashed again at completion to reject changes during the run. No confidence or timestamps are inferred from logits.

Windows are independent: no language model, overlap or decoder/resampler state crosses boundaries. All output frames of the padded final window are decoded because valid output length has not been verified. Python Fourier and native sinc frontends can differ substantially, especially on short windows; this is preserved source behavior, not cross-language parity. The [upstream simple API](https://libsndfile.github.io/libsamplerate/api_simple.html) is not a continuous streaming resampler. Historical performance is in the [evaluator guide](../../evaluator/README.md); no new model latency or CER is claimed.

## Troubleshooting
Identity mismatch: use the exact supported board, never relabel S100P. Missing model: download explicitly per [model guide](../../model/README.md). SDK configure failure: provide matching vendor headers/libraries and compiler. Vocabulary mismatch: restore the bundled file, not a same-length substitute. Tensor-contract error: inspect the artifact and actual SDK descriptors; do not reinterpret integer output or fabricate strides. Existing output: choose another directory. Child success with invalid/fixture report: keep logs and treat the launch as failed.

[Native CLI evidence](../../../../../docs/releases/unified-migration/evidence/2026-09-28-b10-asr-native-cli/) separates API doubles and host fixtures from real SDK/board execution. Full-branch independent acceptance is still open.
