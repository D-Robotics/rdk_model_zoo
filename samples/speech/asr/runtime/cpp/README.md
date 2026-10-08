# ASR native runtime

English | [简体中文](README_cn.md)

<a id="overview"></a>
## C++ inference

Transcribe fixed-length audio windows on S100/S600 using Wav2Vec2. The native program prepares the waveform tensor and decodes model logits with CTC or the legacy token mode.

<a id="directory"></a>
## Directory structure

```text
cpp/
├── inc/  # Files for inc
├── src/  # Files for src
├── tests/  # Automated tests
├── CMakeLists.txt  # Source or data file
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── launcher.py  # Python script
└── run.sh  # Run the sample
```

<a id="supported-boards"></a>
## Supported boards
The native workflow supports S100 and S600, fixed-vocabulary ASR and complete-file chunk processing. X5/S100P have no ASR publication and are rejected. The SDK adapter and launcher use the matching board SDK and model artifacts.

<a id="dependencies"></a>
## Dependencies
C++17, CMake >=3.18, libsndfile and libsamplerate headers/libraries; the CLI additionally uses `nlohmann/json.hpp`. WAV PCM/float support depends on the installed libsndfile build; other format availability depends on its enabled codecs. No OpenCV or gflags is required.

Real inference also requires the matching UCP SDK: `dnn/hb_dnn.h`, `hb_ucp.h`, `hb_ucp_sys.h`, `libdnn` and `libhbucp`. The launcher requires Python 3.10+, NumPy and PyYAML for shared selection utilities, but not Python audio/SciPy or `hbm_runtime`. Dependencies and model preparation are explicit; the launcher does not install or download them.

<a id="build"></a>
## Build
On the matching board, `run.sh --build` verifies identity/model/input before configuring a Release build with `ASR_BUILD_SDK=ON`, `ASR_BUILD_CLI=ON`, tests OFF, then builds and runs `asr_demo`. It retains build logs. Model loading still checks observed metadata. An existing binary can be selected with `--binary`; it must be executable.

For library users, SDK and CLI builds default OFF. `asr_frontend` and `asr_preflight` build without vendor headers; `asr_sdk` requires SDK ON, and `asr_demo` requires both SDK and CLI ON. A CLI request without SDK is rejected. Cross compilation needs the appropriate compiler/sysroot. Custom installations may set `ASR_DNN_INCLUDE`, `ASR_UCP_INCLUDE`, `ASR_UCP_SYS_INCLUDE`, `ASR_DNN_LIBRARY`, `ASR_UCP_LIBRARY`, `ASR_JSON_INCLUDE`, or an installation prefix through CMake. Simultaneously visible X5/UCP headers are rejected.

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
| `--vocab-file` | `samples/speech/asr/test_data/vocab.json` | Fixed vocabulary / 固定词表 |
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

`SdkRunner` requires `make_preflight(model_digest, vocabulary_path)` before any SDK call. Preflight verifies exact local target (including S100P aliases), model bytes and the fixed vocabulary SHA. `load_vocabulary` parses the same hash-checked bytes into 3503 ordered tokens. The native adapter accepts one named model with an unquantized FLOAT32 `[1,30000]` input and `[1,T,3503]` output, validating positive allocation, nonoverlapping float-aligned strides and T before allocation; integer SCALE outputs are supported by the [Python runtime](../python/README.md).

Input padding is cleared and floats copied through observed strides. Cache clean, synchronous UCP task execution and output invalidation are checked; outputs are copied to owned compact vectors. Model/tensor owners unwind partial initialization, including error returns with a nonnull allocation. Success with a null allocation is rejected. Cleanup is nonthrowing and cannot guarantee reclamation after a real SDK release error. Do not concurrently reuse an SDK instance; a Runner capturing it must not outlive it.

<a id="results-interpretation"></a>
## Results and interpretation
Launcher exit 0 requires the child to exit 0 **and** a completed `native-sdk` report whose identities, tensor metadata, chunk geometry and combined text validate. Its new directory contains `launch-report.json`, raw `configure/build/native.stdout.log` and `.stderr.log` for processes actually run, and native results in `result/`. The launch record includes exact argv/cwd, UTC process intervals/return codes, binary/model/audio/vocabulary/report hashes and publisher-checksum status. Preparation-only output explicitly says inference was not executed.

`result/result.json` (`rdk-model-zoo/asr-native-run/v1`) records backend, target/asset, hashes, decoder/frontend/configuration, observed model/tensor metadata and `chunks`. Each chunk has index, original frame offset/count/rate, valid target samples and decoded text; `text` concatenates the full file without extra separators. Failures after native output creation write `failed.json` with completed chunks and error; the launcher retains logs and marks failure. If report storage itself fails, stderr preserves the error. Inputs are hashed again at completion to detect changes during the run. Confidence and timestamp fields are not produced.

Windows are independent: language-model, overlap and decoder/resampler state does not cross boundaries. All output frames of the padded final window are decoded because model metadata supplies no valid-frame count. Python Fourier and native sinc frontends use different algorithms, especially on short windows. The [upstream simple API](https://libsndfile.github.io/libsamplerate/api_simple.html) describes the native resampler. Source-release performance figures and their measurement conditions are in the [evaluator guide](../../evaluator/README.md).

## Troubleshooting
Identity mismatch: use the exact supported board. Missing model: prepare it per [model guide](../../model/README.md). SDK configure failure: provide matching vendor headers/libraries and compiler. Vocabulary mismatch: use the bundled vocabulary. Tensor-contract error: inspect the artifact and SDK descriptors against the FLOAT32 input and stride requirements. Existing output: choose another directory.
