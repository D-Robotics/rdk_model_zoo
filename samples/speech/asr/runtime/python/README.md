# ASR Python runtime

English | [简体中文](README_cn.md)

<a id="environment"></a>
## Environment
Python 3.10+, NumPy, PyYAML, SciPy and SoundFile (with libsndfile). S100/S600 inference additionally requires the matching board-image `hbm_runtime`; no minimum BSP version has been established by this migration. Host contract tests do not certify a BSP. [Prepare the model](../../model/README.md) explicitly. No dependency installation or model download happens during inference.

<a id="usage"></a>
## Usage
Run from the repository root. The default command detects the board, selects its published model and processes the bundled recording. Use a fresh output directory for every run.
```sh
# Repository root; matching board and explicitly downloaded model required.
python3 samples/speech/asr/runtime/python/main.py
python3 samples/speech/asr/runtime/python/main.py --target s600 --decode-mode legacy --output-dir outputs/asr-legacy
```
Success is exit 0 and `status: completed` in `result.json`. Help, `--list-models`, and explicit-target `--dry-run` work without the board SDK; dry-run is not model validation.

<a id="parameters"></a>
## Parameters
| Parameter | Type | Default | Meaning / 含义 |
| --- | --- | --- | --- |
| `--target` | str | `auto` | Detect board; host dry-run requires s100/s600 / 自动识别，主机预检须指定 |
| `--asset-id` | str | `None` | Exact published identity / 精确制品身份 |
| `--model-path` | str | `None` | External path requires asset-id / 外部路径必须同时指定身份 |
| `--audio-file` | Path | `samples/speech/asr/test_data/chi_sound.wav` | Audio input / 音频输入 |
| `--vocab-file` | Path | `samples/speech/asr/test_data/vocab.json` | Hash-pinned vocabulary / 固定哈希词表 |
| `--audio-maxlen` | int | `30000` | Fixed compiled length / 编译固定长度 |
| `--new-rate` | int | `16000` | Fixed sample rate / 固定采样率 |
| `--decode-mode` | str | `ctc` | ctc or legacy / CTC 或源实现兼容模式 |
| `--priority` | int | `0` | Scheduling priority 0–255 / 调度优先级 |
| `--bpu-cores` | int list | `[0]` | Nonnegative core IDs / 非负核心编号 |
| `--output-dir` | Path | `outputs/asr` | Must be new / 必须为新目录 |
| `--list-models` | bool | `false` | List publications without SDK / 无 SDK 列举 |
| `--dry-run` | bool | `false` | Resolve without inference / 只解析不推理 |
The audio/vocabulary defaults resolve relative to this sample, independent of the caller's directory. Relative user paths and output paths resolve from the caller's directory. List and dry-run are mutually exclusive. Fixed frontend settings reject overrides inconsistent with the compiled model; changing these flags cannot resize a model.

<a id="results"></a>
## Results
`result.json` records target, asset ID, locally observed model/audio/vocabulary SHA-256, publisher digest if available, observed tensor metadata, frontend/config and decoder mode. Each `chunks` item includes index, source-frame offset/count/rate, valid resampled sample count and text. `text` concatenates chunk text without inserted separators. Scores and timestamps are not produced. A failure after output-directory creation writes `failed.json` with completed chunks and an error; it is not a successful transcript. Earlier failures only print an error. If storage also prevents writing the failure report, stderr preserves both errors and the command still returns 2. Errors return 2.

<a id="integration-example"></a>
## Integration example
From the repository root on S100, after the explicit model download. The bundled audio and vocabulary are included. Use s600 in the selection for S600. This example performs real inference when used on a board; host verification substitutes only the SDK transport.
```python
from samples.speech.asr.runtime.python.model_binding import resolve_selection, SAMPLE_DIR
from samples.speech.asr.runtime.python.model_runner import RuntimeModelRunner
from samples.speech.asr.runtime.python.vocabulary import load_vocabulary
from samples.speech.asr.runtime.python.audio_io import read_chunks
from samples.speech.asr.runtime.python.asr import ASR

selection = resolve_selection("s100")
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = ASR(runner, binding, load_vocabulary(SAMPLE_DIR / "test_data/vocab.json"))
texts = []
for chunk in read_chunks(SAMPLE_DIR / "test_data/chi_sound.wav", task.config):
    prepared = task.pre_process(chunk.waveform, chunk.sample_rate)
    raw = task.forward({binding.input_name: prepared.tensor})
    texts.append(task.post_process(raw))
print("".join(texts))
```

<a id="stage-io"></a>
## Three-stage I/O
- `pre_process(waveform, sample_rate)` accepts finite float waveform `[frames]` or `[frames,channels]`, bounded by `ceil(30000 × source_rate / 16000)`. It averages channels, uses SciPy Fourier resampling, normalizes by `sqrt(var + 1e-5)` before zero padding, and returns `PreparedChunk.tensor` as owned float32 `[1,30000]` plus geometry.
- `forward({input_name: tensor})` calls the runner once. The runner validates names/shapes/dtypes and returns independently owned raw output. No decoder or activation runs here.
- `post_process(raw)` checks observed `[1,T,3503]` metadata, converts declared integer SCALE output through the shared quantization helper when necessary, takes argmax and returns text. FLOAT32 output is used directly; no softmax is required for argmax.
- `predict(waveform, sample_rate)` composes the three stages for one chunk. File reading, vocabulary loading and saving belong to the caller, not ASR.

CTC collapses adjacent equal IDs before removing blank 0; legacy only removes blank. Blank separates repeated tokens: `[5,5,0,5]` becomes `AA` under CTC and `AAA` under legacy. All nonblank vocabulary strings, including `|` and special tokens, are retained literally. No state or duplicate suppression crosses chunk boundaries. Final padding still produces a full logit sequence; the model has no verified valid-output-length contract, so all frames are decoded. Independent windows can split words; this is not overlap-aware streaming. Python Fourier and the historical C++ sinc resampler are different algorithms.

<a id="troubleshooting"></a>
## Troubleshooting
| Error | Action |
| --- | --- |
| `Host dry-run requires an explicit target` | Use `--target s100` or `s600`. |
| `ASR is published only for s100 and s600` | Do not substitute an S100 artifact for S100P/X5. |
| `An external model path requires the exact --asset-id` | Supply the identity from the model guide. |
| `Output directory must be new` | Choose another output directory; retain earlier evidence. |
| `ASR input must be float32 [1,30000] for the fixed frontend` | Verify the artifact and actual SDK metadata, not just its filename. |
| `Audio file changed during streaming` | Keep the input immutable and rerun to a new directory. |

[Host evidence](../../../../../docs/releases/unified-migration/evidence/2026-09-28-b10-asr-core/) covers frontend and fixture transport. Board inference remains not-run.
