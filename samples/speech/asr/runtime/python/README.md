English | [简体中文](README_cn.md)

# ASR Python runtime

<a id="overview"></a>
## Python inference

Transcribe audio on S100/S600 in independent fixed-length windows. `ASR.from_model` loads the selected model; `predict` prepares one waveform chunk, runs inference and decodes its text.

<a id="directory"></a>
## Directory structure

```text
python/
├── asr.py  # Model stages and prediction
├── audio_io.py  # Audio file reading
├── cli.py  # Arguments, model selection and result presentation
├── decoding.py  # Token decoding
├── frontend.py  # Audio feature preparation
├── main.py  # Command-line entry: construct the model and call predict
├── model_binding.py  # Model selection and physical tensor contracts
├── run.sh  # Locate the Python entry and forward arguments
└── vocabulary.py  # Vocabulary loading and validation
```

<a id="environment"></a>
## Environment
Python 3.10+, NumPy, PyYAML, SciPy and SoundFile (with libsndfile). S100/S600 inference requires the matching board image and its `hbm_runtime`. [Prepare the model](../../model/README.md) before running inference. The runtime does not install dependencies or download models.

From the repository root, install the Python dependencies; use the board image’s version of `hbm_runtime`.

```bash
python3 -m pip install numpy PyYAML scipy soundfile
```

<a id="usage"></a>
## Usage
Run from the repository root. The default command detects the board, selects its published model and processes the bundled recording. Use a fresh output directory for every run.
```sh
# Repository root; matching board and explicitly downloaded model required.
python3 samples/speech/asr/runtime/python/main.py
python3 samples/speech/asr/runtime/python/main.py --target s600 --decode-mode legacy --output-dir outputs/asr-legacy
```
The command returns 0 and records `status: completed` in `result.json` when processing succeeds. Help, `--list-models`, and explicit-target `--dry-run` work without the board SDK; dry-run resolves target and artifact selection without running inference.

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
| `--decode-mode` | str | `ctc` | ctc or legacy / CTC 或逐帧解码模式 |
| `--priority` | int | `0` | Scheduling priority 0–255 / 调度优先级 |
| `--bpu-cores` | int list | `[0]` | Nonnegative core IDs / 非负核心编号 |
| `--output-dir` | Path | `outputs/asr` | Must be new / 必须为新目录 |
| `--list-models` | bool | `false` | List publications without SDK / 无 SDK 列举 |
| `--dry-run` | bool | `false` | Resolve without inference / 只解析不推理 |
The audio/vocabulary defaults resolve relative to this sample, independent of the caller's directory. Relative user paths and output paths resolve from the caller's directory. List and dry-run are mutually exclusive. Fixed frontend settings reject overrides inconsistent with the compiled model; changing these flags cannot resize a model.

<a id="results"></a>
## Results
`result.json` records target, asset ID, locally observed model/audio/vocabulary SHA-256, publisher digest if available, observed tensor metadata, frontend/config and decoder mode. Each `chunks` item includes index, source-frame offset/count/rate, valid resampled sample count and text. `text` concatenates chunk text without inserted separators. Scores and timestamps are not produced. A failure after output-directory creation writes `failed.json` with completed chunks and error details; earlier failures print an error. If the failure report cannot be written, stderr includes both the inference error and the report-write error. Errors return 2.

<a id="integration-example"></a>
## Integration example
From the repository root on S100, after preparing the model. The bundled audio and vocabulary are included. Use `s600` in the selection for S600.
```python
from samples.speech.asr.runtime.python.model_binding import resolve_selection, SAMPLE_DIR
from samples.speech.asr.runtime.python.vocabulary import load_vocabulary
from samples.speech.asr.runtime.python.audio_io import read_chunks
from samples.speech.asr.runtime.python.asr import ASR

selection = resolve_selection("s100")
task = ASR.from_model(selection, load_vocabulary(SAMPLE_DIR / "test_data/vocab.json"))
task.set_scheduling_params(priority=0, bpu_cores=[0])
texts = []
for chunk in read_chunks(SAMPLE_DIR / "test_data/chi_sound.wav", task.config):
    prediction = task.predict(
        chunk.waveform, chunk.sample_rate, return_details=True)
    texts.append(prediction.text)  # prediction.prepared holds this chunk's geometry
print("".join(texts))
```

<a id="stage-io"></a>
## Three-stage I/O
- `preprocess(waveform, sample_rate)` accepts finite float waveform `[frames]` or `[frames,channels]`, bounded by `ceil(30000 × source_rate / 16000)`. It averages channels, uses SciPy Fourier resampling, normalizes by `sqrt(var + 1e-5)` before zero padding, and returns `PreparedChunk.tensor` as owned float32 `[1,30000]` plus geometry.
- `infer({input_name: tensor})` calls the runner once. The runner validates names/shapes/dtypes and returns independently owned raw output. No decoder or activation runs here.
- `postprocess(raw)` checks observed `[1,T,3503]` metadata, takes argmax and returns text. FLOAT32 output is decoded directly as before; declared integer SCALE output is dequantized through the shared quantization helper at float64 comparison precision, so distinct raw integers stay ordered through argmax — float32 would round adjacent magnitudes such as `2**24` and `2**24 + 1` into an artificial tie. No softmax is required for argmax.
- `predict(waveform, sample_rate, *, return_details=False)` composes the three stages for one chunk and returns decoded text by default. With `return_details=True` it returns `ChunkPrediction(text, prepared)`, which also carries the prepared chunk (owned tensor, valid sample count, source geometry). Each call makes one runner request and keeps no per-call state on the model. The CLI uses this form for per-chunk report records. File reading, vocabulary loading and saving belong to the caller, not `ASR`.

The established `pre_process`, `forward`, and `post_process` names remain importable aliases of `preprocess`, `infer`, and `postprocess`.

CTC collapses adjacent equal IDs before removing blank 0; legacy only removes blank. Blank separates repeated tokens: `[5,5,0,5]` becomes `AA` under CTC and `AAA` under legacy. Only exactly equal scores tie, and ties go to the lowest ID: float32 output compares in float32, while integer SCALE output compares at float64 so distinct raw integers cannot round into an artificial tie. All nonblank vocabulary strings, including `|` and special tokens, are retained literally. No state or duplicate suppression crosses chunk boundaries. Final padding still produces a full logit sequence, and all output frames are decoded because model metadata supplies no valid-frame count. Independent windows can split words; chunking does not add overlap-aware stitching. Python uses Fourier resampling, while the C++ runtime uses sinc resampling.

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
