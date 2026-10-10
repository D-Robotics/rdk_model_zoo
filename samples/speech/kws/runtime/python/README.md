English | [简体中文](README_cn.md)

# KWS Python runtime

<a id="overview"></a>
## Python inference

Detect the wake word on S100. `KWS.from_model` initializes the runtime; `predict` computes fbank features, runs the model and returns a probability score.

<a id="directory"></a>
## Directory structure

```text
python/
├── cli.py  # Published selection, arguments, audio loading and reports
├── kws.py  # MDTC frontend, tensor binding, runner and model stages
├── main.py  # Command-line entry: construct the model and call predict
└── run.sh  # Locate the Python entry and forward arguments
```

<a id="environment"></a>
## Environment

Run commands from the repository root. The task needs Python 3.10+, NumPy, PyYAML, SoundFile, PaddlePaddle, PaddleAudio and S100's BSP-provided `hbm_runtime`. Help/list/dry-run only need the lightweight selection layer with PyYAML and NumPy for option checks. No import loads a board SDK before actual execution.

Install the frontend dependencies in an isolated Python environment. An example setup is:

```sh
python3 -m venv /path/to/kws-venv
/path/to/kws-venv/bin/python -m pip install numpy PyYAML soundfile paddlepaddle paddleaudio tqdm scipy resampy scikit-learn
```

PaddleAudio imports dataset/metric modules that need tqdm/scikit-learn even for the fbank workflow. Do not let a launcher install packages into the system. The compiled `hbm_runtime` must come from the board image, with access from the chosen environment.

<a id="usage"></a>
## Usage

Read-only host commands:

```bash
bash samples/speech/kws/runtime/python/run.sh --help
bash samples/speech/kws/runtime/python/run.sh --list-models
bash samples/speech/kws/runtime/python/run.sh --target s100 --dry-run
```

After [explicit model preparation](../../model/README.md), run on S100:

```sh
bash samples/speech/kws/runtime/python/run.sh --target s100 \
  --audio-file samples/speech/kws/test_data/sample.wav --output-dir outputs/kws-run1
```

`run.sh` does not download, install or change the caller's working directory. Default model/audio paths are sample-relative; supplied relative paths and output paths are caller-relative. `--target auto` detects the actual board; host dry-run requires an explicit target. A dry-run does not validate the model bytes or SDK descriptors.

<a id="parameters"></a>
## Parameters

| Option | Default | Meaning |
| --- | --- | --- |
| `--target` | auto | Actual target; only S100 has an artifact |
| `--asset-id` | `null` | Exact `s:kws:s100/kws.hbm` |
| `--model-path` | `null` | Existing file; requires exact asset ID |
| `--audio-file` | `samples/speech/kws/test_data/sample.wav` | Mono 16 kHz audio, decoded float32 |
| `--output-dir` | `outputs/kws` | New directory for `result.json` |
| `--audio-maxlen` | 60000 | Fixed published frontend sample count |
| `--frame-shift` | 10 | Fixed milliseconds |
| `--frame-length` | 25 | Fixed milliseconds |
| `--n-mels` | 80 | Fixed feature width |
| `--priority` | 0 | SDK scheduling priority 0..255 |
| `--bpu-cores` | `[0]` | One or more nonnegative SDK core indexes |
| `--threshold` | 0.5 | Finite [0,1], decision uses score >= threshold |
| `--list-models` / `--dry-run` | false | Mutually exclusive read-only modes |

The frontend options `--audio-maxlen`, `--frame-shift`, `--frame-length` and `--n-mels` must retain these published values for this model. Changing the window or feature dimensions requires a model exported for the new feature contract. Scheduling is applied through the shared runner; unsupported SDK scheduling fails.

<a id="results"></a>
## Results

The CLI writes and prints `result.json`: score/decision, threshold rule, selected identity, actual metadata, model/audio SHA-256 and source/used/padded/truncated sample counts. A returned score is a Python float probability, not a time interval or transcript. The report copies the manifest's `publisher_sha256` value, which may be null. Directory reuse is rejected. A preprocessing, metadata or probability-contract error exits 2; failed predictions do not write a result report.

<a id="integration-example"></a>
## Reusable API

On S100 with a prepared model and frontend dependencies, this example reads the bundled audio, loads the runner and calls all three stages.

```python
from samples.speech.kws.runtime.python.cli import resolve_selection, SAMPLE_DIR
from samples.speech.kws.runtime.python.cli import load_audio
from samples.speech.kws.runtime.python.kws import KWS
selection = resolve_selection("s100")
audio, sample_rate = load_audio(SAMPLE_DIR / "test_data/sample.wav")
task = KWS.from_model(selection)
task.set_scheduling_params(priority=0, bpu_cores=[0])
score = task.predict(audio, sample_rate)
print(score)
# Explicit stages are available when intermediate tensors are needed.
```

<a id="stage-io"></a>
## Stage and lifetime contract

`preprocess` accepts a nonempty finite mono float32 `[N]` waveform with amplitudes in [-1,1] and rate 16000. It copies the first 60000 samples and pads the remainder with zeros. No hidden resampling/channel averaging is performed. PaddleAudio fbank uses 25 ms frames, 10 ms shifts, 80 bins and source defaults (including dither=0 and snip_edges=True), producing 373 frames. The named `[1,373,80]` tensor owns contiguous data.

`infer` performs one shared runner call and returns raw output; it has no sigmoid, dequantization, file access or reduction. The runner validates names/shapes/dtypes/finite values and copies SDK output so it survives later calls. `postprocess` validates the bound output, applies shared SCALE conversion only to integer data, requires probabilities in [0,1] and returns their maximum. Float output is not transformed even if metadata carries a vestigial quant descriptor. No additional sigmoid is applied.

`predict` composes the stages and caches no last-image/audio state. The established `pre_process`, `forward`, and `post_process` names remain importable thin aliases of `preprocess`, `infer`, and `postprocess` — one implementation, two names. One instance is intended for serial use; SDK concurrency is not promised. `kws.py` prepares features (fixed MDTC frontend), owns scoring and runtime initialization; `cli.py` reads audio files, resolves selection and writes reports.

<a id="troubleshooting"></a>
## Troubleshooting

| Error | Action |
| --- | --- |
| Only s100 published / target mismatch | Use a genuine S100 and its artifact; do not impersonate S100P/S600 |
| Missing model or SDK | Perform explicit download and use the BSP runtime environment |
| Missing PaddleAudio dependency | Prepare the frontend environment explicitly; inspect the missing package name |
| Input must be [1,373,80] | Check artifact and frontend versions; do not reshape arbitrary tensors to bypass checks |
| Mono/16000 Hz required | Convert audio explicitly, retaining the converted input identity |
| Invalid probability or SCALE descriptor | Check the exact model/SDK output metadata; no guessed sigmoid or scale |
| Output directory must be new | Select a new run directory |

The supplied clip covers one positive example. Use a labeled positive/negative set and the [evaluator](../../evaluator/README.md) before choosing an operating threshold.
