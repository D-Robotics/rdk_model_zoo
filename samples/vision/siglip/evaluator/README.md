English | [简体中文](./README_cn.md)

# Evaluator — SigLIP vision features

All numeric tables in this document are historical source records copied from the fixed S sample and release benchmark records. They are retained for provenance and comparability; they are not a benchmark rerun in this migration. No evaluator script, dataset download, board, or HBM download was used here.

<a id="dataset"></a>
## Dataset

The historical `pooler_output` zero-shot classification record used ImageNet-1k validation (50,000 images). The historical `last_hidden_state` semantic-consistency record used COCO2014 validation (5,000 images). The source does not publish a preparation script, exact archive revision, directory layout, or evaluator implementation.

```text
# cwd: repository root
# Preparation: not provided; do not infer a download command from this README.
# expected source-owned layouts: ImageNet-1k val and COCO2014 val, as supplied by the evaluator owner
```

<a id="environment"></a>
## Environment

- Historical measurements: RDK S100 and S100P boards, with the CPU/BPU settings recorded below.
- Current comparison recipe: same board, board-image `hbm_runtime`, Python runtime dependencies (`numpy`, `opencv-python`, `PyYAML`), and the fixed source runtime at `platforms/s/samples/vision/siglip/runtime/python`.
- Reuses: unified `model_binding.py`, `model_runner.py`, `tensor_io.py`, and `embedding.py`; legacy `SigLIPConfig`/`SigLIP` from the fixed source runtime.
- Current board and runtime versions: not verified.

<a id="command"></a>
## Evaluation Command

There is no checked-in evaluator command. The following is a copyable same-board raw-output comparison procedure; it is documented only and was not executed. It uses the existing source runtime at `platforms/s/samples/vision/siglip/runtime/python`, saves complete arrays in a unique run directory, and performs no pooling, normalization, dequantization, or score conversion.

```bash
# cwd: repository root; prerequisite: one prepared exact HBM on the board
PYTHONPATH="$PWD:$PWD/platforms/s:$PWD/platforms/s/samples/vision/siglip/runtime/python" python3 - <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import cv2
import numpy as np

from siglip import SigLIP, SigLIPConfig
from samples.vision.siglip.runtime.python.model_binding import resolve_selection
from samples.vision.siglip.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.siglip.runtime.python.embedding import SigLIPTask

repo = Path.cwd()
model_path = repo / "samples/vision/siglip/model/s100/bpu-siglip-base-patch16-224.hbm"
image_path = repo / "samples/vision/siglip/test_data/dog.jpg"
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
raw_dir = repo / "evaluator-output" / f"siglip-raw-{run_id}"
raw_dir.mkdir(parents=True, exist_ok=False)
image = cv2.imread(str(image_path))
if image is None or not model_path.is_file():
    raise RuntimeError("prepare the image and exact HBM before running")

target = "s100"
variant = "base-patch16-224"
submodel = "pooler_output"
image_size = 224
priority = 0
bpu_cores = [0]

legacy = SigLIP(SigLIPConfig(str(model_path), image_size=image_size, submodel=submodel))
legacy.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)
legacy_inputs = legacy.pre_process(image)
legacy_raw_nested = legacy.forward(legacy_inputs)
legacy_raw = np.asarray(legacy_raw_nested[submodel]["_output_0"])
np.save(raw_dir / "legacy.npy", legacy_raw, allow_pickle=False)

selection = resolve_selection(target, variant=variant, model_path=model_path,
                              asset_id="s:siglip:s100/bpu-siglip-base-patch16-224.hbm",
                              submodel=submodel, image_size=image_size)
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)
task = SigLIPTask(runner, binding)
prepared = task.pre_process(image)
unified_raw_mapping = task.forward(prepared.tensors)
unified_raw = task.post_process(unified_raw_mapping)
np.save(raw_dir / "unified.npy", unified_raw, allow_pickle=False)

if legacy_raw.shape != unified_raw.shape or legacy_raw.dtype != unified_raw.dtype:
    raise AssertionError((legacy_raw.shape, legacy_raw.dtype, unified_raw.shape, unified_raw.dtype))
if np.issubdtype(legacy_raw.dtype, np.floating):
    np.testing.assert_allclose(legacy_raw, unified_raw, rtol=0.0, atol=1e-5)
else:
    np.testing.assert_array_equal(legacy_raw, unified_raw)
print({"legacy": str(raw_dir / "legacy.npy"), "unified": str(raw_dir / "unified.npy"), "comparison": "passed", "run_id": run_id})
PY
# expect: two complete .npy raw arrays and a passing comparison line; assertion failure exits nonzero; this procedure was not-run here
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `target` | str | `s100` in the example | Explicit board target; repeat on `s100p` for the second supported target. |
| `variant` | str | `base-patch16-224` in the example | One of the eight published variants. |
| `submodel` | str | `pooler_output` in the example | Compare one fixed packed submodel at a time. |
| `image_size` | int | `224` in the example | Must equal the selected variant. |
| `raw_dir` | path | `evaluator-output/siglip-raw-<UTC microsecond run id>` | Unique destination for complete legacy/unified arrays. |

<a id="metrics"></a>
## Metrics

| Metric | Definition | Conditions |
| --- | --- | --- |
| `pooler_output` latency | One BPU `perf` measurement for the global feature submodel. | Single thread; input resolution and output shape in the table; board CPU/BPU settings below. |
| `last_hidden_state` latency | One BPU `perf` measurement for the patch-feature submodel. | Single thread; input resolution and output shape in the table; board CPU/BPU settings below. |
| TOP1/TOP5 | Zero-shot ImageNet classification accuracy from global embeddings. | ImageNet-1k val, 50,000 images; RGB `(127,127,127)` letterbox for floating-point and BPU paths. |
| Cosine Similarity | Mean/min~max and 1% low similarity of patch features against the reference. | COCO2014 val, 5,000 images; same RGB letterbox preprocessing. |
| MSE | Mean/min~max and 1% low mean-squared error of patch features against the reference. | COCO2014 val, 5,000 images; same RGB letterbox preprocessing. |

Historical board settings:

- S100: CPU `6 x A78AE @ 1.5GHz`, BPU `1 x Nash-E @ 1.0GHz`.
- S100P: CPU `6 x A78AE @ 2.0GHz`, BPU `1 x Nash-M @ 1.5GHz`.
- The source records performance governor commands for CPU policies 0/4 and BPU `28108000.bpu`; the commands were not run here.

### Historical `pooler_output` performance (not a current rerun)

| Model Name | Input Size | Embedding Size | Params total / vision | RDK S100 | RDK S100P |
|---|---|---|---|---|---|
| siglip-base-patch16-224 | `(1,3,224,224)` | `(1,1,768)` | `0.2 B / 0.09 B` | 26.8 ms | 18.8 ms |
| siglip-base-patch16-384 | `(1,3,384,384)` | `(1,1,768)` | `0.2 B / 0.09 B` | 46.7 ms | 32.3 ms |
| siglip-base-patch16-512 | `(1,3,512,512)` | `(1,1,768)` | `0.2 B / 0.09 B` | 81.7 ms | 55.8 ms |
| siglip-large-patch16-256 | `(1,3,256,256)` | `(1,1,1024)` | `0.7 B / 0.32 B` | 68.8 ms | 47.2 ms |
| siglip-large-patch16-384 | `(1,3,384,384)` | `(1,1,1024)` | `0.7 B / 0.32 B` | 132.5 ms | 91.4 ms |
| siglip-so400m-patch14-224 | `(1,3,224,224)` | `(1,1,1152)` | `0.9 B / 0.43 B` | 89.8 ms | 62.2 ms |
| siglip-so400m-patch14-384 | `(1,3,384,384)` | `(1,1,1152)` | `0.9 B / 0.43 B` | 255.7 ms | 175.5 ms |
| siglip-so400m-patch16-256-i18n | `(1,3,256,256)` | `(1,1,1152)` | `1.0 B / 0.43 B` | 89.6 ms | 61.9 ms |

### Historical `last_hidden_state` performance (not a current rerun)

| Model Name | Input Size | Embedding Size | Params total / vision | RDK S100 | RDK S100P |
|---|---|---|---|---|---|
| siglip-base-patch16-224 | `(1,3,224,224)` | `(1,196,768)` | `0.2 B / 0.09 B` | 26.0 ms | 18.3 ms |
| siglip-base-patch16-384 | `(1,3,384,384)` | `(1,576,768)` | `0.2 B / 0.09 B` | 45.9 ms | 31.7 ms |
| siglip-base-patch16-512 | `(1,3,512,512)` | `(1,1024,768)` | `0.2 B / 0.09 B` | 80.8 ms | 55.3 ms |
| siglip-large-patch16-256 | `(1,3,256,256)` | `(1,256,1024)` | `0.7 B / 0.32 B` | 67.6 ms | 46.5 ms |
| siglip-large-patch16-384 | `(1,3,384,384)` | `(1,576,1024)` | `0.7 B / 0.32 B` | 131.3 ms | 90.5 ms |
| siglip-so400m-patch14-224 | `(1,3,224,224)` | `(1,256,1152)` | `0.9 B / 0.43 B` | 88.6 ms | 61.4 ms |
| siglip-so400m-patch14-384 | `(1,3,384,384)` | `(1,729,1152)` | `0.9 B / 0.43 B` | 254.2 ms | 174.5 ms |
| siglip-so400m-patch16-256-i18n | `(1,3,256,256)` | `(1,256,1152)` | `1.0 B / 0.43 B` | 88.3 ms | 61.1 ms |

<a id="outputs"></a>
## Outputs

The comparison procedure writes complete raw arrays to a unique `evaluator-output/siglip-raw-<UTC microsecond run id>/legacy.npy` and `unified.npy`. It writes no reduced summary as a substitute for the arrays. A future evaluator may add a JSON comparison record beside them, but no such result exists today. Shape and dtype must match first; integer raw arrays require exact equality, while floating raw arrays allow `rtol=0` and `atol=1e-5`. The assertion must pass.

<a id="reference-results"></a>
## Reference Results

The following two historical tables preserve every source row and column. They are reference values only, with status `not-run` for this migration. Source: `platforms/s/samples/vision/siglip/evaluator/README.md`, corroborated by `platforms/s/docs/release/benchmarks.yaml`.

### Historical `pooler_output` zero-shot classification (not a current rerun)

| Model Name | PyTorch TOP1 / TOP5 | BPU TOP1 / TOP5 |
|---|---|---|
| siglip-base-patch16-224 | 0.7123 / 0.9143 | 0.7118 / 0.9144 |
| siglip-base-patch16-384 | 0.7411 / 0.9318 | 0.7418 / 0.9319 |
| siglip-base-patch16-512 | 0.7490 / 0.9343 | 0.7482 / 0.9340 |
| siglip-large-patch16-256 | 0.7490 / 0.9238 | 0.7490 / 0.9242 |
| siglip-large-patch16-384 | 0.7584 / 0.9252 | 0.7595 / 0.9256 |
| siglip-so400m-patch14-224 | 0.7659 / 0.9361 | 0.7651 / 0.9357 |
| siglip-so400m-patch14-384 | 0.7872 / 0.9433 | 0.7893 / 0.9447 |
| siglip-so400m-patch16-256-i18n | 0.7678 / 0.9395 | 0.7668 / 0.9397 |

### Historical `last_hidden_state` semantic consistency (not a current rerun)

| Model Name | Cosine Similarity mean (min ~ max), 1% low | MSE mean (min ~ max), 1% low |
|---|---|---|
| siglip-base-patch16-224 | 0.991 (0.951 ~ 0.997), 0.980 | 0.087 (0.024 ~ 0.471), 0.039 |
| siglip-base-patch16-384 | 0.989 (0.960 ~ 0.997), 0.977 | 0.113 (0.029 ~ 0.409), 0.050 |
| siglip-base-patch16-512 | 0.987 (0.956 ~ 0.995), 0.974 | 0.142 (0.045 ~ 0.507), 0.067 |
| siglip-large-patch16-256 | 0.990 (0.933 ~ 0.997), 0.974 | 0.069 (0.018 ~ 0.497), 0.024 |
| siglip-large-patch16-384 | 0.985 (0.900 ~ 0.995), 0.965 | 0.111 (0.034 ~ 0.775), 0.048 |
| siglip-so400m-patch14-224 | 0.984 (0.850 ~ 0.995), 0.961 | 0.104 (0.028 ~ 1.038), 0.041 |
| siglip-so400m-patch14-384 | 0.980 (0.859 ~ 0.993), 0.957 | 0.140 (0.040 ~ 1.093), 0.059 |
| siglip-so400m-patch16-256-i18n | 0.984 (0.878 ~ 0.996), 0.959 | 0.082 (0.018 ~ 0.570), 0.030 |

<a id="boundaries"></a>
## Boundaries

- This directory has no evaluator implementation or dataset preparation script; the comparison is a documented board-side procedure and remains not-run.
- The four tables are historical records, not evidence of current artifact identity, runtime version, or board reproducibility.
- SigLIP is evaluated as a vision feature encoder here. No text encoder, text tokenizer, image-text score, calibration recipe, or C++ evaluator is covered.

## License

The evaluator documentation and comparison helper follow the repository [LICENSE](../../../../LICENSE), Apache-2.0. Source contributor attribution remains Cauchy @吴超.
