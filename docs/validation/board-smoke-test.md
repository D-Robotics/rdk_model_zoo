# Board smoke-test checklist

Purpose: give the first board session a ready-to-run path through the
reference flows — ResNet classification and Ultralytics YOLO detection —
then the same pattern for speech, offline-policy and native-LLM samples.
Every command uses the samples' own native entries and is recorded
automatically.

Preconditions: the board boots its matching image (which provides the board
SDK and `hbm_runtime`), Python 3 with NumPy/OpenCV/PyYAML is installed on
the board, and the repository checkout is present. Downloads need network
access to the model server named in `docs/release/{x5,s}/models.yaml`; an
offline board can instead receive artifacts copied from a connected host
into the paths each `model/` guide prints.

## 0. Set up recording, record identity, inventory models

The examples assume the repository root is `/root/rdk_model_zoo` — replace
with your actual path. `LOGDIR` is defined once as an absolute path so it
keeps working across the `cd` steps in section 5.

```bash
cd /root/rdk_model_zoo                                     # repository root
export LOGDIR="$PWD/outputs/board-$(date +%Y%m%d-%H%M%S)"    # absolute path
mkdir -p "$LOGDIR"

# Record one case: argv + cwd, full output log, and the command's exit code.
run_case() {
  local name="$1"; shift
  if ! mkdir -p "$LOGDIR" 2>/dev/null; then
    echo "run_case[$name]: cannot create LOGDIR=$LOGDIR" >&2
    return 9
  fi
  local cmd_rec
  cmd_rec=$(printf 'cwd=%s\nargv:' "$PWD"; printf ' %q' "$@"; printf '\n')
  if ! printf '%s' "$cmd_rec" > "$LOGDIR/$name.command"; then
    echo "run_case[$name]: cannot write $LOGDIR/$name.command" >&2
    return 9
  fi
  local -a ps
  if "$@" 2>&1 | tee "$LOGDIR/$name.log"; then
    ps=("${PIPESTATUS[@]}")
  else
    ps=("${PIPESTATUS[@]}")
  fi
  local rc="${ps[0]}" trc="${ps[1]}"
  if ! printf '%s\n' "$rc" > "$LOGDIR/$name.exit" 2>/dev/null; then
    echo "run_case[$name]: cannot write $LOGDIR/$name.exit" >&2
    return 9
  fi
  if (( rc != 0 )); then
    echo "run_case[$name] FAILED rc=$rc — see $LOGDIR/$name.log" >&2
  fi
  if (( trc != 0 )); then
    echo "run_case[$name]: tee failed rc=$trc — log may be incomplete" >&2
    if (( rc == 0 )); then
      return "$trc"
    fi
  fi
  return "$rc"
}
```

Record the board, SDK and source identity once (versions are read from the
system where they exist — no assumed values):

```bash
run_case board-identity bash -s <<'BOARD_IDENTITY'
set -eu
date
uname -a
cat /sys/class/boardinfo/soc_name /sys/class/boardinfo/board_type 2>/dev/null || true
cat /etc/version 2>/dev/null || head -2 /etc/os-release
python3 - <<'PY'
import hbm_runtime
import sys

print("hbm_runtime_path=", hbm_runtime.__file__)
print("hbm_runtime_version=", getattr(hbm_runtime, "__version__", "unavailable"))
print("python_version=", sys.version)
PY
git rev-parse HEAD
BOARD_IDENTITY
```

Inventory the published combinations:

- ResNet `--target auto --list-models` prints every published target's
  asset references straight from the manifest — it works on the host as
  well as on the board.
- YOLO resolves `--platform auto` from the board identity, so run it on
  the board; for a host-side preview pass a concrete `--platform
  x5|s100|s100p|s600`.

```bash
run_case resnet-inventory \
  python3 samples/vision/resnet/runtime/python/main.py --target auto --list-models
# on the board:
run_case yolo-inventory \
  python3 samples/vision/ultralytics_yolo/runtime/python/main.py --platform auto --list-models
```

Run the section for the board you are on — X5, S100 or S600 — not all of
them in sequence. S100P has no ResNet publication; a target without a
matching artifact is an explicit error.

## 1. ResNet classification (X5, S100, S600)

Model preparation uses positional `TARGET VARIANT` (equivalent Python
form: `model/download.py --target <t> --variant <v>`); `download.sh x5`
needs no variant (resnet18 only).

### On an X5 board

```bash
run_case resnet-download \
  bash samples/vision/resnet/model/download.sh x5 resnet18
run_case resnet-run \
  python3 samples/vision/resnet/runtime/python/main.py \
    --target x5 \
    --asset-id x5:resnet:resnet18_224x224_nv12.bin \
    --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
    --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
    --label-file datasets/imagenet/imagenet_classes.names \
    --img-save-path "$LOGDIR/resnet18-x5.jpg"
```

### On an S100 board (resnet18 / resnet50 / resnet152)

```bash
# resnet18
run_case resnet18-download \
  bash samples/vision/resnet/model/download.sh s100 resnet18
run_case resnet18-run \
  python3 samples/vision/resnet/runtime/python/main.py \
    --target s100 \
    --asset-id s:resnet18:s100/resnet18_224x224_nv12.hbm \
    --model-path samples/vision/resnet/model/s100/resnet18_224x224_nv12.hbm \
    --test-img samples/vision/resnet/test_data/zebra_cls.jpg \
    --label-file datasets/imagenet/imagenet_classes.names \
    --img-save-path "$LOGDIR/resnet18-s100.jpg"

# resnet50
run_case resnet50-download \
  bash samples/vision/resnet/model/download.sh s100 resnet50
run_case resnet50-run \
  python3 samples/vision/resnet/runtime/python/main.py \
    --target s100 --variant resnet50 \
    --asset-id s:resnet50:s100/resnet50_224x224_nv12.hbm \
    --model-path samples/vision/resnet/model/s100/resnet50_224x224_nv12.hbm \
    --test-img samples/vision/resnet/test_data/zebra_cls.jpg \
    --label-file datasets/imagenet/imagenet_classes.names \
    --img-save-path "$LOGDIR/resnet50-s100.jpg"

# resnet152: same commands with resnet152 and
#   s:resnet152:s100/resnet152_224x224_nv12.hbm / model/s100/resnet152_224x224_nv12.hbm
```

### On an S600 board (literal commands)

```bash
run_case resnet-download \
  bash samples/vision/resnet/model/download.sh s600 resnet18
run_case resnet-run \
  python3 samples/vision/resnet/runtime/python/main.py \
    --target s600 \
    --asset-id s:resnet18:s600/resnet18_224x224_nv12.hbm \
    --model-path samples/vision/resnet/model/s600/resnet18_224x224_nv12.hbm \
    --test-img samples/vision/resnet/test_data/zebra_cls.jpg \
    --label-file datasets/imagenet/imagenet_classes.names \
    --img-save-path "$LOGDIR/resnet18-s600.jpg"
# resnet50/resnet152 on S600: --variant resnet50|resnet152 with
#   s:resnet50:s600/resnet50_224x224_nv12.hbm / s:resnet152:s600/resnet152_224x224_nv12.hbm
#   and the matching model/s600/... paths
```

Record digests and check the per-case exit codes:

```bash
run_case resnet-sha256 sha256sum \
  samples/vision/resnet/model/s100/resnet18_224x224_nv12.hbm \
  samples/vision/resnet/test_data/zebra_cls.jpg
cat "$LOGDIR/resnet-run.exit"        # per-case exit file (use the case name you ran)
```

**Pass criteria:** the case's `.exit` file shows `0`; the log prints a
Top-5 of class IDs, scores and labels; `white_wolf.JPEG` ranks wolf/dog
classes first and the zebra image ranks zebra (class ID 340) first; a
repeated run prints the same Top-5. The S-series C++ flow is
`run_case resnet-cpp bash samples/vision/resnet/runtime/cpp/run.sh`
(see [runtime/cpp](../../samples/vision/resnet/runtime/cpp/README.md)).

## 2. Ultralytics YOLO detection (X5 and supported S targets)

YOLO uses `--platform` for the downloader and the runtime. With an explicit
`--model-path` the model is treated as custom and results render class IDs
unless `--label-file` is supplied — the commands below pass the COCO label
file so detections show class names. Families per target come from the
step-0 inventory (`yolo26` covers detect/seg/pose/cls/obb on all targets;
`yolov8`/`yolo11` cover detect/seg/pose/cls; X5 additionally carries
`yolov5u`, `yolov9`, `yolov10`, `yolo12`, `yolov13`).

### On an X5 board

```bash
run_case yolo-download \
  bash samples/vision/ultralytics_yolo/model/download_model.sh \
    --platform x5 --family yolov8 --task detect --model-size n
run_case yolo-run \
  python3 samples/vision/ultralytics_yolo/runtime/python/main.py \
    --platform x5 --family yolov8 --task detect \
    --model-path samples/vision/ultralytics_yolo/model/yolov8n_detect_bayese_640x640_nv12.bin \
    --label-file datasets/coco/coco_classes.names \
    --test-img samples/vision/ultralytics_yolo/test_data/bus.jpg \
    --img-save-path "$LOGDIR/yolov8n-x5.jpg"
```

### On an S100 board (YOLO26 detection; exact artifact from the inventory)

```bash
run_case yolo-download \
  bash samples/vision/ultralytics_yolo/model/download_model.sh \
    --platform s100 --family yolo26 --task detect --model-size n
run_case yolo-run \
  python3 samples/vision/ultralytics_yolo/runtime/python/main.py \
    --platform s100 --family yolo26 --task detect \
    --model-path samples/vision/ultralytics_yolo/model/nash-e/yolo26n_detect_nashe_640x640_nv12.hbm \
    --label-file datasets/coco/coco_classes.names \
    --test-img samples/vision/ultralytics_yolo/test_data/bus.jpg \
    --img-save-path "$LOGDIR/yolo26n-s100.jpg"
```

On S600 use `--platform s600` and the `model/nash-p/` artifact the
inventory prints. Segmentation/pose/classification/OBB follow the same
command shape with `--task seg|pose|cls|obb` (pose renders the single
`person` label; OBB uses the DOTA label set).

**Pass criteria:** the case's `.exit` file shows `0`; `[Saved]` printed and
the annotated image written; the bus image shows person and bus boxes with
scores and class names; a repeated run is stable.

## 3. Speech (ASR on S100/S600; Paraformer and KWS on S100)

- **ASR — single model** (`asr.hbm` for the selected target; use a fresh
  output directory for each run):

```bash
run_case asr-download bash samples/speech/asr/model/download.sh --target s100
run_case asr-run bash samples/speech/asr/runtime/python/run.sh \
  --target s100 --output-dir outputs/asr-run1
```

  Use `--target s600` on an S600; never rename an S100 HBM. The console
  prints a full-file transcript and the path to `result.json`; errors exit 2
  and a failure after processing starts writes `failed.json`.

- **Paraformer — multi-model pipeline** (encoder/predictor/decoder + vocab;
  fresh output directory each run):

```bash
run_case para-download \
  bash samples/speech/paraformer/model/download_model.sh --target s100
run_case para-run \
  python3 samples/speech/paraformer/runtime/python/main.py --target s100 \
    --output-dir outputs/paraformer-run1
```

- **KWS — single model** (S100 only; one `kws.hbm`):

```bash
run_case kws-download bash samples/speech/kws/model/download.sh --target s100
run_case kws-run python3 samples/speech/kws/runtime/python/main.py \
  --target s100 --audio-file samples/speech/kws/test_data/sample.wav \
  --output-dir outputs/kws-run1
```

  Success writes `result.json` with score, threshold and `detected`
  (`score >= threshold`, default 0.5); the sample guide records the
  bundled clip's source score (~0.985).

## 4. Offline policy (HIMLoco, X5)

Prepared six-frame observations in, actions out; no robot control:

```bash
run_case himloco-download \
  bash samples/robotics/himloco/model/download_model.sh --target x5
run_case himloco-run python samples/robotics/himloco/runtime/python/main.py \
  --target x5 --output-dir outputs/himloco-run1
```

Each run needs a new output directory; mismatched targets or model hashes
fail before SDK construction.

## 5. Native LLM (C++, S-series only)

Gemma4-E2B runs on S100P/S600 through `runtime/cpp/run.sh`. The model
downloader installs under `GEMMA4_HOME` (default `~/gemma4_e2b`) — export
it before the download so the downloaded `model/` and `tokenizer/` files
and the runtime use the same directory. `--build` builds only; start an
app without it. `LOGDIR` is absolute, so recording keeps working after the
`cd`; the sequence returns to the repository root afterwards.

```bash
export GEMMA4_HOME=~/gemma4_e2b_s600      # S100P: ~/gemma4_e2b_s100p
run_case gemma-download /bin/sh -c \
  'cd samples/llm/gemma4-e2b && GEMMA4_SOC=s600 bash model/download_model.sh'
run_case gemma-tokenizers \
  bash samples/llm/gemma4-e2b/third_party/install_tokenizers_cpp.sh
cd samples/llm/gemma4-e2b/runtime/cpp
run_case gemma-build ./run.sh --target s600 --build     # build only
run_case gemma-main  ./run.sh --target s600             # interactive VLM chat
run_case gemma-demo  ./run.sh --target s600 demo text --prompt "Hello"
run_case gemma-server ./run.sh --target s600 server --port=8000
cd /root/rdk_model_zoo                    # back to the repository root
```

Launcher options precede the app name (`main` default, `server`, `demo`,
`text_bench`, `golden_verify`); native flags follow it. `--target auto`
resolves the board; S100 requires manually supplied HBMs. See the
[Gemma C++ guide](../../samples/llm/gemma4-e2b/runtime/cpp/README.md) for
build dependencies.

MiniCPM5-2B (S600 OELLM 2.0 beta C++ entry; S100/S100P OELLM 1.0.0 legacy
entry) follows
[minicpm5-2b](../../samples/llm/minicpm5-2b/README.md) for its SDK
environment, downloads and commands.

ACT/Pi0 VLA samples use their pinned source repositories and operator-supplied
model resources. See the [VLA guides](../../samples/vla/README.md).

## 6. Extend to the remaining samples

Choose a sample from the [sample index](../../samples/README.md), then follow
its model and runtime guides for the target, command-line options, inputs and
outputs. For classification, use the [ResNet guide](../../samples/vision/resnet/README.md);
other samples have their own artifact and input contracts.

## 7. Evidence and failure capture

Each `run_case` writes `<name>.command` (cwd + argv), `<name>.log` (full
output) and `<name>.exit` (the command's real exit code) under `$LOGDIR`;
keep the runtime's own result files (`result.json`, annotated images) and
the SHA-256 records alongside. Pass criteria here are inference success,
expected visual/textual output and repeatability; report the recorded
per-case exit codes and outputs.

`tools/board_validation/b3_classification_compare.py` is a separate,
scope-fixed comparator for four classifier samples (13 X5 variants:
convnext/edgenext/fasternet/fastvit) — it covers neither ResNet, YOLO nor
the full sample set.

On failure: re-run the failing case once (note whether it reproduces);
keep the `.command`/`.log`/`.exit` triple, the board identity log, the
artifact path and manifest row used, and whether `--dry-run` passes for
the same arguments. File an issue with target, image/SDK, model reference,
commit and the collected logs (see
[Issues](https://github.com/D-Robotics/rdk_model_zoo/issues)).
