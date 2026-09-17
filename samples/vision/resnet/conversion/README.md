# ResNet18 model conversion

Conversion is performed on an x86 Linux host in the RDK OpenExplore (OE)
environment. It is not a board runtime operation. The audited X5 and S18
directories both identify TorchVision ResNet18 as the source and point to the
OE classification example, but neither ResNet18 directory contains the
original YAML, calibration set, or a complete replay script. This directory
now supplies the source export step and records the remaining OE-owned steps
without inventing a configuration that could produce a different artifact.

## 1. Prepare the conversion host

Use the OE Docker/toolchain release that matches the target board and record
its image tag, OE version, and host date. The source S18 documentation points to
the OE example at:

```text
samples/ai_toolchain/horizon_model_convert_sample/03_classification/13_resnet18
```

Mount this checkout in that environment. The exact Docker image name and file
name are release-specific; obtain them from the OE release documentation rather
than copying a stale name into this sample. Conversion does not require
`hbm_runtime`, but it does require the OE commands below to be on `PATH`.

The exporter itself needs PyTorch, TorchVision, and ONNX. ONNX Runtime is useful
for an optional host graph smoke test, but is not a replacement for the OE
checker or a board run. Check the versions before exporting; do not let a missing
package trigger an implicit install:

```bash
python3 - <<'PY'
import importlib

for name in ("torch", "torchvision", "onnx"):
    module = importlib.import_module(name)
    print(f"{name}={getattr(module, '__version__', 'unknown')}")
try:
    import onnxruntime
    print(f"onnxruntime={onnxruntime.__version__}")
except ImportError:
    print("onnxruntime=not-installed (optional for the host smoke test)")
PY
```

If the host environment does not provide these packages, install compatible
versions in a user-owned virtual environment using the official PyTorch and
ONNX instructions. The small repository-side helpers are listed in
[`requirements-host.txt`](../requirements-host.txt); the OE image supplies the
compiler and its runtime libraries.

The authoritative OE references are the [RDK S toolchain
overview](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview),
the [D-Robotics toolchain download page](https://toolchain.d-robotics.cc/), and
the [TorchVision ResNet18 model
page](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html).
Use the release-specific package and image from those sources. A generic
load-and-mount sequence is:

```bash
# Set OE_IMAGE_TAR to the image archive supplied with your OE release.
export OE_IMAGE_TAR=/absolute/path/to/oe-image.tar
test -s "$OE_IMAGE_TAR"
docker load -i "$OE_IMAGE_TAR"
docker images

# Set OE_IMAGE to the exact image:tag printed by `docker images`.
: "${OE_IMAGE:?Set OE_IMAGE to the loaded OE image:tag}"
export REPOSITORY_ROOT="${REPOSITORY_ROOT:-$PWD}"
docker run --rm -it --network host --shm-size=15g \
  -v "$REPOSITORY_ROOT":/workspace --workdir /workspace \
  "$OE_IMAGE" /bin/bash
```

Run the remaining commands inside that container from `/workspace`. The
container image, `OE_CONFIG`, and target board must come from the same release
family.

## 2. Export the source ONNX graph

The canonical exporter uses a fixed NCHW input `[1,3,224,224]`, output name
`output`, and ImageNet-1k class count. It uses the TorchScript ONNX exporter
(`dynamo=False`) so graph generation is stable for the OE workflow:

```bash
export WORK="${WORK:-/workspace/resnet18-conversion}"
mkdir -p "$WORK"
python3 samples/vision/resnet/conversion/export_resnet18_onnx.py \
  --output "$WORK/resnet18.onnx" \
  --opset 11 \
  --weights IMAGENET1K_V1
```

`IMAGENET1K_V1` is explicit and may download the official TorchVision weights
when they are not cached. `--weights none` is useful for an offline graph
smoke test, but it creates random weights and cannot produce a meaningful
accuracy claim or establish equivalence with a released RDK artifact:

```bash
export WORK="${WORK:-/workspace/resnet18-conversion}"
mkdir -p "$WORK"
python3 samples/vision/resnet/conversion/export_resnet18_onnx.py \
  --output "$WORK/resnet18-random.onnx" \
  --opset 11 \
  --weights none
```

The exporter checks the ONNX graph with `onnx.checker` by default. Use
`--no-check` only when validation is performed by another recorded tool. The
export smoke test has been run with Torch 2.7.1, TorchVision 0.22.1, ONNX
1.19, and ONNX Runtime 1.23.2; it produced a `[1,1000]` output and matched the
same seeded PyTorch graph within `1e-4` tolerance. That is an ONNX structure
check, not a BPU compilation or accuracy result.

## 3. Obtain the OE conversion inputs

Copy the matching `13_resnet18` files into a working conversion directory and
follow that sample's documented preprocessing and calibration procedure. The
audited RDK sample does not publish its calibration image list, mean/scale
values, or a ResNet18 YAML, so those values must come from the OE release that
produced the artifact being regenerated. Record the resulting files, including
the exact calibration data, rather than assuming the ResNet152 configuration
under another legacy directory applies to ResNet18.

The graph-side input is RGB/NCHW. The deployed runtime-side input is NV12 at
224x224: X5 expects one packed buffer, while S100/S600 expect separate Y and UV
inputs. The OE configuration must perform the graph conversion required for
that target; the runtime Python and C++ code only performs image resize and
NV12 layout conversion and does not apply an unrecorded normalization.

## 4. Run the OE checks and compilation

Use the commands provided by the matching OE sample. The audited X5 notes name
`hb_mapper checker` and `hb_mapper makertbin`; the audited S18 notes name
`hb_compile`. The repository does not contain the historical YAML, so set
`OE_CONFIG` to the actual configuration file shipped by the matching OE
`13_resnet18` sample. This guard makes the required input explicit:

```bash
export WORK="${WORK:-/workspace/resnet18-conversion}"
export ONNX="${ONNX:-$WORK/resnet18.onnx}"
: "${OE_CONFIG:?Set OE_CONFIG to the YAML from your OE 13_resnet18 sample}"
test -s "$ONNX"
test -s "$OE_CONFIG"

# Many OE YAML files resolve relative paths from the current directory. Keep
# the OE sample's documented working directory, or make the copied YAML paths
# absolute. In the copied YAML, set its `working_dir`/output directory to
# "$WORK" (and point `onnx_model` and calibration paths at the files above)
# before running the target-specific compiler block.
```

For an X5 release, first verify the mapper and then compile with the same
configuration. The audited X5 examples use `bayes-e`; confirm the march value
against the selected OE release before running:

```bash
export X5_MARCH="${X5_MARCH:-bayes-e}"
hb_mapper --version
hb_mapper checker --model-type onnx --march "$X5_MARCH" --model "$ONNX"
hb_mapper makertbin --model-type onnx --config "$OE_CONFIG"
find "$WORK" -type f -name '*.bin' -print
```

For an S100 or S600 release, the selected YAML must contain the matching Nash
target (`nash-e` for S100 or `nash-p` for S600). Keep the target choice in the
configuration and in the evidence record; do not infer it from an artifact
filename or path:

```bash
hb_compile --help
hb_compile --config "$OE_CONFIG"
find "$WORK" -type f -name '*.hbm' -print
```

Run only the target-specific block for the artifact being regenerated. If the
OE sample uses a different command spelling or puts the model and calibration
paths in another file, preserve that exact command and file in the evidence
record rather than guessing a replacement. The `find` commands only locate the
output; copy the selected file to the manifest destination after checking its
metadata.

Record these expected stages and artifacts:

| Stage | Artifact to retain |
| --- | --- |
| export | `resnet18.onnx`, exporter arguments, Torch/TorchVision/ONNX versions |
| calibration | the OE sample's calibration directory/list and preprocessing record |
| checker | checker command, target config, and log |
| compile | target config, OE version, and generated output directory |
| deployment | X5 `resnet18_224x224_nv12.bin` or S `resnet18_224x224_nv12.hbm` |
| validation | `hb_perf`/`hrt_model_exec` command, raw output, latency, and accuracy input |

The published filenames are the manifest rows used by this sample. A file with
the same basename is not equivalent until its target, input metadata, output
shape/dtype, and numerical result are compared.

The exporter names its ONNX graph output `output`. The released X5 runtime
metadata names its output `prob` and exposes shape `[1,1000,1,1]`; S artifacts
normally expose `output` with shape `[1,1000]`. These names and shapes are part
of the target artifact contract. A successful ONNX export or compiler command
does not prove that the generated artifact is accepted by the runtime: inspect
the compiled metadata, bind the exact output in the canonical runtime, and run
the matching board check before publishing it.

## 5. Validate before publishing an artifact

Use the OE sample's `hb_perf` and `hrt_model_exec` invocation with the generated
artifact and record their complete output. Then use the canonical runtime
contract on the matching board to confirm:

* X5 metadata exposes one packed NV12 input and an F32 `[1,1000,1,1]` output.
* S100/S600 metadata exposes Y `[1,224,224,1]`, UV `[1,112,112,2]`, and an F32
  `[1,1000]` output.
* The same image, resize type, label file, and Top-K produce the expected
  class IDs and score ordering.

The legacy X5 evaluator records a historical reference of Top-1 71.5% for its
float model and 70.5% for its quantized model, with 2.95 ms latency and 449+
FPS. Those figures describe the published legacy evaluation; the source does
not state whether latency or FPS used single calls, batching, or multiple
threads, so do not derive one value from the other or present them as a fresh
result for a regenerated artifact. The S50/S152 conversion material is not a
ResNet18 recipe and remains outside this sample.

## Known limits

The exporter is a reproducible source-graph helper, not a replacement for the
OE sample. The checked-in source does not establish the exact weights,
calibration set, YAML, target config, or output semantic ownership used for the
published `.bin`/`.hbm` files. Do not label a regenerated model equivalent or
production-ready until those files and board results are recorded. Keep model
URLs and optional hashes in the platform manifests; conversion documentation
should refer to the selected artifact reference rather than create another URL
registry.
