# ResNet18 evaluation

Evaluation has two separate purposes: confirm that one board can execute the
selected artifact with the expected tensor contract, and measure accuracy or
latency with a stated dataset and toolchain. Host tests cover the former
contract logic only; they do not simulate `hbm_runtime` or certify a board.

## Host checks

From the repository root, run the complete ResNet test suite:

```bash
python3 -m unittest discover -s samples/vision/resnet/tests -v
```

These tests cover manifest selection, strict runtime metadata binding, resize
geometry, packed and split NV12 layouts, injected execution, safe labels,
legacy wrapper return shapes, and deterministic score decoding. The exporter
smoke check is separate from board compilation and accuracy.

## Functional board check

Prepare the artifact from the matching manifest row and run the canonical
Python command on its board. For example, on X5:

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file platforms/x5/datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

On S100 or S600, substitute the corresponding `s:resnet18:<target>/...`
reference, artifact path, and `platforms/s/datasets/imagenet/` labels. Save the
board identity, model reference, runtime metadata, raw F32 score tensor, Top-K
output, image path, resize type, and command line. A result is `not-run` when
the board connection, artifact, or required runtime is unavailable; host test
success does not change that status.

The old Python entrypoints are useful for a direct compatibility comparison:

```bash
python3 platforms/x5/samples/vision/resnet/runtime/python/main.py --help
python3 platforms/s/samples/vision/resnet18/runtime/python/main.py --help
```

For a comparison, run both commands with the same image, model bytes, labels,
resize type, and Top-K. Compare class IDs and raw scores before comparing the
printed label formatting. The compatibility classes return the old tuple/list
shapes while using the canonical preprocessing and decoder.

## Native S-series check

The consolidated C++ source is built and run only on S100 or S600:

```bash
bash samples/vision/resnet/runtime/cpp/run.sh
```

The launcher checks the model, image, and label files, runs CMake, and executes
the binary. It does not install `gflags`, OpenCV, or the Horizon DNN runtime and
does not download a model. For S600, set `MODEL_PATH` and `BUILD_DIR` as shown
in the parent README. To audit the compatibility build independently, configure
`platforms/s/samples/vision/resnet18/runtime/cpp` with CMake; that directory now
adds the canonical CMake target and preserves the historical output location.

The native output should be compared with the original S18 binary using the
same S artifact, `zebra_cls.jpg`, label file, and `--top_k 5`. Record the full
configure/build command and the Top-K lines. The C++ source is a consolidated
copy of the audited S18 implementation and uses the shared
`platforms/s/utils/c_utils` source files; no X5 C++ baseline exists in the
audited source.

## Accuracy and performance

For an ImageNet validation measurement, use the same dataset preprocessing as
the OE conversion reference: 224x224 input and the target's NV12 contract.
Record the image list, label mapping, artifact reference, board identity, and
whether the score vector was decoded with the legacy softmax policy. The
published legacy X5 evaluator reports:

| Artifact/evaluation | Top-1 | Latency | FPS | Source |
| --- | ---: | ---: | ---: | --- |
| ResNet18 float reference | 71.5% | 2.95 ms | 449+ | X5 legacy evaluator README |
| ResNet18 quantized reference | 70.5% | 2.95 ms | 449+ | X5 legacy evaluator README |

These are historical published values, not a fresh result for every checkout
or regenerated artifact. The legacy source does not state whether latency or
FPS used single calls, batching, or multiple threads; keep the two fields as
separate source metrics and do not derive FPS from 2.95 ms. The S18 README provides a qualitative smoke check:
`zebra_cls.jpg` should produce a finite non-zero score for a zebra class, but it
does not publish a full ImageNet accuracy number. Invoke the OE tools used by a
conversion, including `hb_perf` and `hrt_model_exec`, and retain their complete
logs when reporting latency or model-level output.

## Result interpretation

The canonical contract accepts only one output tensor with F32 type and 1000
classes. X5 metadata uses output `prob` with shape `[1,1000,1,1]`; S100/S600 use
`output` with `[1,1000]`. Both Python source wrappers apply softmax. Because the
available conversion source does not prove where X5 normalization occurs,
report raw output and decoded output together and keep the policy labeled
`legacy_softmax` over an `unverified_score_vector`.

Do not publish a new accuracy or performance claim from a missing board run,
from a different artifact with the same filename, or from ResNet50/152 legacy
files. Record `not-run` with the blocking condition and keep the historical
reference separate from newly measured evidence.
