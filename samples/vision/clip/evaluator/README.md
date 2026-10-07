English | [简体中文](./README_cn.md)

# Evaluator — CLIP image-text matching

This directory documents the validation path for the CLIP model pair. No published benchmark table exists, so no latency or accuracy values are given.

<a id="dataset"></a>
## Dataset

The default validation input is `samples/vision/clip/test_data/dog.jpg` with prompts `a diagram` and `a dog`. The BPE vocabulary ships with the runtime at `samples/vision/clip/runtime/python/bpe_simple_vocab_16e6.txt.gz`. No larger evaluation dataset or preparation script is published.

```text
# cwd: repository root
samples/vision/clip/test_data/dog.jpg
samples/vision/clip/runtime/python/bpe_simple_vocab_16e6.txt.gz
# prompts: a diagram,a dog
```

<a id="environment"></a>
## Environment

- Target: RDK X5; image encoder through board `hbm_runtime`, text encoder through CPU `onnxruntime`.
- Host tests: Python 3.14.7, NumPy, OpenCV, `ftfy==6.3.1`, and `regex==2026.9.10`; injected runtime fixtures avoid an ONNX Runtime requirement on the host.

<a id="command"></a>
## Evaluation Command

The validation entrypoints are `runtime/python/main.py` and `run.sh`. The explicit command below runs the sample and writes a user-selected image:

```bash
# cwd: repository root; prerequisites: both model assets prepared for X5
python3 samples/vision/clip/runtime/python/main.py \
  --target x5 \
  --test-img samples/vision/clip/test_data/dog.jpg \
  --texts "a diagram,a dog" \
  --img-save-path /tmp/clip-eval/inference.png
# expect: JSON scores/order and /tmp/clip-eval/inference.png; no published numeric benchmark
```

<a id="metrics"></a>
## Metrics

| Metric | Definition | Conditions |
| --- | --- | --- |
| Cosine similarity | Dot product of each text feature with the image feature divided by both L2 norms plus `1e-12`. | One image, N prompts, float32 features; scores remain in prompt order. |
| Rank order | Descending `argsort` of cosine scores. | NumPy argsort ordering; no cross-version tie-order guarantee. |

No latency, top-k, retrieval, or classification metric is published for this sample.

<a id="outputs"></a>
## Outputs

The runtime writes the annotated image exactly at `--img-save-path` and prints JSON `scores`/`order`.

<a id="reference-results"></a>
## Reference Results

There is no published numeric benchmark table. The validation expectation is qualitative: for `dog.jpg`, the score for `a dog` should exceed the score for `a diagram`.

| Reference | Value | Conditions | Source |
| --- | --- | --- | --- |
| Dog prompt ranking | `a dog` ranks above `a diagram` | X5 pair, bundled BPE, `dog.jpg`, cosine ranking | X5 platform source evaluator README |

<a id="boundaries"></a>
## Boundaries

- No standalone dataset evaluator or published benchmark is provided.
- The text encoder is CPU ONNX and requires ONNX Runtime on the board.
- This sample covers image-text similarity only; no C++ path, training, or text-generation task is included.

## License

Evaluator documentation follows the repository [LICENSE](../../../../LICENSE), Apache-2.0.
