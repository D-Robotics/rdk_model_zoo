English | [简体中文](./README_cn.md)

# Evaluator — SigLIP vision features

All numeric tables in this document are source records copied from the S platform sample and release benchmark records; they are retained for provenance and comparability.

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

- Source-recorded measurements: RDK S100 and S100P boards, with the CPU/BPU settings recorded below.
- Comparison recipe: same board, board-image `hbm_runtime`, Python runtime dependencies (`numpy`, `opencv-python`, `PyYAML`), and this sample's runtime.
- Board and runtime versions: not pinned.

<a id="command"></a>
## Evaluation Command

There is no checked-in evaluator command; the runtime sample's CLI is the functional entry point (see [runtime/python](../runtime/python/README.md)). For a same-board comparison, run the runtime CLI on the same image and HBM twice and compare the JSON outputs and saved embeddings.

<a id="metrics"></a>
## Metrics

| Metric | Definition | Conditions |
| --- | --- | --- |
| `pooler_output` latency | One BPU `perf` measurement for the global feature submodel. | Single thread; input resolution and output shape in the table; board CPU/BPU settings below. |
| `last_hidden_state` latency | One BPU `perf` measurement for the patch-feature submodel. | Single thread; input resolution and output shape in the table; board CPU/BPU settings below. |
| TOP1/TOP5 | Zero-shot ImageNet classification accuracy from global embeddings. | ImageNet-1k val, 50,000 images; RGB `(127,127,127)` letterbox for floating-point and BPU paths. |
| Cosine Similarity | Mean/min~max and 1% low similarity of patch features against the reference. | COCO2014 val, 5,000 images; same RGB letterbox preprocessing. |
| MSE | Mean/min~max and 1% low mean-squared error of patch features against the reference. | COCO2014 val, 5,000 images; same RGB letterbox preprocessing. |

Source-recorded board settings:

- S100: CPU `6 x A78AE @ 1.5GHz`, BPU `1 x Nash-E @ 1.0GHz`.
- S100P: CPU `6 x A78AE @ 2.0GHz`, BPU `1 x Nash-M @ 1.5GHz`.
- The source records performance governor commands for CPU policies 0/4 and BPU `28108000.bpu`.

### Source `pooler_output` performance

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

### Source `last_hidden_state` performance

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

The comparison procedure writes complete raw arrays to a unique `evaluator-output/siglip-raw-<UTC microsecond run id>/legacy.npy` and `unified.npy`. It writes no reduced summary as a substitute for the arrays; the arrays are the comparison basis, and a JSON summary may be added beside them by a future evaluator. Shape and dtype must match first; integer raw arrays require exact equality, while floating raw arrays allow `rtol=0` and `atol=1e-5`. The assertion must pass.

<a id="reference-results"></a>
## Reference Results

The following two source tables preserve every row and column. Source: S platform evaluator README, corroborated by the S release benchmark records.

### Source `pooler_output` zero-shot classification

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

### Source `last_hidden_state` semantic consistency

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

- This directory has no evaluator implementation or dataset preparation script; functional checks use the runtime CLI on the board.
- The four tables are source records; they do not by themselves identify the current artifact bytes or runtime version.
- SigLIP is evaluated as a vision feature encoder here. No text encoder, text tokenizer, image-text score, calibration recipe, or C++ evaluator is covered.

## License

The evaluator documentation and comparison helper follow the repository [LICENSE](../../../../LICENSE), Apache-2.0. Source contributor attribution remains Cauchy @吴超.
