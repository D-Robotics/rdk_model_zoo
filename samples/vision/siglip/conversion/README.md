English | [简体中文](./README_cn.md)

# Conversion — SigLIP

<a id="source-model"></a>
## Source Model

The published artifacts are SigLIP vision encoders derived from Google-origin HuggingFace weights. The source README and release manifest do not identify an exact upstream weight version, commit, export version, or per-artifact license. The official model-family context is [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343). This sample contains deployable HBM references, not a source checkpoint or a reproducible exporter.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain & Targets

The source identifies Nash BPU deployment but records no OE release, compiler build, `march` command, or per-target configuration file. General OE resources: [OE environment documentation](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview) and [toolchain download](https://toolchain.d-robotics.cc/). The supported mapping is S100 → Nash-E and S100P → Nash-M; the same eight publication assets are referenced under `s100/`.

| Target | march | OE version | Config |
| --- | --- | --- | --- |
| s100 | Nash-E | unknown | no source config |
| s100p | Nash-M | unknown | no source config |

<a id="export"></a>
## Export (ONNX)

No export script or reproducible source-checkpoint procedure is published in this sample. Therefore no ONNX command, exporter version, or verified ONNX shape is asserted. The deployable input contract is known: `_input_0`, float32 RGB NCHW `(1,3,size,size)`, values `[-1,1]`.

<a id="calibration"></a>
## Calibration

No calibration dataset, sample count, quantization configuration, calibration script, or calibration version is published. The source only states that the vision encoder was quantized and compiled; those steps cannot be reconstructed from this directory.

<a id="compile"></a>
## Compile

No reproducible compile command, compiler version, calibration parameters, or naming recipe is published. The result is a set of precompiled `.hbm` files listed in [`../model/README.md`](../model/README.md).

<a id="validation"></a>
## Post-Conversion Validation

Each published HBM packs two fixed submodels — `pooler_output` (global image embedding) and `last_hidden_state` (patch-level features) — each exposing `_output_0`. The source inspects a downloaded artifact on the board by checking the model info and measuring both submodels separately with single-threaded `perf`:

```bash
# cwd: repository root, on the board, after downloading the artifact
hrt_model_exec model_info --model_file samples/vision/siglip/model/s100/bpu-siglip-base-patch16-224.hbm
hrt_model_exec perf --thread_num 1 --model_name pooler_output --model_file samples/vision/siglip/model/s100/bpu-siglip-base-patch16-224.hbm
hrt_model_exec perf --thread_num 1 --model_name last_hidden_state --model_file samples/vision/siglip/model/s100/bpu-siglip-base-patch16-224.hbm
```

Apply the same three commands to the other seven artifacts by replacing the file name. The source-recorded evaluator tables are preserved as reference records; validate a fresh conversion with the runs below. A validation must bind both packed submodels, check `_input_0` metadata and the selected `_output_0` shape/dtype, then run the board smoke path in [`../runtime/python/README.md`](../runtime/python/README.md) on both S100 and S100P.

<a id="artifacts"></a>
## Artifacts

| Artifact | Target | Lands at |
| --- | --- | --- |
| `bpu-siglip-base-patch16-224.hbm` | s100, s100p | `samples/vision/siglip/model/s100/` |
| `bpu-siglip-base-patch16-384.hbm` | s100, s100p | `samples/vision/siglip/model/s100/` |
| `bpu-siglip-base-patch16-512.hbm` | s100, s100p | `samples/vision/siglip/model/s100/` |
| `bpu-siglip-large-patch16-256.hbm` | s100, s100p | `samples/vision/siglip/model/s100/` |
| `bpu-siglip-large-patch16-384.hbm` | s100, s100p | `samples/vision/siglip/model/s100/` |
| `bpu-siglip-so400m-patch14-224.hbm` | s100, s100p | `samples/vision/siglip/model/s100/` |
| `bpu-siglip-so400m-patch14-384.hbm` | s100, s100p | `samples/vision/siglip/model/s100/` |
| `bpu-siglip-so400m-patch16-256-i18n.hbm` | s100, s100p | `samples/vision/siglip/model/s100/` |

<a id="known-gaps"></a>
## Known Gaps

- Exact source checkpoint/version and export script are missing.
- OE version, compiler build, `march` configuration, calibration dataset/configuration, and compile command are missing.
- Weight/export license metadata and artifact hashes are missing (`sha256: null (unknown)` in the model README).
- The reproducible boundary is selection, input/output contract documentation, and use of already published HBM assets; this directory cannot recreate the HBM files.

## License

Conversion documentation and sample code follow the repository [LICENSE](../../../../LICENSE), Apache-2.0. No additional artifact license is asserted because the source release does not record one.
