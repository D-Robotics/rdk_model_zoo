# Conversion — FCOS

<a id="source-model"></a>
## Source Model

- Framework: the fixed X5 source records FCOS EfficientNet-B0/B2/B3 deployment artifacts; it does not include training checkpoints or an export script.
- Weights: source snapshot `platforms/x5` at `ac115717197920355fc390bb04299b20e6436864`.
- Correspondence: the three released artifacts are 512, 768, and 896 FCOS variants. The source README and screenshots do not identify a checkpoint release or training commit.

<a id="toolchain-targets"></a>
## Toolchain & Targets

| Target | march | OE version | Config |
| --- | --- | --- | --- |
| x5 / B0 512 | bayes-e | not recorded in source | no YAML in source |
| x5 / B2 768 | bayes-e | not recorded in source | no YAML in source |
| x5 / B3 896 | bayes-e | not recorded in source | no YAML in source |

The three PNG files in this directory are copied source `hb_perf` snapshots; they are evidence of historical documentation, not conversion output from this checkout.

<a id="export"></a>
## Export (ONNX)

```text
No export recipe is included in the fixed source. An ONNX checkpoint, export commit,
input preprocessing provenance, and output naming contract must be supplied before
rebuilding any artifact.
```

<a id="calibration"></a>
## Calibration

- Dataset: not recorded; no calibration files or count are present.
- Config: no FCOS YAML is present in the source tree.
- Command: no reproducible calibration command exists in the fixed source.

<a id="compile"></a>
## Compile

```text
No verified compile command is available. The source's `hb_mapper makertbin --model-type onnx --config your_fcos_config.yaml` is only a placeholder and cannot reproduce these artifacts without missing inputs/configuration.
```

<a id="validation"></a>
## Post-Conversion Validation

On a matching X5, inspect a supplied artifact with `hrt_model_exec model_info --model_file <exact-file>` and run the runtime README command. Save the command output, raw fifteen tensor metadata, and result JSON under one UTC evidence directory. This migration performed host source numerical checks only; conversion and board smoke are not-run.

<a id="artifacts"></a>
## Artifacts

| Artifact | Target | Lands at |
| --- | --- | --- |
| `fcos_efficientnetb0_detect_512x512_bayese_nv12.bin` | x5 | `model/` |
| `fcos_efficientnetb2_detect_768x768_bayese_nv12.bin` | x5 | `model/` |
| `fcos_efficientnetb3_detect_896x896_bayese_nv12.bin` | x5 | `model/` |

<a id="known-gaps"></a>
## Known Gaps

- No checkpoint, ONNX export, calibration data, quantization YAML, toolchain version, or reproducible source compile pipeline is present.
- Manifest publisher hashes are unknown. The three source screenshots cannot establish tensor values or numerical equivalence.
- Rebuilding is therefore outside the verified boundary; only the runtime protocol and host fixture are reproducible here.
