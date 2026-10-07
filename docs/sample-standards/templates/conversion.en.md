<!-- Template: conversion/ README (English). Contract: readme-contract.md §4.5.
     Keep anchors; replace ⟪…⟫; delete guidance when done. Missing recipe steps
     MUST be listed under known-gaps — a generic command skeleton must never be
     presented as a verified conversion flow. -->

# Conversion — ⟪model name⟫

<a id="source-model"></a>
## Source Model

> **Must answer:** source framework, weight version/release, where weights come
> from, and how they correspond to the official release.

- Framework: ⟪PyTorch ⟪version⟫ / PaddlePaddle ⟪version⟫ / …⟫
- Weights: ⟪release tag or commit⟫ from ⟪source⟫
- Correspondence: ⟪e.g. official yolov8n.pt ⟪ver⟫⟫

<a id="toolchain-targets"></a>
## Toolchain & Targets

> **Must answer:** OpenExplorer toolchain version, march per target, and the
> config entry point for each target's compile.

| Target | march | OE version | Config |
| --- | --- | --- | --- |
| ⟪target⟫ | ⟪bayes-e / nash-e / nash-m / nash-p⟫ | ⟪version⟫ | ⟪path to yaml/script⟫ |

<a id="export"></a>
## Export (ONNX)

> **Must answer:** environment, actual script/command (cwd), resulting ONNX
> and its expected shape/layout. External exporters belong in Additional Preparation.

```bash
# cwd: ⟪dir⟫
⟪actual export command with its required exporter prepared⟫
# expect: ⟪onnx path + input shape/dtype/layout⟫
```

<a id="calibration"></a>
## Calibration

> **Must answer:** calibration dataset source and size, quantization config,
> actual calibration command and preparation of external calibration tools.

- Dataset: ⟪name/version, N samples, source⟫
- Config: ⟪path⟫
- Command: ⟪command with cwd⟫

<a id="compile"></a>
## Compile

> **Must answer:** the full command producing the .bin/.hbm artifacts and the
> artifact naming convention (`<model>_<resolution>_<chip>.⟪bin|hbm⟫`).

```bash
# cwd: ⟪dir⟫
⟪compile command⟫
# expect: ⟪artifact paths⟫
```

<a id="validation"></a>
## Post-Conversion Validation

> **Must answer:** how to confirm the artifact works (board smoke command,
> reference comparison), with expected output and success criteria.

- Smoke: ⟪command — same as sample quickstart with the fresh artifact⟫
- Expected result: ⟪exit status, output tensors or result file⟫

<a id="artifacts"></a>
## Artifacts

> **Must answer:** produced artifact list with target correspondence and landing
> paths — must agree with model/README.md artifacts.

| Artifact | Target | Lands at |
| --- | --- | --- |
| ⟪file⟫ | ⟪target⟫ | ⟪path⟫ |

<a id="known-gaps"></a>
## Additional Preparation

> **Must answer:** external calibration data, configurations or toolchain examples
> required by this recipe, with concrete acquisition and preparation steps.

- ⟪required external input/configuration and how to prepare it⟫
