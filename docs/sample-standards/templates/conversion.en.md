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

> **Must answer:** environment, script/command (cwd), resulting ONNX and its
> expected shape/layout. If no export recipe exists, say so here and list it in
> known-gaps.

```bash
# cwd: ⟪dir⟫
⟪export command or “No export recipe — see known-gaps”⟫
# expect: ⟪onnx path + input shape/dtype/layout⟫
```

<a id="calibration"></a>
## Calibration

> **Must answer:** calibration dataset source and size, quantization config,
> calibration command. Absent calibration → known-gaps.

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
> reference comparison); what has actually been validated vs not-run.

- Smoke: ⟪command — same as sample quickstart with the fresh artifact⟫
- Status: ⟪validated on ⟪board⟫ (evidence ⟪link⟫) / not-run⟫

<a id="artifacts"></a>
## Artifacts

> **Must answer:** produced artifact list with target correspondence and landing
> paths — must agree with model/README.md artifacts.

| Artifact | Target | Lands at |
| --- | --- | --- |
| ⟪file⟫ | ⟪target⟫ | ⟪path⟫ |

<a id="known-gaps"></a>
## Known Gaps

> **Must answer:** every missing recipe piece (no calibration data, no export
> script, toolchain unavailable) and the reproducible boundary as of now.
> Mandatory section — write “none” only if genuinely complete.

- ⟪gap + what can/cannot be reproduced today⟫
