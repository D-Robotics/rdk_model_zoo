English | [简体中文](README_cn.md)

<!-- Template: sample root README (English). Contract: docs/sample-standards/readme-contract.md §4.1.
     Rules: keep every <a id="…"></a> anchor exactly as given; replace ⟪…⟫ placeholders;
     delete guidance blockquotes when done; a section may only be dropped with a
     not-applicable reason (contract §1). File must be paired with README_cn.md. -->

# ⟪Model Name⟫ (⟪task⟫)

<a id="overview"></a>
## Overview

> **Must answer:** what the model does (one sentence); the algorithm in one short
> paragraph; official paper/repo links; where it sits in this repository.

⟪Introduce the model and its method in one or two sentences, e.g. “ResNet uses residual connections to train deep image classifiers.”⟫

Sources: ⟪paper/upstream repository links⟫

⟪Put boards and variants in the support matrix; put I/O and APIs in the runtime guide. Do not explain editorial choices in the finished README.⟫

<a id="directory"></a>
## Directory structure

⟪List the actual files and their roles.⟫

<a id="support-matrix"></a>
## Support Matrix

> **Must answer:** available targets × variants × languages, their artifacts
> and any required SDK or model configuration. Keep test execution records in
> a separate validation document.

| Target | Variant | Language | Artifact and environment |
| --- | --- | --- | --- |
| ⟪x5 / s100 / s100p / s600⟫ | ⟪variant⟫ | ⟪Python / C++⟫ | ⟪actual artifact and SDK⟫ |

⟪State the required model artifact or SDK for any conditional combination.⟫

<a id="prerequisites"></a>
## Prerequisites

> **Must answer:** board + system image version; toolchain requirements; Python
> dependencies; disk/memory constraints. Concrete version numbers only (“latest”
> is not a version).

- Board: ⟪e.g. RDK X5 (8GB/4GB), system image ≥ ⟪version⟫⟫
- Python: ⟪version⟫ with ⟪deps or “board image built-ins only”⟫
- Model artifact prepared in advance (see [Quick Start](#quickstart)).

<a id="quickstart"></a>
## Quick Start

> **Must answer:** ONE complete path from model preparation to a visible result.
> Every command states its cwd, where prerequisite files come from, parameters,
> output, and how success is judged. Prepare models explicitly using the actual
> script and target-selection arguments.

```bash
# cwd: repository root
⟪actual model-preparation command with its supported arguments⟫
# expect: artifact at ⟪path⟫; state the downloader's checksum result or local SHA-256

# cwd: repository root
python3 samples/⟪domain⟫/⟪name⟫/runtime/python/main.py ⟪actual target-selection arguments⟫ ⟪input⟫
# expect: ⟪observable success criterion, e.g. Top-5 list printed / result file path⟫
```

⟪If a run.sh convenience script exists, mention it AFTER the explicit path.⟫

<a id="expected-results"></a>
## Expected Results

> **Must answer:** what a correct run looks like — real output shapes/values from
> test_data, output file paths and naming. Never invent accuracy numbers.

⟪e.g. Top-5 = […] on test_data/⟪image⟫; result image written to ⟪path⟫⟫

<a id="entry-points"></a>
## Entry Points

> **Must answer:** links + one-line description for each entry; absent entries
> state the reason instead of a link.

- Model preparation: [`model/README.md`](model/README.md) — ⟹ ⟪one line⟫
- Python runtime: [`runtime/python/README.md`](runtime/python/README.md) — ⟹ ⟪one line⟫
- ⟪C++ runtime: [`runtime/cpp/README.md`](runtime/cpp/README.md) — ⟹ ⟪one line⟫⟫
- Conversion: [`conversion/README.md`](conversion/README.md) — ⟹ ⟪one line or absence reason⟫
- Evaluation: [`evaluator/README.md`](evaluator/README.md) — ⟹ ⟪one line or absence reason⟫

<a id="license"></a>
## License

> **Must answer:** license of the model weights and of the sample code; relation
> to the repository top-level LICENSE.

⟪e.g. Code: repository LICENSE; weights: ⟪license⟫ (source: ⟪link⟫).⟫
