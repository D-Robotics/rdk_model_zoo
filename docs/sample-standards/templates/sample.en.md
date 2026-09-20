<!-- Template: sample root README (English). Contract: docs/sample-standards/readme-contract.md §4.1.
     Rules: keep every <a id="…"></a> anchor exactly as given; replace ⟪…⟫ placeholders;
     delete guidance blockquotes when done; a section may only be dropped with a
     not-applicable reason (contract §1). File must be paired with README_cn.md. -->

# ⟪Model Name⟫ (⟪task⟫)

<a id="overview"></a>
## Overview

> **Must answer:** what the model does (one sentence); the algorithm in one short
> paragraph; official paper/repo links; where it sits in this repository.

⟪One-sentence task description, e.g. “Real-time object detection for RDK boards.”⟫

- Algorithm: ⟪one paragraph, no implementation details⟫
- Official source: ⟪paper/repo URL⟫
- Category in this repo: `samples/⟪domain⟫/⟪name⟫`

<a id="support-matrix"></a>
## Support Matrix

> **Must answer:** which targets × variants × languages are **supported**, and of
> those, which are **actually verified** on board. Three cell states only:
> `supported-verified`, `supported-not-run`, `not-supported`. Missing C++ must be
> visible here — never claim dual-language support elsewhere when it is absent.

| Variant | x5 | s100 | s100p | s600 | Python | C++ |
| --- | --- | --- | --- | --- | --- | --- |
| ⟪variant⟫ | ⟪state⟫ | ⟪state⟫ | ⟪state⟫ | ⟪state⟫ | ⟪yes/no⟫ | ⟪yes/no⟫ |

Board-verification evidence: ⟪link to evidence/batch review, or “not-run”⟫.

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
> output, and how success is judged. Model preparation is explicit
> (`model/download.sh --target …`); do not rely on implicit auto-download.

```bash
# cwd: repository root
bash samples/⟪domain⟫/⟪name⟫/model/download.sh --target ⟪target⟫
# expect: artifact at ⟪path⟫ (download log prints sha256 verification)

# cwd: repository root
python3 samples/⟪domain⟫/⟪name⟫/runtime/python/main.py --target ⟪target⟫ ⟪input⟫
# expect: ⟪observable success criterion, e.g. Top-5 list printed / result file path⟫
```

⟪If a run.sh convenience script exists, mention it AFTER the explicit path.⟫

<a id="expected-results"></a>
## Expected Results

> **Must answer:** what a correct run looks like — real output shapes/values from
> test_data, output file paths and naming. Never invent accuracy numbers.

⟪e.g. Top-5 = […] on test_data/⟪image⟫; result image written to ⟪path⟫⟫

<a id="directory"></a>
## Directory Layout

> **Must answer:** one-line responsibility per entry; must match the actual tree
> (local paths are machine-checked).

```text
⟪name⟫/
├── conversion/    # ⟹ ONNX → BPU artifact recipes (per-target)
├── model/         # ⟹ artifact download/preparation + artifact README
├── runtime/       # ⟹ python/ (and cpp/ where provided) inference implementations
├── evaluator/     # ⟹ accuracy/performance evaluation
├── test_data/     # ⟹ sample inputs and expected references
└── README.md      # ⟹ this file
```

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
