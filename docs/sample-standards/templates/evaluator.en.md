English | [简体中文](README_cn.md)

<!-- Keep section anchors, fill ⟪…⟫ and remove guidance before publishing. Maintain the Chinese version alongside this file. -->

# Evaluator — ⟪model name⟫

<a id="dataset"></a>
## Dataset

> **Must answer:** dataset name/version/scale; acquisition and preparation
> steps (cwd, commands); resulting directory layout.

- Dataset: ⟪name ⟪version⟫, ⟪N⟫ items⟫
- Preparation:

```bash
# cwd: ⟪dir⟫
⟪download/prepare command or manual steps⟫
# expect: ⟪layout⟫
```

<a id="directory"></a>
## Directory structure

⟪List the actual files and their roles.⟫

<a id="environment"></a>
## Environment

> **Must answer:** runs on board or host; dependencies beyond the runtime;
> relation to runtime/python (reuses which modules).

- Runs on: ⟪board / host / both⟫
- Extra deps: ⟪list or none⟫
- Reuses: ⟪runtime modules used by the evaluator⟫

<a id="command"></a>
## Evaluation Command

> **Must answer:** cwd, the full command, parameter table with actual defaults,
> expected runtime duration.

```bash
# cwd: ⟪dir⟫
⟪command⟫
# expect: ⟪duration, progress output⟫
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| ⟪arg⟫ | ⟪type⟫ | ⟪default⟫ | ⟪…⟫ |

<a id="metrics"></a>
## Metrics

> **Must answer:** the definition AND measurement conditions of every metric (topk,
> IoU thresholds, data subset) — results are comparable only with full
> conditions stated.

| Metric | Definition | Conditions |
| --- | --- | --- |
| ⟪metric⟫ | ⟪definition⟫ | ⟪topk / IoU / subset / ⟪…⟫⟫ |

<a id="outputs"></a>
## Outputs

> **Must answer:** where result files land and their format.

⟪e.g. ⟪path⟫ JSON: {"metric": value, …}⟫

<a id="reference-results"></a>
## Reference Results

> **Must answer:** published reference values WITH their source (benchmark
> record / release notes). If no reference exists, describe how to obtain it.

| Metric | Reference | Conditions | Source |
| --- | --- | --- | --- |
| ⟪metric⟫ | ⟪value⟫ | ⟪conditions⟫ | ⟪source link⟫ |

<a id="boundaries"></a>
## Supported Evaluation Scope

> **Must answer:** supported datasets, metrics and input conditions; identify
> the concrete external tools and preparation needed for additional evaluation.

- ⟪supported evaluation and required tools/data⟫
