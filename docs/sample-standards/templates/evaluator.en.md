<!-- Template: evaluator/ README (English). Contract: readme-contract.md §4.6.
     Keep anchors; replace ⟪…⟫; delete guidance when done. Reference results are
     cited from published records or marked not-run — never invented. An empty
     or placeholder-only evaluator does not count as an implementation. -->

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

> **Must answer:** the definition AND test conditions of every metric (topk,
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
> record / release notes); anything not yet evaluated is explicitly not-run.

| Metric | Reference | Conditions | Source |
| --- | --- | --- | --- |
| ⟪metric⟫ | ⟪value⟫ | ⟪conditions⟫ | ⟪link / not-run⟫ |

<a id="boundaries"></a>
## Boundaries

> **Must answer:** what this evaluator does NOT cover (metrics not implemented,
> datasets not supported); required when coverage is partial.

- ⟪not covered + reason⟫
