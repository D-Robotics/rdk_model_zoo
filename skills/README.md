English | [简体中文](README_cn.md)

# RDK Model Zoo Skills

This directory provides Skills for Model Zoo workflows: model lookup, target repository inspection, integration, development, validation, review and release preparation. Use the repository, branch and hardware specified by the user, and read support information from that checkout's samples, manifests and runtime interfaces.

## Capabilities

| Skill | Responsibility |
|---|---|
| [rdk-model-zoo](rdk-model-zoo/SKILL.md) | Select existing models, query release data and use samples through their actual commands |
| [rdk-model-zoo-repo](rdk-model-zoo-repo/SKILL.md) | Target repository/version/standards context and workflow navigation |
| [rdk-model-zoo-integrate](rdk-model-zoo-integrate/SKILL.md) | Integrate custom model interfaces and hand off to the OE toolchain |
| [rdk-model-zoo-develop](rdk-model-zoo-develop/SKILL.md) | Add, maintain, fix and standardize deliveries |
| [rdk-model-zoo-validate](rdk-model-zoo-validate/SKILL.md) | Scoped functional/accuracy/performance validation and evidence |
| [rdk-model-zoo-review](rdk-model-zoo-review/SKILL.md) | sample-audit and change-review, read-only by default |
| [rdk-model-zoo-release](rdk-model-zoo-release/SKILL.md) | Model/Skills release preparation and distribution |

Use the corresponding OE Skills for quantization, compilation and tuning. This Pack provides the Model Zoo repository and delivery workflows.

## Versions

| Component | Version |
|---|---|
| Pack | 1.1.0 |
| rdk-model-zoo | 1.1.2 |
| rdk-model-zoo-repo | 1.1.1 |
| rdk-model-zoo-integrate | 1.0.1 |
| rdk-model-zoo-develop | 1.1.1 |
| rdk-model-zoo-validate | 1.1.1 |
| rdk-model-zoo-review | 1.1.1 |
| rdk-model-zoo-release | 1.0.1 |

The [`pack.json`](pack.json) manifest is authoritative for versions.

## Target repositories and platforms

First resolve `SKILL_ROOT` (the current Skill installation directory) and `REPO_ROOT` (the user's Model Zoo workspace). Common targets include `rdk_x5`/`x5-v*` for X5, `rdk_s`/`s-v*` for S100/S100P/S600, `rdk_x3`/`x3-v*` for X3, and user-specified `rdk_x5_legacy` or historical refs. Inspect the README, code, manifests and metadata at the requested ref. Locate S100, S100P and S600 support information at the specific sample, artifact and runtime level.

Resolve manifests from paths that actually exist at the target ref. Common locations
include `docs/manifests/`, `docs/release/`, `release/` and platform directories
`docs/release/{x5,s,x3}/`; some checkouts also contain platform manifests under
`platforms/`. `inspect_repo` reports candidate manifests, platform manifests,
branch roles, workspace status and an optional sample file inventory.
`read_catalog` reads actual manifests with PyYAML: a single candidate can be
selected automatically; use `--manifest` to select among multiple candidates.
Use `--benchmark-manifest` for benchmark data and `--model` to filter by model ID
or name. For example: `--manifest docs/release/s/models.yaml`.

## Directories and resources

Each Skill contains SKILL.md, a governance card, references loaded as needed, templates (where applicable) and evals/tasks.yaml. Each of the three runtime helper tools is distributed independently with its owning Skill and does not depend on sibling Skill files. After copying a Skill directory into a supported discovery directory, it can resolve its own references/assets independently.

To distribute an individual Skill, copy the complete `skills/<name>/` directory, including `SKILL.md`, references, applicable assets, scripts and templates. Copies of shared references are included in each Skill directory.

## Command-line tools

```bash
python3 skills/rdk-model-zoo-repo/scripts/inspect_repo.py --repo "$REPO_ROOT"
python3 skills/rdk-model-zoo/scripts/read_catalog.py --repo "$REPO_ROOT" --model "$MODEL_QUERY"
python3 skills/rdk-model-zoo-validate/scripts/validate_evidence.py "$RECEIPT" --evidence-root "$EVIDENCE_ROOT"
```

The variables denote the target checkout, query, receipt file and evidence directory,
respectively. `read_catalog` requires PyYAML, `validate_evidence` requires
jsonschema, and `inspect_repo` uses Python and Git.

## Installation and release

Copy the complete `skills/<name>/` directory into a Skill discovery directory supported by the Agent, then reload the session to use it. Standards, model manifests and source information come from the user's specified Model Zoo checkout version.

## Sources and licensing

[Sources and licensing](NOTICE.md) records this Pack's reference sources and documentation/script licenses.
