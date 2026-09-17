# RDK Skills v1.0.0

> Component: RDK Model Zoo Skills. Seven workflows for Model Zoo usage, repository context, integration, development, validation, review, and release preparation.

## Overview

Model Zoo now maintains one Skills Pack on `rdk_x5` for X5, S100/S100P/S600, X3, and historical repository layouts. The maintenance branch never determines the user's target platform. Skills Pack versions remain independent from model release versions.

## Highlights

- Preserve the `rdk-model-zoo` entry at member version 1.1.0; add six focused workflows at 1.0.0.
- Resolve target repository identity separately from installed skill resources, with explicit platform conflicts and manifest ambiguity.
- Support current and historical manifest locations without inventing missing metrics or board evidence.
- Include self-contained references, templates and helpers in each independently installable skill.
- Isolate non-model Release events from the model Pages concurrency group. Model versions, manifests and existing tags are unchanged.

## Install

From the source repository:

```bash
npx skills add D-Robotics/rdk_model_zoo --skill rdk-model-zoo
```

After the Hub onboarding PR merges, the same seven names are available through `D-Robotics/rdk-skills`. The Hub's existing `rdk-model-zoo` entry changes its maintenance source from Device Skills to Model Zoo; remove an older duplicate local installation before installing the replacement.

## Verification

- Reference synchronization, seven package checks, independent resource closure and 50 local tests passed.
- 75 core Codex behavior cases executed. Initial grading includes failures and incomplete fixtures; this is not a claim of complete behavioral acceptance.
- Fourteen paired controls were attempted; one release baseline timed out. Focused reruns and known limitations are preserved in `skills/verification/REPORT.md` and the evidence archive linked from `skills/evals/REPORT.md`.
- Evaluation expansion stopped at the maintainer's request to prioritize shipping. No real hardware, model accuracy, performance, or external actuator validation was performed.

## Component tags

| Component | Version |
| --- | --- |
| RDK Model Zoo Skills Pack | v1.0.0 |
| rdk-model-zoo entry | 1.1.0 |
| Other six members | 1.0.0 |

## Compatibility notes

S-family compatibility remains specific to the sample, artifact and runtime. Historical layouts remain valid for their target versions. Quantization and compilation are handed to existing platform toolchains. Some evaluation prompts have routing ambiguity or conflicting requirements; missing real artifacts and production Hub automation are not represented as verified capabilities. This release does not update model releases or replace their Latest designation.
