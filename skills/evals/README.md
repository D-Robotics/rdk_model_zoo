English | [简体中文](README_cn.md)

# Agent behavior evaluation

The seven Model Zoo Skills define 75 core behavior cases in their respective
`evals/tasks.yaml` files. Each case records the input, fixture and expected routing
or behavior. Results and evidence for individual cases are available in the
[behavior report](REPORT.md). Archive checksums are available in the
[SHA256 file](codex-evidence-2026-09-17.sha256).

## Evaluation records

Each result records `run_id`, `case_id`, evaluation variant (`baseline`,
`agents-only`, `full`), Agent/model/version, fixture summary, actual primary Skill,
tool trace and artifact paths, assertion status and reviewer. Assertion statuses
are `pass`, `fail`, `fixture-invalid` and `not-run`; assertions that cannot run retain
the reason. In negative routing cases, `expect.skill: none` means that the Skill
is not used as the primary Skill.

Evaluations use fixed target repositories, commits and input conditions, recording
the Agent configuration, available tools and session context. Each variant uses
an independent session; synthetic scenarios use separate temporary fixtures.
Review checks expected behavior case by case against actual tool calls, file
changes, logs and final artifacts.

## Current report

The [behavior report](REPORT.md) records execution of the 75 core cases,
comparison results, scoring status and known limitations. It also links to an
archive containing inputs, commit identities, tool events, file changes and
scoring records. After modifying a Skill, run new Agent evaluations for affected
cases and related negative routing scenarios, and record the results in the
corresponding report.
