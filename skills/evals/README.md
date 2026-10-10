English | [简体中文](README_cn.md)

# Agent behavior evaluation

The seven Model Zoo Skills define 75 core behavior cases in their respective
`evals/tasks.yaml` files. Each case records the input, fixture and expected routing
or behavior.

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

## Running an evaluation

After modifying a Skill, run new Agent evaluations for the affected cases and
the related negative routing scenarios, then record the results in a report
following the record structure above.
