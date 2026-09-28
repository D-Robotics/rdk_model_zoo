# VLA fixed-submodule integration

Base: `2f7106cd`. Author integration record; independent whole-branch review open.

ACT and Pi0 remain real mode-160000 Git submodules at their exact original commits:
`326ea043be204de25223d95c7d918efe8672dc66` and
`a32de276bc1681a2b1531012de111eaa1c16acb6`. Both use
`https://github.com/D-Robotics/rdk_LeRobot_tools`. They moved from archival paths to
`samples/vla/act` and `samples/vla/pi0`, with matching .gitmodules section/path
updates. Initialization from the public URL succeeded at those exact commits.
Both submodule worktrees are clean; no upstream file was edited.

The outer bilingual overview, ACT and Pi0 guides preserve practical initialization,
source-environment distinctions, conversion entry points, offline/live operation,
required operator assets, outputs, historical results, licenses and local links to
complete upstream workflows/images. Old archival paths retain navigation guides.
Root/sample indexes now expose VLA separately from the 49 in-repository samples.

Key distinction: the ACT pin is the older S100/v2.1-data/SO101 source. S600 ACT is
inside the Pi0 checkout under models/act and uses LeRobot v0.5.2/v3.0-data/SO100.
Pi0 is the S600 three-HBM chain; its single-input numeric evidence is not a task
success rate. Model Zoo manifest rows remain manual with no published assets.
Source acquisition does not imply board/model readiness.

Ruling: preserve the complete upstream Git layouts rather than rewriting nested
README or policy code to satisfy the in-repository sample template. Refactor is
not-applicable for these fixed external source integrations; their parent
integration has dedicated tests for gitlink mode/pin, .gitmodules URL/path,
guide/pin presence, manual manifest state and absence of duplicate old gitlinks.
This is a separate source-integration contract, not a rule exemption for native
samples or a board-pass claim. Changing an upstream pin requires intentional
review. AGENTS.md documents the distinction for future migration work.

Two integration tests failed before relocation/registration and pass afterward.
[Evidence](evidence/2026-09-28-b11-vla-integration/checks.json) records initialized
HEADs, clean status, source file counts, all outer-guide relative links and the
host commands. The ordinary 49-sample migration contract still passes with zero
violations and zero exemptions. No upstream code, package installer, conversion,
model inference, robot-control command or remote-board action was executed.

VLA source integration and outer guides are author-complete. Review remains
not-run and Closed=no. Gemma/MiniCPM runtime migration, H8 shared/resource work,
remaining author audits and final independent whole-branch review remain open.
