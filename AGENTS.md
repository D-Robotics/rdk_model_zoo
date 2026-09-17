# RDK Model Zoo workspace guidance

- Treat the checked-out `REPO_ROOT` and the user’s requested ref as the target. The `rdk_x5` default branch is the Skills maintenance source, not a default hardware target.
- Before platform-specific work, read the target README, relevant sample, and available Manifest/metadata. Resolve X5, S100/S100P/S600, X3, or legacy from those facts; a branch name is only a hint. Treat S-family support per sample, artifact, and runtime.
- Current manifests normally live under `docs/manifests/`; historical refs may retain `docs/release/` or `release/`. Preserve the selected ref’s layout and source attribution.
- Keep `SKILL_ROOT` (installed Skill resources) separate from `REPO_ROOT` (model repository files). Explicit user constraints win; report conflicts and do not switch branches, reset the worktree, or trigger external actions to make them fit.
- Use the smallest applicable workflow. Read-only investigation is allowed by default; downloads, board runs, remote writes, and publishing require explicit scope and authorization. Existing user/task authorization remains in force; do not ask again for the same approved action.
