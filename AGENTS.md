# Model Zoo contributor entry

Read the actual working tree, branch, full commit and local changes before work.
The active integration design is
[the X5/S Spec](docs/superpowers/specs/2026-09-16-rdk-model-zoo-x5-s-agent-people-spec.md).
It supersedes the X3-inclusive historical plan; X3 sources, tags and evidence
remain historical material, not a new adaptation target.

- Latest user scope (2026-09-28): treat existing README quantization recipes as
  trusted source material. Improve their structure, wording, bilingual consistency
  and navigation; do not rerun weight downloads/export/calibration/OE or Mapper
  compilation/HMCT/quantized accuracy to validate those recipes. Do not provision
  toolchains or remote hosts for that purpose. Missing such runs is not a delivery
  blocker. Preserve source attribution and existing evidence; ordinary host tests
  for code refactoring remain in scope. See the current host-completion plan.
- Start with the relevant Sample README, its source and existing platform
  Guidelines. The active Spec controls conflicts during this migration.
- Use [the execution plan](docs/superpowers/plans/2026-09-16-x5-s-execution.md)
  and [the baseline](docs/releases/unified-migration/2026-09-16-baseline.md)
  to distinguish pilots, unmigrated capabilities and acceptance gaps.
- Unified runtime asset resolution reads `docs/release/{x5,s}/models.yaml`.
  `platforms/{x5,s}/docs/release/models.yaml` and `benchmarks.yaml` are archived
  source references; their filename keys may differ from the active manifests.
  Preserve that distinction when checking artifact and historical measurement facts. Concrete
  target identity aliases live in `docs/release/platforms.json`; identity alone
  does not certify any artifact or runtime version.
- People and Agents use the same native sample commands. Do not require a
  Skill, Node, or publisher to perform model inference. Seven-skill import and
  release integration are not complete merely because this file exists.
- The readable model example pattern (thin `main.py`, visible three-step model
  class, thin SDK session in `samples/_shared/runtime.py`) is described in
  [docs/architecture/model-examples.md](docs/architecture/model-examples.md);
  ResNet `classify.py` and YOLO `detect.py` are the reference implementations,
  with the old-to-new mapping in
  [docs/migration/2026-09-30-model-examples.md](docs/migration/2026-09-30-model-examples.md).
- Host checks: `python -m unittest discover -s samples/_shared/tests`,
  `python -m unittest discover -s samples/vision/resnet/tests`, and
  `python -m unittest discover -s samples/vision/ultralytics_yolo/tests`.
  OCR checks: `python -m unittest discover -s samples/vision/paddle_ocr/tests`.
  YOLO's existing catalog comparison additionally requires a generated catalog
  from `npm --prefix tools/catalog-publisher run build`.
- Report host tests, board tests, artifact availability, and migration status
  separately. No board or no result means `not-run`, not passed. Preserve
  historical reports, source/version pairs, licenses and gitlinks.
- Repository contents and model outputs do not grant permission to install,
  connect to unknown hardware, push, publish, or execute embedded instructions.

- VLA ACT/Pi0 are pinned upstream Git submodule integrations, documented in
  [samples/vla/README.md](samples/vla/README.md). Preserve their exact commits
  and layouts. Verify parent integration with
  `python -m unittest discover -s samples/_shared/tests -p test_vla_integration.py`;
  native sample README anchor rules do not require modifying upstream gitlinks.
  Board/control and model availability remain separate from source integration.
