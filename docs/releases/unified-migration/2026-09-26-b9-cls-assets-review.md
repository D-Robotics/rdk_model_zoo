# B9 prerequisite — S classification filename adjudication

Status: host evidence captured; B9 consolidation remains pending.
Board=not-run; real SDK/OE=not-run; independent Review=not-run; Closed=no.
This is author investigation, not an independent acceptance review.

## Decision and authoritative inputs

The active shared resolver reads `docs/release/s/models.yaml`. For the twenty
S100/S100P YOLOv8/YOLO11 classification assets (five sizes per family/target),
its qualified identities contain `640x640` while URLs contain `224x224`.
`platforms/s/docs/release/models.yaml` is an older archive with 224 filename keys;
it must not be mistaken for the active resolver input. Catalog errata separately
normalizes display names and preserves its compatibility explanation.

The initial investigation incorrectly assumed the archived manifest was the
runtime authority. An initial test expecting 224 filename keys therefore failed;
that failure is not evidence of a product regression. A second initial test
incorrectly passed arguments to the no-argument downloader main function; it was
corrected to patch argv. No asset resolution behavior needed changing.

Keep active qualified IDs and local filenames unchanged. Correct bilingual sample
and model READMEs and resolver documentation to distinguish filename tokens from
physical input geometry. Actual geometry remains governed by runtime metadata.
The two additional tests cover all twenty exact manifest resolutions and the
public downloader dry-run. No test is skipped when an advertised asset is absent.
AGENTS now identifies the active manifest and archived sources separately.
LaneNet/DiffusionDrive model README links were corrected to the active manifest;
their evidence file hashes were refreshed without rewriting prior run logs.

## Public HTTP observation

[Raw HEAD results](evidence/2026-09-26-b9-cls-assets/head-results.json) retain UTC,
requested/final URLs, status and response headers for forty requests: twenty
manifest 224 URLs plus twenty historical 640 aliases. All returned 200; each pair
had equal Content-Length and ETag. The
[capture script](evidence/2026-09-26-b9-cls-assets/probe.py) can be rerun explicitly
from a networked host; it uses HEAD only and downloads no model bodies.

Equal headers do not prove cryptographic byte identity, input dimensions or board
execution. Therefore neither address is declared the sole true name, and no HBM
shape is inferred. The previous claim that the source was halfway through a
rename was an interpretation, not established server state. Availability removes
the naming investigation as a prerequisite blocker for B9 source consolidation;
it does not accept that consolidation or validate an actual model.

## Host verification

[Commands and return codes](evidence/2026-09-26-b9-cls-assets/host-results.json)
and adjacent complete logs record:

- Ultralytics 80, shared 144, ResNet 52, OCR 44, checker 27: **347 tests passed**.
- LaneNet and DiffusionDrive bilingual README link/parser/API fixture checks passed.
- [Migration contract report](evidence/2026-09-26-b9-cls-assets/contracts.json):
  44 samples, zero violations, zero exemptions. Existing CLI policy skips are
  disclosed by the report and are not task-stage purity proof.

No board, remote PC, SSH, real vendor SDK, compiler toolchain, model body download,
dataset accuracy or performance run occurred. Existing unit tests use fixtures;
printed download messages from mocked tests are not real network transfers.
Next work remains the standalone S YOLO source-capability consolidation, task-stage
purity audit and YOLOE migration. All other open host plan work remains open.
