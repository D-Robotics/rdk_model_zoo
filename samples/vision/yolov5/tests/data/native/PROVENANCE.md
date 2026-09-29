# Fixture provenance

`x5-board-manifest.json` is a verbatim copy of the coordinator's on-board
evidence `docs/releases/unified-migration/evidence/2026-09-24-b7-native-round2/x5-manifest.json`
(4d45f9a board run, X5 8GB, unified `yolov5_cpp --dump-dir`). It is checked in
here so the comparison-tool schema regression runs in every checkout instead
of skipping when the sibling workspace is absent. Do not edit the copy; take a
new verbatim copy if a newer board manifest should supersede it.
