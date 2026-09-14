# S-series release manifest

This directory contains the S-series model and benchmark inventory for the
`s-v1.1.1` release. The manifests are derived from the `rdk_s` repository
contents and the model download scripts present in the release source tree.

For the Ultralytics entry, this release aligns 94 existing model filenames and
URLs with the committed `yolov8`/`yolov9`/`yolov10` naming, removes the two
unsupported `yolo9n` assets, and records the two published `yolov9t` S100 and
S100P assets. These are source-backed metadata corrections; benchmark values
are unchanged.

`models.yaml` records one entry per releasable sample and every model artifact
exposed by its committed download helper. An asset with `sha256: null` has no
trusted digest recorded in the repository; `null` does not mean that the file
was verified. A local file copied by a download helper is retained as an asset
with no invented URL.

`benchmarks.yaml` records numeric performance and accuracy values already
published in the repository's model READMEs, evaluator notes, or conversion
records. Each record points to an immutable commit and an exact Markdown
heading. The 473 evidence records in this release use commit
`53d924f4c88175ec77634d24e5d711a3a0901eb6`, resolved from the immutable
`s-v1.1.0` tag. Missing conditions are left out rather than inferred.

The inventory includes ACT and Pi0 as manual model entries but omits the
contents of their external gitlinks. External gitlinks are not counted as
S-series model assets. The source tree itself is unchanged by that omission.

This release is a documentation and manifest update. It does not certify
runtime behavior, board compatibility, or a repository-wide board test result;
no board tests were run for this release. The public GitHub Release attaches
both `models.yaml` and `benchmarks.yaml`.

See the [S v1.1.1 release notes](../releases/s-v1.1.1.md) for the audited
totals and validation scope. Current branch location: `docs/release/`.
Historical tags such as `s-v1.0.0` keep their original `release/` paths. See
the [documentation index](../README.md) for the shared release policy and
website maintenance instructions.
