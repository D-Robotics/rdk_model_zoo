# S-series release manifest

This directory freezes the S-series model and benchmark inventory for
`s-v1.0.0`. The manifests are derived from the `rdk_s` repository contents and
the model download scripts that exist at the release source ref.

`models.yaml` records one entry per releasable sample and every model artifact
exposed by its committed download helper. An asset with `sha256: null` has no
trusted digest recorded in the repository; `null` does not mean that the file
was verified. A local file copied by a download helper is retained as an asset
with no invented URL.

`benchmarks.yaml` records numeric performance and accuracy values already
published in the repository's model READMEs, evaluator notes, or conversion
records. Each record points to an immutable release ref and an exact Markdown
heading. Missing conditions are left out rather than inferred.

The inventory intentionally omits the ACT/Pi0 external gitlink samples.
External gitlinks are not counted as S-series model assets in this baseline.
The source tree itself is unchanged by that omission.

This release is a documentation and manifest baseline. It does not certify
runtime behavior, board compatibility, or a repository-wide board test result.
The public GitHub Release attaches both `models.yaml` and `benchmarks.yaml`.

Current branch location: `docs/release/`. Historical tag `s-v1.0.0` keeps
its original `release/` paths. See the [documentation index](../README.md)
for the shared release policy and website maintenance instructions.
