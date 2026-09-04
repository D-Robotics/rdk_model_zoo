# RDK X3 release manifests

`x3-v1.0.0` is a historical baseline for the `rdk_x3` branch. The branch
contains legacy, non-normalized demos and documentation, so these manifests
preserve the evidence available in that snapshot without claiming current
support or build reproducibility.

## Files

- [`models.yaml`](models.yaml) lists the 15 logical model families, 19
  downloadable external binary assets, and one manual FCOS asset.
- [`benchmarks.yaml`](benchmarks.yaml) records X3-only metrics copied from the
  exact Markdown sections named in each record's `source` object.
- [`schemas/models.schema.json`](schemas/models.schema.json) and
  [`schemas/benchmarks.schema.json`](schemas/benchmarks.schema.json) define the
  manifest shape shared by the release lines.

## Evidence rules

Only URLs explicitly present in the X3 branch are included. The inventory does
not add checksums, upstream variants, datasets, or test conditions that are not
documented by the branch. The FCOS binary is retained as a manual asset because
its download document contains a local `cp` instruction rather than a public
URL. ACT and Pi0 are not present on this branch.

The metrics are documentation transcriptions. No RDK board tests were run for
this release, and the manifests do not imply that the legacy demos are
currently buildable or validated on hardware.
