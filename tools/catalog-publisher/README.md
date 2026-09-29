# Catalog data publisher

This module belongs to `rdk_model_zoo`. It is the sole generator of the dashboard data consumed by `model_zoo_doc/catalog`.

## Local development

Use Node.js 22.12 or later in the Node 22 series.

```sh
cd tools/catalog-publisher
npm ci
npm run check
```

The check validates platform manifests and source references, runs tests and TypeScript checks, then rebuilds and verifies a reproducible artifact. Output is `dist/catalog.json` plus `dist/catalog.meta.json` (SHA256, byte length, data version and provenance).

## Data sources

`sources.json` defines one source per platform. All three currently use `worktree` mode: the build reads files from the checked-out repository itself — including uncommitted local changes — never from a branch, tag or remote. (`sources.json` also supports `tag` mode, which reads an immutable annotated release tag through `git show`; no source uses it by default, but per-platform pins select it, see below.) The build resolves paths from `sources.json`; `platforms/registry.json` is cross-checked against it by tests, not read by the build.

| Platform | Manifests read from the worktree | Version file | Generated source links |
| --- | --- | --- | --- |
| `x5` | `docs/release/x5/models.yaml`, `benchmarks.yaml` | `docs/release/x5/VERSION` | `blob/develop/...` |
| `s` | `docs/release/s/models.yaml`, `benchmarks.yaml` | `docs/release/s/VERSION` | `blob/develop/...` |
| `x3` | `platforms/x3/release/models.yaml`, `benchmarks.yaml` | `platforms/x3/VERSION` | `blob/main/platforms/x3/...` |

New models, fixes and measurements for X5 and S are maintained in the unified manifests `docs/release/{x5,s}` together with their per-platform `VERSION` files (the build rejects a `VERSION` that disagrees with the manifest's release version). Do not edit `platforms/{x5,s}/docs/release/*.yaml`: those are archived frozen snapshots of the `rdk_x5`/`rdk_s` delivery branches, kept as historical source references, and `sources.json` never reads them. `platforms/x3/release` is read because X3 is an archived historical distribution with no unified successor; it is not a new-adaptation target.

### Worktree reads versus generated links

`link_ref`/`link_prefix` do not select what is read. They only label what the artifact emits: the per-record `source_ref`/`source_path_prefix` (rendered by consumers as `blob/<ref>/<prefix>/...` repository links) and the per-platform `ref` in `catalog.meta.json` provenance. A worktree build therefore reads the current checkout while reporting whichever ref the links name; a `--pin` build reads the pinned tag's tree instead, and its links name that tag. An integration-branch worktree build emits `develop` links for X5/S even though its content comes from the integration branch's own files — such links do not prove that `develop` already contains those candidates; that becomes true only after the branch merges. What was actually read is pinned by the per-platform `manifest_sha256` in the provenance, which is what reproducibility is checked against.

## Data identity and historical releases

The catalog format version is independent of platform versions. `catalog-v1.0.0-<content fingerprint>` changes when any emitted model, observation or source provenance changes, even if platform release tags have not changed. The metadata SHA256 checks the complete serialized artifact. Platform tags remain recorded separately.

Historical annotated tags can be selected per platform; their original root directory layout is preserved:

```sh
npm run catalog:build -- --pin x5=x5-v1.1.2 --out dist/historical
```

For a tag whose tree carries the platform under a `platforms/<id>` prefix, name that prefix after a colon: `--pin x5=<annotated-tag>:platforms/x5`. The same override works for `s` and `x3`.

A pinned build resolves each platform's VERSION from the layout its tag actually carries — colocated with the manifest directory, or at the platform root in legacy layouts; the worktree's configured `version_file` is never applied to a tag. A tag that publishes manifests but no VERSION is rejected, and the resolved version must agree with the manifest's release version. Do not rewrite published tags or rename existing download URLs as part of a source layout migration.

## Import into the documentation repository

After the source check succeeds, run from `model_zoo_doc/catalog`:

```sh
npm run catalog:import -- /path/to/rdk_model_zoo/tools/catalog-publisher/dist
npm run check
```

Commit the imported snapshot and lock in the documentation repository. A normal docs build verifies its own checked-in snapshot and never reads a sibling checkout or generates model data.

The CI workflow (`.github/workflows/model-catalog-data.yml`) runs `npm run check` on three triggers: `pull_request` (path-filtered), pushes to `main` only (path-filtered), and manual `workflow_dispatch`. It uploads the two artifact files only for non-PR runs — a push to `main` or a manual dispatch; a pull request validates but uploads nothing. Pushes of integration work branches trigger no workflow at all, so nothing built on this branch is published anywhere by a push. Enabling that workflow remotely and publishing the website are separate actions; this migration only prepares the local files. The old website redirect is in `../catalog-redirect`.

## Adding hardware

Add a complete platform tree and registry entry, then extend `sources.json`, catalog platform/hardware types and normalization mappings with tests. Choosing a new featured board is a presentation choice; it does not require moving the dashboard or relocating the maintained manifests.

Static manifest checks do not replace board inference testing. Missing measurements remain missing; no values are synthesized.
