English | [简体中文](README_cn.md)

# Catalog data publisher

This module belongs to `rdk_model_zoo`. It is the sole generator of the dashboard data consumed by `model_zoo_doc/catalog`.

## Local development

Use Node.js 22.12 or later in the Node 22 series.

```sh
cd scripts/tools/catalog-publisher
npm ci
npm run check
```

The check validates platform manifests and source references, runs tests and TypeScript checks, then rebuilds and verifies a reproducible artifact. Output is `dist/catalog.json` plus `dist/catalog.meta.json` (SHA256, byte length, data version and provenance).

## Data sources

`sources.json` defines one input per platform. X5 and S use `worktree` mode and read the checked-out manifests. X3 uses `commit` mode at `6fcef2b87c12435e11fbd7327ea70d4efd917b1c`; Git reads that tree with `git show`. Full history is required. `tag` mode selects an immutable annotated release tag; a missing object reports the required `git fetch origin <sha>` command.

| Platform | Manifests read from the worktree | Version file | Generated source links |
| --- | --- | --- | --- |
| `x5` | `docs/release/x5/models.yaml`, `benchmarks.yaml` | `docs/release/x5/VERSION` | `blob/<resolved commit>/...` |
| `s` | `docs/release/s/models.yaml`, `benchmarks.yaml` | `docs/release/s/VERSION` | `blob/<resolved commit>/...` |
| `x3` | `6fcef2b8…/platforms/x3/release/models.yaml`, `benchmarks.yaml` | `6fcef2b8…/platforms/x3/VERSION` | `blob/6fcef2b87c12435e11fbd7327ea70d4efd917b1c/platforms/x3/...` |

Maintain X5 and S models and benchmarks in `docs/release/{x5,s}` with the matching per-platform `VERSION`. The build checks that each version file agrees with its manifest release version. X3 catalog inputs use the commit tree configured in `sources.json`.

### Worktree reads versus generated links

`link_ref`/`link_prefix` do not select what is read. They only label what the artifact emits: the per-record `source_ref`/`source_path_prefix` (rendered by consumers as `blob/<ref>/<prefix>/...` repository links) and the per-platform `ref` in `catalog.meta.json` provenance. A worktree build therefore reads the current checkout while its links name an immutable ref; a `--pin` build reads the pinned tag's tree instead, and its links name that tag.

For worktree sources, `sources.json` configures `link_ref: "HEAD"`, and the loader resolves it — before anything is emitted — to the complete 40-hex commit the checkout actually builds from (`git rev-parse HEAD^{commit}`). Generating on `develop`, on `main` or from any PR checkout therefore yields links to that exact commit; the artifact never carries the literal `HEAD`, a branch name, or any other moving target. A checkout without Git context, or a configured ref that does not resolve, fails the build explicitly. Historical sources are unaffected: the X3 commit pin and every `--pin` tag keep their original immutable links. Ref resolution changes only these labels — never a model, asset or benchmark value — and what was actually read stays pinned by the per-platform `manifest_sha256` in the provenance, which is what reproducibility is checked against.

## Data identity and historical releases

The catalog format version is independent of platform versions. `catalog-v1.0.0-<content fingerprint>` changes when any emitted model, observation or source provenance changes, even if platform release tags have not changed. The metadata SHA256 checks the complete serialized artifact. Platform tags remain recorded separately.

Historical annotated tags can be selected per platform; their original root directory layout is preserved:

```sh
npm run catalog:build -- --pin x5=x5-v1.1.2 --out dist/historical
```

For a tag whose tree carries the platform under a `platforms/<id>` prefix, name that prefix after a colon: `--pin x5=<annotated-tag>:platforms/x5`. The same override works for `s` and `x3`.

A pinned build resolves each platform's VERSION from the layout its tag actually carries — colocated with the manifest directory, or at the platform root in legacy layouts; the worktree's configured `version_file` is never applied to a tag. A tag that publishes manifests but no VERSION is rejected, and the resolved version must agree with the manifest's release version. Do not rewrite published tags or rename existing download URLs when the source layout changes.

## Import into the documentation repository

After the source check succeeds, run from `model_zoo_doc/catalog`:

```sh
npm run catalog:import -- /path/to/rdk_model_zoo/scripts/tools/catalog-publisher/dist
npm run check
```

Commit the imported snapshot and lock in the documentation repository. A normal docs build verifies its own checked-in snapshot and never reads a sibling checkout or generates model data.

The workflow `.github/workflows/model-catalog-data.yml` runs `npm run check` for pull requests, pushes to `develop` and `main`, and manual dispatch. Push and manual runs upload `catalog.json` and `catalog.meta.json` as `model-catalog-data`; pull requests run validation. Website import follows the documentation-repository commands above.

## Adding hardware

Add a complete platform distribution, then extend `sources.json`, catalog platform/hardware types and normalization mappings with tests. Choosing a new featured board is a presentation choice; it does not require moving the dashboard or relocating the maintained manifests.

Static manifest checks do not replace board inference testing. Missing measurements remain missing; no values are synthesized.
