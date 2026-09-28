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

All current platform sources are read from `main` through `sources.json`: `platforms/x5`, `platforms/s`, and `platforms/x3`. New models, fixes and measurements are maintained there. Historical platform branches remain compatibility references.

## Data identity and historical releases

The catalog format version is independent of platform versions. `catalog-v1.0.0-<content fingerprint>` changes when any emitted model, observation or source provenance changes, even if platform release tags have not changed. The metadata SHA256 checks the complete serialized artifact. Platform tags remain recorded separately.

Historical annotated tags can be selected per platform; their original root directory layout is preserved:

```sh
npm run catalog:build -- --pin x5=x5-v1.1.2 --out dist/historical
```

For future tags containing the new layout, use `--pin x5=<annotated-tag>:platforms/x5`. The same override works for `s` and `x3`. Do not rewrite published tags or rename existing download URLs as part of a source layout migration.

## Import into the documentation repository

After the source check succeeds, run from `model_zoo_doc/catalog`:

```sh
npm run catalog:import -- /path/to/rdk_model_zoo/tools/catalog-publisher/dist
npm run check
```

Commit the imported snapshot and lock in the documentation repository. A normal docs build verifies its own checked-in snapshot and never reads a sibling checkout or generates model data.

The main CI workflow uploads the two artifact files for non-PR runs. Enabling that workflow remotely and publishing the website are separate actions; this migration only prepares the local files. The old website redirect is in `../catalog-redirect`.

## Adding hardware

Add a complete platform tree and registry entry, then extend `sources.json`, catalog platform/hardware types and normalization mappings with tests. Choosing a new featured board is a presentation choice; it does not require changing the main branch or moving the dashboard.

Static manifest checks do not replace board inference testing. Missing measurements remain missing; no values are synthesized.
