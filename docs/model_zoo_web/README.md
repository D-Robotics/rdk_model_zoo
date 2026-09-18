# RDK Model Zoo Web

This directory contains the standalone source for the static RDK Model Zoo
website. It is the primary catalog UI source in this repository.

```text
src/       catalog UI and empty report-library shell
public/    brand logo and interface icons
scripts/   deterministic build and validation scripts
dist/      generated deployment artifact; ignored by Git
```

The formal model migration has not started. The generated site intentionally
contains zero models, zero downloadable model artifacts, zero benchmark
records, and zero OE reports. It does not import model metadata from another
repository directory.

The catalog header does not expose an OE report entry. The model-detail view
keeps a non-interactive report placeholder until the report URL and hosting
contract are agreed; do not add an empty `href`, a fake `#` target, or a link
to local report payloads.

```bash
npm ci --prefix docs/model_zoo_web
npm --prefix docs/model_zoo_web run check
```

Compiled models, conversion outputs, and complete OE reports must not be
committed here. When migration begins, their stable OSS URLs and checksums can
be supplied through a reviewed catalog-generation step. This project also does
not define or own a GitHub Pages deployment.
