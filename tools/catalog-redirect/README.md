# Catalog redirect artifact (local only, not deployed)

`https://d-robotics.github.io/rdk_model_zoo/` was the published dashboard URL. It is a single-page application, so every link it ever handed out is a query string on that one path — `?model=…`, `?task=…`, `?benchmark=performance`, and combinations of them. Those links exist in shared chats, issue trackers, and release notes and cannot be recalled.

The dashboard now ships as a static module of the documentation site at `/model_zoo_doc/models/`. The query vocabulary did not change, so forwarding the query string and fragment verbatim preserves every deep link.

`index.html` in this directory is that forwarder.

## Status: not published

**No workflow in this repository deploys this file, and none should without an explicit decision.** It is kept here as a reviewable artifact so the redirect can be published deliberately later:

- The `rdk_model_zoo` GitHub Pages site has to keep serving *something* at its root, which means a deployment to that Pages environment — a change to how this repository publishes, not a data build.
- Publishing it is a user-visible, outward-facing action that affects a live URL.

The model repository deliberately has no Pages deployment left: `.github/workflows/model-catalog-data.yml` builds and uploads the catalog data package and stops there.

## How to publish it (when approved)

The redirect must be served from the root of the old Pages site, with `index.html` at the path prefix `/rdk_model_zoo/`. Configure the Pages source for that site to publish this directory's contents as its root, then verify a live deep link such as

```
https://d-robotics.github.io/rdk_model_zoo/?model=yolov8&benchmark=performance
```

lands on the equivalent dashboard view with the query string intact. The target base URL is a single constant at the top of the inline script in `index.html`; update it if the documentation site's base path changes.

## Why a redirect instead of a copy of the old site

Keeping two copies of the dashboard would mean two generators, two data pipelines, and two places for the numbers to drift. The dashboard now consumes a checksummed data package published by `tools/catalog-publisher`, and that package has exactly one consumer.
