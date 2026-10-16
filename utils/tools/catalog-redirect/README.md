English | [简体中文](README_cn.md)

# Catalog redirect artifact (local only, not deployed)

`https://d-robotics.github.io/rdk_model_zoo/` was the published dashboard URL. It is a single-page application, so every link it ever handed out is a query string on that one path — `?model=…`, `?task=…`, `?benchmark=performance`, and combinations of them. Those links exist in shared chats, issue trackers, and release notes and cannot be recalled.

This directory preserves an earlier proposal to forward the dashboard to
`/model_zoo_doc/models/`. The current website is maintained in
[`model_zoo_web/`](../../../model_zoo_web/README.md) on `main` and continues
to use `/rdk_model_zoo/`.

`index.html` in this directory is that forwarder.

## Status: not published

**No workflow in this repository deploys this file, and none should without an explicit decision.** It is kept here as a reviewable artifact so the redirect can be published deliberately later:

- The `rdk_model_zoo` GitHub Pages site has to keep serving *something* at its root, which means a deployment to that Pages environment; this changes how the repository publishes, beyond the data-build workflow.
- Publishing it is a user-visible, outward-facing action that affects a live URL.

The dedicated `model-zoo-web-pages.yml` workflow publishes the website.
`model-catalog-data.yml` still builds and uploads the separate catalog data
package. Neither workflow deploys this redirect.

## How to publish it (when approved)

The redirect must be served from the root of the old Pages site, with `index.html` at the path prefix `/rdk_model_zoo/`. Configure the Pages source for that site to publish this directory's contents as its root, then verify a live deep link such as

```
https://d-robotics.github.io/rdk_model_zoo/?model=yolov8&benchmark=performance
```

lands on the equivalent dashboard view with the query string intact. The target base URL is a single constant at the top of the inline script in `index.html`; update it if the documentation site's base path changes.

## Why a redirect instead of a copy of the old site

The proposal accompanied the catalog data publisher used by the documentation
site. That publisher and `docs/release/` are retained independently of the
website, which consumes its own `model_zoo_web/data/` records.
