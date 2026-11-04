English | [简体中文](README_cn.md)

# Vendor precompiled artifacts

<a id="source-model"></a>
## Model sources

Use the vendor S600 ZIPs in the [inventory](../model/README.md#artifacts). Upstream identity and per-variant licenses are recorded in each manifest; undisclosed full HF revisions remain null.

<a id="directory"></a>
## Directory

This directory explains vendor provenance; hashes are in `../model/manifests/` and SDK configurations in `../runtime/cpp/configs/`.

<a id="toolchain-targets"></a>
## Toolchain and targets

SDK 2.0.0 / S600 nash-p. Actual HBDK compiler, cores, prefill chunk and context are recorded per package rather than copied from another size.

<a id="export"></a>
## Export

Use the exported vendor HBMs directly.

<a id="calibration"></a>
## Calibration

Released packages need no new calibration. They do not supply complete calibration corpora or experiment logs; accuracy/PPL from other artifacts is not assigned to them.

<a id="compile"></a>
## Compile

HBMs are precompiled. Changing target, cores, context or chunk size requires a separate SDK conversion and new artifact identity.

<a id="validation"></a>
## Artifact validation

Follow [preparation](../model/README.md#preparation) to verify the ZIP and every member. HBM metadata is read on the toolchain host without graph execution.

<a id="artifacts"></a>
## Artifacts

OSS ZIPs retain vendor bytes; this repository supplies manifests, native launchers and deployment configurations.

<a id="known-gaps"></a>
## Custom conversion preparation

Custom quantization needs separately pinned upstream revision, SDK recipe, calibration set and artifacts; do not relabel an artifact and reuse vendor performance.
