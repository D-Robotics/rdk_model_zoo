English | [简体中文](README_cn.md)

# Official precompiled models

[Sample](../README.md)

<a id="source-model"></a>
## Source models

The precompiled packages come from the OELLM SDK 2.0.0 S600 release. Upstream models are Google Gemma4 E2B/E4B; the ZIP supplies no full HF revision, so the manifests retain null.

<a id="directory"></a>
## Directory

This directory documents precompiled provenance. Four complete manifests are in `../model/manifests/`; matching SDK configurations are in `../runtime/cpp/configs/`.

<a id="toolchain-targets"></a>
## Toolchain and targets

S600/nash-p, four cores, chunk 512 and context 4096. Each manifest records its actual HBDK compiler and HBM metadata; execution uses the matching SDK 2.0.0 Runtime.

<a id="export"></a>
## Export

The active entry uses exported vendor HBMs directly. Custom export requires a separate experiment based on the SDK Gemma recipe.

<a id="calibration"></a>
## Calibration and quantization

The SDK Gemma route is AutoClip/GPTQ → HPP → HBDK. These ZIPs do not provide calibration corpora or full reproduction records; calibration and accuracy from the retired self-converted package are not assigned to them.

<a id="compile"></a>
## Compile

The vendor HBMs need no recompilation. Changing core count, context or chunk size requires a new SDK conversion and matching configuration.

<a id="validation"></a>
## Artifact validation

Follow [model preparation](../model/README.md#preparation) to verify the ZIP and every member SHA256. This step does not execute board inference.

<a id="artifacts"></a>
## Artifacts

Vendor ZIP bytes remain unchanged. Preparation adds absolute-path `oellm_config.json`, `model-manifest.json` and `SAMPLE_LICENSE`. The self-converted W4/W8 package is retired from the active binding.

<a id="known-gaps"></a>
## Preparing custom conversions

For re-quantization, separately pin the upstream revision, SDK recipe, calibration set, accuracy results and new artifact digests. This sample delivers the vendor precompiled route.
