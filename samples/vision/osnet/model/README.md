English | [简体中文](README_cn.md)

# OSNet model preparation

<a id="artifacts"></a>
## Artifacts

Four technically accepted OSNet x1.0 models are hosted on Archive. Models, manifests and OE reports passed authenticated readback and anonymous HTTPS verification on 2026-11-02. Native download URLs are registered in `docs/release/{x5,s}/models.yaml`; models remain research-only. Source integration and formal website release approval are pending. [Upload receipt](../evaluator/evidence/2026-11-02-archive-upload.json).

| Target | Variant | File |
| --- | --- | --- |
| x5 | spatial-rgb-late8 | spatial-rgb-late8.bin |
| s100 | s100-f32-int16 | osnet_x1_0_nashe_256x128_f32_int16.hbm |
| s100p | s100p-f32-int16 | osnet_x1_0_nashm_256x128_f32_int16.hbm |
| s600 | s600-f32-int16 | osnet_x1_0_nashp_256x128_f32_int16.hbm |

Other historical X5 identities remain in artifacts.json; failed INT8 experiments are not recommended release models.

<a id="directory"></a>
## Directory

`download.py` / `download.sh`: preparation entry; `artifacts.json` records identity and `artifacts.py` checks SHA-256. Model binaries are not committed.

<a id="preparation"></a>
## Preparation

Run at repository root and select the target's recommended model:

```bash
python3 samples/vision/osnet/model/download.py --target x5 --download
python3 samples/vision/osnet/model/download.py --target s600 --download
```

Existing bundles can also be imported offline. `--candidate-dir` accepts the delivery root or one platform bundle directory. `--target` selects its recommended variant; `--variant` can select an explicit variant for that target.

```bash
python3 samples/vision/osnet/model/download.py --target x5 \
  --candidate-dir /absolute/path/accepted-delivery
```

```bash
python3 samples/vision/osnet/model/download.py --target s600 \
  --candidate-dir /absolute/path/accepted-delivery/s600
```

`--download --target TARGET` reads the native manifest and verifies its pinned SHA-256. The public X5 filename differs from the local `spatial-rgb-late8.bin` cache name; their bytes are identical. Existing X5 private ZIP/TAR imports remain available:

```bash
python3 samples/vision/osnet/model/download.py --bundle /absolute/path/x5-readback.tar \
  --format readback --target x5 --fixtures
```

<a id="accompanying-files"></a>
## Accompanying files

Candidate preparation and published downloads copy models only. Python takes a permitted person crop via `--test-img`. C++ verification separately requires images.dat and 32×512 raw references captured from the selected model; Market1501 fixtures remain private research data.

<a id="local-paths"></a>
## Local paths

Default destination: `samples/vision/osnet/model/<registered filename>`. Use `--output-dir` to prepare elsewhere and `--model-path` at inference. Preparation does not identify hardware or execute inference; execution validates the board.

<a id="formats-checksums"></a>
## Formats and checksums

Candidate platform, SHA-256 and actual model bytes must match artifacts.json. A delivery filename may differ from the registered local cache name. Different existing files are not replaced. Published downloads use the same protection in the shared assets utility.

See [artifacts.json](artifacts.json) for full identities and failed experiments, and the [license review](../third_party/LICENSE_REVIEW.md) for scope.
