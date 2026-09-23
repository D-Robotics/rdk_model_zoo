English | [简体中文](./README_cn.md)

# Model Artifacts — SigLIP

<a id="artifacts"></a>
## Artifacts

Each row is one manifest asset. The `s100/` path is the release storage identity used for both supported S100 (Nash-E) and S100P (Nash-M); it is not a fallback that removes S100P support. Each HBM is a packed two-submodel vision feature artifact.

| Artifact | Format | Target(s) | Stage(s) | Source |
| --- | --- | --- | --- | --- |
| `s100/bpu-siglip-base-patch16-224.hbm` | HBM | s100, s100p | `pooler_output`, `last_hidden_state` | download |
| `s100/bpu-siglip-base-patch16-384.hbm` | HBM | s100, s100p | `pooler_output`, `last_hidden_state` | download |
| `s100/bpu-siglip-base-patch16-512.hbm` | HBM | s100, s100p | `pooler_output`, `last_hidden_state` | download |
| `s100/bpu-siglip-large-patch16-256.hbm` | HBM | s100, s100p | `pooler_output`, `last_hidden_state` | download |
| `s100/bpu-siglip-large-patch16-384.hbm` | HBM | s100, s100p | `pooler_output`, `last_hidden_state` | download |
| `s100/bpu-siglip-so400m-patch14-224.hbm` | HBM | s100, s100p | `pooler_output`, `last_hidden_state` | download |
| `s100/bpu-siglip-so400m-patch14-384.hbm` | HBM | s100, s100p | `pooler_output`, `last_hidden_state` | download |
| `s100/bpu-siglip-so400m-patch16-256-i18n.hbm` | HBM | s100, s100p | `pooler_output`, `last_hidden_state` | download |

The exact manifest URLs are:

| Variant | Release URL |
| --- | --- |
| `base-patch16-224` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-base-patch16-224.hbm> |
| `base-patch16-384` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-base-patch16-384.hbm> |
| `base-patch16-512` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-base-patch16-512.hbm> |
| `large-patch16-256` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-large-patch16-256.hbm> |
| `large-patch16-384` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-large-patch16-384.hbm> |
| `so400m-patch14-224` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-so400m-patch14-224.hbm> |
| `so400m-patch14-384` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-so400m-patch14-384.hbm> |
| `so400m-patch16-256-i18n` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-so400m-patch16-256-i18n.hbm> |

<a id="preparation"></a>
## Preparation

Run from the repository root. The script has no implicit default download in the runtime. It accepts `--target s100|s100p`, `--variant`, and `--output-dir`; the shell wrapper accepts target then variant as positional arguments.

```bash
# cwd: repository root; source: exact URL recorded in docs/release/s/models.yaml
python3 samples/vision/siglip/model/download.py --target s100 --variant base-patch16-224
# expect: samples/vision/siglip/model/s100/bpu-siglip-base-patch16-224.hbm and an observed SHA-256

# cwd: repository root; the same published bytes are used on S100P
bash samples/vision/siglip/model/download.sh s100p so400m-patch14-384
# expect: samples/vision/siglip/model/s100/bpu-siglip-so400m-patch14-384.hbm
```

The manifest has no publisher SHA-256 values. The downloader prints the observed digest and explicitly reports that it cannot verify origin. On I/O, selection, or download errors it exits 2. If the primary route is unavailable, manually place the exact HBM from the release URL at the path above and pass both `--asset-id s:siglip:s100/bpu-siglip-<variant>.hbm` and `--model-path <path>` to the runtime; manual provenance remains unverified.

<a id="accompanying-files"></a>
## Accompanying Files

| File | Role | Required |
| --- | --- | --- |
| `download.py` | Resolves manifest identity and downloads one selected HBM. | No, if artifacts are already present; required for scripted preparation. |
| `download.sh` | Positional compatibility wrapper for `download.py`. | No. |
| `../test_data/dog.jpg` | Runtime smoke input; it is not part of the HBM. | No, only for the sample CLI. |

<a id="local-paths"></a>
## Local Paths

- Artifacts: `samples/vision/siglip/model/s100/bpu-siglip-<variant>.hbm`.
- Runtime default `--model-path`: the same `model/s100/bpu-siglip-base-patch16-224.hbm` path selected by `resolve_selection`.
- The explicit `--model-path` option must be paired with its exact manifest `--asset-id`.

<a id="formats-checksums"></a>
## Formats & Checksums

The source inventory and release manifest record all publisher hashes as unknown. No hash is copied between artifacts.

| Artifact | Format | SHA-256 | Source of value |
| --- | --- | --- | --- |
| `s100/bpu-siglip-base-patch16-224.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `s100/bpu-siglip-base-patch16-384.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `s100/bpu-siglip-base-patch16-512.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `s100/bpu-siglip-large-patch16-256.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `s100/bpu-siglip-large-patch16-384.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `s100/bpu-siglip-so400m-patch14-224.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `s100/bpu-siglip-so400m-patch14-384.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `s100/bpu-siglip-so400m-patch16-256-i18n.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |

Release URLs are the exact `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-<variant>.hbm` entries in that manifest. Formal download was not executed for this migration.

## License

The sample preparation code is Apache-2.0 under the repository [LICENSE](../../../../LICENSE). The source release does not record a per-artifact model license or weight version; this README therefore does not invent one.
