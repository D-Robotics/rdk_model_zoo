[English](README.md) | [简体中文](README_cn.md)

# Model files

<a id="artifacts"></a>
## S600 published archive

Archive: `minicpm5-2b_s600_oellm2_w8_ctx4096_20260908.tar.gz`

SHA256: `8f2bef6fc7d2290f05055570e7dcfb1cf4e8c07c9a6bb0d0acc24dd202a83841`

Size: 2457817532 bytes.

[Download model](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/minicpm5-2b_s600_oellm2_w8_ctx4096_20260908.tar.gz)

<a id="preparation"></a>
## Explicit preparation

Run from this directory on the host or board. Allow 6 GB free space. The helper verifies the archive, a pinned SHA256SUMS manifest, and every extracted member. An existing directory is reused only if all checks pass; a different existing directory is left intact and reported as an error.

```bash
BOARD=s600 bash download_model.sh
# Optional destination; the checksum remains fixed.
MODEL_DIR=/data/minicpm5-s600 bash download_model.sh
```

<a id="accompanying-files"></a>
## Accompanying files

The default destination is `model/s600`. It contains the HBM, FP16 embedding, tokenizer files, model metadata, LICENSE, NOTICE and MODEL_INFO.json. SDK libraries are not included. This HBM is S600/Nash-p only.

## S100 / S100P

```bash
BOARD=s100 bash download_model.sh
BOARD=s100p bash download_model.sh
```

Each target has its own HBM and tokenizer, extracted to `model/s100` or `model/s100p`. Use the [legacy runtime](../runtime/legacy/README.md), not the S600 2.0 entry point. SDK files are excluded. The script pins archive and manifest SHA256 hashes and verifies every extracted file.

- [S100 archive](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/minicpm5-2b_s100_oellm1_w8_ctx4096_20260909.tar.gz): 2275420219 bytes; SHA256 `dc130ae21dc1f1cc266c526e090b03b4d3be4af8cf4157c4b5841b7f47ab818d`.

- [S100P archive](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/minicpm5-2b_s100p_oellm1_w8_ctx4096_20260909.tar.gz): 2270216392 bytes; SHA256 `65c89ae7ead48711a392fa556f602358ffffe16438e94051d1f8cca46d4e096a`.

<a id="local-paths"></a>
## Local layout

```text
model/<board>/
  minicpm5-2b_ctx4096_<board>.hbm
  tokenizer/
  LICENSE
  NOTICE
  MODEL_INFO.json
  SHA256SUMS
```
<a id="formats-checksums"></a>
## Formats, checksums and overrides

HBM is the board model; the S600 FP16 embedding is companion data. Tokenizer and metadata must match the archive. The helper checks archive SHA-256, the pinned SHA256SUMS digest and files named by that manifest; directory presence alone is not success. Failure preserves an existing destination.

| Environment | Behavior |
| --- | --- |
| `BOARD` | `s100` / `s100p` / `s600`; source default is s600; select explicitly |
| `MODEL_DIR` | Defaults to `<board>` below this directory; use separate target directories |
| `MINICPM5_MODEL_URL` | Override archive URL without changing pinned hashes; not for an arbitrary new model |

Requires Bash, curl, tar and sha256sum. The SDK is not included. Versions, sizes and digests above are the source release's published records.
