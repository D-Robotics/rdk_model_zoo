# Model Download

[简体中文](./README_cn.md) | **English**

<a id="artifacts"></a>
## Published combinations

S100P (`nash-m`) and S600 (`nash-p`) each have Vision/Text HBMs; S100 (`nash-e`) has no default public HBM.
Identical filenames and layout do not imply cross-target compatibility. The current download script reuses nonempty files;
it does not automatically verify the target or SHA-256 of existing files.

<a id="preparation"></a>
## Explicit preparation

From the repository root, enter the sample model directory and select the target explicitly. Below is S600;
use `GEMMA4_SOC=s100p` and a separate data directory for S100P.

```bash
cd samples/llm/gemma4-e2b/model
export GEMMA4_HOME=~/gemma4_e2b_s600
GEMMA4_SOC=s600 bash download_model.sh --dry-run
# Run explicitly when ready to download:
GEMMA4_SOC=s600 bash download_model.sh
```

The public `rdk_s100` archive contains the validated S100P (`nash-m`)
HBMs, while `rdk_s600` contains the validated S600 (`nash-p`) HBMs. Both
are selected by explicit `GEMMA4_SOC`, not by the preparation computer’s identity. For S100 (`nash-e`), pre-place matching HBMs
under `$GEMMA4_HOME/model` or provide their directory URL explicitly:

```bash
GEMMA4_HOME=~/gemma4_e2b_s100 GEMMA4_SOC=s100 GEMMA4_MODEL_BASE_URL=https://your-server/path/to/s100/model bash download_model.sh
```

## Preparation interface

| Setting / argument | Meaning |
| --- | --- |
| `GEMMA4_SOC` | Required: `s100`, `s100p` or `s600`; no board probing or implicit S100P fallback |
| `GEMMA4_HOME` | Destination root; defaults to `~/gemma4_e2b`; use separate roots for different targets |
| `--dry-run` | Print reuse/download decisions and URLs; no network requests or directory creation |
| `--help` | Print usage without preparing files |
| `GEMMA4_MODEL_BASE_URL` | Override the target-specific HBM directory URL; needed for missing S100 HBMs |
| `GEMMA4_COMMON_MODEL_BASE_URL` | Override the common embedding directory URL |
| `GEMMA4_TOKENIZER_BASE_URL` | Override the tokenizer directory URL |

Run with Bash; actual transfers require `wget` on PATH. Preview needs neither
`wget` nor board SDKs. Unknown arguments or missing targets are errors. S100
without a custom HBM URL requires both HBMs already present, including in preview.
The helper does not validate model contents: a reused nonempty file still needs
to be the correct target. Remove stale `.part` files before switching a URL or
artifact version, because `wget -c` resumes them. Transfer failures retain partial
files and return a nonzero status; empty transfers never replace final files.

<a id="accompanying-files"></a>
## Accompanying files

The token embedding table and tokenizer are shared across all targets
and are downloaded from the common archive when missing.

<a id="local-paths"></a>
## Local paths

Files below `$GEMMA4_HOME` (default `~/gemma4_e2b` if unset):

```bash
model/gemma4-e2b_vit_ptq.hbm
model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm
model/tok_embeddings.bin
tokenizer/tokenizer.json
tokenizer/tokenizer_config.json
```

## Files

| File | Size | Description |
| --- | --- | --- |
| `model/gemma4-e2b_vit_ptq.hbm` | 329–377 MB | Platform-specific Vision encoder HBM |
| `model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm` | 4.5 GB | Text LLM HBM |
| `model/tok_embeddings.bin` | 1.5 GB | External token embedding table |
| `tokenizer/` | ~32 MB | `tokenizer.json`, chat template, config |

<a id="formats-checksums"></a>
## Integrity Check (optional)

```bash
sha256sum "$GEMMA4_HOME"/model/*.hbm
# S100P Vision: 470791849d21cffadb388cc61c8f4b1452078c1722d302fd8a8ac775ee9769f1
# S100P Text:   3e4d4940051e4e8dc0cb434e972e7aae75d49504da3fac435e303f68af73a25f
# S600 Vision:  a5998ca829cff121aa5672567b20e7be9f527da5b0220962b0fe7467bf8ff7b7
# S600 Text:    aab1831b1ea2b86763d5457890d89c55b684e4ba4834c1e008c668813d1cf646
```

The four HBM hashes above are preserved from the S source release's README.
The active release manifest records `sha256: null` for these files; the downloader prints the observed digest, which you can compare with the published digests above.
HBM files are board models; `tok_embeddings.bin` is companion embedding data, not an X5 inference BIN.
Downloads use a `.part` file renamed on completion; existing nonempty files are skipped.
`GEMMA4_MODEL_BASE_URL`, `GEMMA4_COMMON_MODEL_BASE_URL` and `GEMMA4_TOKENIZER_BASE_URL` explicitly override the three sources.
For the model conversion/quantization workflow, see the [conversion guide](../conversion/README.md).
