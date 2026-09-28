# Paraformer model package

[简体中文](README_cn.md) · [Python integration](../runtime/python/README.md)

The active [S manifest](../../../../docs/release/s/models.yaml) publishes a six-file
S100 package. This directory provides explicit preparation; inference never
silently downloads it. X5, S100P and S600 have no published Paraformer set here.

| File under `s100/` | Role | Source |
| --- | --- | --- |
| `paraformer_large_encoder_400x560_s100.hbm` | encoder | active manifest URL (`encoder_int16.hbm`) |
| `paraformer_large_predictor_400x512_s100.hbm` | predictor | active manifest URL (`predictor_int16.hbm`) |
| `paraformer_large_decoder_400x512_s100.hbm` | decoder | active manifest URL (`decoder_int16.hbm`) |
| `tokens.json` | 8,404 ordered token strings | active manifest URL |
| `am.mvn` | frontend CMVN statistics | bundled pinned S source |
| `paraformer_config.yaml` | source frontend/model configuration | bundled pinned S source |

## Preview and prepare

Use Python with NumPy and PyYAML; no SDK, board or publisher build is required to
prepare files on a host. From repository root, preview all sources/destinations
without network access or writes:

```bash
bash samples/speech/paraformer/model/download_model.sh --target s100 --dry-run
```

To explicitly download the four remote files and copy both local frontend files:

```bash
bash samples/speech/paraformer/model/download_model.sh --target s100
```

The default output root is this directory, so all six files are placed in
`model/s100/`. `--output-dir /path/to/package` instead writes under
`/path/to/package/s100/`. `--target` accepts only `s100` and defaults to it;
`--dry-run` prints exactly six lines and creates nothing. Set `PYTHON` to an
interpreter path when using the shell wrapper, or directly run `download.py` with
your chosen Python. `--help` works without SDK or model files.

Existing files are never overwritten. Remote files use temporary downloads and
atomic installation through the shared asset helper. Local files are byte-checked
against pinned SHA-256 before copying and installed without replacing an existing
path. A mismatched local configuration or vocabulary exits with code 2 and leaves
the existing file intact. A later failure can leave earlier complete files in the
output directory; inspect the error, retain any user files, and resume after
correcting the failing input. Do not treat a partially prepared package as ready.

The script prints each observed digest and a final success line only after all
six entries are processed. A successful download is not SDK compatibility or
inference validation. The active manifest has no publisher hashes for the HBM
files; their observed digests do not independently authenticate origin. A vocabulary
that differs from this migration's observed fixed package is rejected, but that
local pin does not become a publisher-provided hash.

## Fixed source and byte identities

`am.mvn` and `paraformer_config.yaml` are copied without changes from S commit
`380e1a2bf42041af54be6f34935e50197cfadff9`. Their SHA-256 values are:

- `am.mvn`: `29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5`
- `paraformer_config.yaml`: `1d9057edeaba9e131cb98f26011606497cf3af187d8943525ddb5ee36c836b1b`
- Downloaded `tokens.json`: `2b20c2b12572d682afff84ce1c8d560f67b8b32a4c1f21567411d141ed352127`

The source frontend uses 16 kHz, 80 mel bands, 25 ms windows, 10 ms shifts,
LFR stacking 7 with step 6, and at most 400 frames of width 560. Its real CPU frontend has passed seven source comparisons; see the Python guide. The pipeline's zero context bias preserves
the source deployment; it does not implement user-supplied contextual hotwords.
See [physical tensor contracts](../runtime/python/README.md) for the three models.
INT16 names describe the compiled model recipe, not permission to guess I/O dtype.

## Current validation

Host tests exercise real preparation logic with substituted HTTP response bytes,
checking all six files, rerun reuse, protected-file rejection and write-free preview.
Synthetic model bytes in those tests are not real HBM assets. The actual published
vocabulary was downloaded and its 8,404 unique tokens validated. Neither the full
HBM package download nor real SDK inference was executed in this step. Source
auxiliary bytes, preview commands and help are checked separately in the
[binding/package evidence](../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-binding-review.md).
