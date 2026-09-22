# Model artifacts (fixture)

<a id="artifacts"></a>
## Artifacts

| File | Target | Stage | Source |
| --- | --- | --- | --- |
| `fixture1_224x224_nv12.bin` | x5 | single | download |
| `fixture1_224x224_nv12.hbm` | s100 | single | download |

<a id="preparation"></a>
## Preparation

From the repository root:

```bash
bash samples/tools/fixture/good_sample/model/download.sh --target x5
```

The script fails loudly on checksum mismatch; there is no silent fallback.

<a id="accompanying-files"></a>
## Accompanying files

`labels.txt` maps output indexes to fixture class names and is required at
run time.

<a id="local-paths"></a>
## Local paths

After preparation the artifacts live in this directory; the runtime default
`--model-path` resolves against it when paired with `--asset-id`.

<a id="formats-checksums"></a>
## Formats and checksums

| File | Format | SHA-256 |
| --- | --- | --- |
| `fixture1_224x224_nv12.bin` | bayes-e `.bin` | null (unknown) |

No checksum has been published for the fixture artifact; unknown values stay
`null (unknown)` and are never copied across artifacts.
