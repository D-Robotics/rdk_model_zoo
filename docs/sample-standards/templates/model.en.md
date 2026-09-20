<!-- Template: model/ README (English). Contract: readme-contract.md §4.2.
     Keep anchors; replace ⟪…⟫; delete guidance when done. sha256 discipline:
     unknown values stay `sha256: null (unknown)` — never guessed or copied. -->

# Model Artifacts — ⟪model name⟫

<a id="artifacts"></a>
## Artifacts

> **Must answer:** one row per artifact file mapping it to the target(s) and
> pipeline stage(s) it serves; how it is obtained (download / manual / prepared
> by conversion). Must agree with the manifest row for this sample.

| Artifact | Format | Target(s) | Stage(s) | Source |
| --- | --- | --- | --- | --- |
| ⟪file⟫ | ⟪.bin/.hbm/…⟫ | ⟪x5 / s100 / …⟫ | ⟪single / det / rec / …⟫ | ⟪download \| manual \| conversion⟫ |

<a id="preparation"></a>
## Preparation

> **Must answer:** the exact command (cwd, `--target`) or the manual steps; what
> happens on hash mismatch; alternative route if the primary one fails.

```bash
# cwd: repository root
bash samples/⟪domain⟫/⟪name⟫/model/download.sh --target ⟪target⟫
# expect: files listed above under model/, sha256 verified against manifest
```

⟪Manual-only artifacts: where to obtain them (internal archive, vendor portal),
which version, and where to place them.⟫

<a id="accompanying-files"></a>
## Accompanying Files

> **Must answer:** every non-artifact file required at runtime (vocab, labels,
> mvn stats, configs) with a one-line role and whether it is required.

| File | Role | Required |
| --- | --- | --- |
| ⟪file⟫ | ⟪one line⟫ | ⟪yes/no⟫ |

<a id="local-paths"></a>
## Local Paths

> **Must answer:** where files land after preparation and what the runtime
> default parameters point to (these are machine-checked for consistency).

- Artifacts: `samples/⟪domain⟫/⟪name⟫/model/⟪…⟫`
- Runtime default `--model-path` (or equivalent): ⟪path⟫

<a id="formats-checksums"></a>
## Formats & Checksums

> **Must answer:** format per artifact and its known SHA-256 **with source**.
> Unknown values are written as `sha256: null (unknown)` — never fabricated,
> never copied from a sibling artifact.

| Artifact | Format | SHA-256 | Source of value |
| --- | --- | --- | --- |
| ⟪file⟫ | ⟪format⟫ | ⟪hash or `null (unknown)`⟫ | ⟪release record / manifest / unknown⟫ |
