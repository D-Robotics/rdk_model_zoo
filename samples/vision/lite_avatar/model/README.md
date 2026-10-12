English | [简体中文](README_cn.md)

# LiteAvatar model artifacts

[Python runtime](../runtime/python/README.md) · [Conversion](../conversion/README.md)

LiteAvatar runs on **S100** with a five-file package: four HBM artifacts
prepared locally plus the bundled CMVN file. The [S
manifest](../../../../docs/release/s/models.yaml) records the published
digests; preparation is one of the two explicit routes below. S100P, S600
and X5 have no published set.

<a id="artifacts"></a>
## Artifacts

| File under `model/` | Target | Role | Source |
| --- | --- | --- | --- |
| `s100/lite_avatar_encoder_400x560_s100.hbm` | s100 | audio encoder (paraformer, 50 hidden states) | local bundle or self-compile ([conversion](../conversion/README.md)) |
| `s100/lite_avatar_audio2mouth_30x50x512_s100.hbm` | s100 | hidden states to mouth params | local bundle or self-compile ([conversion](../conversion/README.md)) |
| `s100/lite_avatar_net_encode_384x384_s100.hbm` | s100 | reference-frame face encoder | local bundle or self-compile ([conversion](../conversion/README.md)) |
| `s100/lite_avatar_net_decode_inorm_fullrange_384x384_s100.hbm` | s100 | face decoder (INT16 BPU + FP32 CPU InstanceNorm) | local bundle or self-compile ([conversion](../conversion/README.md)) |
| `am.mvn` | s100 | frontend CMVN statistics | bundled in this directory |

All four HBM files need preparation (`am.mvn` is already committed); all
four expose FLOAT32 inputs/outputs at the runtime boundary, with the INT16
quantization internal to the artifacts.

<a id="directory"></a>
## Directory structure

```text
model/
├── README.md   # English instructions
├── README_cn.md  # Chinese instructions
├── verify.py   # Golden-digest check of the prepared package
├── am.mvn      # Bundled CMVN statistics (pinned digest)
└── s100/       # Prepared HBM artifacts
```

<a id="preparation"></a>
## Preparation

Both routes end with the four files under `model/s100/`; verify with:

```bash
# input: the four HBM files under model/s100/
# success: exit 0 with each digest match printed
python3 samples/vision/lite_avatar/model/verify.py
```

`verify.py` checks each file's SHA-256 against the manifest digests in
[formats and checksums](#formats-checksums) and exits nonzero on any
missing or mismatched artifact.

**Route A — local bundle.** Place a copy of the four compiled artifacts
under `model/s100/` with the published names, then run `verify.py`.

**Route B — self-compilation.** Compile the set yourself with the
OpenExplorer toolchain by following [conversion](../conversion/README.md)
end to end, then copy the four HBMs under `model/s100/` with the published
names. A self-built set legitimately differs from the golden digests;
identify it explicitly with a local hash manifest so verification stays
exact:

```bash
cd samples/vision/lite_avatar/model
python3 - <<'PY' > lite-avatar-rebuild.json
import hashlib, json, pathlib
print(json.dumps({
    f"s100/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(pathlib.Path("s100").glob("*.hbm"))
}, indent=2))
PY
bash ../runtime/python/run.sh --target s100 \
    --models-manifest lite-avatar-rebuild.json \
    --audio ../test_data/asr_example.wav \
    --output outputs/lite_avatar/avatar.mp4
```

The manifest is a JSON object keyed by (a subset of) the published
filenames with 64-hex SHA-256 values. Each listed file is verified against
its recorded digest, and the SDK validates the model's input/output shapes
and target when it loads.

<a id="accompanying-files"></a>
## Accompanying files

`am.mvn` is the Paraformer frontend's CMVN statistics file, committed in
this directory (same bytes as the Paraformer sample's copy) and selected by
the runtime's `--cmvn-path` default — no preparation needed. The avatar
data (background video, face box, reference frames) is separate and
prepared per [test data](../test_data/README.md).

<a id="local-paths"></a>
## Local paths

After preparation, the four artifacts live under `model/s100/` relative to
the sample root; a self-built set additionally carries its hash manifest at
`model/lite-avatar-rebuild.json`. The `--models-manifest` example in
[preparation](#preparation) and the runtime paths in
[runtime/python/README.md](../runtime/python/README.md) point at these
locations.

<a id="formats-checksums"></a>
## Formats and checksums

| File | Format | SHA-256 |
| --- | --- | --- |
| `s100/lite_avatar_encoder_400x560_s100.hbm` | nash-e `.hbm`, F32 feats `[1,400,560]` in / F32 hidden states `[1,400,50,512]` out | `0a6833a1e24d9f407e4a7662c305c6b292f2d947aef03adce2043bec1091aef0` |
| `s100/lite_avatar_audio2mouth_30x50x512_s100.hbm` | nash-e `.hbm`, F32 `[1,30,50,512]` in / F32 mouth params `[1,30,32]` + feat out | `ddded3350ef1e9c8e068263d841b3a15a1a5f59b027c7d50f6dd3c8985fa50e5` |
| `s100/lite_avatar_net_encode_384x384_s100.hbm` | nash-e `.hbm`, F32 RGB `[1,3,384,384]` in / F32 pyramid `f0..f3` out | `b244f61b63bbfdfb219bf03270cb19b9a7a0d8e1604dbbc6fb0591956b673ecb` |
| `s100/lite_avatar_net_decode_inorm_fullrange_384x384_s100.hbm` | nash-e `.hbm`, F32 pyramid + mouth params in / F32 mouth image `[1,3,384,384]` out (21 InstanceNorm nodes on CPU) | `27e46cf9d6514f587d54c16cfa813583d89e59c9d5b5d4138efdf51e4d20df3c` |
| `am.mvn` | Kaldi CMVN statistics (text) | `29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5` |

Every HBM is a post-training quantization (INT16 activations, INT8
weights) of the pinned upstream TorchScript export, compiled for nash-e.
The `lite_avatar` rows of the [S
manifest](../../../../docs/release/s/models.yaml) carry these digests and
are the publishing authority.
