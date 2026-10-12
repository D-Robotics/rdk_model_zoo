English | [简体中文](README_cn.md)

# LiteAvatar model artifacts

[Python runtime](../runtime/python/README.md) · [Conversion](../conversion/README.md)

LiteAvatar runs on **S100**, **S100P** and **S600** with a per-target
package: four HBM artifacts prepared locally plus the shared bundled CMVN
file. The [S manifest](../../../../docs/release/s/models.yaml) records the
published digests; preparation is one of the two explicit routes below. X5
has no published set.

<a id="artifacts"></a>
## Artifacts

| File under `model/` | Target | Role | Source |
| --- | --- | --- | --- |
| `s100/lite_avatar_encoder_400x560_s100.hbm` | s100 | audio encoder (paraformer, 50 hidden states) | local bundle or self-compile ([conversion](../conversion/README.md)) |
| `s100/lite_avatar_audio2mouth_30x50x512_s100.hbm` | s100 | hidden states to mouth params | local bundle or self-compile ([conversion](../conversion/README.md)) |
| `s100/lite_avatar_net_encode_384x384_s100.hbm` | s100 | reference-frame face encoder | local bundle or self-compile ([conversion](../conversion/README.md)) |
| `s100/lite_avatar_net_decode_inorm_fullrange_384x384_s100.hbm` | s100 | face decoder | local bundle or self-compile ([conversion](../conversion/README.md)) |
| `s100p/lite_avatar_encoder_400x560_s100p.hbm` | s100p | audio encoder (paraformer, 50 hidden states) | self-compile ([conversion](../conversion/README.md)) |
| `s100p/lite_avatar_audio2mouth_30x50x512_s100p.hbm` | s100p | hidden states to mouth params | self-compile ([conversion](../conversion/README.md)) |
| `s100p/lite_avatar_net_encode_384x384_s100p.hbm` | s100p | reference-frame face encoder | self-compile ([conversion](../conversion/README.md)) |
| `s100p/lite_avatar_net_decode_inorm_fullrange_384x384_s100p.hbm` | s100p | face decoder | self-compile ([conversion](../conversion/README.md)) |
| `s600/lite_avatar_encoder_400x560_s600.hbm` | s600 | audio encoder (paraformer, 50 hidden states) | self-compile ([conversion](../conversion/README.md)) |
| `s600/lite_avatar_audio2mouth_30x50x512_s600.hbm` | s600 | hidden states to mouth params | self-compile ([conversion](../conversion/README.md)) |
| `s600/lite_avatar_net_encode_384x384_s600.hbm` | s600 | reference-frame face encoder | self-compile ([conversion](../conversion/README.md)) |
| `s600/lite_avatar_net_decode_inorm_fullrange_384x384_s600.hbm` | s600 | face decoder | self-compile ([conversion](../conversion/README.md)) |
| `am.mvn` | all | frontend CMVN statistics | bundled in this directory |

All four HBM files need preparation (`am.mvn` is already committed); all
four expose FLOAT32 inputs/outputs at the runtime boundary, with the
quantized graphs internal to the artifacts.

<a id="directory"></a>
## Directory structure

```text
model/
├── README.md   # English instructions
├── README_cn.md  # Chinese instructions
├── verify.py   # Golden-digest check of the prepared package
├── am.mvn      # Bundled CMVN statistics (pinned digest)
├── s100/       # Prepared HBM artifacts (nash-e)
├── s100p/      # Prepared HBM artifacts (nash-m)
└── s600/       # Prepared HBM artifacts (nash-p)
```

<a id="preparation"></a>
## Preparation

Both routes end with the four files under `model/<target>/` — `s100`
(the default), `s100p` or `s600`. Verify the prepared package with:

```bash
# input: the four HBM files under model/<target>/ (default s100)
# success: exit 0 with each digest match printed
python3 samples/vision/lite_avatar/model/verify.py --target s100p
```

`verify.py` checks each file's SHA-256 against the manifest digests in
[formats and checksums](#formats-checksums) and exits nonzero on any
missing or mismatched artifact.

**Route A — local bundle.** Place a copy of the four compiled artifacts
under `model/<target>/` with the published names, then run
`verify.py --target <target>`.

**Route B — self-compilation.** Compile the set yourself with the
OpenExplorer toolchain by following [conversion](../conversion/README.md)
end to end, then copy the four HBMs under `model/<target>/` with the
published names. A self-built set legitimately differs from the golden
digests; identify it explicitly with a local hash manifest so verification
stays exact (one target at a time):

```bash
TARGET=s100p   # s100 | s100p | s600
cd samples/vision/lite_avatar/model
python3 - "$TARGET" <<'PY' > lite-avatar-rebuild.json
import hashlib, json, pathlib, sys
target = sys.argv[1]
print(json.dumps({
    f"{target}/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(pathlib.Path(target).glob("*.hbm"))
}, indent=2))
PY
bash ../runtime/python/run.sh --target "$TARGET" \
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

After preparation, each target's four artifacts live under
`model/<target>/` relative to the sample root; a self-built set also
carries its hash manifest at `model/lite-avatar-rebuild.json`. The
`--models-manifest` example in [preparation](#preparation) and the runtime
paths in [runtime/python/README.md](../runtime/python/README.md) point at
these locations.

<a id="formats-checksums"></a>
## Formats and checksums

| File | Format | SHA-256 |
| --- | --- | --- |
| `s100/lite_avatar_encoder_400x560_s100.hbm` | nash-e `.hbm`, F32 feats `[1,400,560]` in / F32 hidden states `[1,400,50,512]` out | `0a6833a1e24d9f407e4a7662c305c6b292f2d947aef03adce2043bec1091aef0` |
| `s100/lite_avatar_audio2mouth_30x50x512_s100.hbm` | nash-e `.hbm`, F32 `[1,30,50,512]` in / F32 mouth params `[1,30,32]` + feat out | `ddded3350ef1e9c8e068263d841b3a15a1a5f59b027c7d50f6dd3c8985fa50e5` |
| `s100/lite_avatar_net_encode_384x384_s100.hbm` | nash-e `.hbm`, F32 RGB `[1,3,384,384]` in / F32 pyramid `f0..f3` out | `b244f61b63bbfdfb219bf03270cb19b9a7a0d8e1604dbbc6fb0591956b673ecb` |
| `s100/lite_avatar_net_decode_inorm_fullrange_384x384_s100.hbm` | nash-e `.hbm`, F32 pyramid + mouth params in / F32 mouth image `[1,3,384,384]` out (InstanceNorm path FP32; 21 nodes configured `run_on_cpu`) | `27e46cf9d6514f587d54c16cfa813583d89e59c9d5b5d4138efdf51e4d20df3c` |
| `s100p/lite_avatar_encoder_400x560_s100p.hbm` | nash-m `.hbm`, F32 feats `[1,400,560]` in / F32 hidden states `[1,400,50,512]` out | `0427e93d2ddd443630f50eabbcbe53929f934e96441bfb3d3657c285df679381` |
| `s100p/lite_avatar_audio2mouth_30x50x512_s100p.hbm` | nash-m `.hbm`, F32 `[1,30,50,512]` in / F32 mouth params `[1,30,32]` + feat out | `ca559da118a24e43194e9474b7daaae74f71b392b781fe767852b0c9c4091531` |
| `s100p/lite_avatar_net_encode_384x384_s100p.hbm` | nash-m `.hbm`, F32 RGB `[1,3,384,384]` in / F32 pyramid `f0..f3` out | `14f8c4179db98f30957cdc5fe2acb7b9bdb064fcb7334700585aa0468a8a9dbe` |
| `s100p/lite_avatar_net_decode_inorm_fullrange_384x384_s100p.hbm` | nash-m `.hbm`, F32 pyramid + mouth params in / F32 mouth image `[1,3,384,384]` out (InstanceNorm path FP32; 21 nodes configured `run_on_cpu`) | `143f290edbfc254fe103075d7d28e941396b331aa19b6c32632754e14289684a` |
| `s600/lite_avatar_encoder_400x560_s600.hbm` | nash-p `.hbm`, F32 feats `[1,400,560]` in / F32 hidden states `[1,400,50,512]` out | `60325abcf55190382becff0c1c530e5fcb1e466a1647563033b0fd31171d3beb` |
| `s600/lite_avatar_audio2mouth_30x50x512_s600.hbm` | nash-p `.hbm`, F32 `[1,30,50,512]` in / F32 mouth params `[1,30,32]` + feat out | `b47eb04b28219ab113cc8362c3f85b883ab752a3e6321d8c0f995137973acce5` |
| `s600/lite_avatar_net_encode_384x384_s600.hbm` | nash-p `.hbm`, F32 RGB `[1,3,384,384]` in / F32 pyramid `f0..f3` out | `aecf34b9610d24aafdce81b611a022208b91a09855e0f8aedae8fc85f5f4a678` |
| `s600/lite_avatar_net_decode_inorm_fullrange_384x384_s600.hbm` | nash-p `.hbm`, F32 pyramid + mouth params in / F32 mouth image `[1,3,384,384]` out (InstanceNorm path FP32; 21 nodes configured `run_on_cpu`) | `3d828941c7b2a4fba736e264a210d8b290a9a804b0c334a0eee630a2563fc8ec` |
| `am.mvn` | Kaldi CMVN statistics (text) | `29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5` |


Every HBM is a post-training quantization with INT16 configured on all
nodes and INT8 weights (the compiler still assigns a few ops other dtypes),
of the pinned upstream TorchScript export, compiled for the
target march (nash-e S100, nash-m S100P, nash-p S600). The `lite_avatar`
rows of the [S
manifest](../../../../docs/release/s/models.yaml) carry these digests and
are the publishing authority.
