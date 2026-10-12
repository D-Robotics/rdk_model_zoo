English | [简体中文](README_cn.md)

# Faster R-CNN model files

<a id="artifacts"></a>
## Model pairs

| Target | march | PartA | PartB |
| --- | --- | --- | --- |
| x5 | bayes-e | `../conversion/build/x5/parta/frcnn_mbv3_320_parta.bin` | `../conversion/build/x5/partb/frcnn_mbv3_320_partb.bin` |
| s100 | nash-e | `../conversion/build/s100/parta/frcnn_mbv3_320_parta.hbm` | `../conversion/build/s100/partb/frcnn_mbv3_320_partb.hbm` |
| s100p | nash-m | `../conversion/build/s100p/parta/frcnn_mbv3_320_parta.hbm` | `../conversion/build/s100p/partb/frcnn_mbv3_320_partb.hbm` |
| s600 | nash-p | `../conversion/build/s600/parta/frcnn_mbv3_320_parta.hbm` | `../conversion/build/s600/partb/frcnn_mbv3_320_partb.hbm` |
Prepare one PartA/PartB pair from the [conversion recipe](../conversion/README.md) for each target you use.

<a id="directory"></a>
## Directory structure

```text
frcnn_mbv3_320.json  # Target layouts and reference digests
prepare.py          # Install compiled pairs and inspect file digests
coco91_labels.txt    # COCO category names
x5/                 # .bin pair
s100/               # .hbm pair
s100p/              # .hbm pair
s600/               # .hbm pair
```

<a id="preparation"></a>
## Prepare a pair

Compile the target using the conversion guide. From the repository root, copy the X5 pair:
```bash
python3 samples/vision/fasterrcnn/model/prepare.py --target x5 \
  --from-build samples/vision/fasterrcnn/conversion/build
```
For S targets, use the matching build directory and `.hbm` suffix. The two paths must belong to the same target. Run a self-compiled pair explicitly:
```bash
python3 samples/vision/fasterrcnn/runtime/python/main.py --target x5 \
  --parta-path samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_parta.bin \
  --partb-path samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_partb.bin
```



`prepare.py --target` accepts one or more of `x5`, `s100`, `s100p`, `s600`; omitting it selects all four. `--from-build` accepts the conversion directory or its `build/` directory. `--dry-run` previews copying; `--force` replaces existing different files. A reported digest difference (exit 2) means use the printed explicit-path command.

After compiling the S family, prepare its three pairs and run the S100P pair as follows. Select just `--target s100`, `s100p` or `s600` to install one target.

```bash
python3 samples/vision/fasterrcnn/model/prepare.py --target s100 s100p s600 \
  --from-build samples/vision/fasterrcnn/conversion/build
python3 samples/vision/fasterrcnn/runtime/python/main.py --target s100p \
  --parta-path samples/vision/fasterrcnn/model/s100p/frcnn_mbv3_320_parta.hbm \
  --partb-path samples/vision/fasterrcnn/model/s100p/frcnn_mbv3_320_partb.hbm
```

<a id="accompanying-files"></a>
## Metadata and labels

`frcnn_mbv3_320.json` maps targets to paired artifacts, head layouts and reference hashes. `coco91_labels.txt` maps original COCO category IDs to names; line 0 is background and line 1 is person. Keep both files with the Sample.

<a id="local-paths"></a>
## Local paths

Runtime defaults read `model/<target>/frcnn_mbv3_320_parta.{bin,hbm}` and `frcnn_mbv3_320_partb.{bin,hbm}`. `--target auto` resolves the running board; an explicit target selects its directory. Both `--parta-path` and `--partb-path` override the default pair together.

<a id="formats-checksums"></a>
## Formats and SHA-256

| Target | Stage | File | SHA-256 |
| --- | --- | --- | --- |
| x5 | parta | `frcnn_mbv3_320_parta.bin` | `a6f6509e5ab3b42f709d10062cfdb0a930b9eebb3ceb1d1a4bff3baeda42af86` |
| x5 | partb | `frcnn_mbv3_320_partb.bin` | `2ade35e91d66debf616594a28e2cae051b5e1ede617dfa914b25d06c3255724e` |
| s100 | parta | `frcnn_mbv3_320_parta.hbm` | `ca4c565497b3b027cad2241110e689c5eb54e24939a328f6e4fb8bd013e51337` |
| s100 | partb | `frcnn_mbv3_320_partb.hbm` | `d4d21cebaece91c6195794b7062f607d665ef10c3055ce22d512ba14a5307ff4` |
| s100p | parta | `frcnn_mbv3_320_parta.hbm` | `e7f4611d5283ff71673f8b0d67953111e70d2878405b8054bf56536d12924710` |
| s100p | partb | `frcnn_mbv3_320_partb.hbm` | `ed91c61f81f2ac21a2d9dd8d20170f6a85243fab4157247e0943c8af535809f7` |
| s600 | parta | `frcnn_mbv3_320_parta.hbm` | `a0c857181e95b275190aca68c4149814c7381ceb7e3c9aac1cdd8a22cd2e0e84` |
| s600 | partb | `frcnn_mbv3_320_partb.hbm` | `2ae5360e0b1bbc1d463cfd12f4764936e2c4e9dd5741cc35d0df705d848603ef` |

Reference digests are recorded in `frcnn_mbv3_320.json`. `sha256sum` prints your local file digests. Default selection checks the recorded pair; explicit paths use the supplied files and their live tensor metadata.

