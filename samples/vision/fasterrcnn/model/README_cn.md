[English](README.md) | 简体中文

# Faster R-CNN model files

<a id="artifacts"></a>
## 模型组合

每个板卡目标使用一组 PartA/PartB。所有组合均已发布到公开档案站，可按下文直接下载，也可用[转换配方](../conversion/README_cn.md)重建。

| Target | march | PartA | PartB |
| --- | --- | --- | --- |
| x5 | bayes-e | `x5/frcnn_mbv3_320_parta.bin` | `x5/frcnn_mbv3_320_partb.bin` |
| s100 | nash-e | `s100/frcnn_mbv3_320_parta.hbm` | `s100/frcnn_mbv3_320_partb.hbm` |
| s100p | nash-m | `s100p/frcnn_mbv3_320_parta.hbm` | `s100p/frcnn_mbv3_320_partb.hbm` |
| s600 | nash-p | `s600/frcnn_mbv3_320_parta.hbm` | `s600/frcnn_mbv3_320_partb.hbm` |

<a id="directory"></a>
## 目录结构

```text
download.py           # 下载已发布模型并校验哈希
download.sh           # download.py 的 Shell 入口
frcnn_mbv3_320.json   # 板卡布局与参考哈希
prepare.py            # 安装编译产物并查看文件哈希
coco91_labels.txt     # COCO 类别名称
x5/                   # .bin 组合
s100/                 # .hbm 组合
s100p/                # .hbm 组合
s600/                 # .hbm 组合
```

<a id="preparation"></a>
## 准备模型

### 下载已发布模型

在仓库根目录下载 X5 模型对：
下载使用 Python 3.10 或更高版本。`--output-dir DIR` 将所选板卡的子目录保存到指定目录；省略时使用本目录。

```bash
python3 -m venv .venv-model-download
. .venv-model-download/bin/activate
python3 -m pip install PyYAML==6.0.3
python3 samples/vision/fasterrcnn/model/download.py --target x5
```
`--target` 可指定一个或多个 `x5`、`s100`、`s100p`、`s600`，省略时下载全部四组到 `model/<target>/`（runtime 的默认查找路径）。每个文件按发布清单获取、校验 SHA-256 后原子安装；已存在的文件只做校验，不会被覆盖。`download.sh` 接受相同参数。下载只需带 PyYAML 的 Python 3，不需要板端运行时依赖。

### 编译并安装（可选）

先按转换指南编译目标模型，在仓库根目录复制 X5 产物：
```bash
python3 samples/vision/fasterrcnn/model/prepare.py --target x5 \
  --from-build samples/vision/fasterrcnn/conversion/build
```
S 系列使用对应的构建目录及 `.hbm` 后缀，两个文件须属于同一板卡目标。自行编译的模型同时指定两个路径运行：
```bash
python3 samples/vision/fasterrcnn/runtime/python/main.py --target x5 \
  --parta-path samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_parta.bin \
  --partb-path samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_partb.bin
```



`prepare.py --target` 可指定一个或多个 `x5`、`s100`、`s100p`、`s600`，省略时选择四个平台。`--from-build` 接收转换目录或其中的 `build/` 目录；`--dry-run` 查看复制路径，`--force` 替换已有的不同文件。报告哈希不同（返回 2）时，使用终端给出的显式路径命令。

完成 S 系列编译后，可以一次准备三组模型，并按下面的命令在 S100P 上运行。仅准备一块板卡时，选择 `--target s100`、`s100p` 或 `s600`。

```bash
python3 samples/vision/fasterrcnn/model/prepare.py --target s100 s100p s600 \
  --from-build samples/vision/fasterrcnn/conversion/build
python3 samples/vision/fasterrcnn/runtime/python/main.py --target s100p \
  --parta-path samples/vision/fasterrcnn/model/s100p/frcnn_mbv3_320_parta.hbm \
  --partb-path samples/vision/fasterrcnn/model/s100p/frcnn_mbv3_320_partb.hbm
```

<a id="accompanying-files"></a>
## 伴随文件

`frcnn_mbv3_320.json` 记录板卡、模型组合、头部布局及参考哈希。`coco91_labels.txt` 将原始 COCO ID 映射为名称，第 0 行是背景，第 1 行是行人。两个文件随 Sample 一起保留。

<a id="local-paths"></a>
## 本地路径

Runtime 默认读取 `model/<target>/frcnn_mbv3_320_parta.{bin,hbm}` 与 `frcnn_mbv3_320_partb.{bin,hbm}`。`--target auto` 识别当前板卡，显式目标选择对应目录。通过 `--parta-path` 和 `--partb-path` 一起指定其他模型路径。

<a id="formats-checksums"></a>
## 格式与 SHA-256

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

参考哈希记录在 `frcnn_mbv3_320.json` 与发布清单中，即已发布产物的哈希；`download.py` 下载时逐一校验。可通过 `sha256sum` 查看本地文件哈希。默认选择校验记录的模型组合；显式路径使用所提供文件，并检查其实际张量信息。

