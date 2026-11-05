[English](README.md) | 简体中文

# OSNet 模型准备

<a id="artifacts"></a>
## 模型制品

OSNet x1.0 的四平台已验收模型已上传 Archive，模型、清单和 OE 报告于 2026-11-02 完成服务器回读与匿名 HTTPS 校验。原生下载地址登记在 `docs/release/{x5,s}/models.yaml`，模型仅供研究；本地源码整合和研究用途目录登记已获用户放行，远端推送与站点上线待执行。[上传回执](../evaluator/evidence/2026-11-02-archive-upload.json)。

| Target | Variant | File |
| --- | --- | --- |
| x5 | spatial-rgb-late8 | spatial-rgb-late8.bin |
| s100 | s100-f32-int16 | osnet_x1_0_nashe_256x128_f32_int16.hbm |
| s100p | s100p-f32-int16 | osnet_x1_0_nashm_256x128_f32_int16.hbm |
| s600 | s600-f32-int16 | osnet_x1_0_nashp_256x128_f32_int16.hbm |

X5 其他历史规格的身份保留在 artifacts.json；失败的 INT8 实验不作为推荐发布模型。

<a id="directory"></a>
## 目录

`download.py` / `download.sh`: 模型准备入口；`artifacts.json` 为模型身份，`artifacts.py` 执行 SHA-256 校验。模型文件不提交到 Git。

<a id="preparation"></a>
## 模型准备

在仓库根目录执行，按板卡选择推荐模型：

```bash
python3 samples/vision/osnet/model/download.py --target x5 --download
python3 samples/vision/osnet/model/download.py --target s600 --download
```

已有交付包也可离线导入。`--candidate-dir` 接受 accepted-delivery 根目录或其中单个平台目录；`--target` 选择推荐版本，`--variant` 可显式选择同一板卡的版本。

```bash
python3 samples/vision/osnet/model/download.py --target x5 \
  --candidate-dir /absolute/path/accepted-delivery
```

```bash
python3 samples/vision/osnet/model/download.py --target s600 \
  --candidate-dir /absolute/path/accepted-delivery/s600
```

`--download --target TARGET` 从原生清单获取模型并校验固定 SHA-256。X5 公网文件名与本地 `spatial-rgb-late8.bin` 不同，模型字节相同。原 X5 私有 ZIP/TAR 导入仍可用：

```bash
python3 samples/vision/osnet/model/download.py --bundle /absolute/path/x5-readback.tar \
  --format readback --target x5 --fixtures
```

<a id="accompanying-files"></a>
## 配套文件

候选目录和正式下载仅准备模型。Python 使用 `--test-img` 指定有权使用的行人裁剪图。C++ 数值验证另需自行准备 images.dat 与该模型对应的 32×512 原始特征；Market1501 fixture 仅限私有研究。

<a id="local-paths"></a>
## 本地路径

默认保存为 `samples/vision/osnet/model/<registered filename>`。`--output-dir` 修改准备目录，运行时用 `--model-path` 指向新位置。模型准备不识别硬件，也不执行推理；运行时才校验目标板卡。

<a id="formats-checksums"></a>
## 格式与校验和

候选包的 platform、SHA-256 和模型实际字节须与 artifacts.json 一致；交付文件名可以与已登记的本地缓存名不同；已有不同内容的目标文件不覆盖。正式下载沿用公共 assets 工具的相同保护。

完整身份、失败实验及许可范围见 [artifacts.json](artifacts.json) 和[许可核验](../third_party/LICENSE_REVIEW.md)。
