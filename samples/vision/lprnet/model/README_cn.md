# LPRNet 模型

<a id="artifacts"></a>
## 制品清单

| target | stage | asset ID | 文件 | 获取方式 |
|---|---|---|---|---|
| X5 | recognition | `x5:lprnet:lpr.bin` | `lpr.bin` | 发布下载 |

S100/S100P/S600 manifest 没有 LPRNet 制品。随源提供的 `test_data/test_input.dat` 是输入制品，不是模型。

<a id="preparation"></a>
## 准备步骤

在仓库根目录中，于允许联网的环境显式运行下载器：

```bash
python3 -m samples.vision.lprnet.model.download \
  --target x5 --asset-id x5:lprnet:lpr.bin \
  --output-dir samples/vision/lprnet/model
```

成功判断是 `samples/vision/lprnet/model/lpr.bin` 存在且脚本打印 `Prepared ...`。2026-09-24 板端运行（一块 X5 8GB、一块 X5 4GB，均 rc=0）执行的就是本文档的模块命令，观测 SHA-256 见下。runtime 不会下载或覆盖模型文件。

<a id="accompanying-files"></a>
## 伴随文件

- `../test_data/test_input.dat`：源提供的 float32 输入 tensor，27072 字节，reshape 为 `1x3x24x94`。
- `../test_data/example.jpg`：源视觉参考，不会送入 runtime。

<a id="local-paths"></a>
## 本地路径

runtime 默认路径为 `samples/vision/lprnet/model/lpr.bin`。外部路径必须同时给出精确 `--asset-id x5:lprnet:lpr.bin`，即使发布者校验值未知也保留制品身份。

<a id="formats-checksums"></a>
## 格式与校验值

`lpr.bin` 是 X5 `bin` 部署制品。active manifest URL 为 `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/LPRNet/lpr.bin`，发布者字段为 `sha256: null (unknown)`。两次 2026-09-24 板端运行的观测 SHA-256 均为 `f480391507b6d15274bfff90900acc3271f4e46ac9afd4bf7d0fee4aa50f91bc`（[8GB 复验](../../../../docs/releases/unified-migration/evidence/2026-09-24-b7-binding-recheck/)、[4GB 运行](../../../../docs/releases/unified-migration/evidence/2026-09-24-b7-other-x5-variants/)）；本地观测 digest 只能说明本次使用的字节，不代表发布者认证。
