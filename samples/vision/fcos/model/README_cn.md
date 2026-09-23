# FCOS 模型制品

<a id="artifacts"></a>
## 制品清单

| 制品 | 格式 | target | 阶段 | 来源 |
| --- | --- | --- | --- | --- |
| `fcos_efficientnetb0_detect_512x512_bayese_nv12.bin` | bin | x5 | FCOS detector | manifest 下载 |
| `fcos_efficientnetb2_detect_768x768_bayese_nv12.bin` | bin | x5 | FCOS detector | manifest 下载 |
| `fcos_efficientnetb3_detect_896x896_bayese_nv12.bin` | bin | x5 | FCOS detector | manifest 下载 |

精确引用为 `x5:fcos:<filename>`。本 sample 不提交模型二进制。

<a id="preparation"></a>
## 准备步骤

```bash
# cwd：仓库根目录
bash samples/vision/fcos/model/download.sh --target x5 --variant efficientnetb0
# 预期：文件写入 samples/vision/fcos/model/；已有文件先校验，不覆盖

# cwd：仓库根目录
bash samples/vision/fcos/model/fulldownload.sh --target x5
# 预期：三个制品均写入上述目录

# 与 manifest 兼容的单变体入口
bash samples/vision/fcos/model/download_model.sh --target x5 --variant efficientnetb2
```

下载器使用 manifest 的 URL 和 hash。当前三行均为 `sha256: null (unknown)`，本地观察到的摘要不能证明发布来源；缺 URL 或已记录 hash 不匹配时会在安装前失败。

Shell wrapper 会从自身目录定位 `download.py`，调用板端镜像提供的 `python3`，不依赖仓库内主机 `.venv`。`download_model.sh` 是 `docs/release/x5/models.yaml` 登记的兼容入口。

<a id="accompanying-files"></a>
## 伴随文件

| 文件 | 作用 | 必需 |
| --- | --- | --- |
| `runtime/python/main.py` | 运行入口和结果可视化 | 是 |
| `test_data/bus.jpg` | 内置 BGR 冒烟输入 | 否，可换用户图片 |
| `datasets/coco/coco_classes.names` | 绘图时的可选标签 | 否，仍会输出数字 ID |

<a id="local-paths"></a>
## 本地路径

- 制品位置：`samples/vision/fcos/model/<manifest filename>`。
- 运行时默认使用 B0 路径；执行仍必须给出 `--target x5` 和精确 asset ID。
- 外部 `--model-path` 只有和精确限定的 `--asset-id` 一起使用才接受；文件名不能代表身份。

<a id="formats-checksums"></a>
## 格式与校验值

| 制品 | 格式 | SHA-256 | 来源 |
| --- | --- | --- | --- |
| B0 512 | `.bin` | `null (unknown)` | `docs/release/x5/models.yaml` |
| B2 768 | `.bin` | `null (unknown)` | `docs/release/x5/models.yaml` |
| B3 896 | `.bin` | `null (unknown)` | `docs/release/x5/models.yaml` |
