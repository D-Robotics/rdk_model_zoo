# MODNet 模型

<a id="artifacts"></a>
## 制品清单

| target | stage | asset ID | 文件 | 获取方式 |
|---|---|---|---|---|
| X5 | matting | `x5:modnet:modnet_512x512_rgb.bin` | `modnet_512x512_rgb.bin` | 外部手工制品 |

没有发布 S100/S100P/S600 MODNet 制品。

<a id="preparation"></a>
## 准备步骤

active manifest 没有公开 URL。通过授权模型提供方取得外部模型，确认身份为 `x5:modnet:modnet_512x512_rgb.bin`，并放到 `samples/vision/modnet/model/modnet_512x512_rgb.bin`。仓库 helper 刻意不下载：

```bash
# cwd：仓库根目录
python3 -m samples.vision.modnet.model.download \
  --target x5 --asset-id x5:modnet:modnet_512x512_rgb.bin \
  --output-dir samples/vision/modnet/model
```

它会打印手工准备要求并返回 `2`；由于没有 URL，这是预期行为。

<a id="accompanying-files"></a>
## 伴随文件

- `../test_data/person.jpg`：源输入图像。
- `../test_data/bg.jpg`：可选背景合成图像。

<a id="local-paths"></a>
## 本地路径

runtime 默认路径为 `samples/vision/modnet/model/modnet_512x512_rgb.bin`。外部路径必须配合精确 `--asset-id x5:modnet:modnet_512x512_rgb.bin`；单凭文件名不能确定协议身份。

<a id="formats-checksums"></a>
## 格式与校验值

期望制品是 X5 `bin` 部署模型。manifest 记录 `url: null` 和 `sha256: null (unknown)`。任何本地观测都不视为发布者认证。
