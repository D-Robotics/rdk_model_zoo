# ViT 模型

<a id="artifacts"></a>
## 制品

| Variant | Target | Stage | File | Format |
| --- | --- | --- | --- | --- |
| int8 | s100 | classifier | `s100/vit_cifar10_batch1_int8.hbm` | hbm |
| int16 | s100 | classifier | `s100/vit_cifar10_batch1_int16.hbm` | hbm |

发布事实来源： [S manifest](../../../../docs/release/s/models.yaml), sample `vit`.

- `int8`: [download](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ViT/vit_cifar10_batch1_int8.hbm)
- `int16`: [download](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ViT/vit_cifar10_batch1_int16.hbm)


<a id="preparation"></a>
## 准备

```bash
# cwd: repository root
bash samples/vision/vit/model/download.sh s100 int8
python3 samples/vision/vit/model/download.py --target s100 --variant int16
```

默认目标 s100、变体 int8。download.sh 接受目标/变体位置参数，download.py 使用命名选项。网络/文件错误返回 2。离线可转移准确制品到同一路径，保留观测哈希和来源 URL，不借用其他目标构建。

<a id="accompanying-files"></a>
## 伴随文件

`../test_data/cifar10_classes.names` 是整数键 Python 字典字面量，通过 literal_eval 安全解析。十个标签对应 ID 0–9；`airplane_0000.png` 是默认功能图片，推理不下载二者。

<a id="local-paths"></a>
## 本地路径

运行时从本目录解析模型，不依赖 cwd。外部 `--model-path` 必须同时提供准确 `--asset-id`，不能只按文件名猜协议。

```bash
python3 samples/vision/vit/runtime/python/main.py --dry-run --target s100 --asset-id s:vit:s100/vit_cifar10_batch1_int16.hbm --model-path /path/to/vit_cifar10_batch1_int16.hbm
```

<a id="formats-checksums"></a>
## 格式与哈希

两制品在 manifest 均为 `sha256: null (unknown)`。下载器打印观测 SHA-256，仅标识本地字节，不构成发布者独立验证。已有文件按共享下载策略保留；运行前拒绝不兼容张量 metadata。
