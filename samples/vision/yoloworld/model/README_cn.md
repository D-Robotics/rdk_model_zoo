# YOLOWorld 模型资产

<a id="artifacts"></a>
## 资产

活动清单 `docs/release/x5/models.yaml` 只有一个模型资产：
`x5:yoloworld:yolo_world.bin`，格式为 `.bin`，URL 位于 X5 归档，且
`sha256: null`。发布者摘要未知；本地观察摘要不能证明来源。离线词向量伴随
文件是 `test_data/offline_vocabulary_embeddings.json`，不是清单模型资产。

<a id="preparation"></a>
## 准备

```bash
bash samples/vision/yoloworld/model/download.sh --target x5
```

`download.py` 使用仓库资产 helper，原子写入，拒绝无效已有文件，打印观察到的
SHA-256，并明确发布者摘要未知。`runtime/python/main.py` 不会自动调用它。
只有带精确资产 ID 的已识别文件才能手工放置；不要把未识别模型改名放入目录。

<a id="accompanying-files"></a>
## 伴随文件

源 fixture 图片为 `../test_data/dog.jpeg`，源词向量为
`../test_data/offline_vocabulary_embeddings.json`。每个 prompt 必须对应有限的
F32 宽度 512 向量；它提供文本嵌入和 ID 映射，不能用 COCO 标签名替换。

<a id="local-paths"></a>
<a id="formats-checksums"></a>
## 本地路径、格式与摘要

默认模型路径为 `model/yolo_world.bin`；显式 `--model-path` 必须同时给出
`--asset-id x5:yoloworld:yolo_world.bin`。编译产物是 X5 `.bin`。清单发布者
摘要未知，因此本文不声称校验通过。伴随 JSON 是 UTF-8 JSON，清单未记录摘要。

## 入口

`download.py`、`download.sh` 和兼容别名 `download_model.sh` 是显式准备入口；
runtime 不会调用它们。
