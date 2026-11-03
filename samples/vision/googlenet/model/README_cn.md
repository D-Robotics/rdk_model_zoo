[English](README.md) | 简体中文

# GoogLeNet 模型制品

<a id="artifacts"></a>
## 制品

此处列出 X5 发布清单中的单阶段分类制品；目标与变体选择见根支持矩阵。

| Variant | Filename | Target | Format | Source |
| --- | --- | --- | --- | --- |
| `googlenet` | `googlenet_224x224_nv12.bin` | x5 | bin | download |

<a id="directory"></a>
## 目录结构

```text
model/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── __init__.py  # Python 脚本
├── download.py  # 准备模型文件
└── download.sh  # 模型准备命令
```

<a id="preparation"></a>
## 准备

cwd：仓库根。Shell 使用位置参数，Python 使用命名选项。两种形式省略变体均选择 `googlenet`。

```bash
bash samples/vision/googlenet/model/download.sh x5 googlenet
python3 samples/vision/googlenet/model/download.py --target x5 --variant googlenet
```

输出：退出码 0，打印目标文件与观测 SHA-256。下载先写临时文件，不覆盖已有字节；大小或哈希不匹配时显式报错。服务器不可用时可手动传入匹配的发布文件至 `model/`，记录来源及观测哈希，不能用其他变体替代。

<a id="accompanying-files"></a>
## 配套文件

CLI 读取 `datasets/imagenet/imagenet_classes.names` 标签与默认 `test_data/indigo_bunting.JPEG` 图片。API 接收内存图像和可选标签；没有标签时返回类别 ID 字符串。无需 tokenizer 或额外模型。

<a id="local-paths"></a>
## 本地路径

文件保存为 `samples/vision/googlenet/model/<filename>`，默认运行选择该目录的 `googlenet_224x224_nv12.bin`。外部 `--model-path` 必须同时指定精确 `--asset-id`，如 `x5:googlenet:googlenet_224x224_nv12.bin`。

<a id="formats-checksums"></a>
## 格式与校验

上述每个文件在 `docs/release/x5/models.yaml` 中均为 `format: bin`、`sha256: null (unknown)`。下载器打印收到字节的摘要，用于本地身份核对。

<!-- converted-all-20261103 -->

## 本轮 S 系列转换规格（2026-11-03）

这些独立规格已完成转换、完整 OE 和实际 HBM 接口核验。源权重、ONNX、每目标配置、校准数量和摘要见 [converted_targets_20261103.json](../conversion/converted_targets_20261103.json)。

| 目标 | Native variant | 组件 | 校准图片 |
| --- | --- | --- | ---: |
| s100 | `torchvision-021-crop224-int16-o1-20261103` | model | 200 |
| s100p | `torchvision-021-crop224-int16-o1-20261103` | model | 200 |
| s600 | `torchvision-021-crop224-int16-o1-20261103` | model | 200 |

```bash
python3 samples/vision/googlenet/model/download.py --target s100 --variant torchvision-021-crop224-int16-o1-20261103
python3 samples/vision/googlenet/runtime/python/main.py --target s100 --variant torchvision-021-crop224-int16-o1-20261103 --asset-id s:googlenet:s100/googlenet_tv021_crop224_nashe_int16_20261103_o1.hbm --model-path /path/to/googlenet_tv021_crop224_nashe_int16_20261103_o1.hbm --test-img samples/vision/googlenet/test_data/indigo_bunting.JPEG
```

命令中的路径需要替换为实际文件。每套多组件模型须同时准备全部组件。旧默认与既有指标保留原身份；本轮规格的板端 smoke、数据集精度、Runtime 1/2 并发和 C++ 1/2 流均为 `not-run`。
