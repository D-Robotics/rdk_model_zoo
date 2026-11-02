[English](README.md) | 简体中文

# ResNeXt 模型制品

<a id="artifacts"></a>
## 制品

此处列出 X5 发布清单中的单阶段分类制品；目标与变体选择见根支持矩阵。

| Variant | Filename | Target | Format | Source |
| --- | --- | --- | --- | --- |
| `50_32x4d` | `ResNeXt50_32x4d_224x224_nv12.bin` | x5 | bin | download |

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

cwd：仓库根。Shell 使用位置参数，Python 使用命名选项。两种形式省略变体均选择 `50_32x4d`。

```bash
bash samples/vision/resnext/model/download.sh x5 50_32x4d
python3 samples/vision/resnext/model/download.py --target x5 --variant 50_32x4d
```

输出：退出码 0，打印目标文件与观测 SHA-256。下载先写临时文件，不覆盖已有字节；大小或哈希不匹配时显式报错。服务器不可用时可手动传入匹配的发布文件至 `model/`，记录来源及观测哈希，不能用其他变体替代。

<a id="accompanying-files"></a>
## 配套文件

CLI 读取 `datasets/imagenet/imagenet_classes.names` 标签与默认 `test_data/bee_eater.JPEG` 图片。API 接收内存图像和可选标签；没有标签时返回类别 ID 字符串。无需 tokenizer 或额外模型。

<a id="local-paths"></a>
## 本地路径

文件保存为 `samples/vision/resnext/model/<filename>`，默认运行选择该目录的 `ResNeXt50_32x4d_224x224_nv12.bin`。外部 `--model-path` 必须同时指定精确 `--asset-id`，如 `x5:resnext:ResNeXt50_32x4d_224x224_nv12.bin`。

<a id="formats-checksums"></a>
## 格式与校验

上述每个文件在 `docs/release/x5/models.yaml` 中均为 `format: bin`、`sha256: null (unknown)`。下载器打印收到字节的摘要，用于本地身份核对。

<!-- converted-batch-20261102 -->
## 已转换新增规格（2026-11-02）

以下 timm 权重单独转换，已通过严格权重加载、结构重参数化及两组输入的
PyTorch/ONNX 主机对照、checker 和编译。板端冒烟、精度、Runtime 1/2 并发和
C++ 1/2 流均为 **not-run**，历史指标和截图不适用于这些产物。原有默认规格保留。
新规格需要 Pillow，使用 bicubic 短边 256、四舍五入居中裁剪 224；Mapper 融合
该权重实际的归一化参数。输出为 float32 1,000 类 logits，由原生 Runtime 做 softmax。

| 规格 | 模型文件 | 权重版本 | 板端 |
|---|---|---|---|
| `timm-50-32x4d-a1h-in1k` | `resnext50_32x4d_a1h_in1k_bayese_224x224_nv12.bin` | [`cc736cb09401`](https://huggingface.co/timm/resnext50_32x4d.a1h_in1k/blob/cc736cb0940158d05966023116d90fca4c38dae3/README.md) | not-run |

```bash
python3 samples/vision/resnext/model/download.py --target x5 --variant timm-50-32x4d-a1h-in1k
python3 samples/vision/resnext/runtime/python/main.py --target x5 --variant timm-50-32x4d-a1h-in1k --dry-run
```

精确 URL、SHA-256 和大小登记在 `docs/release/x5/models.yaml`；来源及主机对照
记录见 `conversion/converted.json`。模型许可逐项保留在公开 release 清单中。
<!-- /converted-batch-20261102 -->
