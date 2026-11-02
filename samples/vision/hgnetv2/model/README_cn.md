[English](README.md) | 简体中文

# HGNetV2 模型制品

<a id="artifacts"></a>
## 制品

此处列出 X5 发布清单中的单阶段分类制品；目标与变体选择见根支持矩阵。

| Variant | Filename | Target | Format | Source |
| --- | --- | --- | --- | --- |
| `b0` | `hgnetv2_b0_224x224_nv12.bin` | x5 | bin | download |
| `b1` | `hgnetv2_b1_224x224_nv12.bin` | x5 | bin | download |
| `b2` | `hgnetv2_b2_224x224_nv12.bin` | x5 | bin | download |
| `b3` | `hgnetv2_b3_224x224_nv12.bin` | x5 | bin | download |
| `b4` | `hgnetv2_b4_224x224_nv12.bin` | x5 | bin | download |

### X5 文件体积

X5 发布模型表（`rdk_x5 @ e3f9fa3fb5a795b2531bdb84fa60d03768af5956`）
以 MB 列出下列 BIN 文件的近似体积。下载校验使用
`docs/release/x5/models.yaml` 中的精确字节数。

| 变种 | 文件名 | 体积 |
| --- | --- | --- |
| HGNetV2 b0 | `hgnetv2_b0_224x224_nv12.bin` | ~5.9 MB |
| HGNetV2 b1 | `hgnetv2_b1_224x224_nv12.bin` | ~6.2 MB |
| HGNetV2 b2 | `hgnetv2_b2_224x224_nv12.bin` | ~11 MB |
| HGNetV2 b3 | `hgnetv2_b3_224x224_nv12.bin` | ~16 MB |
| HGNetV2 b4 | `hgnetv2_b4_224x224_nv12.bin` | ~19 MB |

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

cwd：仓库根。Shell 使用位置参数，Python 使用命名选项。两种形式省略变体均选择 `b0`。

```bash
bash samples/vision/hgnetv2/model/download.sh x5 b0
python3 samples/vision/hgnetv2/model/download.py --target x5 --variant b4
```

输出：退出码 0，打印目标文件与观测 SHA-256。下载先写临时文件，不覆盖已有字节；大小或哈希不匹配时显式报错。服务器不可用时可手动传入匹配的发布文件至 `model/`，记录来源及观测哈希，不能用其他变体替代。

<a id="accompanying-files"></a>
## 配套文件

CLI 读取 `datasets/imagenet/imagenet_classes.names` 标签与默认 `test_data/sandbar.JPEG` 图片。API 接收内存图像和可选标签；没有标签时返回类别 ID 字符串。无需 tokenizer 或额外模型。

<a id="local-paths"></a>
## 本地路径

文件保存为 `samples/vision/hgnetv2/model/<filename>`，默认运行选择该目录的 `hgnetv2_b0_224x224_nv12.bin`。外部 `--model-path` 必须同时指定精确 `--asset-id`，如 `x5:hgnetv2:hgnetv2_b0_224x224_nv12.bin`。

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
| `timm-b0-ssld-stage2-ft-in1k` | `hgnetv2_b0_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`9e0a904b210d`](https://huggingface.co/timm/hgnetv2_b0.ssld_stage2_ft_in1k/blob/9e0a904b210d65e4fcd6582f8ca6e0921fac2650/README.md) | not-run |
| `timm-b1-ssld-stage2-ft-in1k` | `hgnetv2_b1_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`ec7014d7f1d0`](https://huggingface.co/timm/hgnetv2_b1.ssld_stage2_ft_in1k/blob/ec7014d7f1d0dd1ea2bf3012bc51dc95c5919fd2/README.md) | not-run |
| `timm-b2-ssld-stage2-ft-in1k` | `hgnetv2_b2_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`7a3092bc8711`](https://huggingface.co/timm/hgnetv2_b2.ssld_stage2_ft_in1k/blob/7a3092bc87118c7974bb9000b1eb6d9c560f8f75/README.md) | not-run |
| `timm-b3-ssld-stage2-ft-in1k` | `hgnetv2_b3_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`1aac877b3ea5`](https://huggingface.co/timm/hgnetv2_b3.ssld_stage2_ft_in1k/blob/1aac877b3ea55ab0f06d57cad9889d1e601bd472/README.md) | not-run |
| `timm-b4-ssld-stage2-ft-in1k` | `hgnetv2_b4_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`3c808d9d9d98`](https://huggingface.co/timm/hgnetv2_b4.ssld_stage2_ft_in1k/blob/3c808d9d9d9873ed5e37b00509ba2a4882e816e8/README.md) | not-run |

```bash
python3 samples/vision/hgnetv2/model/download.py --target x5 --variant timm-b0-ssld-stage2-ft-in1k
python3 samples/vision/hgnetv2/runtime/python/main.py --target x5 --variant timm-b0-ssld-stage2-ft-in1k --dry-run
```

精确 URL、SHA-256 和大小登记在 `docs/release/x5/models.yaml`；来源及主机对照
记录见 `conversion/converted.json`。模型许可逐项保留在公开 release 清单中。
<!-- /converted-batch-20261102 -->
