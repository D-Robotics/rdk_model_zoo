[English](README.md) | 简体中文

# RepGhost 模型制品

<a id="artifacts"></a>
## 制品

五个文件均为 X5 发布清单中的单阶段分类模型；S 系列没有 RepGhost 制品。

| Variant | Filename | Target | Format | Source |
| --- | --- | --- | --- | --- |
| `100` | `RepGhost_100_224x224_nv12.bin` | x5 | bin | download |
| `111` | `RepGhost_111_224x224_nv12.bin` | x5 | bin | download |
| `130` | `RepGhost_130_224x224_nv12.bin` | x5 | bin | download |
| `150` | `RepGhost_150_224x224_nv12.bin` | x5 | bin | download |
| `200` | `RepGhost_200_224x224_nv12.bin` | x5 | bin | download |

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

cwd：仓库根。Shell 使用位置参数，Python 使用命名选项。两种形式省略变体均选择 `100`。

```bash
bash samples/vision/repghost/model/download.sh x5 100
python3 samples/vision/repghost/model/download.py --target x5 --variant 200
```

输出：退出码 0，打印目标文件与观测 SHA-256。下载先写临时文件，不覆盖已有字节；大小或哈希不匹配时显式报错。服务器不可用时可手动传入匹配的发布文件至 `model/`，记录来源及观测哈希，不能用其他变体替代。

<a id="accompanying-files"></a>
## 配套文件

CLI 读取 `datasets/imagenet/imagenet_classes.names` 标签与默认 `test_data/ibex.JPEG` 图片。API 接收内存图像和可选标签；没有标签时返回类别 ID 字符串。无需 tokenizer 或额外模型。

<a id="local-paths"></a>
## 本地路径

文件保存为 `samples/vision/repghost/model/<filename>`，默认运行选择该目录的 `RepGhost_100_224x224_nv12.bin`。外部 `--model-path` 必须同时指定精确 `--asset-id`，如 `x5:repghost:RepGhost_100_224x224_nv12.bin`。

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
| `timm-100-in1k` | `repghostnet_100_in1k_bayese_224x224_nv12.bin` | [`3c79eee5f138`](https://huggingface.co/timm/repghostnet_100.in1k/blob/3c79eee5f138318c2cb58957105b91bb50259b29/README.md) | not-run |
| `timm-111-in1k` | `repghostnet_111_in1k_bayese_224x224_nv12.bin` | [`03ffb284f8ea`](https://huggingface.co/timm/repghostnet_111.in1k/blob/03ffb284f8ea7ae1c41a1c2617d4dab98d7cd9f4/README.md) | not-run |
| `timm-130-in1k` | `repghostnet_130_in1k_bayese_224x224_nv12.bin` | [`efb0a7881fad`](https://huggingface.co/timm/repghostnet_130.in1k/blob/efb0a7881fada70cec140ce2810f9c0a7d416704/README.md) | not-run |
| `timm-150-in1k` | `repghostnet_150_in1k_bayese_224x224_nv12.bin` | [`577d935410a6`](https://huggingface.co/timm/repghostnet_150.in1k/blob/577d935410a68c6efc6a0fee9f11f88f942ac690/README.md) | not-run |
| `timm-200-in1k` | `repghostnet_200_in1k_bayese_224x224_nv12.bin` | [`0c520f5e8de7`](https://huggingface.co/timm/repghostnet_200.in1k/blob/0c520f5e8de77f9bfedc23f2104c344b1d685530/README.md) | not-run |

```bash
python3 samples/vision/repghost/model/download.py --target x5 --variant timm-100-in1k
python3 samples/vision/repghost/runtime/python/main.py --target x5 --variant timm-100-in1k --dry-run
```

精确 URL、SHA-256 和大小登记在 `docs/release/x5/models.yaml`；来源及主机对照
记录见 `conversion/converted.json`。模型许可逐项保留在公开 release 清单中。
<!-- /converted-batch-20261102 -->

<!-- converted-all-20261103 -->

## 本轮 S 系列转换规格（2026-11-03）

这些独立规格已完成转换、完整 OE 和实际 HBM 接口核验。源权重、ONNX、每目标配置、校准数量和摘要见 [converted_targets_20261103.json](../conversion/converted_targets_20261103.json)。

| 目标 | Native variant | 组件 | 校准图片 |
| --- | --- | --- | ---: |
| s100 | `timm-100-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-100-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-100-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-111-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-111-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-111-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-130-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-130-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-130-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-150-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-150-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-150-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-200-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-200-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-200-in1k-s-int16-o1-20261103` | model | 200 |

```bash
python3 samples/vision/repghost/model/download.py --target s100 --variant timm-100-in1k-s-int16-o1-20261103
python3 samples/vision/repghost/runtime/python/main.py --target s100 --variant timm-100-in1k-s-int16-o1-20261103 --asset-id s:repghost:s100/repghostnet_100_in1k_nashe_224x224_nv12_float_int16_20261103_o1.hbm --model-path /path/to/repghostnet_100_in1k_nashe_224x224_nv12_float_int16_20261103_o1.hbm --test-img samples/vision/repghost/test_data/ibex.JPEG
```

命令中的路径需要替换为实际文件。每套多组件模型须同时准备全部组件。旧默认与既有指标保留原身份；本轮规格的板端 smoke、数据集精度、Runtime 1/2 并发和 C++ 1/2 流均为 `not-run`。
