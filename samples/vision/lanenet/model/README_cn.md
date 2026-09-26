[English](README.md) | [简体中文](README_cn.md)

# LaneNet 模型资产

<a id="artifacts"></a>
## 已发布资产

| 资产 ID | 目标 | 本地文件名 | 发布方 SHA-256 |
| --- | --- | --- | --- |
| `s:lanenet:s100/lanenet256x512.hbm` | S100 / nash-e | `s100/lanenet256x512.hbm` | 未知 |

实际下载 URL 与资产身份以 [S 发布清单](../../../../platforms/s/docs/release/models.yaml)为准。这是唯一已发布的 LaneNet 资产。S100P、S600、X5 在此均无对应资产；重命名或移动 S100 HBM 不会增加平台支持。原入口保留于[源模型目录](../../../../platforms/s/samples/vision/lanenet/model)。

<a id="preparation"></a>
## 显式准备

从仓库根目录执行：

```bash
bash samples/vision/lanenet/model/download.sh --target s100
```

兼容包装入口 `download_model.sh` 委托同一个下载器。推理不触发下载。指定其他存储根目录：

```bash
python3 -m samples.vision.lanenet.model.download --target s100 --output-dir /data/lanenet-models
```

| 下载参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--target` | `s100` | 唯一支持的已发布目标 |
| `--asset-id` | `s:lanenet:s100/lanenet256x512.hbm` | 必须精确匹配清单身份 |
| `--output-dir` | 当前 `model` 目录 | 自动追加资产相对路径 `s100/lanenet256x512.hbm` |

<a id="accompanying-files"></a>
## 配套文件

无需类别名称文件。运行需要 HBM 与输入图片；源输入图像为 [test_data/lane.jpg](../test_data/lane.jpg)。结果不是具名类别或车道实例列表。[转换目录](../conversion/README_cn.md)保留编译 YAML 和历史 checkpoint URL；checkpoint 本身不是可部署模型，源导出脚本仍缺失。

<a id="local-paths"></a>
## 本地路径与身份

默认推理解析到本目录的 `s100/lanenet256x512.hbm`。外部路径必须同时提供明确契约身份：

```bash
python3 -m samples.vision.lanenet.runtime.python.main --target s100 --asset-id s:lanenet:s100/lanenet256x512.hbm --model-path /data/lanenet-models/s100/lanenet256x512.hbm --dry-run
```

Dry-run 不打开 HBM；实际加载会检查板卡身份与模型真实元数据。匹配的资产 ID 只是调用者声明了预期契约，不证明任意提供的字节属于已发布模型。本地编译产物同样需要完整元数据与数值验证；不能把测得的本地摘要写成发布方校验和。

<a id="formats-checksums"></a>
## 格式与校验和

HBM 契约为单个 float32 RGB NCHW `[1,3,256,512]` 输入，由 sample 完成 ImageNet 归一化。必需输出为 float32 `[1,3,256,512]` 嵌入张量，以及离散 int64 `[1,1,256,512]` 或 `[1,256,512]` 二值张量。Python 按 `instance_seg_logits`、`binary_seg_pred` 绑定，原生代码按唯一形状/类型绑定角色。额外观察到的输出原样保留，不依据源文案虚构第三个名称。

下载器打印实际 SHA-256。清单没有发布方 SHA-256，因此它可用于跨主机跟踪字节身份，但不能独立认证下载内容。运行报告记录实际摘要供后续比较。本次主机迁移验证未下载模型，也未在板端加载模型。
