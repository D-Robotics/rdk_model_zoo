# Ultralytics YOLO 模型制品

[English](README.md) · [Sample](../README_cn.md) · [Python 运行](../runtime/python/README_cn.md)

<a id="artifacts"></a>
## 已发布制品

这些脚本准备推理所用的编译模型，不负责训练权重的导出或编译。下载地址和精确制品身份来自 [X5 清单](../../../../docs/release/x5/models.yaml) 或 [S 清单](../../../../docs/release/s/models.yaml)。需要完整仓库；单独复制本目录会缺少注册表和共享下载器。

| 目标 | 格式 | 本目录下的默认位置 | 硬件/编译目标 |
|---|---|---|---|
| `x5` | `.bin` | 文件直接放在本目录 | X5 / Bayes |
| `s100` | `.hbm` | `nash-e/` | S100 / Nash-e |
| `s100p` | `.hbm` | `nash-m/` | S100P / Nash-m |
| `s600` | `.hbm` | `nash-p/` | S600 / Nash-p |

| 系列 | 已发布任务 | 尺度与限制 |
|---|---|---|
| `yolo26` | detect、seg、pose、cls、obb | n/s/m/l/x；每个目标 25 个制品 |
| `yolov5u` | detect | n/s/m/l/x |
| `yolov8`、`yolo11` | detect、seg、pose、cls | n/s/m/l/x |
| `yolov9` | detect、seg | 检测 t/s/m/c/e，S600 无 t；分割仅 c/e，S600 无分割 |
| `yolov10` | detect | n/s/m/b/l/x |
| `yolo12` | detect | n/s/m/l/x |
| `yolov13` | detect | n/s/l/x；仅 X5 |

按清单选择精确的目标、系列、任务和尺度，并使用为该目标编译的制品。

<a id="preparation"></a>
## 准备模型

以下命令都从**仓库根目录**执行。脚本使用 `python3`、读取清单所需的 PyYAML 和仓库内的 Python 辅助模块；帮助和 dry-run 不需要板端运行时或网络。实际下载需要网络和目标目录写权限。先查看一个模型的下载计划：

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform x5 --family yolov8 --task detect --model-size n --dry-run
```
计划显示目标、制品数量、本地路径、URL 和路径是否存在。下载该组合：

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform x5 --family yolov8 --task detect --model-size n
```
下载成功退出码为 0。查看运行入口 `--asset-id` 所接受的精确清单引用：

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py --platform x5 --list-models
```
| 参数 | 含义与默认值 |
|---|---|
| `--platform` | x5/s100/s100p/s600；省略时检测板卡。主机上显式指定。 |
| `--family` | 默认 yolo11；配合 `--all` 时省略表示所有系列。 |
| `--task` | detect/seg/pose/cls/obb；省略时 X5 下载 detect/seg/pose/cls，S 只下载 detect。 |
| `--model-size` | 通常默认 n；YOLOv9 检测为 X5 t、S s。YOLOv9 分割默认 c，可显式选择 c/e。 |
| `--model-dir` | 基础存储目录，默认是本 Sample 的 model 目录。 |
| `--all` | 全部已发布制品，可按 family 限定；此模式不按 task、size 筛选。 |
| `--dry-run` | 仅打印计划，不下载、不加载模型。 |

某系列不支持平台默认的全部任务时，请明确传入任务。不支持的组合会报错，不会换一个近似模型。仍兼容历史位置参数形式；具名选项优先：

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh s600 yolov8 cls n --dry-run
```
完整下载可能较大，去掉 `--dry-run` 前先检查列表：

```bash
bash samples/vision/ultralytics_yolo/model/fulldownload.sh \
  --platform x5 --family yolo26 --dry-run
bash samples/vision/ultralytics_yolo/model/fulldownload.sh \
  --platform s100 --dry-run
```
`fulldownload.sh` 向同一个解析器传入 `--all`，无需维护另一份模型列表。

<a id="accompanying-files"></a>
## 输入、标签与后续步骤

[测试数据目录](../test_data) 包含检测/分割/姿态使用的 `bus.jpg`、分类使用的 `zebra_cls.jpg`、COCO/ImageNet/DOTA 类别名称和结果示意图。标签用于解释输出；自定义模型须使用匹配的类别顺序。

后续见 [Python 推理](../runtime/python/README_cn.md)、[C++ 可用范围](../runtime/cpp/README_cn.md)、[模型转换](../conversion/README_cn.md) 和 [数据集评估](../evaluator/README_cn.md)。ONNX 和训练权重不能直接替代板端二进制，源模型与工具链要求见转换说明。

<a id="local-paths"></a>
## 存储位置与离线使用

默认目录不随当前 shell 工作目录变化。自定义基础目录时，S 目标仍添加对应 march 子目录：

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform s600 --family yolo26 --task cls --model-size n \
  --model-dir /tmp/rdk-models --dry-run
```
此例解析到 `/tmp/rdk-models/nash-p/` 下。去掉 `--dry-run` 下载，再把编译文件复制到匹配的板卡，运行时通过 `--model-path` 传入完整路径。显式运行路径不会被自动替换或下载。下载目标已存在时，下载器校验文件且不覆盖；不需要或损坏的文件请先移到其他位置再重试。

模型存放在默认目录之外时，运行入口通过 `--model-path` 接收完整路径。制品扩展名和 nash-e/m/p 目录须匹配目标。

<a id="formats-checksums"></a>
## 格式与完整性

X5 使用 packed NV12 输入，S 使用独立 Y/UV 输入。非分类文件名使用 640×640。YOLO26 分类文件名在全部目标上使用 224×224，S600 分类标识同样使用 224×224。S100/S100P v8/v11 分类清单 ID 保留 640×640 兼容名称，下载 URL 则使用 224×224。实际输入几何由运行元数据决定，文件名标记不是形状覆盖。

下载器拒绝空文件；清单提供发布方 SHA-256 时会校验。本地摘要用于标识下载字节。`--dry-run` 报告路径是否存在，不下载或读取文件内容。下载使用临时 `.part` 文件，校验成功后才安装最终文件。超时、HTTP 错误或 URL 不可用会返回准备错误。

<a id="maintained-scope"></a>
## 运行输出要求

运行入口 `--asset-id` 接受已登记的 Ultralytics YOLO/YOLO26 身份；使用上文的
family/task/size 命令准备模型。检测、DFL 分割/姿态使用有限浮点输出，量化
metadata 为 NONE 或不存在。Python 后处理消费运行时提供的浮点输出。选择输出类型
匹配的制品；运行时在加载时核对张量 shape 和 dtype，并按目标选择对应制品。
