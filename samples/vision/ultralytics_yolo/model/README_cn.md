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
| `yolov13` | detect | n/s/l/x；X5 与 S100（S100 使用 iMoonLab 原始记录） |

有发布制品不代表所有组合均完成板测。不能用另一目标的模型替代。统一 X5 完整列表共 92 个制品，其中 YOLO26 为 25 个；历史 X5 通用下载包装器仍保留原来的 67 个。

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
| `--asset-id` | 默认无；精确选择下文的独立 S 源制品，不能与 `--all` 同用。 |
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

[测试数据目录](../test_data) 包含检测/分割/姿态使用的 `bus.jpg`、分类使用的 `zebra_cls.jpg`、COCO/ImageNet/DOTA 类别名称和历史结果图。标签用于解释输出，不是模型权重；自定义模型须使用匹配的类别顺序。历史示意图不证明本次运行成功。

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

历史平台下载包装器仍写入原模型目录。统一运行入口使用本 Sample 的 model 目录，复用旧下载请传 `--model-path`。把 `.bin` 改名为 `.hbm` 或修改 nash-e/m/p 名称不能改变硬件目标。

<a id="formats-checksums"></a>
## 格式与完整性

X5 使用 packed NV12 输入，S 使用独立 Y/UV 输入。非分类文件名使用 640×640。YOLO26 分类文件名在全部目标上使用 224×224，S600 分类标识同样使用 224×224。S100/S100P v8/v11 分类清单 ID 保留 640×640 兼容名称，下载 URL 则使用 224×224。2026-09-26 只读 HEAD 核对发现全部 20 对地址均可访问，长度与 ETag 相同；这不是完整字节的密码学比较，也不能证明张量尺寸。实际输入几何由运行元数据决定，文件名标记不是形状覆盖。既有限定 ID 与本地路径继续保留。

下载器拒绝空文件；清单提供发布方 SHA-256 时会校验。部分制品没有发布方哈希，此时本地摘要只能标识字节，**不能证明官方来源**。dry-run 只查看路径是否存在，`present` 不代表完整性已验证。下载使用临时 `.part` 文件，校验成功后才安装最终文件。超时、HTTP 错误或 URL 不可用属于准备失败，不能据此换用另一制品。


<a id="standalone-assets"></a>
## S 独立样例的原始制品身份

现有十个独立样例制品可直接选择，不改名为 Ultralytics 家族制品。这是主机准备和入口绑定能力；
B9 数值与 C++ 归并仍在核查，本轮没有新增板端验证。

| 原 sample ID | task / family | 已发布目标 / 变体 | 源 Python NMS 默认值 |
|---|---|---|---|
| `yolo11` | detect / yolo11 | S100、S600 / n | 0.45 |
| `yolo11_pose` | pose / yolo11 | S100、S600 / n | 0.70 |
| `yolo11_seg` | seg / yolo11 | S100、S600 / n | 0.70 |
| `yolov13_imoonlab` | detect / yolov13 | S100 / n,s,l,x | 0.45 |

下载地址仍以 S 主清单为准。文件保存到 `model/standalone/<原sample-id>/<清单filename>`，
避免同名的旧制品覆盖家族目录中的文件。S600 原始文件在 `s600/` 路径下仍带 `nashe`：
这里只保留发布记录，不代表已检查 HBM 的实际 march。没有登记独立 S100P 制品。

从仓库根目录分别检查准备计划与推理路由：

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform s100 --asset-id s:yolo11_pose:s100/yolo11n_pose_nashe_640x640_nv12.hbm --dry-run
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s100 --task pose \
  --asset-id s:yolo11_pose:s100/yolo11n_pose_nashe_640x640_nv12.hbm --dry-run
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform s100 --family yolov13 --task detect --model-size n --dry-run
```

需要显式下载时，只去掉下载命令中的 `--dry-run`。下载器新增的 `--asset-id` 仅接受上述独立
制品 ID，拒绝与 `--all` 或冲突的 family/task/size 同用。`--all` 枚举家族清单（现包含 S100
的 iMoonLab），不重复下载独立 YOLO11 制品；这些制品须用精确 ID 选择。原默认下载集合不变。
运行入口同时接受原家族 ID 与这些源 ID，须传正确 `--task`。选择源 ID 时，未显式设置的
NMS 使用表中默认值；显式 `--nms-thres` 优先。普通 `--family yolo11` 仍采用 S 默认 0.45。
输入图片与调度参数保留统一 CLI 默认值；对照源流程时显式指定原图片和 `--priority 0 --bpu-cores 0`。

显式 `--model-path` 保留本地路径且不自动准备模型；同时给 asset ID 只是声明预期选择，
不是文件身份的哈希证明。dry-run 不验证 SDK。源检测/分割封装支持量化输出；源姿态 API 返回
关键点 logits，统一 API 返回概率。完整数值/源能力对齐仍在核查，详见
[未关闭的归并记录](../../../../docs/releases/unified-migration/2026-09-27-b9-source-consolidation-review.md)。
