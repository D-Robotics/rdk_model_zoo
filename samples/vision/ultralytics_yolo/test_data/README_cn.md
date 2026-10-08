[English](README.md) | 简体中文

# Ultralytics YOLO 测试数据

本目录提供 11 个固定文件：输入图片、运行时显示标签和参考结果插图。

1. **输入图片**：`bus.jpg` 与 `zebra_cls.jpg`，用于 Runtime、C++ 与转换指南中的 `--test-img`。
2. **显示标签**：三份 `.names` 表，将模型输出 ID 映射为打印和绘图使用的名称。
3. **参考插图**：`result_detect*.jpg` 与四张 `ultralytics_YOLO_*_demo` 截图，用于查看各任务的结果样式。

<a id="files"></a>

## 目录结构

```text
test_data/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── coco_classes.names  # 源码或数据文件
├── imagenet_classes.names  # 源码或数据文件
└── ultralytics_dota_classes.names  # 源码或数据文件
```

## 文件清单与逐字节标识

| 文件 | 角色 | 像素 | SHA-256 |
| --- | --- | --- | --- |
| `bus.jpg` | 默认检测/分割/姿态 CLI 输入 | 810×1080 | `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63` |
| `zebra_cls.jpg` | 分类 CLI 输入 | 376×376 | `53c9f26d927b507fb3b9b68005fd8dd3ba329a0528f9c1f51acbde55e9525462` |
| `coco_classes.names` | detect/seg/pose 显示标签（80 项） | — | `634a1132eb33f8091d60f2c346ababe8b905ae08387037aed883953b7329af84` |
| `imagenet_classes.names` | cls 显示标签（1000 项） | — | `e6ac1e05778e809de37a089105170398e5d0b5f7e337165c94aeaacb80fbba14` |
| `ultralytics_dota_classes.names` | obb 显示标签（15 项） | — | `a6c8b62b2dae0ddc151a4cbae52b8db84542e2ca0e254e5214d86be9d180b6b9` |
| `result_detect.jpg` | S 参考检测示意图（bus 场景） | 810×1080 | `5d792a474744924eb31b37e2f0d888931955151785cfba953104549462afa218` |
| `result_detect_yolo26.jpg` | S 参考 `ultralytics_yolo26` 检测示意图 | 810×1080 | `2631c66105d3e65373db1b60e2f968e4a54663b039646f38bfcf2355d53375c3` |
| `ultralytics_YOLO_Detect_demo.jpg` | 检测参考截图，[sample 指南](../README_cn.md#expected-results)引用 | 2900×1888 | `926ad7e306cc471adf5d3b3e237199cd02d1b4648a96c1a214d984e9747cba6f` |
| `ultralytics_YOLO_Pose_demo.jpg` | 姿态参考截图，参考结果样式 | 2898×1888 | `194ffd24552ae45ae6a526248fc7293f8e44de0bb209b5647f2016c67f782b1c` |
| `ultralytics_YOLO_Seg_demo.jpg` | 分割参考截图，参考结果样式 | 2898×1888 | `a80979ccab290ec5f56116ab554fc1ce1ceed1f0cabcb139371611dab4507731` |
| `ultralytics_YOLO_CLS_demo.png` | 分类参考截图，参考结果样式 | 2846×1268 | `7ea5f2288ad274dc3666b2b8351414dbeb48787348fc7ec272299a0c6b61329f` |

上述观测 SHA-256 用于核对文件内容及检测字节变化。

本仓内已核验的逐字节同一关系：

- `bus.jpg` 与 [`datasets/coco/assets/bus.jpg`](../../../../datasets/coco/README.md) 以及归档交付副本 `platforms/x5/samples/vision/ultralytics_yolo/test_data/`、`platforms/s/samples/vision/ultralytics_yolo/test_data/` 逐字节相同。
- `zebra_cls.jpg` 与 [`datasets/imagenet/asset/zebra_cls.jpg`](../../../../datasets/imagenet/README.md) 及 `samples/vision/resnet/test_data/zebra_cls.jpg` 逐字节相同。
- `coco_classes.names` 与 [`datasets/coco/coco_classes.names`](../../../../datasets/coco/README.md) 逐字节相同；`imagenet_classes.names` 与 [`datasets/imagenet/imagenet_classes.names`](../../../../datasets/imagenet/README.md) 逐字节相同。每套词表在本仓只有一份内容。
- `result_detect.jpg` 与归档 S 交付中的同名文件（`platforms/s/samples/vision/ultralytics_yolo/test_data/`）逐字节相同；`result_detect_yolo26.jpg` 是 S `ultralytics_yolo26` 交付的 `result_detect.jpg`，在本目录以独立文件名携带。两图字节不同、图上标签也不同，均按各自来源身份保留。

<a id="labels"></a>
## 标签表：仅用于显示

Python CLI 自动加载这些表。`main.py` 对 detect/seg/pose 默认使用 `coco_classes.names`，对 cls 默认使用 `imagenet_classes.names`，对 obb 默认使用 `ultralytics_dota_classes.names`；任何任务都可用 `--label-file` 覆盖。

### coco_classes.names

80 行非空文本，每行一个显示名；第 N 行命名模型输出索引 N−1（从 0 开始）。顺序即标准 Ultralytics COCO-80 输出顺序（索引 0 `person` … 索引 79 `toothbrush`）。其中 6 项使用历史 VOC 风格拼写而非 COCO 规范名——索引 3 `motorbike`、4 `aeroplane`、57 `sofa`、58 `pottedplant`、60 `diningtable`、62 `tvmonitor`——见 [COCO 数据集指南](../../../../datasets/coco/README.md)。两种拼写命名的是同一批输出列；不要对该文件重新编号。

### imagenet_classes.names

覆盖全部 1000 类的 Python dict 字面量，形如 `{index: '逗号分隔的同义名'}`，键为标准 ILSVRC-2012 顺序的 0–999（索引 0 `tench, Tinca tinca`、索引 340 `zebra`、索引 999 `toilet tissue, toilet paper, bathroom tissue`）；文件末尾无换行。值为人类可读的显示名，不是 WordNet synset ID（`n########`），也不能当作 synset 使用。加载器既接受该 dict 形式，也接受一行一名的列表。

### ultralytics_dota_classes.names

15 行，命名 YOLO26 OBB 模型的 15 个输出列，顺序即该模型自身的输出顺序：`plane` 0、`ship` 1、`storage-tank` 2、`baseball-diamond` 3、`tennis-court` 4、`basketball-court` 5、`ground-track-field` 6、`harbor` 7、`bridge` 8、`large-vehicle` 9、`small-vehicle` 10、`helicopter` 11、`roundabout` 12、`soccer-ball-field` 13、`swimming-pool` 14。DOTA 原生标注没有官方数字类别 ID 表；这套编号只是模型输出顺序，别无他意。它与 [`datasets/dotav1/dota_classes.names`](../../../../datasets/dotav1/README.md)（本仓固定列表）的顺序**不同**：两者仅在索引 0 相同。用其中一份去命名按另一份顺序工作的输出，会静默错标 15 类中的 14 类。完整顺序警告见 [DOTA 数据集指南](../../../../datasets/dotav1/README.md)。

### 自定义模型与不匹配行为

若编译模型的类别数或顺序不同，请传入匹配的 `--label-file`（每行一名，按模型输出顺序；cls 另可接受 dict 字面量格式），自定义 detect/seg/obb 图还需传匹配的 `--classes-num`。顺序错误不会大声失败：预测的框和分数不变，但所有打印与绘制的名称都会错位。当类别 ID 超出已加载标签表范围时，实际行为如下：

- 检测文本报告打印数字 ID 代替名称。
- OBB 绘图打印数字 ID；分类打印 `Unknown(<id>)`。
- detect/seg 渲染图片时直接按索引取标签表，因此会在写出结果图之前以 `IndexError` 中止。

<a id="inputs"></a>
## 使用随仓与自定义图片

`--test-img` 接受任何 OpenCV 可读的三通道 BGR 图片；任务会将其缩放到模型输入几何（默认 letterbox，例外见 [Python runtime 指南](../runtime/python/README_cn.md#parameters)），所有结果都会还原为原图像素坐标。文件名永远不是输入覆盖：输入几何由模型元数据决定。

随仓图片就是文档指定的固定输入。从仓库根目录在匹配板卡上运行；完全不带参数时自动识别板卡，并以 yolo11 默认尺度对 `bus.jpg` 做检测：

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py
```

分类使用 `zebra_cls.jpg`：

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s600 --family yolo26 --task cls \
  --test-img samples/vision/ultralytics_yolo/test_data/zebra_cls.jpg --topk 5
```

使用自己的图片时，把路径传给 `--test-img`；下面是 runtime 指南中“自定义图片 + 显式本地模型”的既有示例形式：

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s600 --family yolo11 --task detect \
  --model-path /models/yolo11n_nashp_640x640_nv12.hbm \
  --test-img /data/image.jpg --img-save-path /tmp/yolo-s600.jpg
```

把 `/data/image.jpg` 换成任何可读图片，把模型路径换成你准备好的制品（见[模型说明](../model/README_cn.md)）。要在自定义图片上使用自定义类别表，在同样形式上加 `--label-file`：

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolov8 --task detect \
  --test-img /data/image.jpg --label-file /data/my_classes.txt \
  --img-save-path /tmp/custom-detect.jpg
```

本目录不随仓提供航拍/OBB 输入：obb 请自备航拍图片，也不要把 `bus.jpg` 的 OBB 输出当作航拍检测预期——[YOLO26 OBB 阶段](../runtime/python/README_cn.md#yolo26-obb-stages)用随仓 bus 图仅演示 API 调用方式。

### 输出位置与终端语义

detect/seg/pose/obb 将渲染图写入 `--img-save-path`——默认 `result.jpg`，相对调用时的工作目录——会创建其父目录、覆盖同名旧文件，成功时打印 `[Saved] Result saved to: <path>`。分类只打印结果，不写图片。退出码 0 表示命令执行完成；空检测列表是合法结果。以下为占位值表示的终端格式——只是格式模板，不是测量值：

```text
Detection Report: N objects found
  [0] <class name>: <score> | Box (<x1>, <y1>, <x2>, <y2>)
```

```text
Top-K Classification Results:
  [0] <label>: <probability>
```

`<score>` 与 `<probability>` 是任务各自后处理之后的模型置信度（检测经阈值/NMS 过滤；分类经 Softmax），分别保留两位与四位小数；框使用原图像素坐标，类别 ID 从 0 开始。除每个任务都会打印的模型/协议信息外，detect 打印逐目标报告、分类打印 Top-K；seg、pose、obb 通过渲染图表达结果。数据集精度与性能指标通过[评估器](../evaluator/README_cn.md)测量。

C++ 参考程序接受同样的图片作为位置参数（其[文档](../runtime/cpp/README_cn.md)将它们与 `test_data/bus.jpg`、`test_data/zebra_cls.jpg` 配对），但**不读取**本目录的 `.names` 文件：detect/segment 编译了 80 项 COCO 名称数组（使用规范拼写 `motorcycle`、`airplane`、`couch`、`potted plant`、`dining table`、`tv`——与本目录的 VOC 风格同义词有 6 项不同，顺序相同）；pose 编译 17 个 COCO 关键点名；分类编译 `common/imagenet_labels.h` 中 1000 项 ImageNet 顺序。转换后的编译校验同样以 `test_data/bus.jpg` 作为示例输入（见[转换指南](../conversion/README_cn.md#validation)）。

<a id="boundaries"></a>
## 数据准备

检测、分割和姿态评估使用 COCO 标注 JSON。ImageNet 真值通过 `--val-txt`
或按类别顺序排列的 synset ID `--label-file` 提供；评估器的标签文件使用
`n########` 标识，运行时显示名使用本目录的 `imagenet_classes.names`。OBB
入口导出预测。数据准备见 [COCO](../../../../datasets/coco/README_cn.md)、
[ImageNet](../../../../datasets/imagenet/README_cn.md) 与
[DOTA](../../../../datasets/dotav1/README_cn.md)，参数见
[评估器](../evaluator/README_cn.md#dataset)。

校准样本按[转换配方](../conversion/README_cn.md#calibration)从代表性数据集选取。
生成推理结果时指定新的输出路径，并比较本次成功运行写出的文件。主机回归测试的
合成输入在 `../tests/` 中构造。

<a id="provenance"></a>
## 来源

检测结果参考图使用随仓 `bus.jpg`。生成自己的检测、姿态、分割或分类结果时，请为 `--img-save-path` 指定新的输出路径。

`ultralytics_YOLO_*_demo` 文件为示例截图。运行路径与阈值请按当前运行时参数表和命令设置。

数据集计分使用[评估指南](../evaluator/README_cn.md)中带标注的验证输入。
