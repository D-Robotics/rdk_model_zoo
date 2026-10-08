[English](README.md) | 简体中文

# Ultralytics YOLO 模型评估


评估器复用[Python运行时](../runtime/python/README.md)的任务实现。检测、实例分割、姿态估计使用COCO指标；分类使用ImageNet Top-1/Top-5；YOLO26旋转框只导出预测。本目录不下载模型或数据集、不编译模型，也不测量纯BPU延迟。

<a id="dataset"></a>
## 数据集准备

准备与模型类别顺序一致的验证集。获取和整理方法见统一的[COCO](../../../../datasets/coco/README_cn.md)与[ImageNet](../../../../datasets/imagenet/README_cn.md)指南；X5/S 平台快照保留作溯源（X5 COCO、S COCO、X5 ImageNet、S ImageNet）。数据集不随仓库分发，使用时遵守各自许可。

以下示例约定本地目录如下；请将`/data`、`/models`替换为实际准备路径：

```text
/data/coco/val2017/000000000139.jpg
/data/coco/annotations/instances_val2017.json
/data/coco/annotations/person_keypoints_val2017.json
/data/imagenet/val/...
/data/imagenet/val.txt
/data/dota/images/...
/models/                         # 为当前具体板型编译的模型
```

COCO val2017含5,000张图像。检测/分割使用instances标注，姿态使用person_keypoints标注。有标注时按标注中的文件名和图像ID读取；不提供标注时，COCO导出要求图像文件名主体是数字。`--limit N`选择子集，0表示全部。即使标注只有部分类别，仍按标准COCO输出索引映射category ID，不支持自定义类别顺序。

ImageNet的named格式每行是`<相对图像路径> <从0开始的类别索引>`。也可以用`--label-file`按模型类别顺序列出synset ID，此时文件名应包含对应`n########`标识。两种标签来源必须且只能选择一种。原X5每行一个标签的验证列表使用`--val-format ordered --val-txt FILE --label-offset -1`，标签行顺序必须对应排序后的图像文件名。已从0编号的标签不要再减1。

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── eval_batch.py  # Python 脚本
├── eval_common.py  # Python 脚本
├── eval_yolo_cls.py  # Python 脚本
├── eval_yolo_det.py  # Python 脚本
├── eval_yolo_obb.py  # Python 脚本
├── eval_yolo_pose.py  # Python 脚本
└── eval_yolo_seg.py  # Python 脚本
```

<a id="environment"></a>
## 环境

实际推理在目标RDK板上运行，需要系统镜像匹配的`hbm_runtime`、Python、NumPy、OpenCV、SciPy及完整仓库。主机编译器包不能代替板端SDK。COCO评估器即使只导出预测，也会导入pycocotools：

```bash
python3 -m pip install pycocotools
```

将该用户态依赖安装到实际使用的Python环境；脚本不会自动安装依赖。先按[模型准备](../model/README_cn.md)获取当前目标/任务/家族的制品。`--help`不会加载板端SDK。下面命令已对照parser检查，未作为新一轮板端精度测试执行。

<a id="command"></a>
## 执行评估

所有命令的工作目录均为**仓库根目录**。`/models/...`表示已准备的本地文件，命令不会下载它。X5使用.bin，S系列使用当前具体目标的.hbm；修改平台参数不会转换模型。

### 检测：COCO框AP

```bash
python3 samples/vision/ultralytics_yolo/evaluator/eval_yolo_det.py \
  --platform x5 --family yolov8 --model-path /models/yolov8n_detect.bin \
  --image-dir /data/coco/val2017 \
  --annotation /data/coco/annotations/instances_val2017.json \
  --conf-thres 0.25 --nms-thres 0.70 --json-save-path /tmp/yolo-det.json
```

### 实例分割：COCO框及掩码AP

```bash
python3 samples/vision/ultralytics_yolo/evaluator/eval_yolo_seg.py \
  --platform s100 --family yolo11 --model-path /models/yolo11n_seg.hbm \
  --image-dir /data/coco/val2017 \
  --annotation /data/coco/annotations/instances_val2017.json \
  --conf-thres 0.25 --nms-thres 0.70 --json-save-path /tmp/yolo-seg.json
```

### 姿态估计：COCO关键点AP

```bash
python3 samples/vision/ultralytics_yolo/evaluator/eval_yolo_pose.py \
  --platform s100p --family yolov8 --model-path /models/yolov8n_pose.hbm \
  --image-dir /data/coco/val2017 \
  --annotation /data/coco/annotations/person_keypoints_val2017.json \
  --category-id 1 --conf-thres 0.25 --nms-thres 0.70 \
  --json-save-path /tmp/yolo-pose.json
```

### 分类：ImageNet Top-1/Top-5

```bash
python3 samples/vision/ultralytics_yolo/evaluator/eval_yolo_cls.py \
  --platform s600 --family yolo26 --model-path /models/yolo26n_cls.hbm \
  --image-dir /data/imagenet/val --val-txt /data/imagenet/val.txt \
  --val-format named --label-offset 0 --topk 5 \
  --json-save-path /tmp/yolo-cls.json
```

报告Top-5时使用`--topk 5`或更大值。当前实现跳过无法读取或缺少有效真值的图像，须核对输出total与预期子集数量。total=0及零准确率不是有效精度结果。

### YOLO26旋转框：预测导出

```bash
python3 samples/vision/ultralytics_yolo/evaluator/eval_yolo_obb.py \
  --platform x5 --family yolo26 --model-path /models/yolo26n_obb.bin \
  --image-dir /data/dota/images --conf-thres 0.25 --nms-thres 0.70 \
  --json-save-path /tmp/yolo-obb.json
```

输出旋转矩形和多边形坐标，**不计算DOTA AP**。`--label-path`仅兼容旧命令，不参与评分。除非自定义输出协议明确要求，否则保留默认角度符号和偏移。

### 批量评估

`eval_batch.py`只扫描`--model-dir`直接包含的文件，通过`_detect_`、`_seg_`、`_pose_`、`_cls_`、`_obb_`识别任务。S模型应直接指定nash-e/m/p子目录。每次使用相同任务/数据集的模型目录，额外参数会传给所有选中的评估器。

```bash
python3 samples/vision/ultralytics_yolo/evaluator/eval_batch.py \
  --platform x5 --family yolov8 --model-dir /models/coco-detect \
  --image-dir /data/coco/val2017 \
  --annotation /data/coco/annotations/instances_val2017.json --suffix val2017
```

批量命令打印选择结果后交互确认；`--yes`跳过确认。JSON写在各模型旁边。完整验证集可能耗时数小时，取决于板型、模型和存储，。先用`--limit 10`检查流程，再取消限制跑全量；子集结果必须标注为子集。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--platform` | 自动检测板卡 | x5/s100/s100p/s600，必须与实际硬件一致 |
| `--family` | 按已知文件名识别 | 自定义文件名应显式指定任务协议 |
| `--model-path`、`--image-dir` | 必填 | 编译制品及验证图像 |
| `--annotation` | 不提供 | COCO标注；省略只导出预测 |
| `--conf-thres` | wrapper默认，通常0.25 | 检测/分割/姿态/旋转框分数阈值 |
| `--nms-thres` | 所有目标0.70 | 评估IoU阈值；S运行CLI默认则为0.45 |
| `--limit` | 0 | 全部图像，或前N张选中图像 |
| `--json-save-path` | `results_TASK.json` | 相对启动cwd的输出；父目录需先创建 |
| `--category-id` | 1 | 姿态person类别 |
| `--val-format`、`--label-offset` | named、0 | 分类标签格式 |
| `--topk`、`--log-interval` | 5、1000 | 分类返回数量及进度输出间隔 |
| `--angle-sign`、`--angle-offset` | 1、0 | 旋转框角度变换 |

各脚本`--help`列出任务专有协议参数。YOLO26采用直接LTRB，其他受支持检测家族使用DFL；S YOLOv10无NMS，X5保留NMS。不能强行选择不支持的家族/任务，也不能用阈值掩盖张量协议错误。

<a id="metrics"></a>
## 指标与比较条件

COCO通过`pycocotools.COCOeval`的bbox/segm/keypoints计算AP/AR，只评估所选图像ID。分类top1/top5为处理过且有标签图像的正确比例，不是百分数。OBB JSON是中间预测，不是精度指标。运行墙钟时间包含Python与数据处理，并非BPU推理延迟。

记录源码提交、目标/系统/SDK、制品摘要、数据集/划分、实际处理数量、resize策略和阈值后再比较结果。这些条件改变时，不能直接对比历史表。数据集精度使用对应数据集评估命令计算。

<a id="outputs"></a>
## 输出与成功判断

检测JSON包含COCO image_id、category_id、`[x,y,width,height]`和score；分割保存编码掩码，姿态保存关键点和分数。分类JSON包含total、top1、top5、elapsed_sec。旋转框包含file_name、image_id、category_id、score、rrect、polygon；矩形角度为弧度，多边形为原图像素坐标。

当前脚本会覆盖同名结果，应每次选择唯一输出路径，并保存含COCO指标摘要的stdout。退出0仅代表流程完成；无标注、空预测、分类total=0都不是精度通过。COCO空预测写为`[]`并明确跳过指标计算。

<a id="reference-results"></a>
## 参考结果与验证范围

制品清单与参考测量见 [X5 发布数据](../../../../docs/release/x5/)及 [S 发布数据](../../../../docs/release/s/)。

制品与基准数据见 [X5 benchmark 清单](../../../../docs/release/x5/benchmarks.yaml)及 [S benchmark 清单](../../../../docs/release/s/benchmarks.yaml)；[sample说明](../README_cn.md)列出 YOLOv8n/YOLO26n 检测对照及目标条件。

## 故障排查与代码入口

- 缺pycocotools：安装到运行评估脚本的解释器，包括仅导出COCO预测的情况。
- 缺真值或类别错位：核对named/ordered/synset、偏移和模型类别顺序，不猜测任意文件名的标签。
- COCO文件名/ID错误：使用配套图像和标注；无标注导出要求数字文件名主体。
- 空输出：先用runtime检查单图，再核对阈值、任务、制品；保留空输出事实，不能据此宣称精度。
- 写文件失败：创建可写父目录并显式指定输出路径。

`eval_common.py`负责共享参数、类别映射与图像选择，`eval_yolo_*.py`分别负责指标/输出格式并调用runtime，`eval_batch.py`仅派发命令。导出与张量布局变更属于[conversion](../conversion/README_cn.md)及runtime binding，不应混入指标代码。


## BenchMark - Performance

### RDK X5

#### 目标检测 (Obeject Detection)
| Model | Size(Pixels) | Classes |  BPU Task Latency  /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | params(M) | FLOPs(B) |
|----------|---------|----|---------|---------|----------|----------|
| YOLOv5nu | 640×640 | 80 | 6.3 ms / 157.4 FPS (1 thread  ) <br/> 6.8 ms / 291.8 FPS (2 threads)  | 5 ms |  2.6  M  |  7.7   B |
| YOLOv5su | 640×640 | 80 | 12.3 ms / 81.0 FPS (1 thread  ) <br/> 18.9 ms / 105.6 FPS (2 threads) | 5 ms |  9.1  M  |  24.0  B |
| YOLOv5mu | 640×640 | 80 | 26.5 ms / 37.7 FPS (1 thread  ) <br/> 47.1 ms / 42.4 FPS (2 threads)  | 5 ms |  25.1 M  |  64.2  B |
| YOLOv5lu | 640×640 | 80 | 52.7 ms / 19.0 FPS (1 thread  ) <br/> 99.1 ms / 20.1 FPS (2 threads)  | 5 ms |  53.2 M  |  135.0 B |
| YOLOv5xu | 640×640 | 80 | 91.1 ms / 11.0 FPS (1 thread  ) <br/> 175.7 ms / 11.4 FPS (2 threads) | 5 ms |  97.2 M  |  246.4 B |
| YOLOv8n  | 640×640 | 80 | 7.0 ms / 141.9 FPS (1 thread  ) <br/> 8.0 ms / 247.2 FPS (2 threads)  | 5 ms |  3.2  M  |  8.7   B |
| YOLOv8s  | 640×640 | 80 | 13.6 ms / 73.5 FPS (1 thread  ) <br/> 21.4 ms / 93.2 FPS (2 threads)  | 5 ms |  11.2 M  |  28.6  B |
| YOLOv8m  | 640×640 | 80 | 30.6 ms / 32.6 FPS (1 thread  ) <br/> 55.3 ms / 36.1 FPS (2 threads)  | 5 ms |  25.9 M  |  78.9  B |
| YOLOv8l  | 640×640 | 80 | 59.4 ms / 16.8 FPS (1 thread  ) <br/> 112.7 ms / 17.7 FPS (2 threads) | 5 ms |  43.7 M  |  165.2 B |
| YOLOv8x  | 640×640 | 80 | 92.4 ms / 10.8 FPS (1 thread  ) <br/> 178.3 ms / 11.2 FPS (2 threads) | 5 ms |  68.2 M  |  257.8 B |
| YOLOv9t  | 640×640 | 80 | 6.9 ms / 144.0 FPS (1 thread  ) <br/> 7.9 ms / 250.6 FPS (2 threads)  | 5 ms |  2.1  M  |  8.2   B |
| YOLOv9s  | 640×640 | 80 | 13.0 ms / 77.0 FPS (1 thread  ) <br/> 20.1 ms / 98.9 FPS (2 threads)  | 5 ms |  7.2  M  |  26.9  B |
| YOLOv9m  | 640×640 | 80 | 32.5 ms / 30.8 FPS (1 thread  ) <br/> 59.0 ms / 33.8 FPS (2 threads)  | 5 ms |  20.1 M  |  76.8  B |
| YOLOv9c  | 640×640 | 80 | 40.3 ms / 24.8 FPS (1 thread  ) <br/> 74.6 ms / 26.7 FPS (2 threads)  | 5 ms |  25.3 M  |  102.7 B |
| YOLOv9e  | 640×640 | 80 | 119.5 ms / 8.4 FPS (1 thread  ) <br/> 232.5 ms / 8.6 FPS (2 threads)  | 5 ms |  57.4 M  |  189.5 B |
| YOLOv10n | 640×640 | 80 | 8.7 ms / 114.2 FPS (1 thread  ) <br/> 11.6 ms / 171.9 FPS (2 threads) | 5 ms |  2.3  M  |  6.7   B |
| YOLOv10s | 640×640 | 80 | 14.9 ms / 67.1 FPS (1 thread  ) <br/> 23.8 ms / 83.7 FPS (2 threads)  | 5 ms |  7.2  M  |  21.6  B |
| YOLOv10m | 640×640 | 80 | 29.4 ms / 34.0 FPS (1 thread  ) <br/> 52.6 ms / 37.9 FPS (2 threads)  | 5 ms |  15.4 M  |  59.1  B |
| YOLOv10b | 640×640 | 80 | 40.0 ms / 25.0 FPS (1 thread  ) <br/> 74.2 ms / 26.9 FPS (2 threads)  | 5 ms |  19.1 M  |  92.0  B |
| YOLOv10l | 640×640 | 80 | 49.8 ms / 20.1 FPS (1 thread  ) <br/> 93.6 ms / 21.3 FPS (2 threads)  | 5 ms |  24.4 M  |  120.3 B |
| YOLOv10x | 640×640 | 80 | 68.9 ms / 14.5 FPS (1 thread  ) <br/> 131.5 ms / 15.2 FPS (2 threads) | 5 ms |  29.5 M  |  160.4 B |
| YOLO11n  | 640×640 | 80 | 8.2 ms / 121.6 FPS (1 thread  ) <br/> 10.5 ms / 188.9 FPS (2 threads) | 5 ms |  2.6  M  |  6.5   B |
| YOLO11s  | 640×640 | 80 | 15.7 ms / 63.4 FPS (1 thread  ) <br/> 25.6 ms / 77.7 FPS (2 threads)  | 5 ms |  9.4  M  |  21.5  B |
| YOLO11m  | 640×640 | 80 | 34.5 ms / 29.0 FPS (1 thread  ) <br/> 63.0 ms / 31.7 FPS (2 threads)  | 5 ms |  20.1 M  |  68.0  B |
| YOLO11l  | 640×640 | 80 | 45.0 ms / 22.2 FPS (1 thread  ) <br/> 84.0 ms / 23.7 FPS (2 threads)  | 5 ms |  25.3 M  |  86.9  B |
| YOLO11x  | 640×640 | 80 | 95.6 ms / 10.5 FPS (1 thread  ) <br/> 184.8 ms / 10.8 FPS (2 threads) | 5 ms |  56.9 M  |  194.9 B |
| YOLO12n  | 640×640 | 80 | 39.4 ms / 25.3 FPS (1 thread  ) <br/> 72.7 ms / 27.4 FPS (2 threads)  | 5 ms |  2.6  M  |  6.5   B |
| YOLO12s  | 640×640 | 80 | 63.4 ms / 15.8 FPS (1 thread  ) <br/> 120.6 ms / 16.5 FPS (2 threads) | 5 ms |  9.3  M  |  21.4  B |
| YOLO12m  | 640×640 | 80 | 102.3 ms / 9.8 FPS (1 thread  ) <br/> 198.1 ms / 10.1 FPS (2 threads) | 5 ms |  20.2 M  |  67.5  B |
| YOLO12l  | 640×640 | 80 | 181.6 ms / 5.5 FPS (1 thread  ) <br/> 356.4 ms / 5.6 FPS (2 threads)  | 5 ms |  26.4 M  |  88.9  B |
| YOLO12x  | 640×640 | 80 | 311.9 ms / 3.2 FPS (1 thread  ) <br/> 616.3 ms / 3.2 FPS (2 threads)  | 5 ms |  59.1 M  |  199.0 B |
| YOLOv13n | 640×640 | 80 | 44.6 ms / 22.4 FPS (1 thread  ) <br/> 83.1 ms / 24.0 FPS (2 threads)  | 5 ms |  2.5  M  |  6.4   B |
| YOLOv13s | 640×640 | 80 | 63.6 ms / 15.7 FPS (1 thread  ) <br/> 120.7 ms / 16.5 FPS (2 threads) | 5 ms |  9.0  M  |  20.8  B |
| YOLOv13l | 640×640 | 80 | 171.6 ms / 5.8 FPS (1 thread  ) <br/> 336.7 ms / 5.9 FPS (2 threads)  | 5 ms |  27.6 M  |  88.4  B |
| YOLOv13x | 640×640 | 80 | 308.4 ms / 3.2 FPS (1 thread  ) <br/> 609.2 ms / 3.3 FPS (2 threads)  | 5 ms |  64.0 M  |  199.2 B |

#### 实例分割 (Instance Segmentation)

| Model | Size(Pixels) | Classes |  BPU Task Latency  /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | params(M) | FLOPs(B) |
|----------|---------|----|---------|---------|----------|----------|
| YOLOv8n-Seg | 640×640 | 80 | 10.4 ms / 96.0 FPS (1 thread  ) <br/> 10.9 ms / 181.9 FPS (2 threads) | 20 ms | 3.4  M | 12.6  B |
| YOLOv8s-Seg | 640×640 | 80 | 19.6 ms / 50.9 FPS (1 thread  ) <br/> 29.0 ms / 68.7 FPS (2 threads)  | 20 ms | 11.8 M | 42.6  B |
| YOLOv8m-Seg | 640×640 | 80 | 40.4 ms / 24.7 FPS (1 thread  ) <br/> 70.4 ms / 28.3 FPS (2 threads)  | 20 ms | 27.3 M | 100.2 B |
| YOLOv8l-Seg | 640×640 | 80 | 74.9 ms / 13.3 FPS (1 thread  ) <br/> 139.4 ms / 14.3 FPS (2 threads) | 20 ms | 46.0 M | 220.5 B |
| YOLOv8x-Seg | 640×640 | 80 | 115.6 ms / 8.6 FPS (1 thread  ) <br/> 221.1 ms / 9.0 FPS (2 threads)  | 20 ms | 71.8 M | 344.1 B |
| YOLOv9c-Seg | 640×640 | 80 | 55.9 ms / 17.9 FPS (1 thread  ) <br/> 101.3 ms / 19.7 FPS (2 threads) | 20 ms | 27.7 M | 158.0 B |
| YOLOv9e-Seg | 640×640 | 80 | 135.4 ms / 7.4 FPS (1 thread  ) <br/> 260.0 ms / 7.7 FPS (2 threads)  | 20 ms | 59.7 M | 244.8 B |
| YOLO11n-Seg | 640×640 | 80 | 11.7 ms / 85.6 FPS (1 thread  ) <br/> 13.0 ms / 152.6 FPS (2 threads) | 20 ms | 2.9  M | 10.4  B |
| YOLO11s-Seg | 640×640 | 80 | 21.7 ms / 46.0 FPS (1 thread  ) <br/> 33.1 ms / 60.3 FPS (2 threads)  | 20 ms | 10.1 M | 35.5  B |
| YOLO11m-Seg | 640×640 | 80 | 50.3 ms / 19.9 FPS (1 thread  ) <br/> 90.2 ms / 22.1 FPS (2 threads)  | 20 ms | 22.4 M | 123.3 B |
| YOLO11l-Seg | 640×640 | 80 | 60.6 ms / 16.5 FPS (1 thread  ) <br/> 110.8 ms / 18.0 FPS (2 threads) | 20 ms | 27.6 M | 142.2 B |
| YOLO11x-Seg | 640×640 | 80 | 129.1 ms / 7.7 FPS (1 thread  ) <br/> 247.4 ms / 8.1 FPS (2 threads)  | 20 ms | 62.1 M | 319.0 B |



#### 姿态估计 (Pose Estimation)
| Model | Size(Pixels) | Classes |  BPU Task Latency  /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | params(M) | FLOPs(B) |
|----------|---------|----|---------|---------|----------|----------|
| YOLOv8n-Pose | 640×640 | 1 | 7.0 ms / 143.1 FPS (1 thread  ) <br/> 8.2 ms / 241.8 FPS (2 threads)  | 10 ms | 3.3  M | 9.2   B |
| YOLOv8s-Pose | 640×640 | 1 | 14.1 ms / 70.6 FPS (1 thread  ) <br/> 22.6 ms / 88.2 FPS (2 threads)  | 10 ms | 11.6 M | 30.2  B |
| YOLOv8m-Pose | 640×640 | 1 | 31.5 ms / 31.7 FPS (1 thread  ) <br/> 57.2 ms / 34.9 FPS (2 threads)  | 10 ms | 26.4 M | 81.0  B |
| YOLOv8l-Pose | 640×640 | 1 | 60.2 ms / 16.6 FPS (1 thread  ) <br/> 114.4 ms / 17.4 FPS (2 threads) | 10 ms | 44.4 M | 168.6 B |
| YOLOv8x-Pose | 640×640 | 1 | 93.9 ms / 10.7 FPS (1 thread  ) <br/> 181.5 ms / 11.0 FPS (2 threads) | 10 ms | 69.4 M | 263.2 B |
| YOLO11n-Pose | 640×640 | 1 | 8.3 ms / 119.8 FPS (1 thread  ) <br/> 10.9 ms / 182.2 FPS (2 threads) | 10 ms | 2.9  M | 7.6   B |
| YOLO11s-Pose | 640×640 | 1 | 16.3 ms / 61.1 FPS (1 thread  ) <br/> 27.0 ms / 73.9 FPS (2 threads)  | 10 ms | 9.9  M | 23.2  B |
| YOLO11m-Pose | 640×640 | 1 | 35.6 ms / 28.0 FPS (1 thread  ) <br/> 65.4 ms / 30.5 FPS (2 threads)  | 10 ms | 20.9 M | 71.7  B |
| YOLO11l-Pose | 640×640 | 1 | 46.3 ms / 21.6 FPS (1 thread  ) <br/> 86.6 ms / 23.0 FPS (2 threads)  | 10 ms | 26.2 M | 90.7  B |
| YOLO11x-Pose | 640×640 | 1 | 97.8 ms / 10.2 FPS (1 thread  ) <br/> 189.4 ms / 10.5 FPS (2 threads) | 10 ms | 58.8 M | 203.3 B |


### 图像分类 (Image Classification)
| Model | Size(Pixels) | Classes |  BPU Task Latency  /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | params(M) | FLOPs(B) |
|----------|---------|----|---------|---------|----------|----------|
| YOLOv8n-CLS | 224x224 | 1000 | 0.7 ms / 1374.6 FPS (1 thread  ) <br/> 1.0 ms / 2023.2 FPS (2 threads) | 0.5 ms | 2.7  M | 4.3   B |
| YOLOv8s-CLS | 224x224 | 1000 | 1.4 ms / 701.0 FPS (1 thread  ) <br/> 2.3 ms / 848.0 FPS (2 threads)   | 0.5 ms | 6.4  M | 13.5  B |
| YOLOv8m-CLS | 224x224 | 1000 | 3.7 ms / 269.5 FPS (1 thread  ) <br/> 6.9 ms / 290.6 FPS (2 threads)   | 0.5 ms | 17.0 M | 42.7  B |
| YOLOv8l-CLS | 224x224 | 1000 | 7.9 ms / 126.6 FPS (1 thread  ) <br/> 15.2 ms / 130.8 FPS (2 threads)  | 0.5 ms | 37.5 M | 99.7  B |
| YOLOv8x-CLS | 224x224 | 1000 | 13.1 ms / 76.4 FPS (1 thread  ) <br/> 25.5 ms / 78.3 FPS (2 threads)   | 0.5 ms | 57.4 M | 154.8 B |
| YOLO11n-CLS | 224x224 | 1000 | 1.0 ms / 949.5 FPS (1 thread  ) <br/> 1.6 ms / 1238.4 FPS (2 threads)  | 0.5 ms | 2.8  M | 4.2   B |
| YOLO11s-CLS | 224x224 | 1000 | 2.1 ms / 484.3 FPS (1 thread  ) <br/> 3.5 ms / 572.2 FPS (2 threads)   | 0.5 ms | 6.7  M | 13.0  B |
| YOLO11m-CLS | 224x224 | 1000 | 3.8 ms / 262.6 FPS (1 thread  ) <br/> 7.1 ms / 282.2 FPS (2 threads)   | 0.5 ms | 11.6 M | 40.3  B |
| YOLO11l-CLS | 224x224 | 1000 | 5.0 ms / 200.3 FPS (1 thread  ) <br/> 9.4 ms / 211.2 FPS (2 threads)   | 0.5 ms | 14.1 M | 50.4  B |
| YOLO11x-CLS | 224x224 | 1000 | 10.0 ms / 100.2 FPS (1 thread  ) <br/> 19.3 ms / 103.2 FPS (2 threads) | 0.5 ms | 29.6 M | 111.3 B |


### Performance Test Instructions
1. 此处测试的均为YUV420SP (nv12) 输入的模型的性能数据.
2. BPU延迟与BPU吞吐量.
 - 单线程延迟为单帧,单线程,单BPU核心的延迟,BPU推理一个任务最理想的情况.
 - 多线程帧率为多个线程同时向BPU塞任务, 每个BPU核心可以处理多个线程的任务, 一般工程中2个线程可以控制单帧延迟较小,同时吃满所有BPU到100%,在吞吐量(FPS)和帧延迟间得到一个较好的平衡.
 - 表格中一般记录到吞吐量不再随线程数明显增加的数据.
 - BPU延迟和BPU吞吐量使用以下命令在板端测试
```bash
hrt_model_exec perf --thread_num 2 --model_file yolov8n_detect_bayese_640x640_nv12_modified.bin

python3 /path/to/rdk_model_zoo/utils/tools/batch_perf/batch_perf.py --max 3 --file source/reference_bin_models/
```
3. 测试板卡为最佳状态.
 - RDK X5 状态: CPU为8 × A55 @ 1.5GHz, 全核心Performance调度, BPU为1 × Bayes-e @ 1.0GHz, 10 TOPS @ int8.


## Benchmark - Accuracy
### RDK X5
#### Object Detection (COCO2017)
| Model | Pytorch | YUV420SP<br/>Python | YUV420SP<br/>C/C++ |
|---------|---------|-------|---------|
| YOLOv5nu | 0.275 | 0.260(94.55%) | (%) |
| YOLOv5su | 0.362 | 0.354(97.79%) | (%) |
| YOLOv5mu | 0.417 | 0.407(97.60%) | (%) |
| YOLOv5lu | 0.449 | 0.442(98.44%) | (%) |
| YOLOv5xu | 0.458 | 0.443(96.72%) | (%) |
| YOLOv8n  | 0.306 | 0.292(95.42%) | (%) |
| YOLOv8s  | 0.384 | 0.372(96.88%) | (%) |
| YOLOv8m  | 0.433 | 0.423(97.69%) | (%) |
| YOLOv8l  | 0.454 | 0.440(96.92%) | (%) |
| YOLOv8x  | 0.465 | 0.448(96.34%) | (%) |
| YOLOv9t  | 0.309 | 0.298(96.44%) | (%) |
| YOLOv9s  | 0.394 | 0.382(96.95%) | (%) |
| YOLOv9m  | 0.441 | 0.427(96.83%) | (%) |
| YOLOv9c  | 0.452 | 0.435(96.24%) | (%) |
| YOLOv9e  | 0.472 | 0.458(97.03%) | (%) |
| YOLOv10n | 0.299 | 0.282(94.31%) | (%) |
| YOLOv10s | 0.381 | 0.364(95.54%) | (%) |
| YOLOv10m | 0.418 | 0.379(90.67%) | (%) |
| YOLOv10b | 0.435 | 0.391(89.89%) | (%) |
| YOLOv10l | 0.438 | 0.396(90.41%) | (%) |
| YOLOv10x | 0.451 | 0.417(92.46%) | (%) |
| YOLO11n  | 0.323 | 0.308(95.36%) | (%) |
| YOLO11s  | 0.394 | 0.380(96.45%) | (%) |
| YOLO11m  | 0.437 | 0.422(96.57%) | (%) |
| YOLO11l  | 0.452 | 0.432(95.58%) | (%) |
| YOLO11x  | 0.466 | 0.446(95.71%) | (%) |
| YOLO12n  | 0.334 | 0.312(93.41%) | (%) |
| YOLO12s  | 0.397 | 0.379(95.47%) | (%) |
| YOLO12m  | 0.444 | 0.428(96.40%) | (%) |
| YOLO12l  | 0.454 | 0.433(95.37%) | (%) |
| YOLO12x  | 0.466 | 0.444(95.28%) | (%) |
| YOLOv13n | 0.342 | 0.254(--.--%)* | (%) |
| YOLOv13s | 0.402 | 0.392(97.51%) | (%) |
| YOLOv13l | 0.458 | 0.447(97.60%) | (%) |
| YOLOv13x | 0.473 | 0.459(97.04%) | (%) |

#### Instance Segmentation (COCO2017)
| Model | Pytorch<br/>BBox / Mask | YUV420SP - Python<br/>BBox / Mask | YUV420SP - C/C++<br/>BBox / Mask |
|---------|---------|-------|---------|
| YOLOv8n-Seg | 0.300 / 0.241 | 0.284(94.67%) / 0.219(90.87%) | (%) / (%) |
| YOLOv8s-Seg | 0.380 / 0.299 | 0.371(97.63%) / 0.287(95.99%) | (%) / (%) |
| YOLOv8m-Seg | 0.423 / 0.330 | 0.408(96.45%) / 0.311(94.24%) | (%) / (%) |
| YOLOv8l-Seg | 0.444 / 0.344 | 0.431(97.07%) / 0.332(96.51%) | (%) / (%) |
| YOLOv8x-Seg | 0.456 / 0.351 | 0.439(96.27%) / 0.336(95.73%) | (%) / (%) |
| YOLOv9c-Seg | 0.446 / 0.345 | 0.423(94.84%) / 0.321(93.04%) | (%) / (%) |
| YOLOv9e-Seg | 0.471 / 0.118 | 0.332(--.--%) / 0.268(--.--%)*| (%) / (%) |
| YOLO11n-Seg | 0.319 / 0.258 | 0.296(92.79%) / 0.227(87.98%) | (%) / (%) |
| YOLO11s-Seg | 0.388 / 0.306 | 0.377(97.16%) / 0.291(95.10%) | (%) / (%) |
| YOLO11m-Seg | 0.436 / 0.340 | 0.422(96.79%) / 0.322(94.71%) | (%) / (%) |
| YOLO11l-Seg | 0.452 / 0.350 | 0.432(95.58%) / 0.328(93.71%) | (%) / (%) |
| YOLO11x-Seg | 0.466 / 0.358 | 0.447(95.92%) / 0.338(94.41%) | (%) / (%) |


#### Pose Estimation (COCO2017)
| Model | Pytorch | YUV420SP - Python | YUV420SP - C/C++ |
|---------|---------|-------|---------|
| YOLOv8n-Pose | 0.476 | 0.462(97.06%) | (%) |
| YOLOv8s-Pose | 0.578 | 0.553(95.67%) | (%) |
| YOLOv8m-Pose | 0.631 | 0.605(95.88%) | (%) |
| YOLOv8l-Pose | 0.656 | 0.636(96.95%) | (%) |
| YOLOv8x-Pose | 0.670 | 0.655(97.76%) | (%) |
| YOLO11n-Pose | 0.465 | 0.452(97.20%) | (%) |
| YOLO11s-Pose | 0.560 | 0.530(94.64%) | (%) |
| YOLO11m-Pose | 0.626 | 0.600(95.85%) | (%) |
| YOLO11l-Pose | 0.636 | 0.619(97.33%) | (%) |
| YOLO11x-Pose | 0.672 | 0.654(97.32%) | (%) |


#### Image Classification (ImageNet2012)
| Model | Pytorch | YUV420SP - Python<br/>TOP1 / TOP5 | YUV420SP - C/C++<br/>TOP1 / TOP5 |
|---------|---------|-------|---------|
| YOLOv8n-CLS | 0.690 / 0.883 | 0.525(76.09%) / 0.762(86.30%) | (%) / (%) |
| YOLOv8s-CLS | 0.738 / 0.917 | 0.611(82.79%) / 0.837(91.28%) | (%) / (%) |
| YOLOv8m-CLS | 0.768 / 0.935 | 0.682(88.80%) / 0.883(94.44%) | (%) / (%) |
| YOLOv8l-CLS | 0.768 / 0.935 | 0.724(94.27%) / 0.909(97.22%) | (%) / (%) |
| YOLOv8x-CLS | 0.790 / 0.946 | 0.737(93.29%) / 0.917(96.93%) | (%) / (%) |
| YOLO11n-CLS | 0.700 / 0.894 | 0.495(70.71%) / 0.736(82.33%) | (%) / (%) |
| YOLO11s-CLS | 0.754 / 0.927 | 0.665(88.20%) / 0.873(94.17%) | (%) / (%) |
| YOLO11m-CLS | 0.773 / 0.939 | 0.695(89.91%) / 0.896(95.42%) | (%) / (%) |
| YOLO11l-CLS | 0.783 / 0.943 | 0.707(90.29%) / 0.902(95.65%) | (%) / (%) |
| YOLO11x-CLS | 0.795 / 0.949 | 0.732(92.08%) / 0.917(96.63%) | (%) / (%) |


##### Accuracy test conditions

1. 所有的精度数据使用微软官方的无修改的`pycocotools`库进行计算, Det和Seg取的精度标准为`Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]`的数据, Pose取的精度标准为`Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]`的数据.
2. 所有的测试数据均使用`COCO2017`数据集的val验证集的5000张照片, 在板端直接推理, dump保存为json文件, 送入第三方测试工具`pycocotools`库进行计算, 分数的阈值为0.25, nms的阈值为0.7.
3. pycocotools计算的精度比ultralytics计算的精度会低一些是正常现象, 主要原因是pycocotools是取矩形面积, ultralytics是取梯形面积, 我们主要是关注同样的一套计算方式去测试定点模型和浮点模型的精度, 从而来评估量化过程中的精度损失.
4. BPU模型在量化NCHW-RGB888输入转换为YUV420SP(nv12)输入后, 也会有一部分精度损失, 这是由于色彩空间转化导致的, 在训练时加入这种色彩空间转化的损失可以避免这种精度损失.
5. Python接口和C/C++接口的精度结果有细微差异, 主要在于Python和C/C++的一些数据结构进行memcpy和转化的过程中, 对浮点数的处理方式不同, 导致的细微差异.
6. 测试脚本请参考RDK Model Zoo的eval部分: https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/tools/eval_pycocotools
7. 本表格是使用PTQ(训练后量化)使用50张图片进行校准和编译的结果, 用于模拟普通开发者第一次直接编译的精度情况, 并没有进行精度调优或者QAT(量化感知训练), 满足常规使用验证需求, 不代表精度上限.



## 补充参考数据

各表按发布时的模型、板卡和测量条件列出，不同配置的数据分别保留。

### Obeject Detection

| Device   | Model           | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-----------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S600     | YOLO11l Detect | 640×640        |       80 | 3.229 ms / 307.877 FPS (1 thread ) <br/> 9.113 ms / 1277.270 FPS (12 threads) | 2.0 ms                         | 25.3 M      | 86.9 M     |
| S600     | YOLO11m Detect | 640×640        |       80 | 2.584 ms / 384.284 FPS (1 thread ) <br/> 7.054 ms / 1642.643 FPS (12 threads) | 2.0 ms                         | 20.1 M      | 68.0 M     |
| S600     | YOLO11n Detect | 640×640        |       80 | 0.800 ms / 1221.628 FPS (1 thread ) <br/> 1.798 ms / 6142.695 FPS (12 threads) | 2.0 ms                         | 2.6 M       | 6.5 M      |
| S600     | YOLO11s Detect | 640×640        |       80 | 1.204 ms / 817.752 FPS (1 thread ) <br/> 3.008 ms / 3761.944 FPS (12 threads) | 2.0 ms                         | 9.4 M       | 21.5 M     |
| S600     | YOLO11x Detect | 640×640        |       80 | 6.543 ms / 152.382 FPS (1 thread ) <br/> 18.792 ms / 622.258 FPS (12 threads) | 2.0 ms                         | 56.9 M      | 194.9 M    |
| S600     | YOLO12l Detect | 640×640        |       80 | 6.296 ms / 158.364 FPS (1 thread ) <br/> 19.661 ms / 595.057 FPS (12 threads) | 2.0 ms                         | 26.4 M      | 88.9 M     |
| S600     | YOLO12m Detect | 640×640        |       80 | 4.027 ms / 247.181 FPS (1 thread ) <br/> 12.026 ms / 969.575 FPS (12 threads) | 2.0 ms                         | 20.2 M      | 67.5 M     |
| S600     | YOLO12n Detect | 640×640        |       80 | 1.207 ms / 819.555 FPS (1 thread ) <br/> 2.986 ms / 3779.575 FPS (12 threads) | 2.0 ms                         | 2.6 M       | 7.7 M      |
| S600     | YOLO12s Detect | 640×640        |       80 | 1.957 ms / 505.866 FPS (1 thread ) <br/> 5.155 ms / 2235.861 FPS (12 threads) | 2.0 ms                         | 9.3 M       | 21.4 M     |
| S600     | YOLO12x Detect | 640×640        |       80 | 11.073 ms / 90.110 FPS (1 thread ) <br/> 35.574 ms / 329.594 FPS (12 threads) | 2.0 ms                         | 59.1 M      | 199.0 M    |
| S600     | YOLOv10b Detect | 640×640        |       80 | 2.972 ms / 334.349 FPS (1 thread ) <br/> 8.042 ms / 1442.637 FPS (12 threads) | 2.0 ms                         | 19.1 M      | 92.0 M     |
| S600     | YOLOv10l Detect | 640×640        |       80 | 3.723 ms / 267.232 FPS (1 thread ) <br/> 10.360 ms / 1123.873 FPS (12 threads) | 2.0 ms                         | 24.4 M      | 120.3 M    |
| S600     | YOLOv10m Detect | 640×640        |       80 | 2.342 ms / 423.743 FPS (1 thread ) <br/> 6.201 ms / 1867.309 FPS (12 threads) | 2.0 ms                         | 15.4 M      | 59.1 M     |
| S600     | YOLOv10n Detect | 640×640        |       80 | 0.781 ms / 1251.157 FPS (1 thread ) <br/> 1.690 ms / 6457.445 FPS (12 threads) | 2.0 ms                         | 2.3 M       | 6.7 M      |
| S600     | YOLOv10s Detect | 640×640        |       80 | 1.166 ms / 847.781 FPS (1 thread ) <br/> 2.802 ms / 4025.117 FPS (12 threads) | 2.0 ms                         | 7.2 M       | 21.6 M     |
| S600     | YOLOv10x Detect | 640×640        |       80 | 5.314 ms / 187.525 FPS (1 thread ) <br/> 15.173 ms / 769.177 FPS (12 threads) | 2.0 ms                         | 29.5 M      | 160.4 M    |
| S600     | YOLOv5lu Detect | 640×640        |       80 | 3.980 ms / 250.292 FPS (1 thread ) <br/> 11.117 ms / 1048.306 FPS (12 threads) | 2.0 ms                         | 53.2 M      | 135.0 M    |
| S600     | YOLOv5mu Detect | 640×640        |       80 | 2.240 ms / 442.539 FPS (1 thread ) <br/> 5.935 ms / 1941.804 FPS (12 threads) | 2.0 ms                         | 25.1 M      | 64.2 M     |
| S600     | YOLOv5nu Detect | 640×640        |       80 | 0.719 ms / 1355.583 FPS (1 thread ) <br/> 1.528 ms / 7182.102 FPS (12 threads) | 2.0 ms                         | 2.6 M       | 7.7 M      |
| S600     | YOLOv5su Detect | 640×640        |       80 | 1.096 ms / 898.017 FPS (1 thread ) <br/> 2.656 ms / 4213.542 FPS (12 threads) | 2.0 ms                         | 9.1 M       | 24.0 M     |
| S600     | YOLOv5xu Detect | 640×640        |       80 | 7.333 ms / 136.001 FPS (1 thread ) <br/> 21.084 ms / 554.451 FPS (12 threads) | 2.0 ms                         | 97.2 M      | 246.4 M    |
| S600     | YOLOv8l Detect | 640×640        |       80 | 4.514 ms / 220.583 FPS (1 thread ) <br/> 12.603 ms / 924.740 FPS (12 threads) | 2.0 ms                         | 43.7 M      | 165.2 M    |
| S600     | YOLOv8m Detect | 640×640        |       80 | 2.677 ms / 371.027 FPS (1 thread ) <br/> 7.180 ms / 1614.922 FPS (12 threads) | 2.0 ms                         | 25.9 M      | 78.9 M     |
| S600     | YOLOv8n Detect | 640×640        |       80 | 0.776 ms / 1258.503 FPS (1 thread ) <br/> 1.674 ms / 6512.325 FPS (12 threads) | 2.0 ms                         | 3.2 M       | 8.7 M      |
| S600     | YOLOv8s Detect | 640×640        |       80 | 1.223 ms / 805.412 FPS (1 thread ) <br/> 2.934 ms / 3839.066 FPS (12 threads) | 2.0 ms                         | 11.2 M      | 28.6 M     |
| S600     | YOLOv8x Detect | 640×640        |       80 | 7.210 ms / 138.386 FPS (1 thread ) <br/> 20.629 ms / 567.201 FPS (12 threads) | 2.0 ms                         | 68.2 M      | 257.8 M    |
| S600     | YOLOv9c Detect | 640×640        |       80 | 3.149 ms / 316.157 FPS (1 thread ) <br/> 8.773 ms / 1325.812 FPS (12 threads) | 2.0 ms                         | 25.3 M      | 102.7 M    |
| S600     | YOLOv9e Detect | 640×640        |       80 | 8.560 ms / 116.557 FPS (1 thread ) <br/> 31.421 ms / 372.969 FPS (12 threads) | 2.0 ms                         | 57.4 M      | 189.5 M    |
| S600     | YOLOv9m Detect | 640×640        |       80 | 2.747 ms / 361.925 FPS (1 thread ) <br/> 7.518 ms / 1541.699 FPS (12 threads) | 2.0 ms                         | 20.1 M      | 76.8 M     |
| S600     | YOLOv9s Detect | 640×640        |       80 | 1.468 ms / 672.601 FPS (1 thread ) <br/> 3.911 ms / 2921.499 FPS (12 threads) | 2.0 ms                         | 7.2 M       | 26.9 M     |

#### Segmentation

### Segmentation

| Device   | Model        | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|--------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S600     | YOLO11l Seg | 640×640        |       80 | 4.268 ms / 233.082 FPS (1 thread ) <br/> 12.015 ms / 967.109 FPS (12 threads) | 5.0 ms                         | 27.6 M      | 142.2 M    |
| S600     | YOLO11m Seg | 640×640        |       80 | 3.624 ms / 274.240 FPS (1 thread ) <br/> 9.977 ms / 1161.123 FPS (12 threads) | 5.0 ms                         | 22.4 M      | 123.3 M    |
| S600     | YOLO11n Seg | 640×640        |       80 | 0.959 ms / 1024.863 FPS (1 thread ) <br/> 2.379 ms / 4586.104 FPS (12 threads) | 5.0 ms                         | 2.9 M       | 10.4 M     |
| S600     | YOLO11s Seg | 640×640        |       80 | 1.513 ms / 651.042 FPS (1 thread ) <br/> 3.845 ms / 2908.329 FPS (12 threads) | 5.0 ms                         | 10.1 M      | 35.5 M     |
| S600     | YOLO11x Seg | 640×640        |       80 | 8.831 ms / 112.923 FPS (1 thread ) <br/> 25.480 ms / 459.263 FPS (12 threads) | 5.0 ms                         | 62.1 M      | 319.0 M    |
| S600     | YOLOv8l Seg | 640×640        |       80 | 5.585 ms / 178.335 FPS (1 thread ) <br/> 15.661 ms / 743.323 FPS (12 threads) | 5.0 ms                         | 46.0 M      | 220.5 M    |
| S600     | YOLOv8m Seg | 640×640        |       80 | 3.298 ms / 301.000 FPS (1 thread ) <br/> 8.931 ms / 1296.706 FPS (12 threads) | 5.0 ms                         | 27.3 M      | 100.2 M    |
| S600     | YOLOv8n Seg | 640×640        |       80 | 0.940 ms / 1038.438 FPS (1 thread ) <br/> 2.236 ms / 4790.304 FPS (12 threads) | 5.0 ms                         | 3.4 M       | 12.6 M     |
| S600     | YOLOv8s Seg | 640×640        |       80 | 1.540 ms / 642.329 FPS (1 thread ) <br/> 3.883 ms / 2873.027 FPS (12 threads) | 5.0 ms                         | 11.8 M      | 42.6 M     |
| S600     | YOLOv8x Seg | 640×640        |       80 | 8.884 ms / 112.265 FPS (1 thread ) <br/> 25.517 ms / 458.336 FPS (12 threads) | 5.0 ms                         | 71.8 M      | 344.1 M    |

#### Pose Estimation

### Pose Estimation

| Device   | Model        | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|--------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S600     | YOLO11l Pose | 640×640        |       80 | 3.319 ms / 299.436 FPS (1 thread ) <br/> 9.278 ms / 1250.641 FPS (12 threads) | 1.0 ms                         | 26.2 M      | 90.7 M     |
| S600     | YOLO11m Pose | 640×640        |       80 | 2.667 ms / 371.922 FPS (1 thread ) <br/> 7.232 ms / 1596.972 FPS (12 threads) | 1.0 ms                         | 20.9 M      | 71.7 M     |
| S600     | YOLO11n Pose | 640×640        |       80 | 0.846 ms / 1159.078 FPS (1 thread ) <br/> 1.925 ms / 5680.205 FPS (12 threads) | 1.0 ms                         | 2.9 M       | 7.6 M      |
| S600     | YOLO11s Pose | 640×640        |       80 | 1.275 ms / 771.542 FPS (1 thread ) <br/> 3.146 ms / 3565.380 FPS (12 threads) | 1.0 ms                         | 9.9 M       | 23.2 M     |
| S600     | YOLO11x Pose | 640×640        |       80 | 6.768 ms / 147.282 FPS (1 thread ) <br/> 19.390 ms / 602.593 FPS (12 threads) | 1.0 ms                         | 58.8 M      | 203.3 M    |
| S600     | YOLOv8l Pose | 640×640        |       80 | 4.621 ms / 215.565 FPS (1 thread ) <br/> 12.895 ms / 902.833 FPS (12 threads) | 1.0 ms                         | 44.4 M      | 168.6 M    |
| S600     | YOLOv8m Pose | 640×640        |       80 | 2.760 ms / 360.059 FPS (1 thread ) <br/> 7.384 ms / 1565.411 FPS (12 threads) | 1.0 ms                         | 26.4 M      | 81.0 M     |
| S600     | YOLOv8n Pose | 640×640        |       80 | 0.808 ms / 1204.101 FPS (1 thread ) <br/> 1.733 ms / 6280.026 FPS (12 threads) | 1.0 ms                         | 3.3 M       | 9.2 M      |
| S600     | YOLOv8s Pose | 640×640        |       80 | 1.292 ms / 761.464 FPS (1 thread ) <br/> 3.148 ms / 3553.976 FPS (12 threads) | 1.0 ms                         | 11.6 M      | 30.2 M     |
| S600     | YOLOv8x Pose | 640×640        |       80 | 7.392 ms / 134.881 FPS (1 thread ) <br/> 21.121 ms / 553.125 FPS (12 threads) | 1.0 ms                         | 69.4 M      | 263.2 M    |

#### Image Classification

### Image Classification

| Device   | Model       | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-------------|----------------|-----------|------------------------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S600     | YOLO11l CLS | 224×224        |     1000 | 0.654 ms / 1497.623 FPS (1 thread ) <br/> 1.418 ms / 7986.583 FPS (12 threads)          | 0.5 ms                         | 42.6 M      | 50.4 M     |
| S600     | YOLO11m CLS | 224×224        |     1000 | 0.534 ms / 1830.295 FPS (1 thread ) <br/> 1.042 ms / 10636.601 FPS (12 threads)         | 0.5 ms                         | 32.8 M      | 39.3 M     |
| S600     | YOLO11n CLS | 224×224        |     1000 | 0.345 ms / 2788.428 FPS (1 thread ) <br/> 0.761 ms / 13779.799 FPS (12 threads)         | 0.5 ms                         | 5.0 M       | 5.5 M      |
| S600     | YOLO11s CLS | 224×224        |     1000 | 0.407 ms / 2380.216 FPS (1 thread ) <br/> 0.771 ms / 14041.000 FPS (12 threads)         | 0.5 ms                         | 13.1 M      | 14.4 M     |
| S600     | YOLO11x CLS | 224×224        |     1000 | 0.994 ms / 991.985 FPS (1 thread ) <br/> 2.669 ms / 4295.902 FPS (12 threads)           | 0.5 ms                         | 97.7 M      | 113.2 M    |
| S600     | YOLOv8l CLS | 224×224        |     1000 | 0.877 ms / 1123.179 FPS (1 thread ) <br/> 2.737 ms / 4181.651 FPS (12 threads)          | 0.5 ms                         | 36.3 M      | 99.0 M     |
| S600     | YOLOv8m CLS | 224×224        |     1000 | 0.568 ms / 1718.996 FPS (1 thread ) <br/> 1.359 ms / 8276.775 FPS (12 threads)          | 0.5 ms                         | 17.0 M      | 42.8 M     |
| S600     | YOLOv8n CLS | 224×224        |     1000 | 0.315 ms / 3054.368 FPS (1 thread ) <br/> 0.612 ms / 17488.633 FPS (12 threads)         | 0.5 ms                         | 2.7 M       | 4.3 M      |
| S600     | YOLOv8s CLS | 224×224        |     1000 | 0.362 ms / 2695.563 FPS (1 thread ) <br/> 0.697 ms / 15550.891 FPS (12 threads)         | 0.5 ms                         | 6.4 M       | 13.5 M     |
| S600     | YOLOv8x CLS | 224×224        |     1000 | 1.249 ms / 792.299 FPS (1 thread ) <br/> 4.156 ms / 2775.157 FPS (12 threads)           | 0.5 ms                         | 57.4 M      | 154.8 M    |

### RDK S100P

### Obeject Detection

| Device   | Model           | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-----------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S100P    | YOLO12n Detect  | 640×640        |        80 | 1.88 ms / 513.70 FPS (1 thread ) <br/> 3.07 ms / 634.97 FPS (2 threads)  | 2.0 ms                         | 2.6 M       | 7.7 M      |
| S100P    | YOLO12s Detect  | 640×640        |        80 | 3.10 ms / 315.83 FPS (1 thread ) <br/> 5.50 ms / 357.85 FPS (2 threads)  | 2.0 ms                         | 9.3 M       | 21.4 M     |
| S100P    | YOLO12m Detect  | 640×640        |        80 | 6.47 ms / 152.80 FPS (1 thread ) <br/> 12.18 ms / 162.62 FPS (2 threads) | 2.0 ms                         | 20.2 M      | 67.5 M     |
| S100P    | YOLO12l Detect  | 640×640        |        80 | 10.23 ms / 97.01 FPS (1 thread ) <br/> 19.67 ms / 101.04 FPS (2 threads) | 2.0 ms                         | 26.4 M      | 88.9 M     |
| S100P    | YOLO12x Detect  | 640×640        |        80 | 17.05 ms / 58.34 FPS (1 thread ) <br/> 33.21 ms / 59.92 FPS (2 threads)  | 2.0 ms                         | 59.1 M      | 199.0 M    |
| S100P    | YOLO11n Detect  | 640×640        |        80 | 1.16 ms / 816.50 FPS (1 thread ) <br/> 1.66 ms / 1155.65 FPS (2 threads) | 2.0 ms                         | 2.6 M       | 6.5 M      |
| S100P    | YOLO11s Detect  | 640×640        |        80 | 1.81 ms / 533.50 FPS (1 thread ) <br/> 2.98 ms / 656.31 FPS (2 threads)  | 2.0 ms                         | 9.4 M       | 21.5 M     |
| S100P    | YOLO11m Detect  | 640×640        |        80 | 3.90 ms / 252.02 FPS (1 thread ) <br/> 7.10 ms / 278.36 FPS (2 threads)  | 2.0 ms                         | 20.1 M      | 68.0 M     |
| S100P    | YOLO11l Detect  | 640×640        |        80 | 4.73 ms / 208.61 FPS (1 thread ) <br/> 8.75 ms / 225.99 FPS (2 threads)  | 2.0 ms                         | 25.3 M      | 86.9 M     |
| S100P    | YOLO11x Detect  | 640×640        |        80 | 8.84 ms / 112.05 FPS (1 thread ) <br/> 16.92 ms / 117.39 FPS (2 threads) | 2.0 ms                         | 56.9 M      | 194.9 M    |
| S100P    | YOLOv10n Detect | 640×640        |        80 | 1.12 ms / 837.97 FPS (1 thread ) <br/> 1.58 ms / 1211.72 FPS (2 threads) | 2.0 ms                         | 2.3 M       | 6.7 M      |
| S100P    | YOLOv10s Detect | 640×640        |        80 | 1.75 ms / 548.80 FPS (1 thread ) <br/> 2.81 ms / 692.74 FPS (2 threads)  | 2.0 ms                         | 7.2 M       | 21.6 M     |
| S100P    | YOLOv10m Detect | 640×640        |        80 | 3.06 ms / 319.65 FPS (1 thread ) <br/> 5.45 ms / 361.32 FPS (2 threads)  | 2.0 ms                         | 15.4 M      | 59.1 M     |
| S100P    | YOLOv10b Detect | 640×640        |        80 | 4.30 ms / 228.16 FPS (1 thread ) <br/> 7.85 ms / 250.93 FPS (2 threads)  | 2.0 ms                         | 19.1 M      | 92.0 M     |
| S100P    | YOLOv10l Detect | 640×640        |        80 | 5.42 ms / 181.96 FPS (1 thread ) <br/> 10.10 ms / 196.04 FPS (2 threads) | 2.0 ms                         | 24.4 M      | 120.3 M    |
| S100P    | YOLOv10x Detect | 640×640        |        80 | 7.33 ms / 135.18 FPS (1 thread ) <br/> 13.90 ms / 142.81 FPS (2 threads) | 2.0 ms                         | 29.5 M      | 160.4 M    |
| S100P    | YOLOv9t Detect  | 640×640        |        80 | 1.29 ms / 736.75 FPS (1 thread ) <br/> 1.90 ms / 1013.70 FPS (2 threads) | 2.0 ms                         | 2.1 M       | 8.2 M      |
| S100P    | YOLOv9s Detect  | 640×640        |        80 | 1.93 ms / 497.53 FPS (1 thread ) <br/> 3.19 ms / 611.75 FPS (2 threads)  | 2.0 ms                         | 7.2 M       | 26.9 M     |
| S100P    | YOLOv9m Detect  | 640×640        |        80 | 3.77 ms / 260.19 FPS (1 thread ) <br/> 6.83 ms / 288.82 FPS (2 threads)  | 2.0 ms                         | 20.1 M      | 76.8 M     |
| S100P    | YOLOv9c Detect  | 640×640        |        80 | 4.76 ms / 206.90 FPS (1 thread ) <br/> 8.77 ms / 225.46 FPS (2 threads)  | 2.0 ms                         | 25.3 M      | 102.7 M    |
| S100P    | YOLOv9e Detect  | 640×640        |        80 | 12.27 ms / 81.00 FPS (1 thread ) <br/> 23.73 ms / 83.75 FPS (2 threads)  | 2.0 ms                         | 57.4 M      | 189.5 M    |
| S100P    | YOLOv8n Detect  | 640×640        |        80 | 1.10 ms / 851.31 FPS (1 thread ) <br/> 1.52 ms / 1258.50 FPS (2 threads) | 2.0 ms                         | 3.2 M       | 8.7 M      |
| S100P    | YOLOv8s Detect  | 640×640        |        80 | 1.83 ms / 524.95 FPS (1 thread ) <br/> 2.95 ms / 660.43 FPS (2 threads)  | 2.0 ms                         | 11.2 M      | 28.6 M     |
| S100P    | YOLOv8m Detect  | 640×640        |        80 | 3.43 ms / 285.34 FPS (1 thread ) <br/> 6.14 ms / 320.93 FPS (2 threads)  | 2.0 ms                         | 25.9 M      | 78.9 M     |
| S100P    | YOLOv8l Detect  | 640×640        |        80 | 6.72 ms / 147.19 FPS (1 thread ) <br/> 12.67 ms / 156.40 FPS (2 threads) | 2.0 ms                         | 43.7 M      | 165.2 M    |
| S100P    | YOLOv8x Detect  | 640×640        |        80 | 10.44 ms / 95.08 FPS (1 thread ) <br/> 20.11 ms / 98.81 FPS (2 threads)  | 2.0 ms                         | 68.2 M      | 257.8 M    |
| S100P    | YOLOv5nu Detect | 640×640        |        80 | 0.99 ms / 954.28 FPS (1 thread ) <br/> 1.34 ms / 1418.24 FPS (2 threads) | 2.0 ms                         | 2.6 M       | 7.7 M      |
| S100P    | YOLOv5su Detect | 640×640        |        80 | 1.60 ms / 602.38 FPS (1 thread ) <br/> 2.56 ms / 763.66 FPS (2 threads)  | 2.0 ms                         | 9.1 M       | 24.0 M     |
| S100P    | YOLOv5mu Detect | 640×640        |        80 | 3.06 ms / 319.05 FPS (1 thread ) <br/> 5.43 ms / 363.38 FPS (2 threads)  | 2.0 ms                         | 25.1 M      | 64.2 M     |
| S100P    | YOLOv5lu Detect | 640×640        |        80 | 6.04 ms / 163.65 FPS (1 thread ) <br/> 11.36 ms / 174.46 FPS (2 threads) | 2.0 ms                         | 53.2 M      | 135.0 M    |
| S100P    | YOLOv5xu Detect | 640×640        |        80 | 10.74 ms / 92.40 FPS (1 thread ) <br/> 20.67 ms / 96.10 FPS (2 threads)  | 2.0 ms                         | 97.2 M      | 246.4 M    |

#### Instance Segmentation

### Instance Segmentation

| Device   | Model       | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S100P    | YOLO11n Seg | 640×640        |        80 | 1.45 ms / 647.24 FPS (1 thread ) <br/> 2.14 ms / 883.83 FPS (2 threads)  | 5.0 ms                         | 2.9 M       | 10.4 M     |
| S100P    | YOLO11s Seg | 640×640        |        80 | 2.31 ms / 413.73 FPS (1 thread ) <br/> 3.88 ms / 501.74 FPS (2 threads)  | 5.0 ms                         | 10.1 M      | 35.5 M     |
| S100P    | YOLO11m Seg | 640×640        |        80 | 5.36 ms / 182.99 FPS (1 thread ) <br/> 9.89 ms / 199.15 FPS (2 threads)  | 5.0 ms                         | 22.4 M      | 123.3 M    |
| S100P    | YOLO11l Seg | 640×640        |        80 | 6.20 ms / 158.60 FPS (1 thread ) <br/> 11.57 ms / 170.73 FPS (2 threads) | 5.0 ms                         | 27.6 M      | 142.2 M    |
| S100P    | YOLO11x Seg | 640×640        |        80 | 11.89 ms / 83.30 FPS (1 thread ) <br/> 22.83 ms / 86.92 FPS (2 threads)  | 5.0 ms                         | 62.1 M      | 319.0 M    |
| S100P    | YOLOv9c Seg | 640×640        |        80 | 6.29 ms / 156.58 FPS (1 thread ) <br/> 11.74 ms / 168.32 FPS (2 threads) | 5.0 ms                         | 27.7 M      | 158.0 M    |
| S100P    | YOLOv9e Seg | 640×640        |        80 | 14.20 ms / 69.78 FPS (1 thread ) <br/> 27.42 ms / 72.35 FPS (2 threads)  | 5.0 ms                         | 59.7 M      | 244.8 M    |
| S100P    | YOLOv8n Seg | 640×640        |        80 | 1.39 ms / 666.05 FPS (1 thread ) <br/> 2.01 ms / 946.83 FPS (2 threads)  | 5.0 ms                         | 3.4 M       | 12.6 M     |
| S100P    | YOLOv8s Seg | 640×640        |        80 | 2.32 ms / 411.56 FPS (1 thread ) <br/> 3.86 ms / 502.69 FPS (2 threads)  | 5.0 ms                         | 11.8 M      | 42.6 M     |
| S100P    | YOLOv8m Seg | 640×640        |        80 | 4.37 ms / 223.73 FPS (1 thread ) <br/> 7.95 ms / 247.14 FPS (2 threads)  | 5.0 ms                         | 27.3 M      | 100.2 M    |
| S100P    | YOLOv8l Seg | 640×640        |        80 | 8.20 ms / 120.19 FPS (1 thread ) <br/> 15.46 ms / 127.84 FPS (2 threads) | 5.0 ms                         | 46.0 M      | 220.5 M    |
| S100P    | YOLOv8x Seg | 640×640        |        80 | 13.01 ms / 76.23 FPS (1 thread ) <br/> 25.11 ms / 79.06 FPS (2 threads)  | 5.0 ms                         | 71.8 M      | 344.1 M    |

#### Pose Estimation

### Pose Estimation

| Device   | Model        | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|--------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S100P    | YOLO11n Pose | 640×640        |        80 | 1.23 ms / 770.10 FPS (1 thread ) <br/> 1.74 ms / 1097.27 FPS (2 threads) | 1.0 ms                         | 2.9 M       | 7.6 M      |
| S100P    | YOLO11s Pose | 640×640        |        80 | 1.92 ms / 501.66 FPS (1 thread ) <br/> 3.11 ms / 627.29 FPS (2 threads)  | 1.0 ms                         | 9.9 M       | 23.2 M     |
| S100P    | YOLO11m Pose | 640×640        |        80 | 4.04 ms / 241.82 FPS (1 thread ) <br/> 7.32 ms / 269.96 FPS (2 threads)  | 1.0 ms                         | 20.9 M      | 71.7 M     |
| S100P    | YOLO11l Pose | 640×640        |        80 | 4.87 ms / 202.29 FPS (1 thread ) <br/> 8.99 ms / 220.09 FPS (2 threads)  | 1.0 ms                         | 26.2 M      | 90.7 M     |
| S100P    | YOLO11x Pose | 640×640        |        80 | 9.15 ms / 108.13 FPS (1 thread ) <br/> 17.45 ms / 113.64 FPS (2 threads) | 1.0 ms                         | 58.8 M      | 203.3 M    |
| S100P    | YOLOv8n Pose | 640×640        |        80 | 1.14 ms / 822.46 FPS (1 thread ) <br/> 1.58 ms / 1206.58 FPS (2 threads) | 1.0 ms                         | 3.3 M       | 9.2 M      |
| S100P    | YOLOv8s Pose | 640×640        |        80 | 1.97 ms / 486.85 FPS (1 thread ) <br/> 3.23 ms / 606.41 FPS (2 threads)  | 1.0 ms                         | 11.6 M      | 30.2 M     |
| S100P    | YOLOv8m Pose | 640×640        |        80 | 3.65 ms / 267.74 FPS (1 thread ) <br/> 6.54 ms / 301.30 FPS (2 threads)  | 1.0 ms                         | 26.4 M      | 81.0 M     |
| S100P    | YOLOv8l Pose | 640×640        |        80 | 6.92 ms / 142.52 FPS (1 thread ) <br/> 12.99 ms / 152.18 FPS (2 threads) | 1.0 ms                         | 44.4 M      | 168.6 M    |
| S100P    | YOLOv8x Pose | 640×640        |        80 | 10.67 ms / 92.89 FPS (1 thread ) <br/> 20.48 ms / 96.97 FPS (2 threads)  | 1.0 ms                         | 69.4 M      | 263.2 M    |

#### Image Classification

### Image Classification

| Device   | Model       | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                                                                   | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-------------|----------------|-----------|-------------------------------------------------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S100P    | YOLO11n CLS | 640×640        |        80 | 0.40 ms / 2368.83 FPS (1 thread ) <br/> 0.46 ms / 4151.62 FPS (2 threads) <br/> 0.56 ms / 5164.09 FPS (3 threads) | 0.5 ms                         | 2.8 M       | 4.2 M      |
| S100P    | YOLO11s CLS | 640×640        |        80 | 0.52 ms / 1843.47 FPS (1 thread ) <br/> 0.61 ms / 3128.03 FPS (2 threads) <br/> 0.81 ms / 3593.24 FPS (3 threads) | 0.5 ms                         | 6.7 M       | 13.0 M     |
| S100P    | YOLO11m CLS | 640×640        |        80 | 0.78 ms / 1248.81 FPS (1 thread ) <br/> 1.01 ms / 1935.47 FPS (2 threads)                                         | 0.5 ms                         | 11.6 M      | 40.3 M     |
| S100P    | YOLO11l CLS | 640×640        |        80 | 0.90 ms / 1088.57 FPS (1 thread ) <br/> 1.27 ms / 1544.42 FPS (2 threads)                                         | 0.5 ms                         | 14.1 M      | 50.4 M     |
| S100P    | YOLO11x CLS | 640×640        |        80 | 1.45 ms / 676.30 FPS (1 thread ) <br/> 2.34 ms / 844.07 FPS (2 threads)                                           | 0.5 ms                         | 29.6 M      | 111.3 M    |
| S100P    | YOLOv8n CLS | 640×640        |        80 | 0.38 ms / 2470.91 FPS (1 thread ) <br/> 0.45 ms / 4304.69 FPS (2 threads) <br/> 0.55 ms / 5272.31 FPS (3 threads) | 0.5 ms                         | 2.7 M       | 4.3 M      |
| S100P    | YOLOv8s CLS | 640×640        |        80 | 0.49 ms / 1953.98 FPS (1 thread ) <br/> 0.57 ms / 3364.17 FPS (2 threads) <br/> 0.70 ms / 4109.05 FPS (3 threads) | 0.5 ms                         | 6.4 M       | 13.5 M     |
| S100P    | YOLOv8m CLS | 640×640        |        80 | 0.80 ms / 1218.07 FPS (1 thread ) <br/> 1.05 ms / 1876.12 FPS (2 threads)                                         | 0.5 ms                         | 17.0 M      | 42.7 M     |
| S100P    | YOLOv8l CLS | 640×640        |        80 | 1.44 ms / 683.65 FPS (1 thread ) <br/> 2.34 ms / 842.80 FPS (2 threads)                                           | 0.5 ms                         | 37.5 M      | 99.7 M     |
| S100P    | YOLOv8x CLS | 640×640        |        80 | 2.09 ms / 470.34 FPS (1 thread ) <br/> 3.55 ms / 559.63 FPS (2 threads)                                           | 0.5 ms                         | 57.4 M      | 154.8 M    |

### RDK S100

### Obeject Detection

| Device   | Model           | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-----------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S100     | YOLO12n Detect  | 640×640        |        80 | 2.65 ms / 368.54 FPS (1 thread ) <br/> 4.43 ms / 443.33 FPS (2 threads)  | 2.0 ms                         | 2.6 M       | 7.7 M      |
| S100     | YOLO12s Detect  | 640×640        |        80 | 4.48 ms / 220.08 FPS (1 thread ) <br/> 8.10 ms / 244.66 FPS (2 threads)  | 2.0 ms                         | 9.3 M       | 21.4 M     |
| S100     | YOLO12m Detect  | 640×640        |        80 | 9.27 ms / 107.09 FPS (1 thread ) <br/> 17.56 ms / 113.12 FPS (2 threads) | 2.0 ms                         | 20.2 M      | 67.5 M     |
| S100     | YOLO12l Detect  | 640×640        |        80 | 14.66 ms / 67.85 FPS (1 thread ) <br/> 28.30 ms / 70.28 FPS (2 threads)  | 2.0 ms                         | 26.4 M      | 88.9 M     |
| S100     | YOLO12x Detect  | 640×640        |        80 | 24.72 ms / 40.33 FPS (1 thread ) <br/> 48.27 ms / 41.26 FPS (2 threads)  | 2.0 ms                         | 59.1 M      | 199.0 M    |
| S100     | YOLO11n Detect  | 640×640        |        80 | 1.62 ms / 596.53 FPS (1 thread ) <br/> 2.39 ms / 813.87 FPS (2 threads)  | 2.0 ms                         | 2.6 M       | 6.5 M      |
| S100     | YOLO11s Detect  | 640×640        |        80 | 2.63 ms / 371.42 FPS (1 thread ) <br/> 4.39 ms / 448.18 FPS (2 threads)  | 2.0 ms                         | 9.4 M       | 21.5 M     |
| S100     | YOLO11m Detect  | 640×640        |        80 | 5.63 ms / 175.69 FPS (1 thread ) <br/> 10.35 ms / 191.62 FPS (2 threads) | 2.0 ms                         | 20.1 M      | 68.0 M     |
| S100     | YOLO11l Detect  | 640×640        |        80 | 6.96 ms / 142.36 FPS (1 thread ) <br/> 13.02 ms / 152.41 FPS (2 threads) | 2.0 ms                         | 25.3 M      | 86.9 M     |
| S100     | YOLO11x Detect  | 640×640        |        80 | 13.13 ms / 75.78 FPS (1 thread ) <br/> 25.24 ms / 78.82 FPS (2 threads)  | 2.0 ms                         | 56.9 M      | 194.9 M    |
| S100     | YOLOv10n Detect | 640×640        |        80 | 1.58 ms / 608.94 FPS (1 thread ) <br/> 2.32 ms / 837.04 FPS (2 threads)  | 2.0 ms                         | 2.3 M       | 6.7 M      |
| S100     | YOLOv10s Detect | 640×640        |        80 | 2.53 ms / 385.50 FPS (1 thread ) <br/> 4.18 ms / 471.09 FPS (2 threads)  | 2.0 ms                         | 7.2 M       | 21.6 M     |
| S100     | YOLOv10m Detect | 640×640        |        80 | 4.49 ms / 219.98 FPS (1 thread ) <br/> 8.11 ms / 244.17 FPS (2 threads)  | 2.0 ms                         | 15.4 M      | 59.1 M     |
| S100     | YOLOv10b Detect | 640×640        |        80 | 6.28 ms / 157.57 FPS (1 thread ) <br/> 11.65 ms / 170.32 FPS (2 threads) | 2.0 ms                         | 19.1 M      | 92.0 M     |
| S100     | YOLOv10l Detect | 640×640        |        80 | 7.95 ms / 124.70 FPS (1 thread ) <br/> 14.98 ms / 132.53 FPS (2 threads) | 2.0 ms                         | 24.4 M      | 120.3 M    |
| S100     | YOLOv10x Detect | 640×640        |        80 | 10.83 ms / 91.79 FPS (1 thread ) <br/> 20.66 ms / 96.17 FPS (2 threads)  | 2.0 ms                         | 29.5 M      | 160.4 M    |
| S100     | YOLOv9t Detect  | 640×640        |        80 | 1.77 ms / 546.03 FPS (1 thread ) <br/> 2.67 ms / 730.68 FPS (2 threads)  | 2.0 ms                         | 2.1 M       | 8.2 M      |
| S100     | YOLOv9s Detect  | 640×640        |        80 | 2.74 ms / 357.91 FPS (1 thread ) <br/> 4.62 ms / 425.97 FPS (2 threads)  | 2.0 ms                         | 7.2 M       | 26.9 M     |
| S100     | YOLOv9m Detect  | 640×640        |        80 | 5.52 ms / 179.23 FPS (1 thread ) <br/> 10.13 ms / 195.30 FPS (2 threads) | 2.0 ms                         | 20.1 M      | 76.8 M     |
| S100     | YOLOv9c Detect  | 640×640        |        80 | 6.98 ms / 142.00 FPS (1 thread ) <br/> 13.05 ms / 151.95 FPS (2 threads) | 2.0 ms                         | 25.3 M      | 102.7 M    |
| S100     | YOLOv9e Detect  | 640×640        |        80 | 17.75 ms / 56.15 FPS (1 thread ) <br/> 34.41 ms / 57.85 FPS (2 threads)  | 2.0 ms                         | 57.4 M      | 189.5 M    |
| S100     | YOLOv8n Detect  | 640×640        |        80 | 1.53 ms / 632.06 FPS (1 thread ) <br/> 2.24 ms / 868.87 FPS (2 threads)  | 2.0 ms                         | 3.2 M       | 8.7 M      |
| S100     | YOLOv8s Detect  | 640×640        |        80 | 2.63 ms / 371.16 FPS (1 thread ) <br/> 4.41 ms / 446.48 FPS (2 threads)  | 2.0 ms                         | 11.2 M      | 28.6 M     |
| S100     | YOLOv8m Detect  | 640×640        |        80 | 5.18 ms / 190.64 FPS (1 thread ) <br/> 9.45 ms / 209.80 FPS (2 threads)  | 2.0 ms                         | 25.9 M      | 78.9 M     |
| S100     | YOLOv8l Detect  | 640×640        |        80 | 9.97 ms / 99.68 FPS (1 thread ) <br/> 19.00 ms / 104.65 FPS (2 threads)  | 2.0 ms                         | 43.7 M      | 165.2 M    |
| S100     | YOLOv8x Detect  | 640×640        |        80 | 15.77 ms / 63.15 FPS (1 thread ) <br/> 30.53 ms / 65.20 FPS (2 threads)  | 2.0 ms                         | 68.2 M      | 257.8 M    |
| S100     | YOLOv5nu Detect | 640×640        |        80 | 1.42 ms / 674.92 FPS (1 thread ) <br/> 2.02 ms / 959.05 FPS (2 threads)  | 2.0 ms                         | 2.6 M       | 7.7 M      |
| S100     | YOLOv5su Detect | 640×640        |        80 | 2.31 ms / 420.83 FPS (1 thread ) <br/> 3.79 ms / 519.22 FPS (2 threads)  | 2.0 ms                         | 9.1 M       | 24.0 M     |
| S100     | YOLOv5mu Detect | 640×640        |        80 | 4.50 ms / 218.77 FPS (1 thread ) <br/> 8.11 ms / 244.06 FPS (2 threads)  | 2.0 ms                         | 25.1 M      | 64.2 M     |
| S100     | YOLOv5lu Detect | 640×640        |        80 | 8.96 ms / 110.78 FPS (1 thread ) <br/> 16.97 ms / 117.15 FPS (2 threads) | 2.0 ms                         | 53.2 M      | 135.0 M    |
| S100     | YOLOv5xu Detect | 640×640        |        80 | 15.97 ms / 62.32 FPS (1 thread ) <br/> 30.90 ms / 64.41 FPS (2 threads)  | 2.0 ms                         | 97.2 M      | 246.4 M    |

#### Instance Segmentation

### Instance Segmentation

| Device   | Model       | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S100     | YOLO11n Seg | 640×640        |        80 | 2.06 ms / 463.93 FPS (1 thread ) <br/> 3.17 ms / 613.76 FPS (2 threads)  | 5.0 ms                         | 2.9 M       | 10.4 M     |
| S100     | YOLO11s Seg | 640×640        |        80 | 3.34 ms / 291.16 FPS (1 thread ) <br/> 5.69 ms / 344.91 FPS (2 threads)  | 5.0 ms                         | 10.1 M      | 35.5 M     |
| S100     | YOLO11m Seg | 640×640        |        80 | 7.86 ms / 125.63 FPS (1 thread ) <br/> 14.67 ms / 135.16 FPS (2 threads) | 5.0 ms                         | 22.4 M      | 123.3 M    |
| S100     | YOLO11l Seg | 640×640        |        80 | 9.17 ms / 108.00 FPS (1 thread ) <br/> 17.29 ms / 114.67 FPS (2 threads) | 5.0 ms                         | 27.6 M      | 142.2 M    |
| S100     | YOLO11x Seg | 640×640        |        80 | 17.74 ms / 56.07 FPS (1 thread ) <br/> 34.33 ms / 57.96 FPS (2 threads)  | 5.0 ms                         | 62.1 M      | 319.0 M    |
| S100     | YOLOv9c Seg | 640×640        |        80 | 9.07 ms / 109.16 FPS (1 thread ) <br/> 17.12 ms / 115.80 FPS (2 threads) | 5.0 ms                         | 27.7 M      | 158.0 M    |
| S100     | YOLOv9e Seg | 640×640        |        80 | 20.15 ms / 49.38 FPS (1 thread ) <br/> 39.07 ms / 50.91 FPS (2 threads)  | 5.0 ms                         | 59.7 M      | 244.8 M    |
| S100     | YOLOv8n Seg | 640×640        |        80 | 1.93 ms / 495.35 FPS (1 thread ) <br/> 2.98 ms / 652.21 FPS (2 threads)  | 5.0 ms                         | 3.4 M       | 12.6 M     |
| S100     | YOLOv8s Seg | 640×640        |        80 | 3.37 ms / 288.70 FPS (1 thread ) <br/> 5.76 ms / 341.12 FPS (2 threads)  | 5.0 ms                         | 11.8 M      | 42.6 M     |
| S100     | YOLOv8m Seg | 640×640        |        80 | 6.65 ms / 148.28 FPS (1 thread ) <br/> 12.29 ms / 161.07 FPS (2 threads) | 5.0 ms                         | 27.3 M      | 100.2 M    |
| S100     | YOLOv8l Seg | 640×640        |        80 | 12.21 ms / 81.34 FPS (1 thread ) <br/> 23.32 ms / 85.17 FPS (2 threads)  | 5.0 ms                         | 46.0 M      | 220.5 M    |
| S100     | YOLOv8x Seg | 640×640        |        80 | 19.51 ms / 51.00 FPS (1 thread ) <br/> 37.80 ms / 52.62 FPS (2 threads)  | 5.0 ms                         | 71.8 M      | 344.1 M    |

#### Pose Estimation

### Pose Estimation

| Device   | Model        | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|--------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S100     | YOLO11n Pose | 640×640        |        80 | 1.69 ms / 568.00 FPS (1 thread ) <br/> 2.48 ms / 780.47 FPS (2 threads)  | 1.0 ms                         | 2.9 M       | 7.6 M      |
| S100     | YOLO11s Pose | 640×640        |        80 | 2.76 ms / 354.06 FPS (1 thread ) <br/> 4.62 ms / 424.92 FPS (2 threads)  | 1.0 ms                         | 9.9 M       | 23.2 M     |
| S100     | YOLO11m Pose | 640×640        |        80 | 5.89 ms / 167.50 FPS (1 thread ) <br/> 10.79 ms / 183.34 FPS (2 threads) | 1.0 ms                         | 20.9 M      | 71.7 M     |
| S100     | YOLO11l Pose | 640×640        |        80 | 7.23 ms / 136.91 FPS (1 thread ) <br/> 13.48 ms / 147.21 FPS (2 threads) | 1.0 ms                         | 26.2 M      | 90.7 M     |
| S100     | YOLO11x Pose | 640×640        |        80 | 13.61 ms / 73.02 FPS (1 thread ) <br/> 26.16 ms / 76.04 FPS (2 threads)  | 1.0 ms                         | 58.8 M      | 203.3 M    |
| S100     | YOLOv8n Pose | 640×640        |        80 | 1.62 ms / 587.59 FPS (1 thread ) <br/> 2.31 ms / 837.95 FPS (2 threads)  | 1.0 ms                         | 3.3 M       | 9.2 M      |
| S100     | YOLOv8s Pose | 640×640        |        80 | 2.83 ms / 344.35 FPS (1 thread ) <br/> 4.71 ms / 417.72 FPS (2 threads)  | 1.0 ms                         | 11.6 M      | 30.2 M     |
| S100     | YOLOv8m Pose | 640×640        |        80 | 5.47 ms / 180.46 FPS (1 thread ) <br/> 9.92 ms / 199.50 FPS (2 threads)  | 1.0 ms                         | 26.4 M      | 81.0 M     |
| S100     | YOLOv8l Pose | 640×640        |        80 | 10.31 ms / 96.20 FPS (1 thread ) <br/> 19.55 ms / 101.60 FPS (2 threads) | 1.0 ms                         | 44.4 M      | 168.6 M    |
| S100     | YOLOv8x Pose | 640×640        |        80 | 16.07 ms / 61.88 FPS (1 thread ) <br/> 31.01 ms / 64.14 FPS (2 threads)  | 1.0 ms                         | 69.4 M      | 263.2 M    |

#### Image Classification

### Image Classification

| Device   | Model       | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                                                                   | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-------------|----------------|-----------|-------------------------------------------------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S100     | YOLO11n CLS | 640×640        |        80 | 0.53 ms / 1827.92 FPS (1 thread ) <br/> 0.62 ms / 3115.65 FPS (2 threads) <br/> 0.70 ms / 4141.22 FPS (3 threads) | 0.5 ms                         | 2.8 M       | 4.2 M      |
| S100     | YOLO11s CLS | 640×640        |        80 | 0.68 ms / 1415.98 FPS (1 thread ) <br/> 0.76 ms / 2553.99 FPS (2 threads) <br/> 1.05 ms / 2767.63 FPS (3 threads) | 0.5 ms                         | 6.7 M       | 13.0 M     |
| S100     | YOLO11m CLS | 640×640        |        80 | 1.02 ms / 955.28 FPS (1 thread ) <br/> 1.35 ms / 1445.18 FPS (2 threads)                                          | 0.5 ms                         | 11.6 M      | 40.3 M     |
| S100     | YOLO11l CLS | 640×640        |        80 | 1.21 ms / 805.52 FPS (1 thread ) <br/> 1.73 ms / 1139.48 FPS (2 threads)                                          | 0.5 ms                         | 14.1 M      | 50.4 M     |
| S100     | YOLO11x CLS | 640×640        |        80 | 1.97 ms / 501.49 FPS (1 thread ) <br/> 3.23 ms / 612.29 FPS (2 threads)                                           | 0.5 ms                         | 29.6 M      | 111.3 M    |
| S100     | YOLOv8n CLS | 640×640        |        80 | 0.49 ms / 1928.23 FPS (1 thread ) <br/> 0.57 ms / 3399.86 FPS (2 threads) <br/> 0.66 ms / 4410.92 FPS (3 threads) | 0.5 ms                         | 2.7 M       | 4.3 M      |
| S100     | YOLOv8s CLS | 640×640        |        80 | 0.62 ms / 1562.83 FPS (1 thread ) <br/> 0.71 ms / 2712.53 FPS (2 threads) <br/> 0.89 ms / 3279.66 FPS (3 threads) | 0.5 ms                         | 6.4 M       | 13.5 M     |
| S100     | YOLOv8m CLS | 640×640        |        80 | 1.00 ms / 970.04 FPS (1 thread ) <br/> 1.31 ms / 1500.86 FPS (2 threads)                                          | 0.5 ms                         | 17.0 M      | 42.7 M     |
| S100     | YOLOv8l CLS | 640×640        |        80 | 1.98 ms / 497.58 FPS (1 thread ) <br/> 3.22 ms / 614.92 FPS (2 threads)                                           | 0.5 ms                         | 37.5 M      | 99.7 M     |
| S100     | YOLOv8x CLS | 640×640        |        80 | 2.77 ms / 357.03 FPS (1 thread ) <br/> 4.81 ms / 412.60 FPS (2 threads)                                           | 0.5 ms                         | 57.4 M      | 154.8 M    |

### RDK X5

### Obeject Detection

| Device   | Model           | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-----------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| X5       | YOLO12n Detect  | 640×640        |        80 | 39.70 ms / 25.17 FPS (1 thread ) <br/> 73.19 ms / 27.24 FPS (2 threads)  | 5.0 ms                         | 2.6 M       | 7.7 M      |
| X5       | YOLO12s Detect  | 640×640        |        80 | 63.74 ms / 15.68 FPS (1 thread ) <br/> 121.24 ms / 16.45 FPS (2 threads) | 5.0 ms                         | 9.3 M       | 21.4 M     |
| X5       | YOLO12m Detect  | 640×640        |        80 | 103.02 ms / 9.70 FPS (1 thread ) <br/> 199.58 ms / 9.99 FPS (2 threads)  | 5.0 ms                         | 20.2 M      | 67.5 M     |
| X5       | YOLO12l Detect  | 640×640        |        80 | 183.00 ms / 5.46 FPS (1 thread ) <br/> 359.03 ms / 5.56 FPS (2 threads)  | 5.0 ms                         | 26.4 M      | 88.9 M     |
| X5       | YOLO12x Detect  | 640×640        |        80 | 315.16 ms / 3.17 FPS (1 thread )                                         | 5.0 ms                         | 59.1 M      | 199.0 M    |
| X5       | YOLO11n Detect  | 640×640        |        80 | 8.25 ms / 121.05 FPS (1 thread ) <br/> 10.56 ms / 188.57 FPS (2 threads) | 5.0 ms                         | 2.6 M       | 6.5 M      |
| X5       | YOLO11s Detect  | 640×640        |        80 | 15.81 ms / 63.16 FPS (1 thread ) <br/> 25.74 ms / 77.43 FPS (2 threads)  | 5.0 ms                         | 9.4 M       | 21.5 M     |
| X5       | YOLO11m Detect  | 640×640        |        80 | 34.68 ms / 28.82 FPS (1 thread ) <br/> 63.30 ms / 31.51 FPS (2 threads)  | 5.0 ms                         | 20.1 M      | 68.0 M     |
| X5       | YOLO11l Detect  | 640×640        |        80 | 45.23 ms / 22.10 FPS (1 thread ) <br/> 84.30 ms / 23.66 FPS (2 threads)  | 5.0 ms                         | 25.3 M      | 86.9 M     |
| X5       | YOLO11x Detect  | 640×640        |        80 | 96.70 ms / 10.34 FPS (1 thread ) <br/> 186.76 ms / 10.68 FPS (2 threads) | 5.0 ms                         | 56.9 M      | 194.9 M    |
| X5       | YOLOv10n Detect | 640×640        |        80 | 8.75 ms / 114.19 FPS (1 thread ) <br/> 11.60 ms / 171.72 FPS (2 threads) | 5.0 ms                         | 2.3 M       | 6.7 M      |
| X5       | YOLOv10s Detect | 640×640        |        80 | 14.84 ms / 67.32 FPS (1 thread ) <br/> 23.85 ms / 83.58 FPS (2 threads)  | 5.0 ms                         | 7.2 M       | 21.6 M     |
| X5       | YOLOv10m Detect | 640×640        |        80 | 29.40 ms / 33.99 FPS (1 thread ) <br/> 52.83 ms / 37.75 FPS (2 threads)  | 5.0 ms                         | 15.4 M      | 59.1 M     |
| X5       | YOLOv10b Detect | 640×640        |        80 | 40.14 ms / 24.90 FPS (1 thread ) <br/> 74.20 ms / 26.88 FPS (2 threads)  | 5.0 ms                         | 19.1 M      | 92.0 M     |
| X5       | YOLOv10l Detect | 640×640        |        80 | 49.89 ms / 20.04 FPS (1 thread ) <br/> 93.66 ms / 21.30 FPS (2 threads)  | 5.0 ms                         | 24.4 M      | 120.3 M    |
| X5       | YOLOv10x Detect | 640×640        |        80 | 68.92 ms / 14.51 FPS (1 thread ) <br/> 131.54 ms / 15.16 FPS (2 threads) | 5.0 ms                         | 29.5 M      | 160.4 M    |
| X5       | YOLOv9t Detect  | 640×640        |        80 | 6.97 ms / 143.14 FPS (1 thread ) <br/> 7.96 ms / 250.11 FPS (2 threads)  | 5.0 ms                         | 2.1 M       | 8.2 M      |
| X5       | YOLOv9s Detect  | 640×640        |        80 | 13.00 ms / 76.81 FPS (1 thread ) <br/> 20.16 ms / 98.81 FPS (2 threads)  | 5.0 ms                         | 7.2 M       | 26.9 M     |
| X5       | YOLOv9m Detect  | 640×640        |        80 | 32.63 ms / 30.63 FPS (1 thread ) <br/> 59.31 ms / 33.62 FPS (2 threads)  | 5.0 ms                         | 20.1 M      | 76.8 M     |
| X5       | YOLOv9c Detect  | 640×640        |        80 | 40.46 ms / 24.71 FPS (1 thread ) <br/> 74.77 ms / 26.67 FPS (2 threads)  | 5.0 ms                         | 25.3 M      | 102.7 M    |
| X5       | YOLOv9e Detect  | 640×640        |        80 | 119.80 ms / 8.35 FPS (1 thread ) <br/> 233.08 ms / 8.56 FPS (2 threads)  | 5.0 ms                         | 57.4 M      | 189.5 M    |
| X5       | YOLOv8n Detect  | 640×640        |        80 | 7.00 ms / 142.60 FPS (1 thread ) <br/> 8.06 ms / 246.82 FPS (2 threads)  | 5.0 ms                         | 3.2 M       | 8.7 M      |
| X5       | YOLOv8s Detect  | 640×640        |        80 | 13.63 ms / 73.30 FPS (1 thread ) <br/> 21.38 ms / 93.20 FPS (2 threads)  | 5.0 ms                         | 11.2 M      | 28.6 M     |
| X5       | YOLOv8m Detect  | 640×640        |        80 | 30.74 ms / 32.51 FPS (1 thread ) <br/> 55.51 ms / 35.93 FPS (2 threads)  | 5.0 ms                         | 25.9 M      | 78.9 M     |
| X5       | YOLOv8l Detect  | 640×640        |        80 | 59.51 ms / 16.80 FPS (1 thread ) <br/> 112.80 ms / 17.68 FPS (2 threads) | 5.0 ms                         | 43.7 M      | 165.2 M    |
| X5       | YOLOv8x Detect  | 640×640        |        80 | 92.72 ms / 10.78 FPS (1 thread ) <br/> 178.95 ms / 11.15 FPS (2 threads) | 5.0 ms                         | 68.2 M      | 257.8 M    |
| X5       | YOLOv5nu Detect | 640×640        |        80 | 6.33 ms / 157.59 FPS (1 thread ) <br/> 6.80 ms / 291.89 FPS (2 threads)  | 5.0 ms                         | 2.6 M       | 7.7 M      |
| X5       | YOLOv5su Detect | 640×640        |        80 | 12.33 ms / 81.04 FPS (1 thread ) <br/> 18.88 ms / 105.56 FPS (2 threads) | 5.0 ms                         | 9.1 M       | 24.0 M     |
| X5       | YOLOv5mu Detect | 640×640        |        80 | 26.57 ms / 37.62 FPS (1 thread ) <br/> 47.20 ms / 42.24 FPS (2 threads)  | 5.0 ms                         | 25.1 M      | 64.2 M     |
| X5       | YOLOv5lu Detect | 640×640        |        80 | 52.83 ms / 18.92 FPS (1 thread ) <br/> 99.42 ms / 20.06 FPS (2 threads)  | 5.0 ms                         | 53.2 M      | 135.0 M    |
| X5       | YOLOv5xu Detect | 640×640        |        80 | 91.55 ms / 10.92 FPS (1 thread ) <br/> 176.49 ms / 11.30 FPS (2 threads) | 5.0 ms                         | 97.2 M      | 246.4 M    |

#### Instance Segmentation

### Instance Segmentation

| Device   | Model       | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| X5       | YOLO11n Seg | 640×640        |        80 | 11.55 ms / 86.39 FPS (1 thread ) <br/> 12.83 ms / 155.10 FPS (2 threads) | 20.0 ms                        | 2.9 M       | 10.4 M     |
| X5       | YOLO11s Seg | 640×640        |        80 | 21.62 ms / 46.22 FPS (1 thread ) <br/> 33.12 ms / 60.20 FPS (2 threads)  | 20.0 ms                        | 10.1 M      | 35.5 M     |
| X5       | YOLO11m Seg | 640×640        |        80 | 50.43 ms / 19.82 FPS (1 thread ) <br/> 90.49 ms / 22.04 FPS (2 threads)  | 20.0 ms                        | 22.4 M      | 123.3 M    |
| X5       | YOLO11l Seg | 640×640        |        80 | 60.60 ms / 16.50 FPS (1 thread ) <br/> 110.99 ms / 17.97 FPS (2 threads) | 20.0 ms                        | 27.6 M      | 142.2 M    |
| X5       | YOLO11x Seg | 640×640        |        80 | 130.40 ms / 7.67 FPS (1 thread ) <br/> 249.71 ms / 7.99 FPS (2 threads)  | 20.0 ms                        | 62.1 M      | 319.0 M    |
| X5       | YOLOv9c Seg | 640×640        |        80 | 55.85 ms / 17.90 FPS (1 thread ) <br/> 101.47 ms / 19.65 FPS (2 threads) | 20.0 ms                        | 27.7 M      | 158.0 M    |
| X5       | YOLOv9e Seg | 640×640        |        80 | 135.34 ms / 7.39 FPS (1 thread ) <br/> 260.08 ms / 7.67 FPS (2 threads)  | 20.0 ms                        | 59.7 M      | 244.8 M    |
| X5       | YOLOv8n Seg | 640×640        |        80 | 10.40 ms / 96.02 FPS (1 thread ) <br/> 10.75 ms / 185.21 FPS (2 threads) | 20.0 ms                        | 3.4 M       | 12.6 M     |
| X5       | YOLOv8s Seg | 640×640        |        80 | 19.56 ms / 51.08 FPS (1 thread ) <br/> 28.99 ms / 68.76 FPS (2 threads)  | 20.0 ms                        | 11.8 M      | 42.6 M     |
| X5       | YOLOv8m Seg | 640×640        |        80 | 40.52 ms / 24.67 FPS (1 thread ) <br/> 70.70 ms / 28.21 FPS (2 threads)  | 20.0 ms                        | 27.3 M      | 100.2 M    |
| X5       | YOLOv8l Seg | 640×640        |        80 | 75.00 ms / 13.33 FPS (1 thread ) <br/> 139.61 ms / 14.29 FPS (2 threads) | 20.0 ms                        | 46.0 M      | 220.5 M    |
| X5       | YOLOv8x Seg | 640×640        |        80 | 115.94 ms / 8.62 FPS (1 thread ) <br/> 221.06 ms / 9.02 FPS (2 threads)  | 20.0 ms                        | 71.8 M      | 344.1 M    |

#### Pose Estimation

### Pose Estimation

| Device   | Model        | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|--------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| X5       | YOLO11n Pose | 640×640        |        80 | 8.36 ms / 119.43 FPS (1 thread ) <br/> 10.97 ms / 181.61 FPS (2 threads) | 10.0 ms                        | 2.9 M       | 7.6 M      |
| X5       | YOLO11s Pose | 640×640        |        80 | 16.35 ms / 61.11 FPS (1 thread ) <br/> 26.99 ms / 73.85 FPS (2 threads)  | 10.0 ms                        | 9.9 M       | 23.2 M     |
| X5       | YOLO11m Pose | 640×640        |        80 | 35.74 ms / 27.97 FPS (1 thread ) <br/> 65.60 ms / 30.40 FPS (2 threads)  | 10.0 ms                        | 20.9 M      | 71.7 M     |
| X5       | YOLO11l Pose | 640×640        |        80 | 46.38 ms / 21.55 FPS (1 thread ) <br/> 86.82 ms / 22.97 FPS (2 threads)  | 10.0 ms                        | 26.2 M      | 90.7 M     |
| X5       | YOLO11x Pose | 640×640        |        80 | 98.88 ms / 10.11 FPS (1 thread ) <br/> 191.38 ms / 10.42 FPS (2 threads) | 10.0 ms                        | 58.8 M      | 203.3 M    |
| X5       | YOLOv8n Pose | 640×640        |        80 | 6.95 ms / 143.64 FPS (1 thread ) <br/> 8.23 ms / 241.76 FPS (2 threads)  | 10.0 ms                        | 3.3 M       | 9.2 M      |
| X5       | YOLOv8s Pose | 640×640        |        80 | 14.16 ms / 70.54 FPS (1 thread ) <br/> 22.62 ms / 88.09 FPS (2 threads)  | 10.0 ms                        | 11.6 M      | 30.2 M     |
| X5       | YOLOv8m Pose | 640×640        |        80 | 31.60 ms / 31.62 FPS (1 thread ) <br/> 57.34 ms / 34.78 FPS (2 threads)  | 10.0 ms                        | 26.4 M      | 81.0 M     |
| X5       | YOLOv8l Pose | 640×640        |        80 | 60.37 ms / 16.56 FPS (1 thread ) <br/> 114.73 ms / 17.38 FPS (2 threads) | 10.0 ms                        | 44.4 M      | 168.6 M    |
| X5       | YOLOv8x Pose | 640×640        |        80 | 94.15 ms / 10.62 FPS (1 thread ) <br/> 182.08 ms / 10.96 FPS (2 threads) | 10.0 ms                        | 69.4 M      | 263.2 M    |

#### Image Classification

### Image Classification

| Device   | Model       | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                           | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-------------|----------------|-----------|---------------------------------------------------------------------------|--------------------------------|-------------|------------|
| X5       | YOLO11n CLS | 640×640        |        80 | 1.06 ms / 939.95 FPS (1 thread ) <br/> 1.61 ms / 1236.07 FPS (2 threads)  | 0.5 ms                         | 2.8 M       | 4.2 M      |
| X5       | YOLO11s CLS | 640×640        |        80 | 2.01 ms / 495.14 FPS (1 thread ) <br/> 3.49 ms / 569.44 FPS (2 threads)   | 0.5 ms                         | 6.7 M       | 13.0 M     |
| X5       | YOLO11m CLS | 640×640        |        80 | 3.82 ms / 261.13 FPS (1 thread ) <br/> 7.09 ms / 280.82 FPS (2 threads)   | 0.5 ms                         | 11.6 M      | 40.3 M     |
| X5       | YOLO11l CLS | 640×640        |        80 | 5.02 ms / 199.15 FPS (1 thread ) <br/> 9.49 ms / 210.12 FPS (2 threads)   | 0.5 ms                         | 14.1 M      | 50.4 M     |
| X5       | YOLO11x CLS | 640×640        |        80 | 10.04 ms / 99.49 FPS (1 thread ) <br/> 19.48 ms / 102.39 FPS (2 threads)  | 0.5 ms                         | 29.6 M      | 111.3 M    |
| X5       | YOLOv8n CLS | 640×640        |        80 | 0.74 ms / 1348.98 FPS (1 thread ) <br/> 0.98 ms / 2018.94 FPS (2 threads) | 0.5 ms                         | 2.7 M       | 4.3 M      |
| X5       | YOLOv8s CLS | 640×640        |        80 | 1.44 ms / 690.86 FPS (1 thread ) <br/> 2.36 ms / 842.52 FPS (2 threads)   | 0.5 ms                         | 6.4 M       | 13.5 M     |
| X5       | YOLOv8m CLS | 640×640        |        80 | 3.66 ms / 272.72 FPS (1 thread ) <br/> 6.78 ms / 294.01 FPS (2 threads)   | 0.5 ms                         | 17.0 M      | 42.7 M     |
| X5       | YOLOv8l CLS | 640×640        |        80 | 7.98 ms / 125.23 FPS (1 thread ) <br/> 15.38 ms / 129.63 FPS (2 threads)  | 0.5 ms                         | 37.5 M      | 99.7 M     |
| X5       | YOLOv8x CLS | 640×640        |        80 | 13.12 ms / 76.18 FPS (1 thread ) <br/> 25.64 ms / 77.78 FPS (2 threads)   | 0.5 ms                         | 57.4 M      | 154.8 M    |



### RDK S100 Performance Data (Performance @ NV12)

| Device | Model | Size <br> (Pixels) | Classes | BPU Task Latency / <br> BPU Throughput (Threads) | CPU Latency | Params <br> (M) | FLOPs <br> (G) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| S100 | YOLO26n Detect | 640x640 | 80 | 1.70 ms / 499.33 FPS (1 thread) <br> 2.31 ms / 779.33 FPS (2 threads) | - | 2.57 | 6.1 |
| S100 | YOLO26s Detect | 640x640 | 80 | 2.87 ms / 314.84 FPS (1 thread) <br> 4.63 ms / 409.84 FPS (2 threads) | - | 10.01 | 22.8 |
| S100 | YOLO26m Detect | 640x640 | 80 | 6.06 ms / 157.14 FPS (1 thread) <br> 10.99 ms / 177.56 FPS (2 threads) | - | 21.90 | 75.4 |
| S100 | YOLO26l Detect | 640x640 | 80 | 7.35 ms / 130.56 FPS (1 thread) <br> 13.51 ms / 144.92 FPS (2 threads) | - | 26.30 | 93.8 |
| S100 | YOLO26x Detect | 640x640 | 80 | 13.91 ms / 70.33 FPS (1 thread) <br> 26.58 ms / 74.37 FPS (2 threads) | - | 58.99 | 209.5 |
| S100 | YOLO26n Seg | 640x640 | 80 | 2.23 ms / 354.15 FPS (1 thread) <br> 2.97 ms / 566.80 FPS (2 threads) | - | 3.13 | 10.5 |
| S100 | YOLO26s Seg | 640x640 | 80 | 3.80 ms / 234.05 FPS (1 thread) <br> 6.17 ms / 299.39 FPS (2 threads) | - | 11.51 | 37.4 |
| S100 | YOLO26m Seg | 640x640 | 80 | 9.08 ms / 103.15 FPS (1 thread) <br> 16.21 ms / 118.81 FPS (2 threads) | - | 27.11 | 132.5 |
| S100 | YOLO26l Seg | 640x640 | 80 | 10.48 ms / 90.13 FPS (1 thread) <br> 19.31 ms / 100.34 FPS (2 threads) | - | 31.52 | 150.9 |
| S100 | YOLO26x Seg | 640x640 | 80 | 19.90 ms / 48.70 FPS (1 thread) <br> 38.17 ms / 51.53 FPS (2 threads) | - | 70.69 | 337.7 |
| S100 | YOLO26n Pose | 640x640 | 1 | 1.81 ms / 486.37 FPS (1 thread) <br> 2.59 ms / 713.83 FPS (2 threads) | - | 3.68 | 10.3 |
| S100 | YOLO26s Pose | 640x640 | 1 | 3.07 ms / 300.62 FPS (1 thread) <br> 5.03 ms / 380.99 FPS (2 threads) | - | 11.81 | 29.2 |
| S100 | YOLO26m Pose | 640x640 | 1 | 6.42 ms / 149.73 FPS (1 thread) <br> 11.68 ms / 167.90 FPS (2 threads) | - | 24.22 | 85.2 |
| S100 | YOLO26l Pose | 640x640 | 1 | 7.75 ms / 124.84 FPS (1 thread) <br> 14.30 ms / 137.49 FPS (2 threads) | - | 28.63 | 103.6 |
| S100 | YOLO26x Pose | 640x640 | 1 | 14.45 ms / 67.87 FPS (1 thread) <br> 27.60 ms / 71.79 FPS (2 threads) | - | 62.73 | 225.3 |
| S100 | YOLO26n Cls | 224x224 | 1000 | 0.56 ms / 1659.68 FPS (1 thread) <br> 0.62 ms / 3112.09 FPS (2 threads) | - | 2.81 | 0.5 |
| S100 | YOLO26s Cls | 224x224 | 1000 | 0.74 ms / 1293.15 FPS (1 thread) <br> 0.83 ms / 2332.95 FPS (2 threads) | - | 6.72 | 1.6 |
| S100 | YOLO26m Cls | 224x224 | 1000 | 1.13 ms / 853.03 FPS (1 thread) <br> 1.56 ms / 1266.73 FPS (2 threads) | - | 11.63 | 5.0 |
| S100 | YOLO26l Cls | 224x224 | 1000 | 1.34 ms / 723.45 FPS (1 thread) <br> 1.97 ms / 999.78 FPS (2 threads) | - | 14.12 | 6.2 |
| S100 | YOLO26x Cls | 224x224 | 1000 | 2.08 ms / 471.88 FPS (1 thread) <br> 3.46 ms / 573.27 FPS (2 threads) | - | 29.64 | 13.7 |
| S100 | YOLO26n Obb | 640x640 | 15 | 1.64 ms / 548.83 FPS (1 thread) <br> 2.24 ms / 842.21 FPS (2 threads) | - | 2.65 | 6.3 |
| S100 | YOLO26s Obb | 640x640 | 15 | 2.80 ms / 337.46 FPS (1 thread) <br> 4.65 ms / 418.28 FPS (2 threads) | - | 10.53 | 24.5 |
| S100 | YOLO26m Obb | 640x640 | 15 | 6.16 ms / 157.61 FPS (1 thread) <br> 11.27 ms / 175.02 FPS (2 threads) | - | 23.49 | 82.2 |
| S100 | YOLO26l Obb | 640x640 | 15 | 7.46 ms / 130.74 FPS (1 thread) <br> 13.79 ms / 143.34 FPS (2 threads) | - | 27.90 | 100.6 |
| S100 | YOLO26x Obb | 640x640 | 15 | 13.81 ms / 71.33 FPS (1 thread) <br> 26.95 ms / 73.69 FPS (2 threads) | - | 62.66 | 225.3 |



### RDK S600 Performance Data (Performance @ NV12)

| Device | Model | Size <br> (Pixels) | Classes | BPU Task Latency / <br> BPU Throughput (Threads) | CPU Latency | Params <br> (M) | FLOPs <br> (G) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| S600 | YOLO26n Detect | 640x640 | 80 | 0.83 ms / 1170.57 FPS (1 thread) <br> 1.85 ms / 6023.19 FPS (12 threads) | - | 2.57 | 6.1 |
| S600 | YOLO26s Detect | 640x640 | 80 | 1.27 ms / 780.57 FPS (1 thread) <br> 3.17 ms / 3576.09 FPS (12 threads) | - | 10.01 | 22.8 |
| S600 | YOLO26m Detect | 640x640 | 80 | 2.61 ms / 380.65 FPS (1 thread) <br> 7.19 ms / 1612.92 FPS (12 threads) | - | 21.90 | 75.4 |
| S600 | YOLO26l Detect | 640x640 | 80 | 3.21 ms / 309.74 FPS (1 thread) <br> 9.02 ms / 1291.03 FPS (12 threads) | - | 26.30 | 93.8 |
| S600 | YOLO26x Detect | 640x640 | 80 | 6.57 ms / 151.87 FPS (1 thread) <br> 19.20 ms / 609.19 FPS (12 threads) | - | 58.99 | 209.5 |
| S600 | YOLO26n Seg | 640x640 | 80 | 1.02 ms / 966.75 FPS (1 thread) <br> 2.54 ms / 4353.03 FPS (12 threads) | - | 3.13 | 10.5 |
| S600 | YOLO26s Seg | 640x640 | 80 | 1.70 ms / 579.68 FPS (1 thread) <br> 4.38 ms / 2585.38 FPS (12 threads) | - | 11.51 | 37.4 |
| S600 | YOLO26m Seg | 640x640 | 80 | 3.91 ms / 254.03 FPS (1 thread) <br> 10.87 ms / 1069.46 FPS (12 threads) | - | 27.11 | 132.5 |
| S600 | YOLO26l Seg | 640x640 | 80 | 4.58 ms / 217.11 FPS (1 thread) <br> 13.00 ms / 895.80 FPS (12 threads) | - | 31.52 | 150.9 |
| S600 | YOLO26x Seg | 640x640 | 80 | 9.32 ms / 107.13 FPS (1 thread) <br> 26.92 ms / 434.77 FPS (12 threads) | - | 70.69 | 337.7 |
| S600 | YOLO26n Pose | 640x640 | 1 | 0.86 ms / 1129.76 FPS (1 thread) <br> 1.93 ms / 5634.91 FPS (12 threads) | - | 3.68 | 10.3 |
| S600 | YOLO26s Pose | 640x640 | 1 | 1.36 ms / 725.97 FPS (1 thread) <br> 3.33 ms / 3384.84 FPS (12 threads) | - | 11.81 | 29.2 |
| S600 | YOLO26m Pose | 640x640 | 1 | 2.73 ms / 363.75 FPS (1 thread) <br> 7.41 ms / 1550.53 FPS (12 threads) | - | 24.22 | 85.2 |
| S600 | YOLO26l Pose | 640x640 | 1 | 3.31 ms / 299.85 FPS (1 thread) <br> 9.26 ms / 1252.19 FPS (12 threads) | - | 28.63 | 103.6 |
| S600 | YOLO26x Pose | 640x640 | 1 | 6.80 ms / 146.52 FPS (1 thread) <br> 19.60 ms / 596.04 FPS (12 threads) | - | 62.73 | 225.3 |
| S600 | YOLO26n Cls | 224x224 | 1000 | 0.34 ms / 2890.55 FPS (1 thread) <br> 0.68 ms / 15743.07 FPS (12 threads) | - | 2.81 | 0.5 |
| S600 | YOLO26s Cls | 224x224 | 1000 | 0.40 ms / 2459.18 FPS (1 thread) <br> 0.78 ms / 13540.05 FPS (12 threads) | - | 6.72 | 1.6 |
| S600 | YOLO26m Cls | 224x224 | 1000 | 0.54 ms / 1825.63 FPS (1 thread) <br> 1.06 ms / 10526.32 FPS (12 threads) | - | 11.63 | 5.0 |
| S600 | YOLO26l Cls | 224x224 | 1000 | 0.65 ms / 1516.99 FPS (1 thread) <br> 1.43 ms / 7924.24 FPS (12 threads) | - | 14.12 | 6.2 |
| S600 | YOLO26x Cls | 224x224 | 1000 | 0.99 ms / 998.72 FPS (1 thread) <br> 2.57 ms / 4463.59 FPS (12 threads) | - | 29.64 | 13.7 |
| S600 | YOLO26n Obb | 640x640 | 15 | 0.79 ms / 1230.91 FPS (1 thread) <br> 1.65 ms / 6496.46 FPS (12 threads) | - | 2.65 | 6.3 |
| S600 | YOLO26s Obb | 640x640 | 15 | 1.26 ms / 784.60 FPS (1 thread) <br> 3.11 ms / 3610.24 FPS (12 threads) | - | 10.53 | 24.5 |
| S600 | YOLO26m Obb | 640x640 | 15 | 2.66 ms / 372.64 FPS (1 thread) <br> 7.21 ms / 1601.00 FPS (12 threads) | - | 23.49 | 82.2 |
| S600 | YOLO26l Obb | 640x640 | 15 | 3.27 ms / 304.08 FPS (1 thread) <br> 9.15 ms / 1266.96 FPS (12 threads) | - | 27.90 | 100.6 |
| S600 | YOLO26x Obb | 640x640 | 15 | 6.73 ms / 148.08 FPS (1 thread) <br> 19.31 ms / 604.86 FPS (12 threads) | - | 62.66 | 225.3 |



### RDK X5 Performance Data

| Device | Model | Size <br> (Pixels) | Classes | BPU Task Latency / <br> BPU Throughput (Threads) | CPU Latency | params <br> (M) | FLOPs <br> (B) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| X5 | YOLO26n Detect | 640x640 | 80 | 11.6 ms / 86.3 FPS (1 thread) <br> 19.1 ms / 104.3 FPS (2 threads) | - | - | - |
| X5 | YOLO26s Detect | 640x640 | 80 | 20.9 ms / 47.7 FPS (1 thread) <br> 37.8 ms / 52.8 FPS (2 threads) | - | - | - |
| X5 | YOLO26m Detect | 640x640 | 80 | 40.1 ms / 24.8 FPS (1 thread) <br> 76.1 ms / 26.1 FPS (2 threads) | - | - | - |
| X5 | YOLO26l Detect | 640x640 | 80 | 51.1 ms / 19.5 FPS (1 thread) <br> 98.0 ms / 20.3 FPS (2 threads) | - | - | - |
| X5 | YOLO26x Detect | 640x640 | 80 | 103.3 ms / 9.6 FPS (1 thread) <br> 202.0 ms / 9.8 FPS (2 threads) | - | - | - |
| X5 | YOLO26n Seg | 640x640 | 80 | 15.5 ms / 64.3 FPS (1 thread) <br> 22.8 ms / 87.6 FPS (2 threads) | - | - | - |
| X5 | YOLO26n Pose | 640x640 | 80 | 12.5 ms / 79.6 FPS (1 thread) <br> 20.1 ms / 98.7 FPS (2 threads) | - | - | - |
| X5 | YOLO26n Cls | 224x224 | 1000 | 1.1 ms / 906.0 FPS (1 thread) <br> 1.7 ms / 1156.8 FPS (2 threads) | - | - | - |



### 参数说明

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--model-path` | BPU 模型的路径 (.bin) | 必填 |
| `--image-path` | 验证集图片目录 | 对应任务默认路径 |
| `--ann-path` | 官方标注 JSON 文件路径 | 对应任务默认路径 |
| `--json-save-path` | 推理结果保存路径 (.json) | yolo26_xxx_results.json |
| `--conf-thres` | 置信度阈值 (建议设为 0.001 以获得准确 mAP) | 0.001 |
| `--limit` | 限制处理的图片数量 (0 表示全部) | 0 |




## 补充参考数据

各表按发布时的模型、板卡和测量条件列出，不同配置的数据分别保留。模型文件大小沿用表中注明的 MB 单位。

### Obeject Detection

| Device   | Model           | Accuracy bbox-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-----------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S600     | YOLO12n Detect  | 0.338 / 0.320 (94.7 %)                                  | 0.128 / 0.104 (81.3 %)                                    | 0.374 / 0.350 (93.6 %)                                     | 0.524 / 0.514 (98.1 %)                                    |
| S600     | YOLO12s Detect  | 0.403 / 0.385 (95.5 %)                                  | 0.201 / 0.158 (78.6 %)                                    | 0.450 / 0.436 (96.9 %)                                     | 0.602 / 0.586 (97.3 %)                                    |
| S600     | YOLO12m Detect  | 0.452 / 0.436 (96.5 %)                                  | 0.251 / 0.228 (90.8 %)                                    | 0.509 / 0.497 (97.6 %)                                     | 0.638 / 0.626 (98.1 %)                                    |
| S600     | YOLO12l Detect  | 0.463 / 0.441 (95.2 %)                                  | 0.268 / 0.226 (84.3 %)                                    | 0.522 / 0.505 (96.7 %)                                     | 0.646 / 0.640 (99.1 %)                                    |
| S600     | YOLO12x Detect  | 0.475 / 0.455 (95.8 %)                                  | 0.276 / 0.238 (86.2 %)                                    | 0.536 / 0.527 (98.3 %)                                     | 0.659 / 0.618 (93.8 %)                                    |
| S600     | YOLO11s Detect  | 0.400 / 0.382 (95.5 %)                                  | 0.198 / 0.165 (83.3 %)                                    | 0.445 / 0.430 (96.6 %)                                     | 0.587 / 0.586 (99.8 %)                                    |
| S600     | YOLO11m Detect  | 0.444 / 0.429 (96.6 %)                                  | 0.247 / 0.220 (89.1 %)                                    | 0.497 / 0.488 (98.2 %)                                     | 0.627 / 0.610 (97.3 %)                                    |
| S600     | YOLO11l Detect  | 0.460 / 0.448 (97.4 %)                                  | 0.267 / 0.238 (89.1 %)                                    | 0.520 / 0.514 (98.8 %)                                     | 0.638 / 0.622 (97.5 %)                                    |
| S600     | YOLO11x Detect  | 0.474 / 0.457 (96.4 %)                                  | 0.283 / 0.244 (86.2 %)                                    | 0.529 / 0.517 (97.7 %)                                     | 0.652 / 0.639 (98.0 %)                                    |
| S600     | YOLOv10n Detect | 0.303 / 0.285 (94.1 %)                                  | 0.099 / 0.075 (75.8 %)                                    | 0.330 / 0.306 (92.7 %)                                     | 0.478 / 0.469 (98.1 %)                                    |
| S600     | YOLOv10s Detect | 0.386 / 0.363 (94.0 %)                                  | 0.175 / 0.144 (82.3 %)                                    | 0.434 / 0.410 (94.5 %)                                     | 0.574 / 0.528 (92.0 %)                                    |
| S600     | YOLOv10m Detect | 0.425 / 0.389 (91.5 %)                                  | 0.221 / 0.202 (91.4 %)                                    | 0.481 / 0.450 (93.6 %)                                     | 0.603 / 0.505 (83.7 %)                                    |
| S600     | YOLOv10b Detect | 0.443 / 0.388 (87.6 %)                                  | 0.242 / 0.189 (78.1 %)                                    | 0.498 / 0.447 (89.8 %)                                     | 0.618 / 0.496 (80.3 %)                                    |
| S600     | YOLOv10l Detect | 0.445 / 0.392 (88.1 %)                                  | 0.258 / 0.215 (83.3 %)                                    | 0.498 / 0.463 (93.0 %)                                     | 0.626 / 0.492 (78.6 %)                                    |
| S600     | YOLOv10x Detect | 0.459 / 0.424 (92.4 %)                                  | 0.258 / 0.228 (88.4 %)                                    | 0.518 / 0.490 (94.6 %)                                     | 0.639 / 0.546 (85.4 %)                                    |
| S600     | YOLOv9s Detect  | 0.400 / 0.387 (96.8 %)                                  | 0.191 / 0.169 (88.5 %)                                    | 0.444 / 0.433 (97.5 %)                                     | 0.583 / 0.564 (96.7 %)                                    |
| S600     | YOLOv9m Detect  | 0.449 / 0.441 (98.2 %)                                  | 0.253 / 0.232 (91.7 %)                                    | 0.504 / 0.496 (98.4 %)                                     | 0.617 / 0.611 (99.0 %)                                    |
| S600     | YOLOv9c Detect  | 0.461 / 0.451 (97.8 %)                                  | 0.269 / 0.251 (93.3 %)                                    | 0.512 / 0.509 (99.4 %)                                     | 0.640 / 0.612 (95.6 %)                                    |
| S600     | YOLOv9e Detect  | 0.481 / 0.470 (97.7 %)                                  | 0.298 / 0.273 (91.6 %)                                    | 0.538 / 0.525 (97.6 %)                                     | 0.662 / 0.650 (98.2 %)                                    |
| S600     | YOLOv8s Detect  | 0.391 / 0.376 (96.2 %)                                  | 0.195 / 0.169 (86.7 %)                                    | 0.437 / 0.423 (96.8 %)                                     | 0.566 / 0.558 (98.6 %)                                    |
| S600     | YOLOv8m Detect  | 0.441 / 0.429 (97.3 %)                                  | 0.249 / 0.222 (89.2 %)                                    | 0.494 / 0.486 (98.4 %)                                     | 0.618 / 0.607 (98.2 %)                                    |
| S600     | YOLOv8l Detect  | 0.461 / 0.448 (97.2 %)                                  | 0.271 / 0.245 (90.4 %)                                    | 0.516 / 0.505 (97.9 %)                                     | 0.651 / 0.642 (98.6 %)                                    |
| S600     | YOLOv8x Detect  | 0.474 / 0.459 (96.8 %)                                  | 0.280 / 0.253 (90.4 %)                                    | 0.527 / 0.516 (97.9 %)                                     | 0.658 / 0.647 (98.3 %)                                    |
| S600     | YOLOv5nu Detect | 0.278 / 0.269 (96.8 %)                                  | 0.093 / 0.083 (89.2 %)                                    | 0.309 / 0.296 (95.8 %)                                     | 0.417 / 0.412 (98.8 %)                                    |
| S600     | YOLOv5su Detect | 0.367 / 0.355 (96.7 %)                                  | 0.169 / 0.140 (82.8 %)                                    | 0.416 / 0.406 (97.6 %)                                     | 0.530 / 0.531 (100.2 %)                                   |
| S600     | YOLOv5mu Detect | 0.425 / 0.415 (97.6 %)                                  | 0.226 / 0.207 (91.6 %)                                    | 0.477 / 0.470 (98.5 %)                                     | 0.603 / 0.604 (100.2 %)                                   |
| S600     | YOLOv5lu Detect | 0.458 / 0.451 (98.5 %)                                  | 0.260 / 0.238 (91.5 %)                                    | 0.516 / 0.511 (99.0 %)                                     | 0.641 / 0.637 (99.4 %)                                    |
| S600     | YOLOv5xu Detect | 0.466 / 0.454 (97.4 %)                                  | 0.281 / 0.246 (87.5 %)                                    | 0.523 / 0.517 (98.9 %)                                     | 0.645 / 0.647 (100.3 %)                                   |



### Instance Segmentation

| Device   | Model       | Accuracy bbox-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S600     | YOLO11s Seg | 0.394 / 0.383 (97.2 %)                                  | 0.184 / 0.162 (88.0 %)                                    | 0.442 / 0.434 (98.2 %)                                     | 0.582 / 0.581 (99.8 %)                                    |
| S600     | YOLO11m Seg | 0.443 / 0.427 (96.4 %)                                  | 0.246 / 0.224 (91.1 %)                                    | 0.497 / 0.484 (97.4 %)                                     | 0.627 / 0.607 (96.8 %)                                    |
| S600     | YOLO11l Seg | 0.460 / 0.443 (96.3 %)                                  | 0.267 / 0.236 (88.4 %)                                    | 0.520 / 0.508 (97.7 %)                                     | 0.638 / 0.614 (96.2 %)                                    |
| S600     | YOLO11x Seg | 0.474 / 0.456 (96.2 %)                                  | 0.283 / 0.244 (86.2 %)                                    | 0.529 / 0.515 (97.4 %)                                     | 0.652 / 0.636 (97.5 %)                                    |
| S600     | YOLOv8s Seg | 0.386 / 0.375 (97.2 %)                                  | 0.180 / 0.162 (90.0 %)                                    | 0.432 / 0.419 (97.0 %)                                     | 0.564 / 0.560 (99.3 %)                                    |
| S600     | YOLOv8m Seg | 0.431 / 0.420 (97.4 %)                                  | 0.228 / 0.209 (91.7 %)                                    | 0.486 / 0.478 (98.4 %)                                     | 0.608 / 0.605 (99.5 %)                                    |
| S600     | YOLOv8l Seg | 0.453 / 0.435 (96.0 %)                                  | 0.258 / 0.223 (86.4 %)                                    | 0.502 / 0.496 (98.8 %)                                     | 0.626 / 0.611 (97.6 %)                                    |
| S600     | YOLOv8x Seg | 0.465 / 0.454 (97.6 %)                                  | 0.268 / 0.236 (88.1 %)                                    | 0.520 / 0.514 (98.8 %)                                     | 0.641 / 0.633 (98.8 %)                                    |



### Instance Segmentation

| Device   | Model       | Accuracy mask-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S600     | YOLO11s Seg | 0.311 / 0.295 (94.9 %)                                  | 0.099 / 0.088 (88.9 %)                                    | 0.350 / 0.327 (93.4 %)                                     | 0.509 / 0.486 (95.5 %)                                    |
| S600     | YOLO11m Seg | 0.347 / 0.322 (92.8 %)                                  | 0.136 / 0.123 (90.4 %)                                    | 0.396 / 0.362 (91.4 %)                                     | 0.549 / 0.506 (92.2 %)                                    |
| S600     | YOLO11l Seg | 0.357 / 0.331 (92.7 %)                                  | 0.143 / 0.126 (88.1 %)                                    | 0.409 / 0.377 (92.2 %)                                     | 0.560 / 0.511 (91.3 %)                                    |
| S600     | YOLO11x Seg | 0.366 / 0.339 (92.6 %)                                  | 0.149 / 0.124 (83.2 %)                                    | 0.420 / 0.384 (91.4 %)                                     | 0.572 / 0.529 (92.5 %)                                    |
| S600     | YOLOv8s Seg | 0.305 / 0.290 (95.1 %)                                  | 0.096 / 0.085 (88.5 %)                                    | 0.343 / 0.317 (92.4 %)                                     | 0.496 / 0.476 (96.0 %)                                    |
| S600     | YOLOv8m Seg | 0.337 / 0.319 (94.7 %)                                  | 0.121 / 0.109 (90.1 %)                                    | 0.386 / 0.360 (93.3 %)                                     | 0.533 / 0.506 (94.9 %)                                    |
| S600     | YOLOv8l Seg | 0.351 / 0.331 (94.3 %)                                  | 0.137 / 0.119 (86.9 %)                                    | 0.398 / 0.374 (94.0 %)                                     | 0.550 / 0.516 (93.8 %)                                    |
| S600     | YOLOv8x Seg | 0.358 / 0.338 (94.4 %)                                  | 0.139 / 0.122 (87.8 %)                                    | 0.409 / 0.382 (93.4 %)                                     | 0.562 / 0.529 (94.1 %)                                    |



### Pose Estimation

| Device   | Model        | Accuracy pose-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy pose-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy pose-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|--------------|---------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S600     | YOLO11n Pose | 0.465 / 0.448 (96.3 %)                                  | 0.386 / 0.374 (96.9 %)                                     | 0.597 / 0.572 (95.8 %)                                    |
| S600     | YOLO11s Pose | 0.559 / 0.541 (96.8 %)                                  | 0.495 / 0.476 (96.2 %)                                     | 0.672 / 0.648 (96.4 %)                                    |
| S600     | YOLO11m Pose | 0.627 / 0.605 (96.5 %)                                  | 0.586 / 0.566 (96.6 %)                                     | 0.711 / 0.688 (96.8 %)                                    |
| S600     | YOLO11l Pose | 0.636 / 0.617 (97.0 %)                                  | 0.592 / 0.568 (95.9 %)                                     | 0.726 / 0.707 (97.4 %)                                    |
| S600     | YOLO11x Pose | 0.672 / 0.651 (96.9 %)                                  | 0.634 / 0.610 (96.2 %)                                     | 0.750 / 0.731 (97.5 %)                                    |
| S600     | YOLOv8n Pose | 0.476 / 0.461 (96.8 %)                                  | 0.391 / 0.373 (95.4 %)                                     | 0.610 / 0.593 (97.2 %)                                    |
| S600     | YOLOv8s Pose | 0.578 / 0.552 (95.5 %)                                  | 0.510 / 0.486 (95.3 %)                                     | 0.692 / 0.667 (96.4 %)                                    |
| S600     | YOLOv8m Pose | 0.630 / 0.605 (96.0 %)                                  | 0.578 / 0.552 (95.5 %)                                     | 0.724 / 0.703 (97.1 %)                                    |
| S600     | YOLOv8l Pose | 0.657 / 0.633 (96.3 %)                                  | 0.607 / 0.581 (95.7 %)                                     | 0.747 / 0.719 (96.3 %)                                    |
| S600     | YOLOv8x Pose | 0.671 / 0.651 (97.0 %)                                  | 0.624 / 0.603 (96.6 %)                                     | 0.757 / 0.739 (97.6 %)                                    |



### Image Classification

| Device   | Model       | Accuracy TOP1 <br/> FP32 / BPU Python   | Accuracy TOP5 <br/> FP32 / BPU Python   |
|----------|-------------|-----------------------------------------|-----------------------------------------|
| S600     | YOLO11n CLS | 0.700 / 0.556 (79.4 %)                  | 0.894 / 0.794 (88.9 %)                  |
| S600     | YOLO11s CLS | 0.754 / 0.666 (88.3 %)                  | 0.927 / 0.875 (94.3 %)                  |
| S600     | YOLO11m CLS | 0.773 / 0.700 (90.5 %)                  | 0.939 / 0.897 (95.5 %)                  |
| S600     | YOLO11l CLS | 0.783 / 0.718 (91.7 %)                  | 0.942 / 0.908 (96.4 %)                  |
| S600     | YOLO11x CLS | 0.795 / 0.731 (92.0 %)                  | 0.949 / 0.917 (96.6 %)                  |
| S600     | YOLOv8n CLS | 0.689 / 0.571 (82.9 %)                  | 0.883 / 0.803 (90.9 %)                  |
| S600     | YOLOv8s CLS | 0.737 / 0.626 (84.9 %)                  | 0.917 / 0.844 (92.0 %)                  |
| S600     | YOLOv8m CLS | 0.768 / 0.700 (91.1 %)                  | 0.935 / 0.899 (96.1 %)                  |
| S600     | YOLOv8l CLS | 0.783 / 0.723 (92.3 %)                  | 0.942 / 0.909 (96.5 %)                  |
| S600     | YOLOv8x CLS | 0.790 / 0.737 (93.3 %)                  | 0.945 / 0.920 (97.4 %)                  |

### RDK S100P

### Obeject Detection

| Device   | Model           | Accuracy bbox-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-----------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S100P    | YOLO12n Detect  | 0.338 / 0.313 (92.4 %)                                  | 0.128 / 0.095 (74.0 %)                                    | 0.374 / 0.342 (91.4 %)                                     | 0.524 / 0.515 (98.3 %)                                    |
| S100P    | YOLO12s Detect  | 0.403 / 0.380 (94.2 %)                                  | 0.201 / 0.152 (75.5 %)                                    | 0.450 / 0.432 (95.9 %)                                     | 0.602 / 0.581 (96.5 %)                                    |
| S100P    | YOLO12m Detect  | 0.452 / 0.423 (93.7 %)                                  | 0.251 / 0.204 (81.3 %)                                    | 0.509 / 0.489 (96.0 %)                                     | 0.638 / 0.616 (96.5 %)                                    |
| S100P    | YOLO12l Detect  | 0.463 / 0.429 (92.8 %)                                  | 0.268 / 0.211 (78.6 %)                                    | 0.522 / 0.492 (94.3 %)                                     | 0.646 / 0.630 (97.7 %)                                    |
| S100P    | YOLO12x Detect  | 0.475 / 0.440 (92.7 %)                                  | 0.276 / 0.222 (80.3 %)                                    | 0.536 / 0.509 (94.9 %)                                     | 0.659 / 0.627 (95.1 %)                                    |
| S100P    | YOLO11n Detect  | 0.327 / 0.306 (93.9 %)                                  | 0.130 / 0.104 (80.0 %)                                    | 0.357 / 0.340 (95.2 %)                                     | 0.511 / 0.500 (97.8 %)                                    |
| S100P    | YOLO11s Detect  | 0.400 / 0.380 (95.0 %)                                  | 0.198 / 0.166 (83.9 %)                                    | 0.445 / 0.427 (96.1 %)                                     | 0.587 / 0.579 (98.6 %)                                    |
| S100P    | YOLO11m Detect  | 0.444 / 0.417 (94.0 %)                                  | 0.247 / 0.214 (87.0 %)                                    | 0.497 / 0.478 (96.1 %)                                     | 0.627 / 0.599 (95.6 %)                                    |
| S100P    | YOLO11l Detect  | 0.460 / 0.434 (94.5 %)                                  | 0.267 / 0.227 (85.2 %)                                    | 0.520 / 0.498 (95.9 %)                                     | 0.638 / 0.611 (95.8 %)                                    |
| S100P    | YOLO11x Detect  | 0.474 / 0.446 (94.0 %)                                  | 0.283 / 0.240 (84.7 %)                                    | 0.529 / 0.506 (95.6 %)                                     | 0.652 / 0.627 (96.1 %)                                    |
| S100P    | YOLOv10n Detect | 0.303 / 0.276 (91.3 %)                                  | 0.099 / 0.064 (64.7 %)                                    | 0.330 / 0.302 (91.5 %)                                     | 0.478 / 0.460 (96.2 %)                                    |
| S100P    | YOLOv10s Detect | 0.386 / 0.354 (91.6 %)                                  | 0.175 / 0.126 (72.2 %)                                    | 0.434 / 0.402 (92.5 %)                                     | 0.574 / 0.527 (91.7 %)                                    |
| S100P    | YOLOv10m Detect | 0.425 / 0.368 (86.7 %)                                  | 0.221 / 0.179 (80.9 %)                                    | 0.481 / 0.431 (89.6 %)                                     | 0.603 / 0.472 (78.2 %)                                    |
| S100P    | YOLOv10b Detect | 0.443 / 0.382 (86.3 %)                                  | 0.242 / 0.194 (80.2 %)                                    | 0.498 / 0.437 (87.7 %)                                     | 0.618 / 0.480 (77.7 %)                                    |
| S100P    | YOLOv10l Detect | 0.445 / 0.372 (83.6 %)                                  | 0.258 / 0.202 (78.5 %)                                    | 0.498 / 0.435 (87.4 %)                                     | 0.626 / 0.463 (74.0 %)                                    |
| S100P    | YOLOv10x Detect | 0.459 / 0.409 (89.2 %)                                  | 0.258 / 0.212 (82.1 %)                                    | 0.518 / 0.475 (91.8 %)                                     | 0.639 / 0.535 (83.6 %)                                    |
| S100P    | YOLOv9t Detect  | 0.313 / 0.301 (96.2 %)                                  | 0.113 / 0.105 (93.6 %)                                    | 0.338 / 0.325 (96.3 %)                                     | 0.483 / 0.461 (95.5 %)                                    |
| S100P    | YOLOv9s Detect  | 0.400 / 0.383 (95.8 %)                                  | 0.191 / 0.165 (86.3 %)                                    | 0.444 / 0.431 (97.0 %)                                     | 0.583 / 0.560 (96.1 %)                                    |
| S100P    | YOLOv9m Detect  | 0.449 / 0.432 (96.1 %)                                  | 0.253 / 0.231 (91.2 %)                                    | 0.504 / 0.487 (96.5 %)                                     | 0.617 / 0.602 (97.5 %)                                    |
| S100P    | YOLOv9c Detect  | 0.461 / 0.446 (96.8 %)                                  | 0.269 / 0.250 (93.2 %)                                    | 0.512 / 0.499 (97.4 %)                                     | 0.640 / 0.618 (96.6 %)                                    |
| S100P    | YOLOv9e Detect  | 0.481 / 0.465 (96.6 %)                                  | 0.298 / 0.270 (90.9 %)                                    | 0.538 / 0.520 (96.7 %)                                     | 0.662 / 0.647 (97.7 %)                                    |
| S100P    | YOLOv8n Detect  | 0.309 / 0.292 (94.4 %)                                  | 0.113 / 0.098 (87.2 %)                                    | 0.338 / 0.321 (94.9 %)                                     | 0.473 / 0.457 (96.7 %)                                    |
| S100P    | YOLOv8s Detect  | 0.391 / 0.373 (95.4 %)                                  | 0.195 / 0.166 (85.1 %)                                    | 0.437 / 0.425 (97.3 %)                                     | 0.566 / 0.558 (98.6 %)                                    |
| S100P    | YOLOv8m Detect  | 0.441 / 0.420 (95.4 %)                                  | 0.249 / 0.213 (85.6 %)                                    | 0.494 / 0.478 (96.7 %)                                     | 0.618 / 0.612 (99.1 %)                                    |
| S100P    | YOLOv8l Detect  | 0.461 / 0.442 (95.8 %)                                  | 0.271 / 0.241 (88.9 %)                                    | 0.516 / 0.499 (96.6 %)                                     | 0.651 / 0.628 (96.4 %)                                    |
| S100P    | YOLOv8x Detect  | 0.474 / 0.448 (94.6 %)                                  | 0.280 / 0.245 (87.6 %)                                    | 0.527 / 0.504 (95.7 %)                                     | 0.658 / 0.640 (97.2 %)                                    |
| S100P    | YOLOv5nu Detect | 0.278 / 0.264 (94.7 %)                                  | 0.093 / 0.080 (85.5 %)                                    | 0.309 / 0.293 (94.8 %)                                     | 0.417 / 0.406 (97.5 %)                                    |
| S100P    | YOLOv5su Detect | 0.367 / 0.349 (95.2 %)                                  | 0.169 / 0.141 (83.3 %)                                    | 0.416 / 0.398 (95.8 %)                                     | 0.530 / 0.524 (98.9 %)                                    |
| S100P    | YOLOv5mu Detect | 0.425 / 0.406 (95.6 %)                                  | 0.226 / 0.194 (86.0 %)                                    | 0.477 / 0.467 (98.0 %)                                     | 0.603 / 0.592 (98.2 %)                                    |
| S100P    | YOLOv5lu Detect | 0.458 / 0.436 (95.1 %)                                  | 0.260 / 0.215 (82.9 %)                                    | 0.516 / 0.500 (96.8 %)                                     | 0.641 / 0.631 (98.4 %)                                    |
| S100P    | YOLOv5xu Detect | 0.466 / 0.445 (95.5 %)                                  | 0.281 / 0.239 (85.0 %)                                    | 0.523 / 0.506 (96.7 %)                                     | 0.645 / 0.638 (99.0 %)                                    |

#### Instance Segmentation

### Instance Segmentation

| Device   | Model       | Accuracy bbox-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S100P    | YOLO11n Seg | 0.322 / 0.294 (91.4 %)                                  | 0.113 / 0.081 (71.8 %)                                    | 0.352 / 0.324 (92.1 %)                                     | 0.502 / 0.490 (97.6 %)                                    |
| S100P    | YOLO11s Seg | 0.394 / 0.372 (94.4 %)                                  | 0.184 / 0.149 (81.2 %)                                    | 0.442 / 0.424 (96.0 %)                                     | 0.582 / 0.577 (99.1 %)                                    |
| S100P    | YOLO11m Seg | 0.443 / 0.414 (93.3 %)                                  | 0.246 / 0.208 (84.3 %)                                    | 0.497 / 0.473 (95.2 %)                                     | 0.627 / 0.599 (95.6 %)                                    |
| S100P    | YOLO11l Seg | 0.460 / 0.430 (93.5 %)                                  | 0.267 / 0.220 (82.5 %)                                    | 0.520 / 0.493 (94.9 %)                                     | 0.638 / 0.610 (95.6 %)                                    |
| S100P    | YOLO11x Seg | 0.474 / 0.441 (93.0 %)                                  | 0.283 / 0.231 (81.7 %)                                    | 0.529 / 0.501 (94.6 %)                                     | 0.652 / 0.625 (95.8 %)                                    |
| S100P    | YOLOv9c Seg | 0.453 / 0.422 (93.0 %)                                  | 0.254 / 0.206 (81.2 %)                                    | 0.508 / 0.483 (94.9 %)                                     | 0.621 / 0.601 (96.8 %)                                    |
| S100P    | YOLOv9e Seg | 0.481 / 0.450 (93.6 %)                                  | 0.292 / 0.245 (83.9 %)                                    | 0.537 / 0.507 (94.3 %)                                     | 0.650 / 0.628 (96.6 %)                                    |
| S100P    | YOLOv8n Seg | 0.304 / 0.282 (92.9 %)                                  | 0.109 / 0.087 (79.7 %)                                    | 0.334 / 0.310 (92.8 %)                                     | 0.461 / 0.440 (95.4 %)                                    |
| S100P    | YOLOv8s Seg | 0.386 / 0.363 (94.0 %)                                  | 0.180 / 0.149 (82.8 %)                                    | 0.432 / 0.410 (94.8 %)                                     | 0.564 / 0.547 (97.0 %)                                    |
| S100P    | YOLOv8m Seg | 0.431 / 0.407 (94.3 %)                                  | 0.228 / 0.191 (83.9 %)                                    | 0.486 / 0.467 (96.0 %)                                     | 0.608 / 0.596 (98.0 %)                                    |
| S100P    | YOLOv8l Seg | 0.453 / 0.426 (94.1 %)                                  | 0.258 / 0.220 (85.0 %)                                    | 0.502 / 0.483 (96.3 %)                                     | 0.626 / 0.607 (97.0 %)                                    |
| S100P    | YOLOv8x Seg | 0.465 / 0.435 (93.5 %)                                  | 0.268 / 0.214 (79.7 %)                                    | 0.520 / 0.496 (95.2 %)                                     | 0.641 / 0.622 (97.1 %)                                    |

##### mask

### Instance Segmentation

| Device   | Model       | Accuracy mask-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S100P    | YOLO11n Seg | 0.262 / 0.226 (86.3 %)                                  | 0.062 / 0.044 (72.0 %)                                    | 0.283 / 0.250 (88.2 %)                                     | 0.444 / 0.394 (88.8 %)                                    |
| S100P    | YOLO11s Seg | 0.311 / 0.287 (92.2 %)                                  | 0.099 / 0.088 (88.9 %)                                    | 0.350 / 0.326 (93.3 %)                                     | 0.509 / 0.474 (93.2 %)                                    |
| S100P    | YOLO11m Seg | 0.347 / 0.315 (90.7 %)                                  | 0.136 / 0.122 (90.3 %)                                    | 0.396 / 0.362 (91.4 %)                                     | 0.549 / 0.493 (89.8 %)                                    |
| S100P    | YOLO11l Seg | 0.357 / 0.325 (91.1 %)                                  | 0.143 / 0.126 (88.1 %)                                    | 0.409 / 0.374 (91.4 %)                                     | 0.560 / 0.504 (90.1 %)                                    |
| S100P    | YOLO11x Seg | 0.366 / 0.331 (90.4 %)                                  | 0.149 / 0.129 (86.9 %)                                    | 0.420 / 0.379 (90.2 %)                                     | 0.572 / 0.520 (90.9 %)                                    |
| S100P    | YOLOv9c Seg | 0.352 / 0.319 (90.7 %)                                  | 0.132 / 0.116 (88.1 %)                                    | 0.404 / 0.367 (90.8 %)                                     | 0.547 / 0.497 (91.0 %)                                    |
| S100P    | YOLOv9e Seg | 0.371 / 0.340 (91.7 %)                                  | 0.155 / 0.136 (87.8 %)                                    | 0.425 / 0.386 (90.7 %)                                     | 0.571 / 0.525 (92.0 %)                                    |
| S100P    | YOLOv8n Seg | 0.246 / 0.221 (89.9 %)                                  | 0.059 / 0.048 (81.8 %)                                    | 0.265 / 0.243 (91.8 %)                                     | 0.409 / 0.364 (89.0 %)                                    |
| S100P    | YOLOv8s Seg | 0.305 / 0.282 (92.6 %)                                  | 0.096 / 0.086 (90.2 %)                                    | 0.343 / 0.316 (92.1 %)                                     | 0.496 / 0.457 (92.1 %)                                    |
| S100P    | YOLOv8m Seg | 0.337 / 0.312 (92.7 %)                                  | 0.121 / 0.110 (90.8 %)                                    | 0.386 / 0.358 (92.8 %)                                     | 0.533 / 0.494 (92.5 %)                                    |
| S100P    | YOLOv8l Seg | 0.351 / 0.326 (92.9 %)                                  | 0.137 / 0.126 (92.1 %)                                    | 0.398 / 0.371 (93.4 %)                                     | 0.550 / 0.509 (92.5 %)                                    |
| S100P    | YOLOv8x Seg | 0.358 / 0.331 (92.3 %)                                  | 0.139 / 0.119 (85.5 %)                                    | 0.409 / 0.379 (92.5 %)                                     | 0.562 / 0.514 (91.4 %)                                    |

#### Pose Estimation

### Pose Estimation

| Device   | Model        | Accuracy pose-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy pose-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy pose-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|--------------|---------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S100P    | YOLO11n Pose | 0.465 / 0.445 (95.7 %)                                  | 0.386 / 0.373 (96.5 %)                                     | 0.597 / 0.568 (95.1 %)                                    |
| S100P    | YOLO11s Pose | 0.559 / 0.533 (95.4 %)                                  | 0.495 / 0.467 (94.5 %)                                     | 0.672 / 0.649 (96.6 %)                                    |
| S100P    | YOLO11m Pose | 0.627 / 0.607 (96.8 %)                                  | 0.586 / 0.563 (96.2 %)                                     | 0.711 / 0.692 (97.3 %)                                    |
| S100P    | YOLO11l Pose | 0.636 / 0.617 (97.0 %)                                  | 0.592 / 0.570 (96.3 %)                                     | 0.726 / 0.704 (97.0 %)                                    |
| S100P    | YOLO11x Pose | 0.672 / 0.648 (96.5 %)                                  | 0.634 / 0.605 (95.5 %)                                     | 0.750 / 0.733 (97.8 %)                                    |
| S100P    | YOLOv8n Pose | 0.476 / 0.460 (96.7 %)                                  | 0.391 / 0.372 (95.0 %)                                     | 0.610 / 0.593 (97.2 %)                                    |
| S100P    | YOLOv8s Pose | 0.578 / 0.550 (95.2 %)                                  | 0.510 / 0.476 (93.4 %)                                     | 0.692 / 0.667 (96.4 %)                                    |
| S100P    | YOLOv8m Pose | 0.630 / 0.605 (96.0 %)                                  | 0.578 / 0.553 (95.7 %)                                     | 0.724 / 0.697 (96.3 %)                                    |
| S100P    | YOLOv8l Pose | 0.657 / 0.631 (96.1 %)                                  | 0.607 / 0.579 (95.3 %)                                     | 0.747 / 0.726 (97.2 %)                                    |
| S100P    | YOLOv8x Pose | 0.671 / 0.649 (96.7 %)                                  | 0.624 / 0.602 (96.4 %)                                     | 0.757 / 0.733 (96.8 %)                                    |

#### Image Classification

### Image Classification

| Device   | Model       | Accuracy TOP1 <br/> FP32 / BPU Python   | Accuracy TOP5 <br/> FP32 / BPU Python   |
|----------|-------------|-----------------------------------------|-----------------------------------------|
| S100P    | YOLO11n CLS | 0.700 / 0.566 (80.8 %)                  | 0.894 / 0.803 (89.8 %)                  |
| S100P    | YOLO11s CLS | 0.754 / 0.661 (87.7 %)                  | 0.927 / 0.872 (94.1 %)                  |
| S100P    | YOLO11m CLS | 0.773 / 0.706 (91.3 %)                  | 0.939 / 0.903 (96.1 %)                  |
| S100P    | YOLO11l CLS | 0.783 / 0.712 (90.8 %)                  | 0.942 / 0.905 (96.1 %)                  |
| S100P    | YOLO11x CLS | 0.795 / 0.734 (92.4 %)                  | 0.949 / 0.919 (96.8 %)                  |
| S100P    | YOLOv8n CLS | 0.689 / 0.577 (83.7 %)                  | 0.883 / 0.808 (91.5 %)                  |
| S100P    | YOLOv8s CLS | 0.737 / 0.631 (85.6 %)                  | 0.917 / 0.850 (92.8 %)                  |
| S100P    | YOLOv8m CLS | 0.768 / 0.703 (91.6 %)                  | 0.935 / 0.899 (96.2 %)                  |
| S100P    | YOLOv8l CLS | 0.783 / 0.723 (92.3 %)                  | 0.942 / 0.910 (96.6 %)                  |
| S100P    | YOLOv8x CLS | 0.790 / 0.742 (93.9 %)                  | 0.945 / 0.923 (97.6 %)                  |



### Obeject Detection

| Device   | Model           | Accuracy bbox-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-----------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S100     | YOLO12n Detect  | 0.338 / 0.311 (92.0 %)                                  | 0.128 / 0.096 (74.9 %)                                    | 0.374 / 0.344 (91.8 %)                                     | 0.524 / 0.507 (96.6 %)                                    |
| S100     | YOLO12s Detect  | 0.403 / 0.380 (94.3 %)                                  | 0.201 / 0.156 (77.4 %)                                    | 0.450 / 0.431 (95.9 %)                                     | 0.602 / 0.573 (95.1 %)                                    |
| S100     | YOLO12m Detect  | 0.452 / 0.424 (93.8 %)                                  | 0.251 / 0.211 (84.2 %)                                    | 0.509 / 0.488 (95.9 %)                                     | 0.638 / 0.609 (95.4 %)                                    |
| S100     | YOLO12l Detect  | 0.463 / 0.431 (93.1 %)                                  | 0.268 / 0.220 (82.0 %)                                    | 0.522 / 0.494 (94.7 %)                                     | 0.646 / 0.629 (97.5 %)                                    |
| S100     | YOLO12x Detect  | 0.475 / 0.441 (92.8 %)                                  | 0.276 / 0.215 (78.0 %)                                    | 0.536 / 0.512 (95.5 %)                                     | 0.659 / 0.619 (94.0 %)                                    |
| S100     | YOLO11n Detect  | 0.327 / 0.309 (94.5 %)                                  | 0.130 / 0.108 (83.2 %)                                    | 0.357 / 0.338 (94.7 %)                                     | 0.511 / 0.497 (97.4 %)                                    |
| S100     | YOLO11s Detect  | 0.400 / 0.380 (95.2 %)                                  | 0.198 / 0.167 (84.5 %)                                    | 0.445 / 0.426 (95.9 %)                                     | 0.587 / 0.575 (97.9 %)                                    |
| S100     | YOLO11m Detect  | 0.444 / 0.417 (94.1 %)                                  | 0.247 / 0.211 (85.7 %)                                    | 0.497 / 0.479 (96.3 %)                                     | 0.627 / 0.590 (94.1 %)                                    |
| S100     | YOLO11l Detect  | 0.460 / 0.433 (94.1 %)                                  | 0.267 / 0.226 (84.9 %)                                    | 0.520 / 0.495 (95.3 %)                                     | 0.638 / 0.605 (94.9 %)                                    |
| S100     | YOLO11x Detect  | 0.474 / 0.445 (93.7 %)                                  | 0.283 / 0.231 (81.4 %)                                    | 0.529 / 0.506 (95.5 %)                                     | 0.652 / 0.623 (95.5 %)                                    |
| S100     | YOLOv10n Detect | 0.303 / 0.278 (91.7 %)                                  | 0.099 / 0.068 (68.6 %)                                    | 0.330 / 0.304 (92.1 %)                                     | 0.478 / 0.455 (95.3 %)                                    |
| S100     | YOLOv10s Detect | 0.386 / 0.354 (91.7 %)                                  | 0.175 / 0.122 (69.7 %)                                    | 0.434 / 0.405 (93.3 %)                                     | 0.574 / 0.529 (92.2 %)                                    |
| S100     | YOLOv10m Detect | 0.425 / 0.374 (88.0 %)                                  | 0.221 / 0.179 (81.0 %)                                    | 0.481 / 0.439 (91.3 %)                                     | 0.603 / 0.490 (81.3 %)                                    |
| S100     | YOLOv10b Detect | 0.443 / 0.380 (85.7 %)                                  | 0.242 / 0.193 (79.7 %)                                    | 0.498 / 0.434 (87.1 %)                                     | 0.618 / 0.469 (75.9 %)                                    |
| S100     | YOLOv10l Detect | 0.445 / 0.380 (85.4 %)                                  | 0.258 / 0.209 (81.3 %)                                    | 0.498 / 0.444 (89.1 %)                                     | 0.626 / 0.476 (76.0 %)                                    |
| S100     | YOLOv10x Detect | 0.459 / 0.413 (90.0 %)                                  | 0.258 / 0.214 (82.9 %)                                    | 0.518 / 0.480 (92.7 %)                                     | 0.639 / 0.539 (84.3 %)                                    |
| S100     | YOLOv9t Detect  | 0.313 / 0.300 (95.8 %)                                  | 0.113 / 0.105 (93.7 %)                                    | 0.338 / 0.325 (96.1 %)                                     | 0.483 / 0.458 (94.9 %)                                    |
| S100     | YOLOv9s Detect  | 0.400 / 0.383 (95.9 %)                                  | 0.191 / 0.160 (84.0 %)                                    | 0.444 / 0.435 (97.9 %)                                     | 0.583 / 0.556 (95.4 %)                                    |
| S100     | YOLOv9m Detect  | 0.449 / 0.434 (96.6 %)                                  | 0.253 / 0.228 (90.3 %)                                    | 0.504 / 0.492 (97.6 %)                                     | 0.617 / 0.593 (96.1 %)                                    |
| S100     | YOLOv9c Detect  | 0.461 / 0.445 (96.5 %)                                  | 0.269 / 0.246 (91.6 %)                                    | 0.512 / 0.500 (97.6 %)                                     | 0.640 / 0.610 (95.2 %)                                    |
| S100     | YOLOv9e Detect  | 0.481 / 0.460 (95.7 %)                                  | 0.298 / 0.266 (89.3 %)                                    | 0.538 / 0.516 (95.9 %)                                     | 0.662 / 0.626 (94.5 %)                                    |
| S100     | YOLOv8n Detect  | 0.309 / 0.291 (94.3 %)                                  | 0.113 / 0.101 (89.3 %)                                    | 0.338 / 0.320 (94.8 %)                                     | 0.473 / 0.448 (94.7 %)                                    |
| S100     | YOLOv8s Detect  | 0.391 / 0.373 (95.5 %)                                  | 0.195 / 0.168 (86.2 %)                                    | 0.437 / 0.421 (96.4 %)                                     | 0.566 / 0.556 (98.3 %)                                    |
| S100     | YOLOv8m Detect  | 0.441 / 0.419 (95.2 %)                                  | 0.249 / 0.213 (85.7 %)                                    | 0.494 / 0.477 (96.5 %)                                     | 0.618 / 0.602 (97.4 %)                                    |
| S100     | YOLOv8l Detect  | 0.461 / 0.441 (95.6 %)                                  | 0.271 / 0.241 (88.9 %)                                    | 0.516 / 0.499 (96.6 %)                                     | 0.651 / 0.625 (96.0 %)                                    |
| S100     | YOLOv8x Detect  | 0.474 / 0.449 (94.7 %)                                  | 0.280 / 0.250 (89.2 %)                                    | 0.527 / 0.505 (95.8 %)                                     | 0.658 / 0.628 (95.5 %)                                    |
| S100     | YOLOv5nu Detect | 0.278 / 0.261 (93.6 %)                                  | 0.093 / 0.081 (86.6 %)                                    | 0.309 / 0.287 (93.0 %)                                     | 0.417 / 0.400 (96.0 %)                                    |
| S100     | YOLOv5su Detect | 0.367 / 0.352 (95.9 %)                                  | 0.169 / 0.144 (85.3 %)                                    | 0.416 / 0.402 (96.7 %)                                     | 0.530 / 0.521 (98.3 %)                                    |
| S100     | YOLOv5mu Detect | 0.425 / 0.406 (95.6 %)                                  | 0.226 / 0.195 (86.4 %)                                    | 0.477 / 0.465 (97.6 %)                                     | 0.603 / 0.594 (98.4 %)                                    |
| S100     | YOLOv5lu Detect | 0.458 / 0.437 (95.4 %)                                  | 0.260 / 0.226 (86.9 %)                                    | 0.516 / 0.499 (96.7 %)                                     | 0.641 / 0.628 (97.9 %)                                    |
| S100     | YOLOv5xu Detect | 0.466 / 0.445 (95.6 %)                                  | 0.281 / 0.238 (84.8 %)                                    | 0.523 / 0.506 (96.9 %)                                     | 0.645 / 0.634 (98.3 %)                                    |

#### Instance Segmentation

### Instance Segmentation

| Device   | Model       | Accuracy bbox-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S100     | YOLO11n Seg | 0.322 / 0.295 (91.6 %)                                  | 0.113 / 0.084 (74.2 %)                                    | 0.352 / 0.322 (91.3 %)                                     | 0.502 / 0.487 (97.0 %)                                    |
| S100     | YOLO11s Seg | 0.394 / 0.369 (93.7 %)                                  | 0.184 / 0.149 (81.0 %)                                    | 0.442 / 0.419 (94.9 %)                                     | 0.582 / 0.571 (98.1 %)                                    |
| S100     | YOLO11m Seg | 0.443 / 0.414 (93.2 %)                                  | 0.246 / 0.206 (83.5 %)                                    | 0.497 / 0.473 (95.3 %)                                     | 0.627 / 0.590 (94.2 %)                                    |
| S100     | YOLO11l Seg | 0.460 / 0.428 (93.1 %)                                  | 0.267 / 0.217 (81.5 %)                                    | 0.520 / 0.490 (94.3 %)                                     | 0.638 / 0.604 (94.8 %)                                    |
| S100     | YOLO11x Seg | 0.474 / 0.440 (92.8 %)                                  | 0.283 / 0.223 (78.7 %)                                    | 0.529 / 0.501 (94.6 %)                                     | 0.652 / 0.621 (95.2 %)                                    |
| S100     | YOLOv9c Seg | 0.453 / 0.420 (92.6 %)                                  | 0.254 / 0.206 (81.2 %)                                    | 0.508 / 0.479 (94.2 %)                                     | 0.621 / 0.584 (93.9 %)                                    |
| S100     | YOLOv9e Seg | 0.481 / 0.449 (93.5 %)                                  | 0.292 / 0.246 (84.1 %)                                    | 0.537 / 0.506 (94.3 %)                                     | 0.650 / 0.620 (95.4 %)                                    |
| S100     | YOLOv8n Seg | 0.304 / 0.283 (93.1 %)                                  | 0.109 / 0.088 (80.3 %)                                    | 0.334 / 0.310 (92.8 %)                                     | 0.461 / 0.441 (95.5 %)                                    |
| S100     | YOLOv8s Seg | 0.386 / 0.363 (94.1 %)                                  | 0.180 / 0.153 (85.2 %)                                    | 0.432 / 0.405 (93.7 %)                                     | 0.564 / 0.550 (97.5 %)                                    |
| S100     | YOLOv8m Seg | 0.431 / 0.407 (94.4 %)                                  | 0.228 / 0.193 (84.7 %)                                    | 0.486 / 0.468 (96.2 %)                                     | 0.608 / 0.591 (97.2 %)                                    |
| S100     | YOLOv8l Seg | 0.453 / 0.425 (93.9 %)                                  | 0.258 / 0.214 (83.0 %)                                    | 0.502 / 0.484 (96.4 %)                                     | 0.626 / 0.592 (94.6 %)                                    |
| S100     | YOLOv8x Seg | 0.465 / 0.434 (93.4 %)                                  | 0.268 / 0.216 (80.6 %)                                    | 0.520 / 0.494 (95.0 %)                                     | 0.641 / 0.613 (95.6 %)                                    |

##### mask

### Instance Segmentation

| Device   | Model       | Accuracy mask-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S100     | YOLO11n Seg | 0.262 / 0.227 (86.7 %)                                  | 0.062 / 0.046 (75.3 %)                                    | 0.283 / 0.249 (88.0 %)                                     | 0.444 / 0.392 (88.3 %)                                    |
| S100     | YOLO11s Seg | 0.311 / 0.285 (91.7 %)                                  | 0.099 / 0.088 (89.3 %)                                    | 0.350 / 0.322 (91.9 %)                                     | 0.509 / 0.470 (92.3 %)                                    |
| S100     | YOLO11m Seg | 0.347 / 0.313 (90.3 %)                                  | 0.136 / 0.121 (89.0 %)                                    | 0.396 / 0.361 (91.2 %)                                     | 0.549 / 0.482 (87.8 %)                                    |
| S100     | YOLO11l Seg | 0.357 / 0.324 (90.7 %)                                  | 0.143 / 0.124 (86.9 %)                                    | 0.409 / 0.372 (90.8 %)                                     | 0.560 / 0.499 (89.1 %)                                    |
| S100     | YOLO11x Seg | 0.366 / 0.332 (90.6 %)                                  | 0.149 / 0.124 (83.2 %)                                    | 0.420 / 0.381 (90.8 %)                                     | 0.572 / 0.516 (90.3 %)                                    |
| S100     | YOLOv9c Seg | 0.352 / 0.317 (90.1 %)                                  | 0.132 / 0.116 (87.9 %)                                    | 0.404 / 0.366 (90.5 %)                                     | 0.547 / 0.485 (88.6 %)                                    |
| S100     | YOLOv9e Seg | 0.371 / 0.340 (91.6 %)                                  | 0.155 / 0.137 (88.2 %)                                    | 0.425 / 0.386 (90.8 %)                                     | 0.571 / 0.521 (91.3 %)                                    |
| S100     | YOLOv8n Seg | 0.246 / 0.220 (89.8 %)                                  | 0.059 / 0.049 (83.0 %)                                    | 0.265 / 0.242 (91.5 %)                                     | 0.409 / 0.365 (89.3 %)                                    |
| S100     | YOLOv8s Seg | 0.305 / 0.281 (92.3 %)                                  | 0.096 / 0.088 (92.3 %)                                    | 0.343 / 0.313 (91.2 %)                                     | 0.496 / 0.459 (92.5 %)                                    |
| S100     | YOLOv8m Seg | 0.337 / 0.311 (92.2 %)                                  | 0.121 / 0.112 (92.1 %)                                    | 0.386 / 0.358 (92.8 %)                                     | 0.533 / 0.484 (90.7 %)                                    |
| S100     | YOLOv8l Seg | 0.351 / 0.326 (92.8 %)                                  | 0.137 / 0.124 (90.8 %)                                    | 0.398 / 0.372 (93.4 %)                                     | 0.550 / 0.495 (90.1 %)                                    |
| S100     | YOLOv8x Seg | 0.358 / 0.330 (92.0 %)                                  | 0.139 / 0.120 (86.6 %)                                    | 0.409 / 0.377 (92.0 %)                                     | 0.562 / 0.508 (90.4 %)                                    |

#### Pose Estimation

### Pose Estimation

| Device   | Model        | Accuracy pose-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy pose-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy pose-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|--------------|---------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S100     | YOLO11n Pose | 0.465 / 0.451 (97.0 %)                                  | 0.386 / 0.375 (97.2 %)                                     | 0.597 / 0.576 (96.5 %)                                    |
| S100     | YOLO11s Pose | 0.559 / 0.531 (95.0 %)                                  | 0.495 / 0.465 (94.0 %)                                     | 0.672 / 0.647 (96.3 %)                                    |
| S100     | YOLO11m Pose | 0.627 / 0.601 (95.9 %)                                  | 0.586 / 0.559 (95.4 %)                                     | 0.711 / 0.690 (97.0 %)                                    |
| S100     | YOLO11l Pose | 0.636 / 0.615 (96.6 %)                                  | 0.592 / 0.571 (96.6 %)                                     | 0.726 / 0.698 (96.1 %)                                    |
| S100     | YOLO11x Pose | 0.672 / 0.651 (96.8 %)                                  | 0.634 / 0.607 (95.8 %)                                     | 0.750 / 0.734 (97.9 %)                                    |
| S100     | YOLOv8n Pose | 0.476 / 0.461 (96.9 %)                                  | 0.391 / 0.372 (95.1 %)                                     | 0.610 / 0.595 (97.6 %)                                    |
| S100     | YOLOv8s Pose | 0.578 / 0.548 (94.9 %)                                  | 0.510 / 0.475 (93.3 %)                                     | 0.692 / 0.667 (96.4 %)                                    |
| S100     | YOLOv8m Pose | 0.630 / 0.604 (95.9 %)                                  | 0.578 / 0.551 (95.4 %)                                     | 0.724 / 0.699 (96.5 %)                                    |
| S100     | YOLOv8l Pose | 0.657 / 0.632 (96.3 %)                                  | 0.607 / 0.578 (95.2 %)                                     | 0.747 / 0.728 (97.5 %)                                    |
| S100     | YOLOv8x Pose | 0.671 / 0.649 (96.7 %)                                  | 0.624 / 0.596 (95.5 %)                                     | 0.757 / 0.739 (97.6 %)                                    |

#### Image Classification

### Image Classification

| Device   | Model       | Accuracy TOP1 <br/> FP32 / BPU Python   | Accuracy TOP5 <br/> FP32 / BPU Python   |
|----------|-------------|-----------------------------------------|-----------------------------------------|
| S100     | YOLO11n CLS | 0.700 / 0.590 (84.3 %)                  | 0.894 / 0.820 (91.7 %)                  |
| S100     | YOLO11s CLS | 0.754 / 0.667 (88.6 %)                  | 0.927 / 0.875 (94.5 %)                  |
| S100     | YOLO11m CLS | 0.773 / 0.706 (91.3 %)                  | 0.939 / 0.902 (96.1 %)                  |
| S100     | YOLO11l CLS | 0.783 / 0.712 (90.9 %)                  | 0.942 / 0.906 (96.1 %)                  |
| S100     | YOLO11x CLS | 0.795 / 0.733 (92.2 %)                  | 0.949 / 0.918 (96.7 %)                  |
| S100     | YOLOv8n CLS | 0.689 / 0.570 (82.7 %)                  | 0.883 / 0.802 (90.8 %)                  |
| S100     | YOLOv8s CLS | 0.737 / 0.636 (86.3 %)                  | 0.917 / 0.852 (92.9 %)                  |
| S100     | YOLOv8m CLS | 0.768 / 0.702 (91.4 %)                  | 0.935 / 0.899 (96.2 %)                  |
| S100     | YOLOv8l CLS | 0.783 / 0.723 (92.3 %)                  | 0.942 / 0.909 (96.5 %)                  |
| S100     | YOLOv8x CLS | 0.790 / 0.742 (93.9 %)                  | 0.945 / 0.921 (97.5 %)                  |



### Obeject Detection

| Device   | Model           | Accuracy bbox-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-----------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| X5       | YOLO12n Detect  | 0.338 / 0.313 (92.5 %)                                  | 0.128 / 0.095 (74.3 %)                                    | 0.374 / 0.343 (91.7 %)                                     | 0.524 / 0.511 (97.4 %)                                    |
| X5       | YOLO12s Detect  | 0.403 / 0.379 (94.0 %)                                  | 0.201 / 0.157 (78.1 %)                                    | 0.450 / 0.427 (95.0 %)                                     | 0.602 / 0.575 (95.5 %)                                    |
| X5       | YOLO12m Detect  | 0.452 / 0.424 (93.8 %)                                  | 0.251 / 0.208 (82.7 %)                                    | 0.509 / 0.489 (96.1 %)                                     | 0.638 / 0.617 (96.7 %)                                    |
| X5       | YOLO12l Detect  | 0.463 / 0.434 (93.8 %)                                  | 0.268 / 0.212 (78.9 %)                                    | 0.522 / 0.499 (95.6 %)                                     | 0.646 / 0.630 (97.6 %)                                    |
| X5       | YOLO12x Detect  | 0.475 / 0.443 (93.3 %)                                  | 0.276 / 0.227 (82.3 %)                                    | 0.536 / 0.513 (95.7 %)                                     | 0.659 / 0.632 (95.9 %)                                    |
| X5       | YOLO11n Detect  | 0.327 / 0.310 (95.1 %)                                  | 0.130 / 0.110 (84.8 %)                                    | 0.357 / 0.341 (95.4 %)                                     | 0.511 / 0.498 (97.5 %)                                    |
| X5       | YOLO11s Detect  | 0.400 / 0.381 (95.2 %)                                  | 0.198 / 0.165 (83.1 %)                                    | 0.445 / 0.426 (95.8 %)                                     | 0.587 / 0.577 (98.3 %)                                    |
| X5       | YOLO11m Detect  | 0.444 / 0.278 (62.7 %)                                  | 0.247 / 0.048 (19.3 %)                                    | 0.497 / 0.299 (60.2 %)                                     | 0.627 / 0.490 (78.2 %)                                    |
| X5       | YOLO11l Detect  | 0.460 / 0.435 (94.7 %)                                  | 0.267 / 0.224 (84.1 %)                                    | 0.520 / 0.499 (96.0 %)                                     | 0.638 / 0.611 (95.8 %)                                    |
| X5       | YOLO11x Detect  | 0.474 / 0.445 (93.9 %)                                  | 0.283 / 0.233 (82.3 %)                                    | 0.529 / 0.505 (95.3 %)                                     | 0.652 / 0.627 (96.2 %)                                    |
| X5       | YOLOv10n Detect | 0.303 / 0.280 (92.5 %)                                  | 0.099 / 0.079 (79.2 %)                                    | 0.330 / 0.302 (91.3 %)                                     | 0.478 / 0.457 (95.7 %)                                    |
| X5       | YOLOv10s Detect | 0.386 / 0.357 (92.4 %)                                  | 0.175 / 0.131 (74.7 %)                                    | 0.434 / 0.406 (93.6 %)                                     | 0.574 / 0.520 (90.6 %)                                    |
| X5       | YOLOv10m Detect | 0.425 / 0.379 (89.1 %)                                  | 0.221 / 0.181 (82.0 %)                                    | 0.481 / 0.439 (91.3 %)                                     | 0.603 / 0.502 (83.2 %)                                    |
| X5       | YOLOv10b Detect | 0.443 / 0.390 (88.1 %)                                  | 0.242 / 0.207 (85.6 %)                                    | 0.498 / 0.435 (87.2 %)                                     | 0.618 / 0.502 (81.2 %)                                    |
| X5       | YOLOv10l Detect | 0.445 / 0.379 (85.1 %)                                  | 0.258 / 0.211 (81.9 %)                                    | 0.498 / 0.440 (88.2 %)                                     | 0.626 / 0.476 (76.1 %)                                    |
| X5       | YOLOv10x Detect | 0.459 / 0.418 (91.3 %)                                  | 0.258 / 0.216 (83.8 %)                                    | 0.518 / 0.480 (92.8 %)                                     | 0.639 / 0.562 (88.0 %)                                    |
| X5       | YOLOv9t Detect  | 0.313 / 0.299 (95.6 %)                                  | 0.113 / 0.105 (93.1 %)                                    | 0.338 / 0.322 (95.5 %)                                     | 0.483 / 0.456 (94.5 %)                                    |
| X5       | YOLOv9s Detect  | 0.400 / 0.384 (96.2 %)                                  | 0.191 / 0.174 (90.9 %)                                    | 0.444 / 0.430 (96.8 %)                                     | 0.583 / 0.557 (95.6 %)                                    |
| X5       | YOLOv9m Detect  | 0.449 / 0.432 (96.3 %)                                  | 0.253 / 0.227 (89.6 %)                                    | 0.504 / 0.488 (96.8 %)                                     | 0.617 / 0.604 (97.9 %)                                    |
| X5       | YOLOv9c Detect  | 0.461 / 0.440 (95.5 %)                                  | 0.269 / 0.242 (90.1 %)                                    | 0.512 / 0.497 (96.9 %)                                     | 0.640 / 0.611 (95.4 %)                                    |
| X5       | YOLOv9e Detect  | 0.481 / 0.462 (96.1 %)                                  | 0.298 / 0.268 (90.1 %)                                    | 0.538 / 0.514 (95.5 %)                                     | 0.662 / 0.642 (97.0 %)                                    |
| X5       | YOLOv8n Detect  | 0.309 / 0.293 (94.7 %)                                  | 0.113 / 0.103 (90.9 %)                                    | 0.338 / 0.323 (95.4 %)                                     | 0.473 / 0.448 (94.7 %)                                    |
| X5       | YOLOv8s Detect  | 0.391 / 0.378 (96.7 %)                                  | 0.195 / 0.174 (89.3 %)                                    | 0.437 / 0.426 (97.5 %)                                     | 0.566 / 0.558 (98.6 %)                                    |
| X5       | YOLOv8m Detect  | 0.441 / 0.425 (96.4 %)                                  | 0.249 / 0.220 (88.6 %)                                    | 0.494 / 0.480 (97.0 %)                                     | 0.618 / 0.612 (99.0 %)                                    |
| X5       | YOLOv8l Detect  | 0.461 / 0.444 (96.2 %)                                  | 0.271 / 0.243 (89.6 %)                                    | 0.516 / 0.501 (97.1 %)                                     | 0.651 / 0.628 (96.4 %)                                    |
| X5       | YOLOv8x Detect  | 0.474 / 0.451 (95.1 %)                                  | 0.280 / 0.251 (89.7 %)                                    | 0.527 / 0.504 (95.6 %)                                     | 0.658 / 0.638 (97.0 %)                                    |
| X5       | YOLOv5nu Detect | 0.278 / 0.212 (76.0 %)                                  | 0.093 / 0.043 (46.2 %)                                    | 0.309 / 0.219 (71.0 %)                                     | 0.417 / 0.356 (85.5 %)                                    |
| X5       | YOLOv5su Detect | 0.367 / 0.354 (96.5 %)                                  | 0.169 / 0.148 (88.0 %)                                    | 0.416 / 0.402 (96.7 %)                                     | 0.530 / 0.523 (98.6 %)                                    |
| X5       | YOLOv5mu Detect | 0.425 / 0.406 (95.6 %)                                  | 0.226 / 0.195 (86.1 %)                                    | 0.477 / 0.461 (96.7 %)                                     | 0.603 / 0.594 (98.5 %)                                    |
| X5       | YOLOv5lu Detect | 0.458 / 0.440 (96.0 %)                                  | 0.260 / 0.226 (87.0 %)                                    | 0.516 / 0.503 (97.3 %)                                     | 0.641 / 0.627 (97.7 %)                                    |
| X5       | YOLOv5xu Detect | 0.466 / 0.448 (96.2 %)                                  | 0.281 / 0.241 (85.8 %)                                    | 0.523 / 0.512 (98.0 %)                                     | 0.645 / 0.639 (99.2 %)                                    |

#### Instance Segmentation

### Instance Segmentation

| Device   | Model       | Accuracy bbox-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| X5       | YOLO11n Seg | 0.322 / 0.294 (91.4 %)                                  | 0.113 / 0.089 (78.8 %)                                    | 0.352 / 0.320 (90.9 %)                                     | 0.502 / 0.479 (95.3 %)                                    |
| X5       | YOLO11s Seg | 0.394 / 0.373 (94.7 %)                                  | 0.184 / 0.155 (84.2 %)                                    | 0.442 / 0.422 (95.5 %)                                     | 0.582 / 0.570 (97.9 %)                                    |
| X5       | YOLO11m Seg | 0.443 / 0.413 (93.2 %)                                  | 0.246 / 0.199 (80.8 %)                                    | 0.497 / 0.472 (95.0 %)                                     | 0.627 / 0.601 (95.9 %)                                    |
| X5       | YOLO11l Seg | 0.460 / 0.430 (93.6 %)                                  | 0.267 / 0.216 (80.9 %)                                    | 0.520 / 0.494 (95.2 %)                                     | 0.638 / 0.609 (95.5 %)                                    |
| X5       | YOLO11x Seg | 0.474 / 0.441 (92.9 %)                                  | 0.283 / 0.225 (79.3 %)                                    | 0.529 / 0.500 (94.4 %)                                     | 0.652 / 0.625 (95.9 %)                                    |
| X5       | YOLOv9c Seg | 0.453 / 0.422 (93.0 %)                                  | 0.254 / 0.205 (80.7 %)                                    | 0.508 / 0.482 (94.7 %)                                     | 0.621 / 0.605 (97.3 %)                                    |
| X5       | YOLOv9e Seg | 0.481 / 0.452 (94.2 %)                                  | 0.292 / 0.254 (86.7 %)                                    | 0.537 / 0.507 (94.4 %)                                     | 0.650 / 0.632 (97.1 %)                                    |
| X5       | YOLOv8n Seg | 0.304 / 0.286 (94.1 %)                                  | 0.109 / 0.091 (83.6 %)                                    | 0.334 / 0.314 (94.0 %)                                     | 0.461 / 0.446 (96.7 %)                                    |
| X5       | YOLOv8s Seg | 0.386 / 0.368 (95.2 %)                                  | 0.180 / 0.155 (86.4 %)                                    | 0.432 / 0.413 (95.4 %)                                     | 0.564 / 0.554 (98.3 %)                                    |
| X5       | YOLOv8m Seg | 0.431 / 0.410 (95.1 %)                                  | 0.228 / 0.197 (86.4 %)                                    | 0.486 / 0.469 (96.5 %)                                     | 0.608 / 0.596 (98.0 %)                                    |
| X5       | YOLOv8l Seg | 0.453 / 0.427 (94.2 %)                                  | 0.258 / 0.216 (83.5 %)                                    | 0.502 / 0.487 (96.9 %)                                     | 0.626 / 0.605 (96.6 %)                                    |
| X5       | YOLOv8x Seg | 0.465 / 0.438 (94.3 %)                                  | 0.268 / 0.219 (81.6 %)                                    | 0.520 / 0.499 (96.0 %)                                     | 0.641 / 0.626 (97.8 %)                                    |

##### mask

### Instance Segmentation

| Device   | Model       | Accuracy mask-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| X5       | YOLO11n Seg | 0.262 / 0.224 (85.6 %)                                  | 0.062 / 0.049 (79.2 %)                                    | 0.283 / 0.245 (86.6 %)                                     | 0.444 / 0.384 (86.5 %)                                    |
| X5       | YOLO11s Seg | 0.311 / 0.288 (92.6 %)                                  | 0.099 / 0.092 (93.0 %)                                    | 0.350 / 0.324 (92.7 %)                                     | 0.509 / 0.470 (92.3 %)                                    |
| X5       | YOLO11m Seg | 0.347 / 0.314 (90.5 %)                                  | 0.136 / 0.115 (84.6 %)                                    | 0.396 / 0.361 (91.2 %)                                     | 0.549 / 0.492 (89.5 %)                                    |
| X5       | YOLO11l Seg | 0.357 / 0.325 (90.9 %)                                  | 0.143 / 0.125 (87.1 %)                                    | 0.409 / 0.373 (91.1 %)                                     | 0.560 / 0.504 (90.0 %)                                    |
| X5       | YOLO11x Seg | 0.366 / 0.332 (90.6 %)                                  | 0.149 / 0.125 (84.5 %)                                    | 0.420 / 0.379 (90.2 %)                                     | 0.572 / 0.520 (91.0 %)                                    |
| X5       | YOLOv9c Seg | 0.352 / 0.319 (90.5 %)                                  | 0.132 / 0.115 (87.0 %)                                    | 0.404 / 0.366 (90.6 %)                                     | 0.547 / 0.500 (91.6 %)                                    |
| X5       | YOLOv9e Seg | 0.371 / 0.342 (92.2 %)                                  | 0.155 / 0.142 (91.6 %)                                    | 0.425 / 0.386 (90.9 %)                                     | 0.571 / 0.527 (92.3 %)                                    |
| X5       | YOLOv8n Seg | 0.246 / 0.222 (90.4 %)                                  | 0.059 / 0.052 (87.6 %)                                    | 0.265 / 0.246 (92.8 %)                                     | 0.409 / 0.368 (89.9 %)                                    |
| X5       | YOLOv8s Seg | 0.305 / 0.284 (93.0 %)                                  | 0.096 / 0.089 (93.3 %)                                    | 0.343 / 0.318 (92.9 %)                                     | 0.496 / 0.462 (93.1 %)                                    |
| X5       | YOLOv8m Seg | 0.337 / 0.314 (93.2 %)                                  | 0.121 / 0.113 (93.6 %)                                    | 0.386 / 0.360 (93.4 %)                                     | 0.533 / 0.493 (92.5 %)                                    |
| X5       | YOLOv8l Seg | 0.351 / 0.327 (93.2 %)                                  | 0.137 / 0.124 (90.5 %)                                    | 0.398 / 0.374 (94.0 %)                                     | 0.550 / 0.506 (92.1 %)                                    |
| X5       | YOLOv8x Seg | 0.358 / 0.332 (92.6 %)                                  | 0.139 / 0.121 (87.1 %)                                    | 0.409 / 0.380 (92.7 %)                                     | 0.562 / 0.517 (91.9 %)                                    |

#### Pose Estimation

### Pose Estimation

| Device   | Model        | Accuracy pose-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy pose-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy pose-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|--------------|---------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| X5       | YOLO11n Pose | 0.465 / 0.453 (97.3 %)                                  | 0.386 / 0.379 (98.2 %)                                     | 0.597 / 0.577 (96.6 %)                                    |
| X5       | YOLO11s Pose | 0.559 / 0.532 (95.1 %)                                  | 0.495 / 0.468 (94.7 %)                                     | 0.672 / 0.644 (95.8 %)                                    |
| X5       | YOLO11m Pose | 0.627 / 0.609 (97.1 %)                                  | 0.586 / 0.565 (96.4 %)                                     | 0.711 / 0.693 (97.4 %)                                    |
| X5       | YOLO11l Pose | 0.636 / 0.619 (97.3 %)                                  | 0.592 / 0.569 (96.3 %)                                     | 0.726 / 0.710 (97.8 %)                                    |
| X5       | YOLO11x Pose | 0.672 / 0.650 (96.8 %)                                  | 0.634 / 0.609 (96.1 %)                                     | 0.750 / 0.733 (97.8 %)                                    |
| X5       | YOLOv8n Pose | 0.476 / 0.459 (96.4 %)                                  | 0.391 / 0.373 (95.3 %)                                     | 0.610 / 0.594 (97.4 %)                                    |
| X5       | YOLOv8s Pose | 0.578 / 0.551 (95.3 %)                                  | 0.510 / 0.478 (93.7 %)                                     | 0.692 / 0.667 (96.5 %)                                    |
| X5       | YOLOv8m Pose | 0.630 / 0.606 (96.2 %)                                  | 0.578 / 0.552 (95.6 %)                                     | 0.724 / 0.692 (95.5 %)                                    |
| X5       | YOLOv8l Pose | 0.657 / 0.632 (96.3 %)                                  | 0.607 / 0.582 (95.8 %)                                     | 0.747 / 0.725 (97.0 %)                                    |
| X5       | YOLOv8x Pose | 0.671 / 0.648 (96.6 %)                                  | 0.624 / 0.599 (96.0 %)                                     | 0.757 / 0.736 (97.2 %)                                    |

#### Image Classification

### Image Classification

| Device   | Model       | Accuracy TOP1 <br/> FP32 / BPU Python   | Accuracy TOP5 <br/> FP32 / BPU Python   |
|----------|-------------|-----------------------------------------|-----------------------------------------|
| X5       | YOLO11n CLS | 0.700 / 0.585 (83.6 %)                  | 0.894 / 0.815 (91.2 %)                  |
| X5       | YOLO11s CLS | 0.754 / 0.663 (88.0 %)                  | 0.927 / 0.873 (94.2 %)                  |
| X5       | YOLO11m CLS | 0.773 / 0.708 (91.5 %)                  | 0.939 / 0.903 (96.1 %)                  |
| X5       | YOLO11l CLS | 0.783 / 0.714 (91.1 %)                  | 0.942 / 0.906 (96.1 %)                  |
| X5       | YOLO11x CLS | 0.795 / 0.733 (92.2 %)                  | 0.949 / 0.917 (96.6 %)                  |
| X5       | YOLOv8n CLS | 0.689 / 0.574 (83.2 %)                  | 0.883 / 0.806 (91.2 %)                  |
| X5       | YOLOv8s CLS | 0.737 / 0.635 (86.1 %)                  | 0.917 / 0.850 (92.8 %)                  |
| X5       | YOLOv8m CLS | 0.768 / 0.702 (91.5 %)                  | 0.935 / 0.899 (96.2 %)                  |
| X5       | YOLOv8l CLS | 0.783 / 0.727 (92.9 %)                  | 0.942 / 0.912 (96.9 %)                  |

| X5       | YOLOv8x CLS | 0.790 / 0.741 (93.8 %)                  | 0.945 / 0.921 (97.5 %)                  |

### RDK S100 Accuracy Data (Accuracy @ NV12 - Detection)

| Device | Model | Accuracy bbox-all <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-small <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-medium <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-large <br> mAP @.50:.95 <br> (FP32 / BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S100 | YOLO26n Detect | 0.319 / 0.286 (89.7 %) | 0.107 / 0.083 (77.6 %) | 0.349 / 0.304 (87.1 %) | 0.508 / 0.473 (93.1 %) |
| S100 | YOLO26s Detect | 0.395 / 0.362 (91.6 %) | 0.183 / 0.163 (89.1 %) | 0.440 / 0.402 (91.4 %) | 0.583 / 0.524 (89.9 %) |
| S100 | YOLO26m Detect | 0.442 / 0.413 (93.4 %) | 0.242 / 0.202 (83.5 %) | 0.489 / 0.456 (93.3 %) | 0.629 / 0.603 (95.9 %) |
| S100 | YOLO26l Detect | 0.456 / 0.440 (96.5 %) | 0.260 / 0.230 (88.5 %) | 0.499 / 0.489 (98.0 %) | 0.627 / 0.623 (99.4 %) |
| S100 | YOLO26x Detect | 0.484 / 0.449 (92.8 %) | 0.292 / 0.246 (84.2 %) | 0.528 / 0.488 (92.4 %) | 0.669 / 0.646 (96.6 %) |



### RDK S100P Accuracy Data (Accuracy @ RGB - Detection)

| Device | Model | Accuracy bbox-all <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-small <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-medium <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-large <br> mAP @.50:.95 <br> (FP32 / BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S100P | YOLO26n Detect | 0.319 / 0.290 (91.0 %) | 0.107 / 0.087 (81.7 %) | 0.349 / 0.313 (89.8 %) | 0.508 / 0.463 (91.1 %) |
| S100P | YOLO26s Detect | 0.395 / 0.367 (93.0 %) | 0.183 / 0.174 (94.8 %) | 0.440 / 0.410 (93.1 %) | 0.583 / 0.530 (91.0 %) |
| S100P | YOLO26m Detect | 0.442 / 0.421 (95.2 %) | 0.242 / 0.224 (92.6 %) | 0.489 / 0.460 (94.0 %) | 0.629 / 0.603 (95.8 %) |
| S100P | YOLO26l Detect | 0.456 / 0.437 (96.0 %) | 0.260 / 0.234 (89.8 %) | 0.499 / 0.484 (97.1 %) | 0.627 / 0.609 (97.0 %) |
| S100P | YOLO26x Detect | 0.484 / 0.466 (96.2 %) | 0.292 / 0.271 (92.8 %) | 0.528 / 0.502 (95.1 %) | 0.669 / 0.654 (97.8 %) |



### RDK S600 Accuracy Data (Accuracy @ NV12 - Detection)

| Device | Model | Accuracy bbox-all <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-small <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-medium <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-large <br> mAP @.50:.95 <br> (FP32 / BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S600 | YOLO26n Detect | 0.319 / 0.285 (89.3 %) | 0.107 / 0.080 (74.8 %) | 0.349 / 0.304 (87.1 %) | 0.508 / 0.460 (90.6 %) |
| S600 | YOLO26s Detect | 0.395 / 0.368 (93.2 %) | 0.183 / 0.172 (94.0 %) | 0.440 / 0.414 (94.1 %) | 0.583 / 0.526 (90.2 %) |
| S600 | YOLO26m Detect | 0.442 / 0.417 (94.3 %) | 0.242 / 0.211 (87.2 %) | 0.489 / 0.460 (94.1 %) | 0.629 / 0.600 (95.4 %) |
| S600 | YOLO26l Detect | 0.456 / 0.429 (94.1 %) | 0.260 / 0.222 (85.4 %) | 0.499 / 0.473 (94.8 %) | 0.627 / 0.610 (97.3 %) |
| S600 | YOLO26x Detect | 0.484 / 0.452 (93.4 %) | 0.292 / 0.246 (84.2 %) | 0.528 / 0.490 (92.8 %) | 0.669 / 0.645 (96.4 %) |



### RDK S100 Accuracy Data (Accuracy @ NV12 - Pose Estimation)

| Device | Model | Accuracy kpt-all <br> mAP @.50:.95 <br> (BPU Python) | Accuracy kpt-medium <br> mAP @.50:.95 <br> (BPU Python) | Accuracy kpt-large <br> mAP @.50:.95 <br> (BPU Python) |
| :--- | :--- | :--- | :--- | :--- |
| S100 | YOLO26n Pose | 0.504 | 0.412 | 0.647 |
| S100 | YOLO26s Pose | 0.575 | 0.498 | 0.697 |
| S100 | YOLO26m Pose | 0.620 | 0.554 | 0.737 |
| S100 | YOLO26l Pose | 0.646 | 0.579 | 0.744 |
| S100 | YOLO26x Pose | 0.663 | 0.601 | 0.775 |



### RDK S600 Accuracy Data (Accuracy @ NV12 - Pose Estimation)

| Device | Model | Accuracy kpt-all <br> mAP @.50:.95 <br> (BPU Python) | Accuracy kpt-medium <br> mAP @.50:.95 <br> (BPU Python) | Accuracy kpt-large <br> mAP @.50:.95 <br> (BPU Python) |
| :--- | :--- | :--- | :--- | :--- |
| S600 | YOLO26n Pose | 0.507 | 0.416 | 0.649 |
| S600 | YOLO26s Pose | 0.578 | 0.505 | 0.697 |
| S600 | YOLO26m Pose | 0.621 | 0.551 | 0.737 |
| S600 | YOLO26l Pose | 0.640 | 0.583 | 0.740 |
| S600 | YOLO26x Pose | 0.665 | 0.604 | 0.769 |



### RDK S100 Accuracy Data (Accuracy @ NV12 - Segmentation)

| Device | Model | Accuracy mask-all <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-small <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-medium <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-large <br> mAP @.50:.95 <br> (FP32 / BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S100 | YOLO26n Seg | - / 0.254 | - / 0.057 | - / 0.269 | - / 0.434 |
| S100 | YOLO26s Seg | - / 0.330 | - / 0.119 | - / 0.367 | - / 0.510 |
| S100 | YOLO26m Seg | - / 0.356 | - / 0.148 | - / 0.399 | - / 0.536 |
| S100 | YOLO26l Seg | - / 0.375 | - / 0.164 | - / 0.419 | - / 0.560 |
| S100 | YOLO26x Seg | - / 0.381 | - / 0.176 | - / 0.426 | - / 0.576 |



### RDK S600 Accuracy Data (Accuracy @ NV12 - Segmentation)

| Device | Model | Accuracy mask-all <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-small <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-medium <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-large <br> mAP @.50:.95 <br> (FP32 / BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S600 | YOLO26n Seg | - / 0.255 | - / 0.061 | - / 0.274 | - / 0.422 |
| S600 | YOLO26s Seg | - / 0.331 | - / 0.121 | - / 0.370 | - / 0.508 |
| S600 | YOLO26m Seg | - / 0.358 | - / 0.156 | - / 0.398 | - / 0.538 |
| S600 | YOLO26l Seg | - / 0.371 | - / 0.158 | - / 0.417 | - / 0.559 |
| S600 | YOLO26x Seg | - / 0.382 | - / 0.171 | - / 0.429 | - / 0.571 |



### RDK S100 Accuracy Data (Accuracy @ NV12 - Classification)

| Device | Model | Top-1 Accuracy | Top-5 Accuracy |
| :--- | :--- | :--- | :--- |
| S100 | YOLO26n Cls | 0.6165 | 0.8359 |
| S100 | YOLO26s Cls | 0.6854 | 0.8853 |
| S100 | YOLO26m Cls | 0.7194 | 0.9080 |
| S100 | YOLO26l Cls | 0.7369 | 0.9168 |
| S100 | YOLO26x Cls | 0.7432 | 0.9222 |



### RDK S600 Accuracy Data (Accuracy @ NV12 - Classification)

| Device | Model | Accuracy TOP1 <br> (BPU Python) | Accuracy TOP5 <br> (BPU Python) |
| :--- | :--- | :--- | :--- |
| S600 | YOLO26n Cls | 0.600 | 0.823 |
| S600 | YOLO26s Cls | 0.649 | 0.863 |
| S600 | YOLO26m Cls | 0.650 | 0.867 |
| S600 | YOLO26l Cls | 0.689 | 0.890 |
| S600 | YOLO26x Cls | 0.666 | 0.876 |



### RDK X5 Accuracy Data (Accuracy @ NV12 - Detection)

| Device | Model | Accuracy bbox-all <br> mAP@.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-small <br> mAP@.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-medium <br> mAP@.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-large <br> mAP@.50:.95 <br> (FP32 / BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| X5 | YOLO26n Detect | 0.319 / 0.284 (89.0 %) | 0.107 / 0.075 (70.1 %) | 0.349 / 0.299 (85.7 %) | 0.508 / 0.467 (91.9 %) |
| X5 | YOLO26s Detect | 0.395 / 0.357 (90.4 %) | 0.183 / 0.154 (84.2 %) | 0.440 / 0.393 (89.3 %) | 0.583 / 0.534 (91.6 %) |
| X5 | YOLO26m Detect | 0.442 / 0.413 (93.4 %) | 0.242 / 0.206 (85.1 %) | 0.489 / 0.454 (92.8 %) | 0.629 / 0.605 (96.1 %) |
| X5 | YOLO26l Detect | 0.456 / 0.431 (94.5 %) | 0.260 / 0.215 (82.7 %) | 0.499 / 0.479 (96.0 %) | 0.627 / 0.618 (98.6 %) |
| X5 | YOLO26x Detect | 0.484 / 0.438 (90.5 %) | 0.292 / 0.230 (78.8 %) | 0.528 / 0.479 (90.7 %) | 0.669 / 0.635 (94.9 %) |



### RDK X5 Accuracy Data (Accuracy @ NV12 - Segmentation)

| Device | Model | Accuracy mask-all <br> mAP@.50:.95 <br> (BPU Python) | Accuracy mask-small <br> mAP@.50:.95 <br> (BPU Python) | Accuracy mask-medium <br> mAP@.50:.95 <br> (BPU Python) | Accuracy mask-large <br> mAP@.50:.95 <br> (BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| X5 | YOLO26n Seg | 0.285 | 0.090 | 0.307 | 0.464 |



### Detection (COCO2017)

| Model | PyTorch AP | Python AP |
| :--- | :--- | :--- |
| YOLOv5nu | 0.275 | 0.260 (94.55%) |
| YOLOv5su | 0.362 | 0.354 (97.79%) |
| YOLOv5mu | 0.417 | 0.407 (97.60%) |
| YOLOv5lu | 0.449 | 0.442 (98.44%) |
| YOLOv5xu | 0.458 | 0.443 (96.72%) |
| YOLOv8n | 0.306 | 0.292 (95.42%) |
| YOLOv8s | 0.384 | 0.372 (96.88%) |
| YOLOv8m | 0.433 | 0.423 (97.69%) |
| YOLOv8l | 0.454 | 0.440 (96.92%) |
| YOLOv8x | 0.465 | 0.448 (96.34%) |
| YOLOv9t | 0.357 | 0.346 (96.92%) |
| YOLOv9s | 0.460 | 0.446 (96.96%) |
| YOLOv9m | 0.504 | 0.485 (96.23%) |
| YOLOv9c | 0.530 | 0.515 (97.17%) |
| YOLOv9e | 0.555 | 0.530 (95.50%) |
| YOLOv10n | 0.387 | 0.357 (92.25%) |
| YOLOv10s | 0.469 | 0.444 (94.67%) |
| YOLOv10m | 0.510 | 0.482 (94.50%) |
| YOLOv10b | 0.525 | 0.504 (96.00%) |
| YOLOv10l | 0.540 | 0.517 (95.74%) |
| YOLOv10x | 0.541 | 0.522 (96.49%) |
| YOLO11n | 0.323 | 0.308 (95.36%) |
| YOLO11s | 0.394 | 0.380 (96.45%) |
| YOLO11m | 0.437 | 0.422 (96.57%) |
| YOLO11l | 0.452 | 0.432 (95.58%) |
| YOLO11x | 0.466 | 0.446 (95.71%) |
| YOLO12n | 0.410 | 0.383 (93.41%) |
| YOLO12s | 0.487 | 0.465 (95.48%) |
| YOLO12m | 0.533 | 0.513 (96.25%) |
| YOLO12l | 0.545 | 0.523 (95.96%) |
| YOLO12x | 0.557 | 0.532 (95.51%) |
| YOLOv13n | 0.409 | 0.385 (94.13%) |
| YOLOv13s | 0.485 | 0.458 (94.43%) |
| YOLOv13l | 0.538 | 0.510 (94.80%) |
| YOLOv13x | 0.551 | 0.526 (95.46%) |




## 补充参考数据

各表按发布时的模型、板卡和测量条件列出，不同配置的数据分别保留。模型文件大小沿用表中注明的 MB 单位。

### RDK S100P (S100 YOLOv13 iMoonLab)

| Model | Size(Pixels) | Classes | BPU Task Latency /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | params(M) | FLOPs(B) |
|---|---|---:|---|---:|---:|---:|
| YOLOv13n | 640x640 | 80 | 2.8 ms / 353.5 FPS (1 thread)<br>3.9 ms / 509.0 FPS (2 threads) | 2 ms | 2.5 | 6.4 |
| YOLOv13s | 640x640 | 80 | 4.3 ms / 231.7 FPS (1 thread)<br>7.1 ms / 278.5 FPS (2 threads) | 2 ms | 9.0 | 20.8 |
| YOLOv13l | 640x640 | 80 | 12.1 ms / 82.5 FPS (1 thread)<br>22.7 ms / 87.7 FPS (2 threads) | 2 ms | 27.6 | 88.4 |
| YOLOv13x | 640x640 | 80 | 19.7 ms / 50.7 FPS (1 thread)<br>37.8 ms / 52.7 FPS (2 threads) | 2 ms | 64.0 | 199.2 |



### RDK S100 (S100 YOLOv13 iMoonLab)

| Model | Size(Pixels) | Classes | BPU Task Latency /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | params(M) | FLOPs(B) |
|---|---|---:|---|---:|---:|---:|
| YOLOv13n | 640x640 | 80 | 3.8 ms / 262.0 FPS (1 thread)<br>5.2 ms / 378.3 FPS (2 threads) | 2 ms | 2.5 | 6.4 |
| YOLOv13s | 640x640 | 80 | 5.8 ms / 169.5 FPS (1 thread)<br>9.7 ms / 204.9 FPS (2 threads) | 2 ms | 9.0 | 20.8 |
| YOLOv13l | 640x640 | 80 | 16.6 ms / 59.8 FPS (1 thread)<br>31.1 ms / 63.9 FPS (2 threads) | 2 ms | 27.6 | 88.4 |
| YOLOv13x | 640x640 | 80 | 26.9 ms / 37.1 FPS (1 thread)<br>51.6 ms / 38.6 FPS (2 threads) | 2 ms | 64.0 | 199.2 |



### Performance Data (Summary) (X3 historical benchmark)

| Model (Official) | Size (px) | Classes | Params (M) | Throughput (FPS) | Post Process Time (Python) |
|---------|---------|-------|-------------------|--------------------|---|
| YOLOv8n-seg | 640×640 | 80 | 3.4  | 175.3 | 6 ms |
| YOLOv8s-seg | 640×640 | 80 | 11.8 | 67.7 | 6 ms |
| YOLOv8m-seg | 640×640 | 80 | 27.3 | 27.0 | 6 ms |
| YOLOv8l-seg | 640×640 | 80 | 46.0 | 14.4 | 6 ms |
| YOLOv8x-seg | 640×640 | 80 | 71.8 | 8.9 | 6 ms |

Note: Detailed performance data is at the end of the document.

### Performance Data (X3 historical benchmark)

| Model | Size (px) | Num Classes | Params (M) | FP Precision (box/mask) | INT8 Precision (box/mask) | Latency/Throughput (Single-threaded) | Latency/Throughput (Multi-threaded) | Post-processing Time (Python) |
|---------|---------|-------|---------|---------|----------|--------------------|--------------------|-------|
| YOLOv8n-seg | 640×640 | 80 | 3.4  | 36.7/30.5 |  | 9 ms / 109.7 FPS (1 thread) | 11.4 ms / 175.3 FPS (2 threads) | 6 ms |
| YOLOv8s-seg | 640×640 | 80 | 11.8 | 44.6/36.8 |  | 18.1 ms / 55.1 FPS (1 thread) | 29.4 ms / 67.7 FPS (2 threads) | 6 ms |
| YOLOv8m-seg | 640×640 | 80 | 27.3 | 49.9/40.8 |  | 40.4 ms / 24.7 FPS (1 thread) | 73.8 ms / 27.0 FPS (2 threads) | 6 ms |
| YOLOv8l-seg | 640×640 | 80 | 46.0 | 52.3/42.6 |  | 72.7 ms / 13.7 FPS (1 thread) | 138.2 ms / 14.4 FPS (2 threads) | 6 ms |
| YOLOv8x-seg | 640×640 | 80 | 71.8 | 53.4/43.4 |  | 115.7 ms / 8.6 FPS (1 thread) | 223.8 ms / 8.9 FPS (2 threads) | 6 ms |

Notes:
1. The X5 is in its optimal state: CPU is 8 × A55 @ 1.8G with full-core Performance scheduling, BPU is 1 × Bayes-e @ 1G with a total equivalent int8 computing power of 10 TOPS.
2. Single-threaded latency is for a single frame, single thread, and single BPU core, representing the ideal delay for BPU inference of a single task.
3. Four-thread engineering frame rate is when four threads simultaneously feed tasks to the dual-core BPU. In general engineering scenarios, four threads can minimize single-frame latency while fully utilizing all BPU cores at 100%, achieving a good balance between throughput (FPS) and frame latency. X5 BPU overall is more powerful, generally 2 threads can eat BPU full, frame delay and throughput are very good.
4. Eight-thread extreme frame rate is when eight threads simultaneously feed tasks to the dual-core BPU on the X3, aiming to test the BPU’s extreme performance. Typically, four cores are already saturated; if eight threads perform significantly better than four, it suggests that the model structure needs to improve the "compute/memory access" ratio, or that DDR bandwidth optimization should be selected during compilation.
5. FP/Q mAP: 50-95 precision is calculated using pycocotools and comes from the COCO dataset. This can refer to Microsoft’s paper, and is used here to assess the degree of accuracy degradation for deployment on the board.
6. Run the following command to test the bin model throughput on the board
```bash
hrt_model_exec perf --thread_num 2 --model_file yolov8n_detect_bayese_640x640_nv12_modified.bin
```
7. Regarding post-processing: At present, the post-processing of Python reconstruction on X5 only requires a single-core single-thread serial about 5ms to complete, that is, it only needs to occupy 2 CPU cores (200% CPU usage, maximum 800% CPU usage), and can complete 400 frames of image post-processing per minute, and post-processing will not constitute a bottleneck.

### RDK X3 & RDK X3 Module (X3 historical benchmark)

| 模型 | 尺寸(像素) | 类别数 | 参数量 | 浮点精度<br/>(box/mask) | 量化精度<br/>(box/mask) | 平均BPU延迟/吞吐量(单线程) <br/> 平均BPU延迟/吞吐量(多线程) | 后处理时间(Python) |
|---------|---------|-------|---------|---------|----------|--------------------|--------------------|
| YOLOv8n-seg | 640×640 | 80 | 3.4 M | 36.7/30.5 |  | 126.7 ms / 7.9 FPS (1 thread) <br/> 129.2 ms / 15.5 FPS (2 threads) <br/> 163.9 ms / 24.1 FPS (4 threads) <br/> 285.8 ms / 27.3 FPS (8 threads) | 6 ms |

说明:
说明:
说明:
1. BPU延迟与BPU吞吐量。
 - 单线程延迟为单帧,单线程,单BPU核心的延迟,BPU推理一个任务最理想的情况。
 - 多线程帧率为多个线程同时向BPU塞任务, 每个BPU核心可以处理多个线程的任务, 一般工程中4个线程可以控制单帧延迟较小,同时吃满所有BPU到100%,在吞吐量(FPS)和帧延迟间得到一个较好的平衡。X5的BPU整体比较厉害, 一般2个线程就可以将BPU吃满, 帧延迟和吞吐量都非常出色。
 - 表格中一般记录到吞吐量不再随线程数明显增加的数据。
 - BPU延迟和BPU吞吐量使用以下命令在板端测试
```bash
hrt_model_exec perf --thread_num 2 --model_file yolov8n_detect_bayese_640x640_nv12_modified.bin
```
2. 测试板卡均为最佳状态。
 - X5的状态为最佳状态：CPU为8 × A55@1.8G, 全核心Performance调度, BPU为1 × Bayes-e@10TOPS.
```bash
sudo bash -c "echo 1 > /sys/devices/system/cpu/cpufreq/boost"  # 1.8Ghz
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor" # Performance Mode
```
 - X3的状态为最佳状态：CPU为4 × A53@1.8G, 全核心Performance调度, BPU为2 × Bernoulli2@5TOPS.
```bash
sudo bash -c "echo 1 > /sys/devices/system/cpu/cpufreq/boost"  # 1.8Ghz
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor" # Performance Mode
```
3. 浮点/定点mAP：50-95精度使用pycocotools计算,来自于COCO数据集,可以参考微软的论文,此处用于评估板端部署的精度下降程度。
4. 关于后处理: 目前在X5上使用Python重构的后处理, 仅需要单核心单线程串行5ms左右即可完成, 也就是说只需要占用2个CPU核心(200%的CPU占用, 最大800%的CPU占用), 每分钟可完成400帧图像的后处理, 后处理不会构成瓶颈.

### Performance Data (Summary) (X3 historical benchmark)

| Model (Official) | Size (px) | Classes | Params (M) | Throughput (FPS) | Post Process Time (Python) |
|---------|---------|-------|-------------------|--------------------|---|
| YOLOv10n | 640×640 | 80 | 6.7  | 132.7 | 4.5 ms |
| YOLOv10s | 640×640 | 80 | 21.6 | 71.0 | 4.5 ms |
| YOLOv10m | 640×640 | 80 | 59.1 | 34.5 | 4.5 ms |
| YOLOv10b | 640×640 | 80 | 92.0 | 25.4 | 4.5 ms |
| YOLOv10l | 640×640 | 80 | 120.3 | 20.0 | 4.5 ms |
| YOLOv10x | 640×640 | 80 | 160.4 | 14.5 | 4.5 ms |



### Performance Data (X3 historical benchmark)

| Model | Size (px) | Num Classes | FLOPs (G) | FP Precision | INT8 Precision (box/mask) | Latency/Throughput (Single-threaded) | Latency/Throughput (Multi-threaded) | Post Process Time(Python) |
|---------|---------|------------|---------|-------------|---------------------------|-------------------------------------|-------------------------------------|--|
| YOLOv10n | 640×640 | 80 | 6.7  | 38.5 |  | 9.3 ms / 107.0 FPS (1 thread) | 15.0 ms / 132.7 FPS (2 threads) | 4.5 ms |
| YOLOv10s | 640×640 | 80 | 21.6 | 46.3 |  | 15.8 ms / 63.0 FPS (1 thread) | 28.1 ms / 71.0 FPS (2 threads) | 4.5 ms |
| YOLOv10m | 640×640 | 80 | 59.1 | 51.1 |  | 30.8 ms / 32.4 FPS (1 thread) | 51.8 ms / 34.5 FPS (2 threads) | 4.5 ms |
| YOLOv10b | 640×640 | 80 | 92.0 | 52.3 |  | 41.1 ms / 24.3 FPS (1 thread) | 78.4 ms / 25.4 FPS (2 threads) | 4.5 ms |
| YOLOv10l | 640×640 | 80 | 120.3 | 53.2 |  | 52.0 ms / 19.2 FPS (1 thread) | 100.0 ms / 20.0 FPS (2 threads) | 4.5 ms |
| YOLOv10x | 640×640 | 80 | 160.4 | 54.4 |  | 70.7 ms / 14.1 FPS (1 thread) | 137.3 ms / 14.5 FPS (2 threads) | 4.5 ms |

Notes:
1. The X5 is in its optimal state: CPU is 8 × A55 @ 1.8G with full-core Performance scheduling, BPU is 1 × Bayes-e @ 1G with a total equivalent int8 computing power of 10 TOPS.
2. Single-threaded latency is for a single frame, single thread, and single BPU core, representing the ideal delay for BPU inference of a single task.
3. Four-thread engineering frame rate is when four threads simultaneously feed tasks to the dual-core BPU. In general engineering scenarios, four threads can minimize single-frame latency while fully utilizing all BPU cores at 100%, achieving a good balance between throughput (FPS) and frame latency. X5 BPU overall is more powerful, generally 2 threads can eat BPU full, frame delay and throughput are very good.
4. Eight-thread extreme frame rate is when eight threads simultaneously feed tasks to the dual-core BPU on the X3, aiming to test the BPU’s extreme performance. Typically, four cores are already saturated; if eight threads perform significantly better than four, it suggests that the model structure needs to improve the "compute/memory access" ratio, or that DDR bandwidth optimization should be selected during compilation.
5. FP/Q mAP: 50-95 precision is calculated using pycocotools and comes from the COCO dataset. This can refer to Microsoft’s paper, and is used here to assess the degree of accuracy degradation for deployment on the board.
6. Run the following command to test the bin model throughput on the board
```bash
hrt_model_exec perf --thread_num 2 --model_file yolov8n_detect_bayese_640x640_nv12_modified.bin
```
7. Regarding post-processing: At present, the post-processing of Python reconstruction on X5 only requires a single-core single-thread serial about 5ms to complete, that is, it only needs to occupy 2 CPU cores (200% CPU usage, maximum 800% CPU usage), and can complete 400 frames of image post-processing per minute, and post-processing will not constitute a bottleneck.

### RDK X3 & RDK X3 Module (X3 historical benchmark)

| 模型 | 尺寸(像素) | 类别数 | FLOPs (G) | 浮点精度<br/>(mAP:50-95) | 量化精度<br/>(mAP:50-95) | BPU延迟/BPU吞吐量(线程) |  后处理时间<br/>(Python) |
|---------|---------|-------|---------|---------|----------|--------------------|--------------------|
| YOLOv10n | 640×640 | 80 | 6.7  | 38.5 G |  | 174.7 ms / 5.7 FPS (1 thread) <br/> 181.5 ms / 11.0 FPS (2 threads) <br/> 240.1 ms / 16.2 FPS (4 threads) <br/> 421.0 ms / 18.1 FPS (8 threads) | 5 ms |

说明:
1. BPU延迟与BPU吞吐量。
 - 单线程延迟为单帧,单线程,单BPU核心的延迟,BPU推理一个任务最理想的情况。
 - 多线程帧率为多个线程同时向BPU塞任务, 每个BPU核心可以处理多个线程的任务, 一般工程中4个线程可以控制单帧延迟较小,同时吃满所有BPU到100%,在吞吐量(FPS)和帧延迟间得到一个较好的平衡。X5的BPU整体比较厉害, 一般2个线程就可以将BPU吃满, 帧延迟和吞吐量都非常出色。
 - 表格中一般记录到吞吐量不再随线程数明显增加的数据。
 - BPU延迟和BPU吞吐量使用以下命令在板端测试
```bash
hrt_model_exec perf --thread_num 2 --model_file yolov8n_detect_bayese_640x640_nv12_modified.bin
```
2. 测试板卡均为最佳状态。
 - X5的状态为最佳状态：CPU为8 × A55@1.8G, 全核心Performance调度, BPU为1 × Bayes-e@10TOPS.
```bash
sudo bash -c "echo 1 > /sys/devices/system/cpu/cpufreq/boost"  # 1.8Ghz
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor" # Performance Mode
```
 - X3的状态为最佳状态：CPU为4 × A53@1.8G, 全核心Performance调度, BPU为2 × Bernoulli2@5TOPS.
```bash
sudo bash -c "echo 1 > /sys/devices/system/cpu/cpufreq/boost"  # 1.8Ghz
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor" # Performance Mode
```
3. 浮点/定点mAP：50-95精度使用pycocotools计算,来自于COCO数据集,可以参考微软的论文,此处用于评估板端部署的精度下降程度。
4. 关于后处理: 目前在X5上使用Python重构的后处理, 仅需要单核心单线程串行5ms左右即可完成, 也就是说只需要占用2个CPU核心(200%的CPU占用, 最大800%的CPU占用), 每分钟可完成400帧图像的后处理, 后处理不会构成瓶颈.

### Performance Data (Summary) (X3 historical benchmark)

| Model (Official) | Size (px) | Classes | Params (M) | Throughput (FPS) | Post Process Time (Python) |
|---------|---------|-------|-------------------|--------------------|---|
| YOLOv8n | 640×640 | 80 | 3.2 | 263.6 | 5 ms |
| YOLOv8s | 640×640 | 80 | 11.2 | 194.9 | 5 ms |
| YOLOv8m | 640×640 | 80 | 25.9 | 35.7 | 5 ms |
| YOLOv8l | 640×640 | 80 | 43.7 | 17.9 | 5 ms |
| YOLOv8x | 640×640 | 80 | 68.2 | 11.2 | 5 ms |

Note: Detailed performance data is at the end of the document.

### Performance Data (X3 historical benchmark)

| Model | Size (px) | Num. Classes | Params (M) | FP Precision | Q Precision | Latency/Throughput (Single-threaded) | Latency/Throughput (Multi-threaded) | Post Process Time (Python) |
|---------|---------|-------|---------|---------|----------|--------------------|--------------------|--------------|
| YOLOv8n | 640×640 | 80 | 3.2 | 37.3 |  | 5.6ms/178.0FPS(1 thread) | 7.5ms/263.6FPS(2 threads) | 5 ms |
| YOLOv8s | 640×640 | 80 | 11.2 | 44.9 |  | 12.4ms/80.2FPS(1 thread) | 21ms/94.9FPS(2 threads) | 5 ms |
| YOLOv8m | 640×640 | 80 | 25.9 | 50.2 |  | 29.9ms/33.4FPS(1 thread) | 55.9ms/35.7FPS(2 threads) | 5 ms |
| YOLOv8l | 640×640 | 80 | 43.7 | 52.9 |  | 57.6ms/17.3FPS(1 thread) | 111.1ms/17.9FPS(2 threads) | 5 ms |
| YOLOv8x | 640×640 | 80 | 68.2 | 53.9 |  | 90.0ms/11.0FPS(1 thread) | 177.5ms/11.2FPS(2 threads) | 5 ms |

Object Detection (Open Image V7)

### Performance Data (X3 historical benchmark)

| Model | Size (px) | Num. Classes | Params (M) | FP Precision | Q Precision | Average Frame Latency/Throughput (Single-threaded) | Average Frame Latency/Throughput (Multi-threaded) |
|------|------|-------|---------|---------|-------------------|--------------------|--------------------|
| YOLOv8n | 640×640 | 600 | 3.5 | 18.4 | - | - | - |
| YOLOv8s | 640×640 | 600 | 11.4 | 27.7 | - | - | - |
| YOLOv8m | 640×640 | 600 | 26.2 | 33.6 | - | - | - |
| YOLOv8l | 640×640 | 600 | 44.1 | 34.9 | - | - | - |
| YOLOv8x | 640×640 | 600 | 68.7 | 36.3 | - | - | - |

Notes:
1. The X5 is in its optimal state: CPU is 8 × A55 @ 1.8G with full-core Performance scheduling, BPU is 1 × Bayes-e @ 1G with a total equivalent int8 computing power of 10 TOPS.
2. Single-threaded latency is for a single frame, single thread, and single BPU core, representing the ideal delay for BPU inference of a single task.
3. Four-thread engineering frame rate is when four threads simultaneously feed tasks to the dual-core BPU. In general engineering scenarios, four threads can minimize single-frame latency while fully utilizing all BPU cores at 100%, achieving a good balance between throughput (FPS) and frame latency. X5 BPU overall is more powerful, generally 2 threads can eat BPU full, frame delay and throughput are very good.
4. Eight-thread extreme frame rate is when eight threads simultaneously feed tasks to the dual-core BPU on the X3, aiming to test the BPU’s extreme performance. Typically, four cores are already saturated; if eight threads perform significantly better than four, it suggests that the model structure needs to improve the "compute/memory access" ratio, or that DDR bandwidth optimization should be selected during compilation.
5. FP/Q mAP: 50-95 precision is calculated using pycocotools and comes from the COCO dataset. This can refer to Microsoft’s paper, and is used here to assess the degree of accuracy degradation for deployment on the board.
6. Run the following command to test the bin model throughput on the board
```bash
hrt_model_exec perf --thread_num 2 --model_file yolov8n_detect_bayese_640x640_nv12_modified.bin
```
7. Regarding post-processing: At present, the post-processing of Python reconstruction on X5 only requires a single-core single-thread serial about 5ms to complete, that is, it only needs to occupy 2 CPU cores (200% CPU usage, maximum 800% CPU usage), and can complete 400 frames of image post-processing per minute, and post-processing will not constitute a bottleneck.

### RDK X3 & RDK X3 Module (X3 historical benchmark)

| 模型(公版) | 尺寸(像素) | 类别数 | 参数量 | BPU吞吐量 | 后处理时间(Python) |
|---------|---------|-------|---------|---------|----------|
| YOLOv8n | 640×640 | 80 | 3.2 M | 34.1 FPS | 6 ms |

注: 详细性能数据见文末.

### RDK X3 & RDK X3 Module (X3 historical benchmark)

| 模型 | 尺寸(像素) | 类别数 | 参数量 | 浮点精度<br/>(mAP:50-95) | 量化精度<br/>(mAP:50-95) | BPU延迟/BPU吞吐量(线程) |  后处理时间<br/>(Python) |
|---------|---------|-------|---------|---------|----------|--------------------|--------------------|
| YOLOv8n | 640×640 | 80 | 3.2 M | 37.3  | - | 99.8 ms / 10.0 FPS (1 thread) <br/> 102.0 ms / 19.6 FPS (2 threads)<br/> 131.4 ms / 30.2 FPS (4 threads)<br/> 231.0 ms / 34.1 FPS (8 threads) | 6 ms |

说明:
说明:
1. BPU延迟与BPU吞吐量。
 - 单线程延迟为单帧,单线程,单BPU核心的延迟,BPU推理一个任务最理想的情况。
 - 多线程帧率为多个线程同时向BPU塞任务, 每个BPU核心可以处理多个线程的任务, 一般工程中4个线程可以控制单帧延迟较小,同时吃满所有BPU到100%,在吞吐量(FPS)和帧延迟间得到一个较好的平衡。X5的BPU整体比较厉害, 一般2个线程就可以将BPU吃满, 帧延迟和吞吐量都非常出色。
 - 表格中一般记录到吞吐量不再随线程数明显增加的数据。
 - BPU延迟和BPU吞吐量使用以下命令在板端测试
```bash
hrt_model_exec perf --thread_num 2 --model_file yolov8n_detect_bayese_640x640_nv12_modified.bin
```
2. 测试板卡均为最佳状态。
 - X5的状态为最佳状态：CPU为8 × A55@1.8G, 全核心Performance调度, BPU为1 × Bayes-e@10TOPS.
```bash
sudo bash -c "echo 1 > /sys/devices/system/cpu/cpufreq/boost"  # 1.8Ghz
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor" # Performance Mode
```
 - X3的状态为最佳状态：CPU为4 × A53@1.8G, 全核心Performance调度, BPU为2 × Bernoulli2@5TOPS.
```bash
sudo bash -c "echo 1 > /sys/devices/system/cpu/cpufreq/boost"  # 1.8Ghz
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor" # Performance Mode
```
3. 浮点/定点mAP：50-95精度使用pycocotools计算,来自于COCO数据集,可以参考微软的论文,此处用于评估板端部署的精度下降程度。
4. 关于后处理: 目前在X5上使用Python重构的后处理, 仅需要单核心单线程串行5ms左右即可完成, 也就是说只需要占用2个CPU核心(200%的CPU占用, 最大800%的CPU占用), 每分钟可完成400帧图像的后处理, 后处理不会构成瓶颈.


## Accuracy Data

### RDK S100 / RDK S100P

Object Detection (COCO2017)

| Model | Pytorch | YUV420SP<br>Python | YUV420SP<br>C/C++ | NCHWRGB<br>C/C++ |
|---|---:|---:|---:|---:|
| YOLOv13n | 0.342 | 0.319 (93.27%) | (%) | (%) |
| YOLOv13s | 0.402 | 0.381 (94.78%) | (%) | (%) |
| YOLOv13l | 0.458 | 0.443 (96.73%) | (%) | (%) |
| YOLOv13x | 0.473 | 0.458 (96.83%) | (%) | (%) |

## Accuracy Test Notes

1. The accuracy data is computed with the official unmodified Microsoft `pycocotools`, using `Average Precision (AP) @[ IoU=0.50:0.95 | area=all | maxDets=100 ]`.
2. The evaluation uses all 5000 images from `COCO2017 val`, runs inference on board, dumps JSON results, and evaluates them with `pycocotools` using `score=0.25` and `nms=0.7`.
3. `pycocotools` AP is usually lower than the Ultralytics built-in numbers due to different area calculation rules. The important signal here is the relative gap between floating-point and quantized models.
4. Some accuracy loss appears when converting NCHW RGB888 inputs to YUV420SP inputs for BPU deployment because of color space conversion. This can be reduced if that transformation is considered during training.
5. Python and C/C++ runtime accuracy may differ slightly because of differences in memory copy and floating-point handling.
6. Evaluation scripts can be referenced here: <https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/tools/eval_pycocotools>
7. The table reflects PTQ results compiled with 50 calibration images. It represents a practical first-pass compilation baseline instead of the upper bound of achievable accuracy.

## Result Check

With the default `test_data/kite.jpg`, the runtime should output boxes that match the visible objects in the image and save a rendered result image. If the boxes are clearly misplaced, all classes are wrong, scores are abnormally low, or the result is empty, check:

- the ONNX export output order
- whether `remove_node_name` matches the current ONNX
- whether `score-thres` and `nms-thres` match the evaluation assumptions

<a id="boundaries"></a>
## 能力边界

这些脚本不验证模型转换正确性、不支持任意自定义类别顺序，也不测量完整应用性能。分类会按上述规则跳过不可读或无标签图像，必须报告实际处理数量。OBB需另行使用DOTA评分器。发布基准与既有固定图板测属于各自源码版本的参考，仅覆盖其记录的任务与尺寸；新板测及全数据集测量按本指南命令执行。

DFL 姿态接口返回关键点概率。当前 COCO JSON 序列化保留历史规则：概率 >0 时 v=1，否则为 0；这不是 0.5 可见性筛选，不会删除低置信度点。绘制阈值与评测序列化是不同操作。


### X5 YOLO26 NV12 测量记录（中文来源）

[Source measurement table](https://github.com/D-Robotics/rdk_model_zoo/blob/cb86079ae5befcef9ca50fb46c8a6d8980106dec/samples/vision/ultralytics_yolo26/evaluator/README_cn.md).

该来源将 YOLO26m 对应到 51.1 ms / 24.8 FPS，将 YOLO26l 对应到 40.1 ms / 19.5 FPS；上方另一份 X5 表的单线程延迟对应关系不同。两份来源存在冲突，不能据此确定哪一份是修正值。输入为 NV12，各行列出尺寸、类别数与线程数。

| Device | Model | Size <br> (Pixels) | Classes | BPU Task Latency / <br> BPU Throughput (Threads) | CPU Latency | params <br> (M) | FLOPs <br> (B) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| X5 | YOLO26n Detect | 640x640 | 80 | 11.6 ms / 86.3 FPS (1 thread) <br> 19.1 ms / 104.3 FPS (2 threads) | - | - | - |
| X5 | YOLO26s Detect | 640x640 | 80 | 20.9 ms / 47.7 FPS (1 thread) <br> 37.8 ms / 52.8 FPS (2 threads) | - | - | - |
| X5 | YOLO26m Detect | 640x640 | 80 | 51.1 ms / 24.8 FPS (1 thread) <br> 76.1 ms / 26.1 FPS (2 threads) | - | - | - |
| X5 | YOLO26l Detect | 640x640 | 80 | 40.1 ms / 19.5 FPS (1 thread) <br> 98.0 ms / 20.3 FPS (2 threads) | - | - | - |
| X5 | YOLO26x Detect | 640x640 | 80 | 103.3 ms / 9.6 FPS (1 thread) <br> 202.0 ms / 9.8 FPS (2 threads) | - | - | - |
| X5 | YOLO26n Seg | 640x640 | 80 | 15.5 ms / 64.3 FPS (1 thread) <br> 22.8 ms / 87.6 FPS (2 threads) | - | - | - |
| X5 | YOLO26n Pose | 640x640 | 80 | 12.5 ms / 79.6 FPS (1 thread) <br> 20.1 ms / 98.7 FPS (2 threads) | - | - | - |
| X5 | YOLO26n Cls | 224x224 | 1000 | 1.1 ms / 906.0 FPS (1 thread) <br> 1.7 ms / 1156.8 FPS (2 threads) | - | - | - |
