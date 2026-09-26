# Ultralytics YOLO 模型评估

[English](README.md) | 简体中文

评估器复用[Python运行时](../runtime/python/README.md)的任务实现。检测、实例分割、姿态估计使用COCO指标；分类使用ImageNet Top-1/Top-5；YOLO26旋转框只导出预测。本目录不下载模型或数据集、不编译模型，也不测量纯BPU延迟。

<a id="dataset"></a>
## 数据集准备

准备与模型类别顺序一致的验证集。获取和整理方法见[X5 COCO（英文）](../../../../platforms/x5/datasets/coco/README.md)、[S COCO（英文）](../../../../platforms/s/datasets/coco/README.md)、[X5 ImageNet](../../../../platforms/x5/datasets/imagenet/README_cn.md)、[S ImageNet](../../../../platforms/s/datasets/imagenet/README_cn.md)。数据集不随仓库分发，使用时遵守各自许可。

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

输出旋转矩形和多边形坐标，**不计算DOTA AP**。`--label-path`仅兼容旧命令，不参与评分。除非经过审查的自定义输出协议要求，否则保留默认角度符号和偏移。

### 批量评估

`eval_batch.py`只扫描`--model-dir`直接包含的文件，通过`_detect_`、`_seg_`、`_pose_`、`_cls_`、`_obb_`识别任务。S模型应直接指定nash-e/m/p子目录。每次使用相同任务/数据集的模型目录，额外参数会传给所有选中的评估器。

```bash
python3 samples/vision/ultralytics_yolo/evaluator/eval_batch.py \
  --platform x5 --family yolov8 --model-dir /models/coco-detect \
  --image-dir /data/coco/val2017 \
  --annotation /data/coco/annotations/instances_val2017.json --suffix val2017
```

批量命令打印选择结果后交互确认；`--yes`跳过确认。JSON写在各模型旁边。完整验证集可能耗时数小时，取决于板型、模型和存储，不保证固定时长。先用`--limit 10`检查流程，再取消限制跑全量；子集结果必须标注为子集。

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

COCO通过`pycocotools.COCOeval`的bbox/segm/keypoints计算AP/AR，只评估所选图像ID。分类top1/top5为处理过且有标签图像的正确比例，不是百分数。OBB JSON是中间预测，不是精度指标。运行墙钟时间包含Python与数据处理，不等于BPU推理延迟。

记录源码提交、目标/系统/SDK、制品摘要、数据集/划分、实际处理数量、resize策略和阈值后再比较结果。这些条件改变时，不能直接对比历史表。固定图片源/统一一致性验证也不等于数据集精度测量。

<a id="outputs"></a>
## 输出与成功判断

检测JSON包含COCO image_id、category_id、`[x,y,width,height]`和score；分割保存编码掩码，姿态保存关键点和分数。分类JSON包含total、top1、top5、elapsed_sec。旋转框包含file_name、image_id、category_id、score、rrect、polygon；矩形角度为弧度，多边形为原图像素坐标。

当前脚本会覆盖同名结果，应每次选择唯一输出路径，并保存含COCO指标摘要的stdout。退出0仅代表流程完成；无标注、空预测、分类total=0都不是精度通过。COCO空预测写为`[]`并明确跳过指标计算。

<a id="reference-results"></a>
## 参考结果与验证范围

源分支完整基准表在仓库内保留：[X5评估与基准](../../../../platforms/x5/samples/vision/ultralytics_yolo/evaluator/README_cn.md)、[S评估与基准](../../../../platforms/s/samples/vision/ultralytics_yolo/evaluator/README_cn.md)、[X5 YOLO26](../../../../platforms/x5/samples/vision/ultralytics_yolo26/README_cn.md)、[S YOLO26](../../../../platforms/s/samples/vision/ultralytics_yolo26/README_cn.md)。制品和历史测量事实见[X5清单](../../../../docs/release/x5/)及[S清单](../../../../docs/release/s/)。

这些是历史发布记录，不是当前统一代码的新测量。[sample说明](../README_cn.md)列出YOLOv8n/YOLO26n检测代表制品的板端对照范围。本次非板端工作没有新执行全数据集精度、延迟或板测。

## 故障排查与代码入口

- 缺pycocotools：安装到运行评估脚本的解释器，包括仅导出COCO预测的情况。
- 缺真值或类别错位：核对named/ordered/synset、偏移和模型类别顺序，不猜测任意文件名的标签。
- COCO文件名/ID错误：使用配套图像和标注；无标注导出要求数字文件名主体。
- 空输出：先用runtime检查单图，再核对阈值、任务、制品；保留空输出事实，不能据此宣称精度。
- 写文件失败：创建可写父目录并显式指定输出路径。

`eval_common.py`负责共享参数、类别映射与图像选择，`eval_yolo_*.py`分别负责指标/输出格式并调用runtime，`eval_batch.py`仅派发命令。导出与张量布局变更属于[conversion](../conversion/README_cn.md)及runtime binding，不应混入指标代码。

<a id="boundaries"></a>
## 能力边界

这些脚本不验证模型转换正确性、不支持任意自定义类别顺序，也不测量完整应用性能。分类会按上述规则跳过不可读或无标签图像，必须报告实际处理数量。OBB需另行使用DOTA评分器。发布基准与既有固定图板测属于各自源码版本的参考，不能作为当前所有任务和尺寸的验收。本次未执行的新板测及全数据集测量仍为待办。
