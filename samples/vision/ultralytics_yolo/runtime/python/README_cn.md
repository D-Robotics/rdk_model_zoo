# Python 运行时

[English](README.md) | 简体中文

<a id="overview"></a>
## Python 推理

运行 Ultralytics 检测、分割、姿态、分类与旋转框模型。`main.py` 取得模型类与配置，显式构造 `Model(config)` 并调用 `predict`；各任务文件包含对应的前处理、推理与解码。

<a id="directory"></a>
## 目录结构

```text
python/
├── classification_decode.py  # 模型阶段与预测
├── decode.py  # 任务输出解码
├── detect.py  # 模型阶段与预测
├── detection_io.py  # 模型阶段与预测
├── geometry.py  # 图像几何与坐标变换
├── main.py  # 命令行入口：构造模型并调用 predict
├── model_binding.py  # 模型选择与物理张量契约
├── model_runner.py  # Runtime 加载与原始张量执行
├── obb_decode.py  # 模型阶段与预测
├── pose_decode.py  # 模型阶段与预测
├── run.sh  # 定位 Python 入口并转发参数
├── segmentation_decode.py  # 模型阶段与预测
├── tensor_io.py  # 模型阶段与预测
├── yolo26_det.py  # 模型阶段与预测
├── yolo26_obb.py  # 模型阶段与预测
├── yolo26_pose.py  # 模型阶段与预测
├── yolo26_seg.py  # 模型阶段与预测
├── yolo_assets.py  # 模型阶段与预测
├── yolo_cli.py  # 参数、模型选择与结果展示
├── yolo_cls.py  # 模型阶段与预测
├── yolo_dispatch.py  # 模型阶段与预测
├── yolo_download.py  # 模型阶段与预测
├── yolo_input.py  # 模型阶段与预测
├── yolo_platform.py  # 模型阶段与预测
├── yolo_pose.py  # 模型阶段与预测
├── yolo_runtime.py  # 模型阶段与预测
├── yolo_seg.py  # 模型阶段与预测
└── yolo_v10detect.py  # 模型阶段与预测
```

<a id="environment"></a>
## 板端准备

将与板卡目标一致的编译制品复制到板端或下载到 Manifest 指定位置。X5 绑定
一个 packed NV12 输入；S 系列绑定命名的 NHWC Y、UV 输入。运行时在推理前
读取模型元数据并检查 batch、尺寸、类型和输出协议；文件名后缀不能代替
元数据检查。

缺少默认模型时，入口会在板端下载默认模型；显式传入 `--model-path`
时绝不下载。主机上想只检查路径而不加载 `hbm_runtime`，可传显式
`--platform` 后使用 `--dry-run` 或 `--list-models`；`--download` 只准备
Manifest 制品然后退出。

板端系统镜像需要 Python 3、NumPy、OpenCV、SciPy 以及匹配的
`hbm_runtime`。模型文件和测试图片必须可由当前用户读取。运行时不会在
请求制品缺失时静默切换平台或模型。

<a id="usage"></a>
## 最短检测命令

在匹配的板卡上从仓库根目录运行：

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolov8 --task detect \
  --test-img samples/vision/ultralytics_yolo/test_data/bus.jpg \
  --img-save-path /tmp/yolov8n-x5-detect.jpg
```

S100/S100P/S600 使用对应的 `--platform`。如果使用自己准备的制品，显式
传入路径：

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s600 --family yolo11 --task detect \
  --model-path /models/yolo11n_nashp_640x640_nv12.hbm \
  --test-img /data/image.jpg --img-save-path /tmp/yolo-s600.jpg
```

便捷脚本在任务名后接受同样的参数，例如
`bash samples/vision/ultralytics_yolo/runtime/python/run.sh detect --platform s100`。不传平台时它会从
`/sys/class/boardinfo` 检测板卡；主机上的列表、下载和 dry-run 可显式选
目标。真正推理若板卡未知或目标不匹配，会在加载模型前停止。

以下为其他任务及默认入口示例。无参数时自动检测板卡，运行 yolo11 默认尺度检测，默认图片是 bus.jpg。OBB 需自行提供航拍图片；在对应板卡上运行各行，不要在一块板上混跑所有目标。

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolo11 --task seg --img-save-path /tmp/yolo-seg.jpg
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s100 --family yolov8 --task pose --img-save-path /tmp/yolo-pose.jpg
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s600 --family yolo26 --task cls \
  --test-img samples/vision/ultralytics_yolo/test_data/zebra_cls.jpg --topk 5
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolo26 --task obb \
  --test-img /data/aerial.jpg --img-save-path /tmp/yolo-obb.jpg
```

## 任务和模型选择

`--family`、`--model-size` 选择 Manifest 中的组合；省略尺寸使用该家族和
平台的默认值。`--asset-id` 接受 `--list-models` 打印的精确
`group:sample:filename` 引用。`--model-path` 直接选择本地制品且不会下载；
自定义文件名应同时指定 `--family`，以便选择正确的解码器。

常见任务名是 `detect`、`seg`、`pose`、`cls`、`obb`，但需以所选家族是否
发布该任务为准。YOLOv8/YOLO11 检测使用三层 DFL logits（`reg=16`）；
YOLO26 检测使用 stride 8/16/32 的直接 LTRB，因此有独立绑定和解码器。
不能把 YOLO26 制品交给 DFL 解码器，也不能根据输出序号猜协议。

已注册的模型系列与解码协议：

| 系列 | 任务 | 解码器 |
|---|---|---|
| `yolo26` | detect, seg, pose, cls, obb | direct-LTRB |
| `yolov5u` | detect | DFL |
| `yolov8` | detect, seg, pose, cls | DFL |
| `yolov9` | detect, seg | DFL |
| `yolov10` | detect | DFL；S 系列走 NMS-free 解码 |
| `yolo11` | detect, seg, pose, cls | DFL（默认系列） |
| `yolo12` | detect | DFL |
| `yolov13` | detect | DFL |

<a id="parameters"></a>
## 命令行参数

默认值列为解析器原值，`null` 表示随后按平台、任务或制品解析，并不表示该功能禁用。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--platform` / `--target` | str | `null` | 自动检测；auto/x5/s100/s100p/s600。准备流程可指定目标，实际推理必须匹配本机。 |
| `--task` | str | `detect` | detect/seg/pose/cls/obb；可用范围取决于系列。 |
| `--family` | str | `null` | 识别本地文件名中的系列，否则 yolo11；显式系列冲突会拒绝。 |
| `--model-size` | str | `null` | 使用发布清单中的系列/任务默认尺度，见模型说明。 |
| `--model-path` | str | `null` | 显式编译文件路径，不自动下载。 |
| `--asset-id` | str | `null` | 精确 group:sample:filename 引用，约束清单选择。 |
| `--input-shape` | HxW | `null` | 仅在缺少运行时尺寸元数据时指定 H×W。 |
| `--test-img` | str | `samples/vision/ultralytics_yolo/test_data/bus.jpg` | 由 OpenCV 读取的 BGR 图片。 |
| `--label-file` | str | `null` | 官方默认：detect/seg 用 COCO，pose 用单一 `person` 标签，cls 用 ImageNet，obb 用 DOTA。显式 `--model-path` 视为自定义模型：不给本项时，绘制/打印类别的路径显示类别 ID（pose 绘制仍为分数/关键点，不含类别文字）；给出时标签数必须等于已绑定模型类别数（支持 json 字典/列表与逐行格式）。 |
| `--img-save-path` | str | `result.jpg` | detect/seg/pose/obb 绘制结果，相对调用目录；cls 不写图片。 |
| `--score-thres` | float | `0.25` | 检测置信度过滤，分类不使用。 |
| `--nms-thres` | float | `null` | 随目标/任务解析；X5 检测 0.70，S 检测 0.45；S YOLOv10 检测无 NMS。 |
| `--resize-type` | int | `null` | 0 拉伸、1 letterbox；解析后的默认值见下文。 |
| `--classes-num` | int | `null` | 使用任务配置默认值；只对 detect/seg/obb 按实际模型类别数覆盖。 |
| `--strides` | comma-separated ints | `[8, 16, 32]` | 特征图 stride，输入如 8,16,32。 |
| `--mc` | int | `32` | 分割 mask 系数数量；YOLO26 固定 32。 |
| `--angle-sign` | float | `1.0` | OBB 角度乘数。 |
| `--angle-offset` | float | `0.0` | OBB 角度偏移，单位为度。 |
| `--regularize` | int | `1` | OBB 旋转框规范化，0 或 1。 |
| `--reg` | int | `16` | DFL 回归 bin 数；修改此值不能改变 direct-LTRB 图。 |
| `--nkpt` | int | `17` | DFL 系列姿态点数；YOLO26 固定 17。 |
| `--topk` | int | `5` | 分类输出数量。 |
| `--kpt-conf-thres` | float | `0.5` | 姿态绘制的可见性阈值，不是张量绑定参数。 |
| `--priority` | int | `0` | BPU 调度优先级，0–255。 |
| `--bpu-cores` | space-separated ints | `[0]` | 一个或多个核心编号，例如 --bpu-cores 0 1。 |
| `--list-models` | flag | `false` | 打印支持模型及精确引用后退出。 |
| `--dry-run` | flag | `false` | 只解析选择，不下载、不推理。 |
| `--download` | flag | `false` | 准备所选发布模型后退出，不推理。 |

非分类任务默认 letterbox。分类中，YOLO26 全目标默认拉伸；其他系列 X5 默认 letterbox，S 默认拉伸。输入尺寸、类别数及 `--reg`、`--strides`、`--mc` 须与模型输出协议一致。YOLO26 非分类任务拒绝偏离 16/17/32 的 DFL/关键点/mask 覆盖参数。YOLOv13 仅在 X5 发布。

<a id="results"></a>
## 输出结果

退出码 0 表示命令完成；空检测列表仍可能是有效结果，并非模型失败。detect/seg/pose/obb 输出由 `--img-save-path` 指定，CLI 会创建父目录并打印 `[Saved]`；已有同名结果会被覆盖。cls 只打印分类，不写结果图。阶段接口不负责绘制或保存。

| 任务 | `predict` 返回值 | 坐标与含义 |
|---|---|---|
| detect | `DetectionResult(boxes_xyxy, scores, class_ids)`，兼容三元组解包 | `(N,4)` 原图像素框、`(N,)` 置信度与从 0 开始的类别 ID |
| seg | `(boxes, scores, ids, masks)` | 原图像素框及对应实例 mask；保持实例顺序配对 |
| pose | `(boxes, scores, ids, xy, confidence)` | 原图像素框和点坐标；发布姿态模型为 17 点，置信度独立返回 |
| cls | `(class_id, probability)` 列表 | Softmax 后按分数排序的 Top-K，不是原始 logits |
| obb | 字典列表：`rrect`、`score`、`id` | `rrect=(cx,cy,w,h,angle)`；尺寸/中心在原图坐标中，角度为弧度 |

数据集精度与性能测量见 [evaluator](../../evaluator/README_cn.md)。本地文件选择与发布范围见 [model](../../model/README_cn.md)。

<a id="integration-example"></a>
## 库接口

DFL 检测流程在 `detect.py`（`YoloDetect`）。`predict` 接受本地图片路径或 BGR `uint8` 数组，
且不会原地修改数组。在匹配的 S600 板卡上从仓库根目录执行下例，并先将
模型路径替换为本地 YOLO11 检测制品：

```python
from samples.vision.ultralytics_yolo.runtime.python.detect import (
    YoloDetect, YoloDetectConfig)
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import (
    resolve_platform)

profile = resolve_platform("s600")
config = YoloDetectConfig(
    model_path="/models/yolo11n_nashp_640x640_nv12.hbm",
    platform=profile,
)
detector = YoloDetect(config)
result = detector.predict("samples/vision/ultralytics_yolo/test_data/bus.jpg")
boxes, scores, class_ids = result
print(boxes.shape, scores.shape, class_ids.shape)
```

三阶段也可显式驱动——`predict` 恰好按各自 transform 串联这三步：

```python
import cv2
import numpy as np

bgr_image = cv2.imread("samples/vision/ultralytics_yolo/test_data/bus.jpg")
if bgr_image is None:
    raise FileNotFoundError("Cannot read test image")
prepared = detector.preprocess(bgr_image)
raw = detector.infer(prepared)
result = detector.post_process(raw, transform=prepared.transform)
for staged, predicted in zip(result, detector.predict(bgr_image)):
    np.testing.assert_allclose(staged, predicted)
```

`YoloDetect` 支持注入 runner，用于接入其他运行时加载器。runner
负责模型执行；图像几何、协议绑定、DFL 解码、按类别 NMS 和坐标还原由共用
任务实现负责。`pre_process` / `forward` / `post_process` 是可读阶段方法的
薄别名（同一实现）。`YOLO26Detect` 共用图片准备与 Runtime 流程，使用直接 LTRB 解码。

<a id="stage-io"></a>
## 代码流程

`YoloDetect`（`detect.py`）和 `YOLO26Detect` 直接串联三个阶段，不保存“上一张图片”的 context。
准备 B 不会覆盖 A 的几何信息；分别保留 prepared，并使用其对应 transform。SDK 调用仍需
由调用方串行安排；逐调用 context 不提供 SDK 推理的线程安全保证。
所有任务类 —— 检测、分类、分割、姿态、OBB，DFL 与 YOLO26 两族 —— 的阶段
统一命名为 `preprocess` / `infer` / `postprocess`，
`pre_process` / `forward` / `post_process` 为薄别名（每个阶段只有一个实现，
在各自任务模块内可见）。

- `preprocess(图片或路径, image_format="BGR")`（别名 `pre_process`）要求非空 uint8 H×W×3 BGR 或可读的本地图片路径，返回 `PreparedDetection.tensors` 和冻结的 `.transform`。后者包含原图/模型/实际缩放尺寸、整数 padding 与横纵缩放比例。X5 张量为 packed NV12；S 为 Y `(1,H,W,1)` 和 UV `(1,H/2,W/2,2)`，H/W 来自模型 metadata。
- `infer(prepared)`（别名 `forward`）只调用一次 runner，返回以角色名索引的 `RawOutputs`。绑定的 runner 校验物理 shape、dtype 和有限值，不反量化、不激活、不解码、不改变布局。数组保留 SDK dtype，借用 SDK 缓冲区；须先完成后处理再发起下一次 SDK 调用，或主动复制需要长期保留的原始数组。
- `postprocess(raw, transform=prepared.transform)`（别名 `post_process`）进行 sigmoid/DFL 或 LTRB 解码、适用的 NMS 和坐标还原。检测与 DFL 分割/姿态绑定均要求模型直接提供浮点输出；整数或 SCALE metadata 在加载时拒绝，后处理不执行手动反量化。
- `predict(图片或路径)` 串联这些方法并返回自有结果数组。注入 runner 返回普通语义映射时，数值须已是浮点；物理浮点输出使用绑定后的 raw 容器。

prepared 支持 `[model_name]` 映射访问，`infer`/`forward(prepared)` 会取出
`.tensors`。显式 `post_process(outputs, 原宽, 原高)` 可无缓存重建同一几何；同时给宽高和 transform
时必须一致。逐图几何保存在 prepared 对象上，没有实例级的 last-transform 属性。
DFL 分割、姿态、分类和 YOLO26 OBB 的阶段接口与完整例子见下文。

```text
main.py
  -> resolve_target / 平台 Manifest 选择（yolo_cli 负责列表、dry-run、
     下载准备与结果展示）
  -> yolo_dispatch.prepare_runtime_model -> Model(config) -> model.predict(image)
  -> ModelRunner + ModelBinding（输入/输出契约，共享 SDK 会话）
  -> geometry.resize_with_transform + NV12 输入绑定
  -> YoloDetect 或 YOLO26Detect 解码 + NMS
  -> DetectionResult -> yolo_cli.present_result -> --img-save-path
```

`model_binding.py` 按既定形状/类型契约识别输出角色，编译器枚举名称
只是物理名称。`geometry.py` 记录实际整数缩放和 padding，使框还原使用
同一个变换。检测张量协议见
[`DETECTION_CONTRACT.md`](../../DETECTION_CONTRACT.md)。

<a id="segmentation-api"></a>
## DFL 分割库接口

YOLOv8/9/11 分割使用 `YoloSeg`；YOLO26 分割采用不同的直接框协议，不适用本例。
在匹配的 S100 板卡上，从仓库根目录执行；先按 [模型说明](../../model/README_cn.md)
准备兼容的本地分割制品，并将例子中的绝对路径换成实际路径。显式设置
本例采用 S 平台默认 `nms_thres=0.45`；需要调整时显式覆盖。

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import resolve_platform
from samples.vision.ultralytics_yolo.runtime.python.yolo_seg import YoloSeg, YoloSegConfig

image_path = Path("samples/vision/ultralytics_yolo/test_data/bus.jpg")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)
segmenter = YoloSeg(YoloSegConfig(
    model_path="/models/yolo11n_seg_nashe_640x640_nv12.hbm",
    platform=resolve_platform("s100"),
    nms_thres=0.45,
))
prepared = segmenter.preprocess(image)
raw = segmenter.infer(prepared)
boxes, scores, ids, masks = segmenter.postprocess(raw, transform=prepared.transform)
expected = segmenter.predict(image)
for staged, predicted in zip((boxes, scores, ids), expected[:3]):
    np.testing.assert_allclose(staged, predicted)
assert len(masks) == len(expected[3])
for staged, predicted in zip(masks, expected[3]):
    np.testing.assert_array_equal(staged, predicted)
print(boxes.shape, scores.shape, ids.shape, [mask.shape for mask in masks])
```

三个方法与检测共用显式 `PreparedDetection`/`RawOutputs` 传输接口，
`YoloSeg(config, runner=...)` 支持注入 runner。工厂读取实际输入/输出 metadata，
在加载真实 SDK 前核对目标身份；推理不下载模型。`preprocess`（别名 `pre_process`）
接收非空 BGR
uint8 H×W×3；`postprocess`（别名 `post_process`）接收 `prepared.transform`，也支持显式传入的
`(原宽, 原高)` 参数。缺失或冲突的几何会报错，不保存上一张图片的状态。

有限输出协议为 stride 8/16/32 的 NHWC 类别 logits `(1,H/s,W/s,C)`、DFL 框
logits `(1,H/s,W/s,64)`、系数 `(1,H/s,W/s,32)`，以及 stride-4 原型
`(1,H/4,W/4,32)` 或 `(1,32,H/4,W/4)`；要求发布模型所用的方形输入。
按 shape 或显式声明的 `DFLSegmentationContract(output_roles=...)` 绑定角色，
不依赖输出枚举顺序。缺失、错误或歧义 metadata、非有限张量、所有整数输出
均拒绝；模型须直接提供浮点张量，后处理只转换 NCHW 原型布局，不做反量化。
注入的普通角色映射须已是有限浮点 NHWC 数组。

返回 `(boxes, scores, ids, masks)`：自有 float32 `(N,4)` xyxy 原图框，裁至
`[0,width]`/`[0,height]`；float32 `(N,)` 概率；int64 `(N,)` 类别 ID；N 个
uint8 **ROI mask**，不是全图 mask。值为 0/1，每个 mask 高宽为
`max(int(y2)-int(y1),0)` × `max(int(x2)-int(x1),0)`。空结果保留数组维度/类型，
`masks=[]`；退化 ROI 保留每个为零的轴，不人为补成 1 像素，始终与对应框配对。
解码使用系数/原型点积阈值 `>0.5`、Lanczos 缩放及可选 5×5 开运算
（`do_morph=True`）。返回的 mask 是逐框 ROI；上游 Ultralytics 的全图 mask
评测是另一种口径。
可选开运算结束后，将 Lanczos 过冲统一转回 0/1，保持前景范围不变。
置信度须为 `(0,1)` 内有限值，NMS 为 `[0,1]` 内有限值。

坐标还原使用实际整数缩放/padding；原型切片先裁至可见图片内容，避免负坐标从另一侧索引，也排除
letterbox padding。板端精度、SDK 兼容性、延迟与数据集指标需按评估指南另行测量。

<a id="pose-api"></a>
## DFL 姿态库接口

YOLOv8/11 姿态使用同一套三阶段接口。以下从仓库根目录在匹配的 S100 板卡执行，
模型必须先按 model 目录说明准备到本地，再替换示例绝对路径。省略 NMS 时采用平台
默认值（S 为 0.45、X5 为 0.70）。YOLO26 姿态使用直接框协议，不适用此 DFL 示例。

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import resolve_platform
from samples.vision.ultralytics_yolo.runtime.python.yolo_pose import YoloPose, YoloPoseConfig

image_path = Path("samples/vision/ultralytics_yolo/test_data/bus.jpg")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)
pose = YoloPose(YoloPoseConfig(
    model_path="/models/yolo11n_pose_nashe_640x640_nv12.hbm",
    platform=resolve_platform("s100"),
))
prepared = pose.preprocess(image)
raw = pose.infer(prepared)
result = pose.postprocess(raw, transform=prepared.transform)
for staged, predicted in zip(result, pose.predict(image)):
    np.testing.assert_allclose(staged, predicted)
boxes, scores, class_ids, keypoints_xy, visibility = result
visible = visibility[..., 0] >= 0.5
print(boxes.shape, keypoints_xy.shape, visibility.shape, visible.sum())
```

`YoloPose(config, runner=...)` 可注入运行器。输入、raw 缓冲区寿命和逐图 transform
约定与检测一致；`post_process(raw, 原宽, 原高)`（`postprocess` 的别名）仍兼容，
缺失或冲突的几何会报错。
9 个模型输出为 stride 8/16/32 的 NHWC `(1,H/s,W/s,1)` 类别 logits、
`(1,H/s,W/s,64)` DFL 框与 `(1,H/s,W/s,51)` 关键点（17 组 x/y/logit）。
要求方形输入、16 个 DFL bin、17 个 COCO 点，以及模型直接提供的有限浮点张量；
按 shape 绑定角色，不依赖物理输出枚举顺序，不支持手动反量化的重复版本。

结果为五元组：float32 `(N,4)` 原图框、float32 `(N,)` 检测概率、int64 `(N,)`
类别 ID（person=0）、float32 `(N,17,2)` 原图点坐标、float32 `(N,17,1)`
关键点概率。所有数组自有存储；空结果保留这些维度。NMS 对框和骨架使用同一索引。
坐标按实际缩放/padding 还原并裁至 `[0,width]`/`[0,height]`。关键点置信度只做一次
稳定 sigmoid；已经是概率，不要再次 sigmoid，也不要按 logits 的零阈值判断可见性。
示例的 0.5 只用于调用方可见性筛选，不改变返回的坐标或删掉关键点。

X5 适配器返回 `(boxes, scores, keypoints)`，最后一项为 `(N,17,3)`
x/y/概率。不提供独立的 S YOLO11Pose logits 返回接口。

<a id="classification-api"></a>
## 分类阶段接口

YOLOv8、YOLO11 和 YOLO26 共用分类流程。在匹配的 S600 板卡上，从仓库根目录运行
下例；先按[模型准备](../../model/README_cn.md)获取制品，并替换绝对路径。
输入尺寸以 SDK metadata 为准，文件名不能证明是 224 还是 640。尤其 S100/S100P
制品 ID 中的 `640` 对应下载 URL 中的 `224`，保留发布身份并读取实际 metadata。

```python
from pathlib import Path
import cv2
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import resolve_platform
from samples.vision.ultralytics_yolo.runtime.python.yolo_cls import YoloCls, YoloClsConfig

image_path = Path("samples/vision/ultralytics_yolo/test_data/zebra_cls.jpg")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)
classifier = YoloCls(YoloClsConfig(
    model_path="/models/yolo11n_cls_nashp_224x224_nv12.hbm",
    platform=resolve_platform("s600"),
    resize_type=0,
    topk=5,
))
inputs = classifier.preprocess(image)
raw = classifier.infer(inputs)
ranked = classifier.postprocess(raw)
assert ranked == classifier.predict(image)
print(ranked)
```

`preprocess`（别名 `pre_process`）返回嵌套 NV12 输入字典，分类无需携带几何还原
信息。`infer`（别名 `forward`）只调用
共用 runner 一次，`raw["logits"]` 保留物理浮点数组及 SDK 缓冲区引用。下一次推理前
完成后处理，或显式复制需要保存的 raw 数组。必须恰好有一个输出：1000 类向量及可选
单例维度，例如 `(1,1000)`、`(1,1000,1,1)`；多维输出的首维 batch 必须为 1。
额外输出、空间特征图、批输入、缺失
shape/dtype metadata、整数输出和 SCALE 描述均拒绝。

`postprocess`（别名 `post_process`）执行一次 SciPy Softmax 和 NumPy 降序排序，返回独立的
Python `(int 类别 ID, float 概率)` 列表。精确平局保持 NumPy 排序顺序。
Top-K 必须是正整数（不能是布尔值），超过类别数则返回全部类别；零、负数和非整数
值会报错。阶段内部不加载标签、不绘图、不写文件；
CLI 负责加载 ImageNet 标签并打印结果。

库接口所有系列/目标默认拉伸（`resize_type=0`）；CLI 对 X5 YOLOv8/11 分类显式
选择 letterbox，对 S 选择拉伸，YOLO26 全目标拉伸。比较库和 CLI 时须统一该参数。
数据集精度按评估指南测量。

<a id="v10-api"></a>
## S 系列 YOLOv10 不执行 NMS

S 系列 YOLOv10 复用 DFL 检测器的三阶段，绑定契约固定 `nms="none"`。
CLI 对 S100/S100P/S600 的 v10 选择该 `nms='none'` 适配器；X5 v10 走 DFL 解码加
按类别 NMS 的路径。两个分派按平台选择，互不通用。

在匹配的 S600 板卡上，先按[模型准备](../../model/README_cn.md)获取制品，
替换下例绝对路径，再从仓库根目录运行：

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import resolve_platform
from samples.vision.ultralytics_yolo.runtime.python.yolo_v10detect import (
    YoloV10Detect, YoloV10DetectConfig,
)

image_path = Path("samples/vision/ultralytics_yolo/test_data/bus.jpg")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)
detector = YoloV10Detect(YoloV10DetectConfig(
    model_path="/models/yolov10n_detect_nashp_640x640_nv12.hbm",
    platform=resolve_platform("s600"),
    score_thres=0.25,
))
prepared = detector.preprocess(image)
raw = detector.infer(prepared)
result = detector.postprocess(raw, transform=prepared.transform)
for staged, predicted in zip(result, detector.predict(image)):
    np.testing.assert_allclose(staged, predicted)
print(result.boxes.shape, result.scores.shape, result.class_ids.shape)
```

六个物理浮点 NHWC 输出按形状绑定为 stride 8/16/32 的分类及 16-bin DFL 框角色，
不按 SDK 枚举顺序猜测含义。模型输入必须为正方形。raw 缓冲区生命周期、显式逐图
transform、拥有独立存储的 `(boxes, scores, class_ids)` 结果与共用检测接口一致。
`pre_process` / `forward` / `post_process` 别名可用；
`post_process(raw, 原图宽, 原图高)` 显式重建几何，`pre_process` 返回支持映射访问的
`PreparedDetection` 对象。

所有达到置信度阈值的 anchor 都保留，包括重叠框。输出按 stride、再按网格遍历顺序
排列，每个 anchor 选一个最高分类。输出保持 anchor 顺序并包含重叠框，不执行 NMS、分数排序或 Top-K 截断。NMS 阈值在此不起作用；传入启用 NMS 的绑定契约会明确拒绝。
`score_thres` 使用共用检测器的有限 `[0,1]` 范围：零保留所有有限 logits 的 anchor，
一不保留任何 anchor。

坐标还原使用实际取整后的缩放宽高及 padding，而非理想浮点比例。
非正方形图片发生取整时，框坐标按修正后的真实几何输出。

<a id="yolo26-pose-api"></a>
## YOLO26 姿态阶段接口

YOLO26 姿态与 DFL 姿态共用前处理、raw runner、显式逐图 context、NMS 和坐标还原。
但它的九个输出在 stride 8/16/32 上分别为单类 logits、**四个直接 LTRB 距离**及
17 组 `(x,y,可见性 logit)`。关键点坐标公式为 `(offset + grid_center) * stride`；
DFL 姿态的倍数和偏移规则不同，绑定契约明确区分两者。文件名和输出枚举顺序都不能
用来猜测协议。

在匹配的 S600 板卡上，从仓库根目录运行下例；先按[模型准备](../../model/README_cn.md)
获取制品并替换绝对路径：

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import resolve_platform
from samples.vision.ultralytics_yolo.runtime.python.yolo26_pose import YOLO26Pose, YOLO26PoseConfig

image_path = Path("samples/vision/ultralytics_yolo/test_data/bus.jpg")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)
pose = YOLO26Pose(YOLO26PoseConfig(
    model_path="/models/yolo26n_pose_nashp_640x640_nv12.hbm",
    platform=resolve_platform("s600"),
    nms_thres=0.45,
))
prepared = pose.preprocess(image)
raw = pose.infer(prepared)
result = pose.postprocess(raw, transform=prepared.transform)
for staged, predicted in zip(result, pose.predict(image)):
    np.testing.assert_allclose(staged, predicted)
boxes, scores, class_ids, keypoint_xy, visibility = result
print(boxes.shape, keypoint_xy.shape, visibility.shape)
```

输入须为非空 BGR uint8 H×W×3。输出须为已反量化的浮点 NHWC，具有完整 shape/dtype
metadata 且不带 SCALE 量化描述。整数、非有限值、DFL 绑定或角色歧义会明确拒绝。
raw 引用 SDK 缓冲区，下一次推理前完成后处理，或复制要保留的 raw 数组。
`preprocess` 返回支持映射访问的 `PreparedDetection`；`pre_process`/`forward`/
`post_process` 别名及后处理显式传原图宽高仍可用，
不依赖最近一次图片的缓存状态。

五个返回值的形状/类型/存储所有权与 DFL 姿态相同：float32 `(N,4)` 框、float32
`(N,)` 检测概率、int64 `(N,)` 类别、float32 `(N,17,2)` 点坐标、float32 `(N,17,1)`
点概率。空结果保持维度。可见性只执行一次稳定 sigmoid；NMS 对框和骨架使用相同索引。
坐标按实际取整缩放/padding 还原并裁到原图范围。X5 适配器返回含整数框的
`{box, score, kpts}` 字典列表；S 适配器返回四元组，关键点合并为 `(N,17,3)`。

库的 NMS 默认为 0.65；CLI 的 X5 默认 0.70、S 默认 0.45，本例显式对齐 S CLI。
置信度须为 `(0,1)` 内有限值，NMS 为 `[0,1]`。置信度阈值严格按传入值使用；
配合实际取整几何和极端 logits 的稳定 sigmoid，存在取整的输入按修正后的
几何输出坐标。板端推理与数据集指标按评估指南测量。

<a id="yolo26-segmentation-api"></a>
## YOLO26 分割阶段接口

YOLO26 分割与 DFL sample 共用阶段传输、严格 runner、NMS 和显式几何，但框为直接
LTRB 距离，mask 算法也不同：系数与原型相乘，执行 sigmoid，把概率双线性放大到模型
尺寸，按模型坐标框裁剪，去掉实际 padding，把概率图缩放到原图，再以 `>0.5` 二值化，
最后取原图框 ROI。不能替换成 DFL 的局部二值 mask 缩放或形态学开运算；插值与阈值
的先后会改变边缘像素。

在匹配的 S600 板卡上，先按[模型准备](../../model/README_cn.md)获取制品并替换
下例路径，然后从仓库根目录运行：

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import resolve_platform
from samples.vision.ultralytics_yolo.runtime.python.yolo26_seg import YOLO26Seg, YOLO26SegConfig

image_path = Path("samples/vision/ultralytics_yolo/test_data/bus.jpg")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)
segmenter = YOLO26Seg(YOLO26SegConfig(
    model_path="/models/yolo26n_seg_nashp_640x640_nv12.hbm",
    platform=resolve_platform("s600"),
    nms_thres=0.45,
))
prepared = segmenter.preprocess(image)
raw = segmenter.infer(prepared)
result = segmenter.postprocess(raw, transform=prepared.transform)
predicted = segmenter.predict(image)
for staged, repeated in zip(result[:3], predicted[:3]):
    np.testing.assert_allclose(staged, repeated)
assert len(result[3]) == len(predicted[3])
for staged, repeated in zip(result[3], predicted[3]):
    np.testing.assert_array_equal(staged, repeated)
boxes, scores, class_ids, masks = result
print(boxes.shape, [mask.shape for mask in masks])
```

契约要求正方形模型输入及十个输出：stride 8/16/32 各包含分类 logits、四个 LTRB
距离、32 个 mask 系数，另有 stride-4 的 32 通道原型。head 为浮点 NHWC；原型可以
是 metadata 形状能证明的浮点 NHWC 或 NCHW。自定义类别数造成角色歧义时，须提供
审定的显式角色映射，不按枚举顺序猜测。整数/SCALE、错误 shape/dtype 和非有限值
会拒绝。forward 保留原型物理布局，postprocess 才归一化布局。

raw 引用 SDK 缓冲区，下一次推理前完成后处理或复制 raw。结果拥有独立的 float32
`(N,4)` 框、float32 `(N,)` 分数、int64 `(N,)` 类别和**布尔** ROI mask 列表。
对已裁到原图的每个框取 `x1,y1,x2,y2 = box.astype(int)`，即可把对应 mask 放到
原图大小空白数组的 `[y1:y2,x1:x2]`。退化 ROI 为 `(0,0)`；无检测结果保持
`(0,4)/(0,)/(0,)/[]`。`YOLO26Seg.predict` 的结果在每个目标上均为 ROI mask；X5 适配器
返回布尔 `(N,H,W)` 全图 mask 堆叠，S 适配器返回逐框 ROI mask。

置信度须为 `(0,1)` 内有限值，NMS 为 `[0,1]`。库 NMS 默认值为 0.65；CLI
传入 X5 0.70 或 S 0.45。解码器使用指定的有效置信度阈值，在 sigmoid 前从
logits 选择类别；mask 使用稳定 sigmoid，依据实际取整后的缩放尺寸及 padding
还原到原图。数据集精度测量见[评估指南](../../evaluator/README_cn.md)。

<a id="yolo26-obb-stages"></a>
## YOLO26 旋转框三阶段

在 S600 板上从仓库根目录运行，先按[模型准备](../../model/README_cn.md)获取
OBB 制品并替换本地路径。随附 bus 图片仅演示 API 调用，不是航拍目标基准图，
也没有预期必须检出的目标。

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.yolo26_obb import YOLO26OBB, YOLO26OBBConfig

image_path = Path("samples/vision/ultralytics_yolo/test_data/bus.jpg")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)
obb = YOLO26OBB(YOLO26OBBConfig(
    model_path="/models/yolo26n_obb_nashp_640x640_nv12.hbm",
    platform="s600",
))
prepared = obb.preprocess(image)
raw = obb.infer(prepared)
records = obb.postprocess(raw, transform=prepared.transform)
predicted = obb.predict(image)
assert len(records) == len(predicted)
for staged, repeated in zip(records, predicted):
    assert staged["id"] == repeated["id"]
    np.testing.assert_allclose(staged["rrect"], repeated["rrect"])
    np.testing.assert_allclose(staged["score"], repeated["score"])
for record in records:
    cx, cy, width, height, angle_radians = record["rrect"]
    print(record["id"], record["score"], record["rrect"])
```

该类复用可读检测器的图像传输与原始 runner 调用（`preprocess`/`infer`，也可通过
`pre_process`/`forward` 别名访问），自身提供旋转框解码：
`preprocess` 返回输入张量及当前图片实际的整数缩放、填充信息。`infer`
复用共用 runner，原样返回借用的 SDK 缓冲区，不解码、不反量化；下一次推理前
完成后处理，或先复制原始输出。`postprocess`（别名 `post_process`）返回独立拥有数据的记录列表：
`rrect=(cx,cy,width,height,angle_radians)`、浮点 `score`、整数 `id`。
无检测时返回 `[]`。交错处理图片时，必须将各自 transform 与输出配对，不能依赖
隐式的“上一次图片”状态。

方形模型需要九个浮点输出：stride 8/16/32 各自的 15 类 logits、4 个直接 LTRB
距离和 1 个**弧度角度**。按形状与角色绑定，不依赖输出枚举顺序；拒绝整数或
SCALE 量化描述、错误形状/类型及非有限值。LTRB 距离取绝对值。
`angle_sign` 乘到角度上；`angle_offset` 使用**度**，转换后相加。
默认 `regularize=True`，宽小于高时交换宽高并将角度加 π/2。
导出器输出弧度角度。

| 行为 | X5 | S100 / S100P / S600 |
| --- | --- | --- |
| 旋转 NMS | 按类别计算旋转 IoU，IoU ≥ 阈值时抑制 | OpenCV `NMSBoxesRotated`，所有类别一起处理 |
| 规范化后的角度 | 归一化到 [−π/2, π/2) | 不再额外归一化 |
| 还原后的中心及宽高 | 分别裁剪到原图宽高范围 | 不裁剪 |

库默认置信度 0.25、NMS 0.2、letterbox 缩放。置信度必须为 `(0,1)` 内有限值，
NMS 为 `[0,1]` 内有限值，角度参数必须有限。OpenCV 求交异常会向调用者报告，
不会静默当成零重叠。逆变换使用实际整数填充和逐轴缩放，修正取整后的 letterbox
坐标；宽高逐轴缩放、角度保持不变。因此 X/Y 比例不同时返回的是
**近似旋转矩形**，不是精确变换后的多边形。

使用上述任务模型类和张量契约。核对 OBB 几何时，保持输入尺寸、缩放策略和模型顺序 DOTA 标签与推理配置一致。

<a id="troubleshooting"></a>
## 故障排查

* **无法导入 `hbm_runtime`：** 使用匹配的 RDK 板端系统镜像并检查 Python
  模块路径。主机 OpenExplore 工具链并非板端运行时。
* **板卡未知或目标不匹配：** 在板端省略 `--platform`，或传入真实目标；
  参数不能覆盖未知硬件身份。
* **找不到模型：** 用 `--list-models` 查看精确 Manifest 引用，用
  `--download` 准备发布制品，或传入已有 `--model-path`。显式路径不会被
  相似模型替换。
* **输入/输出契约被拒绝：** 核对平台、制品家族、静态偶数输入尺寸、NV12
  角色及 DFL/LTRB 输出协议，再核对 `--reg`、`--strides`、`--classes-num`。
* **没有检测框或框异常：** 检查图片颜色顺序、`--resize-type`、
  `--score-thres` 和 `--nms-thres`。解码器会还原到原图坐标，不会把输出
  名字当作几何声明。
* **无法保存结果：** 确认 `--img-save-path` 的父目录可写；相对路径相对
  调用命令所在目录。

完整参数请执行 `python samples/vision/ultralytics_yolo/runtime/python/main.py --help`。`--help`、`--dry-run`、
`--list-models`、`--download` 是不执行板端推理的路径。

模型清单不包含重复的 S 独立版本，见 [范围与输出要求](../../model/README_cn.md#maintained-scope)。
