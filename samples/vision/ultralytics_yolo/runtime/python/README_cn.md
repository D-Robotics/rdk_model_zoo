# Python 运行时

[English](README.md)

这是共用 Ultralytics YOLO Sample 的板端入口。它通过 RDK 系统镜像提供的
`hbm_runtime` 加载 X5 的 `.bin` 或 S100/S100P/S600 的 `.hbm`，把一张 BGR
图片准备为目标板的 NV12 输入，执行任务解码；detect/seg/pose/obb 保存绘制结果，cls 打印 Top-K。脚本不会
安装 Python 依赖。导出和编译请看 [`conversion/README_cn.md`](../../conversion/README_cn.md)。

<a id="environment"></a>
## 板端准备

将与板卡目标一致的编译制品复制到板端或下载到 Manifest 指定位置。X5 绑定
一个 packed NV12 输入；S 系列绑定命名的 NHWC Y、UV 输入。运行时在推理前
读取模型元数据并检查 batch、尺寸、类型和输出协议；文件名后缀不能代替
元数据检查。

缺少默认模型时，历史入口会在板端下载默认模型；显式传入 `--model-path`
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
不能把 YOLO26 制品交给 DFL 解码器，也不能根据输出序号猜协议。当前代表
性板卡证据覆盖 YOLOv8n、YOLO26n 检测，不代表所有尺寸或任务。

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
| `--label-file` | str | `null` | detect/seg/pose 用 COCO，cls 用 ImageNet，obb 用 DOTA；自定义类别顺序须覆盖。 |
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

非分类任务默认 letterbox。分类中，YOLO26 全目标默认拉伸；其他系列 X5 默认 letterbox，S 默认拉伸。输入尺寸与类别数须匹配模型；`--reg`、`--strides`、`--mc` 等不是强行兼容其他模型的开关。YOLO26 非分类任务拒绝偏离 16/17/32 的 DFL/关键点/mask 覆盖参数。YOLOv13 仅在 X5 发布。

<a id="results"></a>
## 输出结果

退出码 0 表示命令完成；空检测列表仍可能是有效结果，不等于模型失败。detect/seg/pose/obb 输出由 `--img-save-path` 指定，CLI 会创建父目录并打印 `[Saved]`；已有同名结果会被覆盖。cls 只打印分类，不写结果图。阶段接口不负责绘制或保存。

| 任务 | `predict` 返回值 | 坐标与含义 |
|---|---|---|
| detect | `DetectionResult(boxes_xyxy, scores, class_ids)`，兼容三元组解包 | `(N,4)` 原图像素框、`(N,)` 置信度与从 0 开始的类别 ID |
| seg | `(boxes, scores, ids, masks)` | 原图像素框及对应实例 mask；保持实例顺序配对 |
| pose | `(boxes, scores, ids, xy, confidence)` | 原图像素框和点坐标；发布姿态模型为 17 点，置信度独立返回 |
| cls | `(class_id, probability)` 列表 | Softmax 后按分数排序的 Top-K，不是原始 logits |
| obb | 字典列表：`rrect`、`score`、`id` | `rrect=(cx,cy,w,h,angle)`；尺寸/中心在原图坐标中，角度为弧度 |

单图绘制不是数据集精度或性能验证，完整评估见 [evaluator](../../evaluator/README_cn.md)。本地文件选择与发布范围见 [model](../../model/README_cn.md)。

<a id="integration-example"></a>
## 库接口

在匹配的 S600 板卡上从仓库根目录执行下例，并先将模型路径替换为本地 YOLO11 检测制品：

```python
import sys
from pathlib import Path
import cv2
import numpy as np

runtime_dir = Path("samples/vision/ultralytics_yolo/runtime/python").resolve()
sys.path.insert(0, str(runtime_dir))
from yolo_platform import resolve_platform
from yolo_detect import YoloDetect, YoloDetectConfig

profile = resolve_platform("s600")
config = YoloDetectConfig(
    model_path="/models/yolo11n_nashp_640x640_nv12.hbm",
    platform=profile,
)
bgr_image = cv2.imread("samples/vision/ultralytics_yolo/test_data/bus.jpg")
if bgr_image is None:
    raise FileNotFoundError("Cannot read test image")
detector = YoloDetect(config)
prepared = detector.pre_process(bgr_image)
raw = detector.forward(prepared.tensors)
result = detector.post_process(raw, transform=prepared.transform)
for staged, predicted in zip(result, detector.predict(bgr_image)):
    np.testing.assert_allclose(staged, predicted)
boxes, scores, class_ids = result
print(boxes.shape, scores.shape, class_ids.shape)
```

`YoloDetect` 支持注入 runner，便于主机测试或接入其他运行时加载器。runner
负责模型执行；图像几何、协议绑定、DFL 解码、按类别 NMS 和坐标还原由共用
任务实现负责。`YOLO26Detect` 共用图片准备和 runner 流程，但使用经过审查
的直接 LTRB 解码。历史 X5/S 模块保留旧类名和 tuple 形状并转发到维护入口。

<a id="stage-io"></a>
## 代码流程

`YoloDetect` 和 `YOLO26Detect` 直接串联三个阶段，不保存“上一张图片”的 context。
准备 B 不会覆盖 A 的几何信息；分别保留 prepared，并使用其对应 transform。SDK 调用仍需
由调用方串行安排；逐调用 context 不代表 SDK 推理线程安全。

- `pre_process(img, image_format="BGR")` 要求非空 uint8 H×W×3 BGR，返回 `PreparedDetection.tensors` 和冻结的 `.transform`。后者包含原图/模型/实际缩放尺寸、整数 padding 与横纵缩放比例。X5 张量为 packed NV12；S 为 Y `(1,H,W,1)` 和 UV `(1,H/2,W/2,2)`，H/W 来自模型 metadata。
- `forward(prepared.tensors)` 只调用一次 runner，返回以角色名索引的 `RawOutputs`。绑定的 runner 校验物理 shape、dtype 和有限值，不反量化、不激活、不解码、不改变布局。数组保留 SDK dtype，借用 SDK 缓冲区；须先完成后处理再发起下一次 SDK 调用，或主动复制需要长期保留的原始数组。
- `post_process(raw, transform=prepared.transform)` 执行声明过的 DFL 仿射反量化，再做 sigmoid/DFL 或 LTRB 解码、适用的 NMS 和坐标还原。DFL 支持有限正数的标量/逐通道 SCALE，校验 axis 与通道数，zero-point 可为空、标量或逐通道；SDK NONE 不做变换。LTRB 仍只接受浮点输出并拒绝 SCALE。整数 logits 缺量化信息时显式报错，不直接转浮点凑结果。
- `predict(img)` 串联这些方法并返回自有结果数组。注入 runner 返回普通语义映射时，数值须已是浮点；物理量化输出应使用绑定后的 raw 容器。

可执行兼容方式：prepared 仍支持 `[model_name]` 映射访问，`forward(prepared)` 会取出
`.tensors`；`legacy.py` 中的 `pre_process_with_transform` 仍返回旧 `(tensors, transform)`
元组。显式 `post_process(outputs, 原宽, 原高)` 可无缓存重建同一几何；同时给宽高和 transform
时必须一致。原 `last_transform`/`last_image_transform` 属性已移除，请保留 prepared。
这些检测签名不改变其他任务 API：seg/pose 解码 mask/关键点，cls 做 Softmax/Top-K，OBB
返回旋转框；它们各自的阶段契约与独立源制品量化处理仍在迁移核查中。

```text
main.py
  -> resolve_target / 平台 Manifest 选择
  -> yolo_dispatch.get_task_types / create_runtime_model
  -> ModelRunner + ModelBinding（输入/输出契约）
  -> geometry.resize_with_transform + NV12 输入绑定
  -> YoloDetect 或 YOLO26Detect 解码 + NMS
  -> DetectionResult -> visualize -> --img-save-path
```

`model_binding.py` 按审查过的形状/类型契约识别输出角色，编译器枚举名称
只是物理名称。`geometry.py` 记录实际整数缩放和 padding，使框还原使用
同一个变换。有限检测协议和旧新符号对应见
[`DETECTION_CONTRACT.md`](../../DETECTION_CONTRACT.md)。

<a id="troubleshooting"></a>
## 故障排查

* **无法导入 `hbm_runtime`：** 使用匹配的 RDK 板端系统镜像并检查 Python
  模块路径。主机 OpenExplore 工具链不等于板端运行时。
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


S 独立 YOLO11 检测/姿态/分割与 S100 iMoonLab YOLOv13 的原始制品，现可通过精确 ID 准备和选择；数值/C++ 归并尚未验收，见 [源制品绑定与边界](../../model/README_cn.md#standalone-assets)。
