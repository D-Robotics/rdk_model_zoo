# Python 运行时

[English](README.md)

这是共用 Ultralytics YOLO Sample 的板端入口。它通过 RDK 系统镜像提供的
`hbm_runtime` 加载 X5 的 `.bin` 或 S100/S100P/S600 的 `.hbm`，把一张 BGR
图片准备为目标板的 NV12 输入，执行任务解码，并保存绘制结果。脚本不会
安装 Python 依赖。导出和编译请看 [`conversion/README_cn.md`](../../conversion/README_cn.md)。

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

返回值是可按旧方式解包的 `DetectionResult`：`boxes_xyxy` 为原图像素坐标
`(N,4)`，`scores` 和 `class_ids` 为 `(N,)`。CLI 绘制检测框并写入
`--img-save-path`（默认 `result.jpg`），同时打印模型、输入协议和检测结果；
模型加载或绑定失败时不会伪造结果。

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

常用参数：

```text
--score-thres 0.25       置信度过滤
--nms-thres 0.45         X5 默认 0.70，S 默认 0.45
--resize-type 0|1        直接缩放或 letterbox，默认随平台协议
--strides 8,16,32        检测特征层 stride
--reg 16                 YOLOv8/YOLO11 DFL bin 数
--priority 0             BPU 调度优先级
--bpu-cores 0            一个或多个 BPU 核
```

`--input-shape HxW` 只用于运行时没有报告输入尺寸的情况；如果模型报告的
尺寸与之冲突会拒绝。`--classes-num`、`--strides`、`--reg`、`--mc`、
`--nkpt` 是模型契约参数，不能用来强行兼容不匹配的制品。

## 库接口

在匹配的 S600 板卡上从仓库根目录执行下例，并先将模型路径替换为本地 YOLO11 检测制品：

```python
import sys
from pathlib import Path
import cv2

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
boxes, scores, class_ids = detector.predict(bgr_image)
print(boxes.shape, scores.shape, class_ids.shape)
```

`YoloDetect` 支持注入 runner，便于主机测试或接入其他运行时加载器。runner
负责模型执行；图像几何、协议绑定、DFL 解码、按类别 NMS 和坐标还原由共用
任务实现负责。`YOLO26Detect` 共用图片准备和 runner 流程，但使用经过审查
的直接 LTRB 解码。历史 X5/S 模块保留旧类名和 tuple 形状并转发到维护入口。

## 代码流程

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
