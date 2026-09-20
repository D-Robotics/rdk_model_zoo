# Phase 1 — A1 utils/ 合并记录（2026-09-21）

来源：`rdk_x5 @ ac11571` 与 `rdk_s @ 380e1a2` 的 `utils/` 树，按计划 A1
合并为 develop 根 `utils/` 过渡兼容层（ADR-0002）。不并入
`samples/_shared/`（spec §5.1：两个真实消费者才准入共享）。

## 逐文件决策

| 文件 | 决策 | 依据 |
| --- | --- | --- |
| `py_utils/README.md` | 取任一侧 | 两 tip 逐字节相同 |
| `py_utils/preprocess.py` | 取任一侧 | 两 tip 逐字节相同 |
| `py_utils/inspect.py` | 取 S | 纯超集（+79 行：`resolve_platform`、`get_soc_name_fallback_free`；无删改） |
| `py_utils/file_io.py` | 取 S | 超集：保留 X5 全部符号（`load_imagenet_labels` 以废弃别名形式存在，语义等价：dict 字符串/逐行两种格式、失败返回空 dict），新增 `save_image`、`load_labels` |
| `py_utils/nn_math.py` | 取 S | 唯一差异 `sigmoid` 的 `cv2.exp`→`np.exp`；两 tip 中无任何 X5 代码引用 `nn_math`，X5 侧 `postprocess.sigmoid` 本就用 `np.exp`，数值语义不变 |
| `py_utils/postprocess.py` | 取 S | 超集：保留 X5 全部函数（`sigmoid` 经 `from .nn_math import sigmoid` 再导出，数值同 X5 的 np.exp 版），新增 `scale_coords_back_obb`、`crop_mask`、`process_mask`、`decode_seg_layer`、`decode_pose_layer`（H1 反量化的移植源 `dequantize_outputs` 两侧同源） |
| `py_utils/visualize.py` | **手工并集** | 见下文 |
| `py_utils/__init__.py` | X5 风格（惰性） | S 版启用全部 wildcard 导入会使包在主机上牵出 `postprocess`→`hbm_runtime`（板端独占）；X5 的注释式 `__init__` 保持主机可导入。两种消费方式（`from utils.py_utils import file_io`、`import utils.py_utils.nn_math`）在两种环境均不受影响 |
| `tools/`（batch_eval_pycocotools、batch_mapper、batch_perf、generate_calibration_data） | 取 X5 原样 | S tip 无 `utils/tools` |
| `c_utils/`（inc+src） | 取 S 原样 | X5 tip 无 `c_utils` |

## visualize.py 并集细节

两侧是真分叉（共享名 `draw_boxes`/`draw_masks`/`draw_pose` 体不同，各自
独有大量函数）。合并结果 = S 版基底 + X5 独有符号逐字追加：

- **S 基底**：`get_topk_predictions`、`print_classification_results`、
  `draw_contours`、`rgb_to_disp_color`、`draw_detections_on_disp`、
  `draw_keypoints`、`draw_text`、`draw_polygon_boxes`、`print_obb_detections`、
  `draw_obb`、`print_pose_detections`、`print_detections`、`draw_mask_result`。
- **X5 追加（逐字）**：`COCO_SKELETON`、`draw_detection_results`、
  `draw_rotated_boxes`、`draw_classification`、`draw_detect_yolo26`、
  `draw_seg_yolo26`、`draw_obb_yolo26`、`draw_pose_yolo26`、`draw_cls_yolo26`、
  `logger = logging.getLogger("YOLO26")`。
- **共享名**：`rdk_colors` 两 tip 逐字节相同；`draw_boxes`/`draw_masks`
  签名相同，取 S 体（文档化超集）；`draw_pose` 取**并集签名**
  `(image, boxes, kpts, skeleton=None, kpt_conf_thres=0.5, scores=None,
  class_ids=None, colors=rdk_colors)` —— X5 参数序（`skeleton` 第 4 位，
  `draw_pose_yolo26` 按位置传参所依赖）+ S 体渲染，仅当传入 `skeleton`
  时画连接线。核对过的全部调用方行为：
  - X5 `ultralytics_yolo/main.py`（keyword `skeleton=`）：得到连接线；
    绿框（未传 `class_ids` 时 S 体回退绿色，与 X5 一致）。
  - S `ultralytics_yolo`/`ultralytics_yolo26` `main.py`（纯 keyword、无
    `skeleton`）：无连接线，渲染与源分支一致。
  - 已知外观级差异（仅 X5 调用方）：关键点由单圈变双圈、标签由
    `person: 0.95` 变 `0.95`。属可视化外观，不影响任何数值/推理结果；
    迁移期 sample 重构时本就优先本地化。

## 验证（2026-09-21，主机）

- `import utils.py_utils`、`from utils.py_utils import file_io, visualize,
  nn_math` 主机成功；`utils.py_utils.postprocess` 主机失败
  （`hbm_runtime` 板端独占）——与两侧源分支行为一致，非回归。
- 并集 API 面逐一 `hasattr` 核对通过（visualize 20 项、file_io 6 项、
  nn_math 2 项、inspect 4 项、preprocess 5 项）。
- `draw_pose` 两种调用风格（X5 位置传参 / S keyword）实跑通过。
- c_utils 主机不可编译验证（需 S 板端 SDK 头文件与库）；按板测门禁
  留待 B 批冒烟，此处 not-run 不伪装。
- `platforms/s/utils/c_utils` 保留不动：已验证的 S100 C++ 构建证据
  （2026-09-17 归档 `ba1d7401…`）绑定该路径；resnet/paddle_ocr 的 CMake
  暂不切到 `utils/c_utils`，统一在收尾第 6 步评估 repoint（见
  x5-s-migration-map 收尾清单）。

## 消费者影响

- 3 个已迁移 sample 不引用 `utils.py_utils`（已本地化），零影响
  （39/43/59/17 主机测试在合并后重跑通过，见 A1 提交后回归）。
- B 批迁移的 sample：分支源码的 `import utils.py_utils.X` 在 develop
  根即可解析；按 spec §5.1 优先 sample 本地化，第二个真实消费者出现
  才考虑升 `_shared/`。
