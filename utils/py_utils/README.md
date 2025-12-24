# Utils Module (Python)

本模块提供一组面向开发者的通用工具函数，用于模型推理流程中常见的文件读写、预处理、后处理、数学计算与可视化等操作，便于在不同模型 Sample 中复用。

## 目录结构与文件职责
### 子目录说明
```bash
py_utils
├── __init__.py
├── file_io.py
├── inspect.py
├── nn_math.py
├── preprocess.py
├── postprocess.py
└── visualize.py
```
- py_utils/

    Python 版本的通用工具模块，提供与 C/C++ utils 功能对齐的实现，主要用于 Python Sample 与快速验证场景。

### 文件职责说明

| 文件名            | 说明             | 包含内容                                 |
| ---------------- | ---------------- | --------------------------------------- |
| `file_io.py`     | 文件与资源读写工具 | 图像加载、标签文件读取、词表加载等         |
| `inspect.py`     | 模型与数据检查工具 | Tensor / 模型输出信息打印、调试与辅助检查  |
| `nn_math.py`     | 数学与数值计算工具 | sigmoid / softmax / 数值归一化等         |
| `preprocess.py`  | 推理前数据预处理   | resize、letterbox、输入格式转换等        |
| `postprocess.py` | 推理后结果处理     | 解码、过滤、NMS、坐标映射等               |
| `visualize.py`   | 可视化工具        | 分类 / 检测 / 分割 / 关键点结果渲染       |


## 函数索引（快速查找）

本节用于帮助开发者快速定位所需功能所在的文件, 不在 README 中重复函数的参数与返回值说明，所有接口行为以 源码中的函数注释 / docstring 为准。


### 文件与标签读写（file_io）
| 函数名                      | 功能说明                                                  | 所在文件         |
| -------------------------- | --------------------------------------------------------- | ------------ |
| `download_model_if_needed` | 检查模型文件是否存在，若不存在则根据给定 URL 自动下载到指定路径 | `file_io.py` |
| `load_image`               | 使用 OpenCV 从文件路径加载图像，返回 BGR 格式的 NumPy 数组    | `file_io.py` |
| `load_class_names`         | 从文本文件中逐行读取类别名称，返回类别名列表                  | `file_io.py` |


### 模型与数据检查（inspect）
| 函数名             | 功能说明                                                                   | 所在文件         |
| ------------------ | ------------------------------------------------------------------------- | ------------ |
| `get_soc_name`     | 读取当前设备的 SoC 名称                                                    | `inspect.py` |
| `print_model_info` | 打印模型的完整结构与运行时信息，包括输入/输出 tensor 形状、数据类型、描述信息等 | `inspect.py` |


### 数学与数值计算（NN Math）
| 函数名                     | 功能说明                                         | 所在文件         |
| -------------------------- | ----------------------------------------------- | ------------ |
| `sigmoid`                  | 对 NumPy 数组逐元素计算 sigmoid 激活函数          | `nn_math.py` |
| `zscore_normalize_lastdim` | 沿最后一个维度执行 Z-score 标准化（均值 0，方差 1） | `nn_math.py` |


### 预处理相关（preprocess）
| 函数名                     | 功能说明                                                                    | 所在文件            |
| ----------------------- | ----------------------------------------------------------------------------- | --------------- |
| `bgr_to_nv12_planes`    | 将 BGR 图像转换为 NV12 格式，输出 Y 平面与交错 UV 平面（含 batch / channel 维度） | `preprocess.py` |
| `resized_image`         | 对输入图像执行 resize 或 letterbox 缩放，输出固定分辨率 BGR 图像                 | `preprocess.py` |
| `split_nv12_bytes`      | 将原始 NV12 字节流拆分为 Y 平面与 UV 平面                                       | `preprocess.py` |
| `letterbox_resize_gray` | 对灰度图像执行等比例缩放并进行对称 padding（letterbox）                          | `preprocess.py` |
| `resize_nv12_yuv`       | 对 NV12 的 Y / UV 平面进行 resize（可选保持宽高比）                             | `preprocess.py` |


### 后处理相关（postprocess）
| 函数名                                 | 功能说明                                                        | 所在文件             |
| ----------------------------------- | ----------------------------------------------------------- | ---------------- |
| `recover_to_original_size`          | 将已 resize / letterbox 的图像恢复回原始分辨率（支持直接 resize 或反向去 padding） | `postprocess.py` |
| `dequantize_tensor`                 | 对量化张量执行反量化（支持 per-tensor / per-channel，依据 QuantParams）      | `postprocess.py` |
| `dequantize_outputs`                | 对输出字典按各自量化参数批量反量化为 float32                                  | `postprocess.py` |
| `scale_coords_back`                 | 将 resized / letterbox 坐标系下的 bbox 映射回原图坐标系（含 clamp）          | `postprocess.py` |
| `NMS`                               | 按类别执行 NMS，返回保留框的索引列表                                        | `postprocess.py` |
| `xywh_to_xyxy`                      | bbox 坐标格式转换：`(cx,cy,w,h)` → `(x1,y1,x2,y2)`                        | `postprocess.py` |
| `filter_classification`             | 对分类输出按 raw 阈值过滤（max logit），返回 scores/ids/valid_indices      | `postprocess.py` |
| `filter_mces`                       | 按 valid_indices 提取 MCES 特征（实例分割系数等）                          | `postprocess.py` |
| `filter_predictions`                | 对检测预测按置信度阈值过滤，输出 `xyxy/score/cls`                          | `postprocess.py` |
| `gen_anchor`                        | 生成方形网格上的 anchor 中心坐标（用于解码）                                | `postprocess.py` |
| `decode_boxes`                      | 解码分布式回归输出为 bbox                                                 | `postprocess.py` |
| `decode_masks`                      | 解码实例分割 masks（protos × mces），并按阈值二值化、裁剪                   | `postprocess.py` |
| `decode_kpts`                       | 解码关键点坐标与分数（支持自动生成 anchor）                                 | `postprocess.py` |
| `decode_layer`                      | 解码单个检测 head 特征层为预测张量 `(N, 5+C)`                              | `postprocess.py` |
| `decode_outputs`                    | 解码所有输出 head 并拼接为统一预测张量                                      | `postprocess.py` |
| `get_bounding_boxes`                | 从多边形轮廓提取最小外接矩形 bbox（minAreaRect）                            | `postprocess.py` |
| `resize_masks_to_boxes`             | 将二值 mask resize 到对应 bbox 尺寸（可选形态学处理）                        | `postprocess.py` |
| `scale_keypoints_to_original_image` | 将关键点从输入尺度映射回原图坐标（支持 resize / letterbox）                     | `postprocess.py` |
| `crop_and_rotate_image`             | 基于旋转框裁剪并透视矫正得到目标区域（可选旋转）                                    | `postprocess.py` |


### 可视化（visualize）
| 函数名                       | 功能说明                                                       | 所在文件           |
| ------------------------- | ---------------------------------------------------------------- | -------------- |
| `rdk_colors`              | 预置颜色表（RGB 元组列表），用于 bbox / mask / keypoint 可视化配色   | `visualize.py` |
| `print_topk_predictions`  | 打印 Top-K 分类结果（对 logits 做 softmax，输出 label + prob）      | `visualize.py` |
| `draw_boxes`              | 在图像上绘制检测框、类别名与置信度（返回绘制后的图像）                | `visualize.py` |
| `draw_masks`              | 将实例 mask 以 alpha 混合方式叠加到图像 ROI（原地修改 image）        | `visualize.py` |
| `draw_contours`           | 从实例 mask 提取轮廓并绘制多边形边界线（原地修改 img）              | `visualize.py` |
| `rgb_to_disp_color`       | RGB → 32-bit ARGB（用于硬件显示叠加层颜色格式）                    | `visualize.py` |
| `draw_detections_on_disp` | 通过硬件 display 对象绘制检测框与文字（依赖 `set_graph_rect/word`） | `visualize.py` |
| `draw_keypoints`          | 绘制关键点                                                        | `visualize.py` |
| `draw_polygon_boxes`      | 在图像上绘制闭合多边形框（返回绘制后的图像副本）                     | `visualize.py` |


## API 查阅说明（源码为准）
本模块面向开发者使用，所有函数接口的功能说明、参数含义、返回值约定，均以源码中的函数注释为参考依据，README 不重复描述具体接口细节。

推荐查阅方式：
- 在 第二部分「函数索引」 中找到目标函数；
- 确认其所在文件（如 preprocess.hpp、postprocess.hpp）；
- 打开对应源码文件，直接阅读函数声明处的注释说明。
