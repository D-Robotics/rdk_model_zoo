# Utils Module (C/C++)
本模块提供一组 面向开发者的通用工具函数，用于模型推理流程中常见的 文件读写、预处理、后处理、数学计算与可视化 等操作，便于在不同模型 Sample 中复用。

## 目录结构与文件职责
### 子目录说明
```text
c_utils
├── inc/    # 头文件（接口声明与函数注释）
└── src/    # 源文件（函数实现）
```

- inc/

    对外暴露的接口声明，所有函数的功能、参数与返回值说明均在此处以注释形式给出。
- src/

    对应接口的具体实现。

### 文件职责说明
| 文件名                     | 说明        | 包含内容                                                  |
| ----------------------- | --------- | ----------------------------------------------------- |
| `file_io.hpp / .cc`     | 文件与资源读写工具 | 图像加载、标签文件读取等                                          |
| `model_types.hpp`       | 公共数据结构定义  | `Classification` / `Detection` / `Keypoint` 等模型相关基础类型 |
| `nn_math.hpp / .cc`     | 数学与数值计算工具 | sigmoid/softmax、数值归一化等通用数学计算                     |
| `preprocess.hpp / .cc`  | 推理前数据预处理  | resize/letterbox、颜色空间转换、tensor 写入与输入输出 tensor 内存准备    |
| `postprocess.hpp / .cc` | 推理后结果处理   | Top-K、反量化、解码、过滤、NMS、坐标映射等                             |
| `visualize.hpp / .cc`   | 可视化工具     | 分类结果打印、检测框/分割/关键点绘制、文本/多边形渲染等                         |
| `runtime.hpp`           | 运行时辅助宏    | HB-DNN / HB-UCP API 调用检查宏（统一错误打印与返回码处理）               |


## 函数索引（快速查找）
本节用于帮助开发者快速定位所需功能所在的文件。

具体接口的参数、返回值与行为说明，请以源码注释为准。

### 公共数据结构（Model Types）
| 类型名              | 用途说明                                           | 所在文件              |
| ---------------- | ---------------------------------------------- | ----------------- |
| `Classification` | Top-K 分类结果结构                              | `model_types.hpp` |
| `Detection`      | 通用目标检测结果结构                             | `model_types.hpp` |
| `Keypoint`       | 关键点结构，表示单个关键点的位置与置信度           | `model_types.hpp` |

- 说明：

    这些结构体用于在 preprocess / postprocess / visualize 等模块之间传递推理结果数据，其字段含义与使用约定以 model_types.hpp 中的注释为准。


### 文件与标签读写（file_io）
| 函数名                           | 功能说明                                                           | 所在文件                |
| ----------------------------- | -------------------------------------------------------------- | ------------------- |
| `load_bgr_image`              | 从磁盘加载图像并返回 BGR 色彩空间的 `cv::Mat`（失败返回空 Mat）   | `file_io.hpp / .cc` |
| `load_imagenet1000_label_map` | 读取 ImageNet-1000 `{id: 'label'}` 文本格式标签映射，返回 `id → label` 的映射表 | `file_io.hpp / .cc` |
| `load_linewise_labels`        | 按行读取纯文本标签文件（每行一个 label，自动去除 `\r`）           | `file_io.hpp / .cc` |
| `load_id2token`               | 从 JSON 词表 `{token: id}` 构建 `id → token` 表（解析失败抛异常）             | `file_io.hpp / .cc` |


### 数学与数值计算（NN Math）

| 函数名                       | 功能说明                                      | 所在文件                |
| ------------------------- | ----------------------------------------- | ------------------- |
| `sigmoid`                 | 对单个 logit 值计算 sigmoid，用于将原始分数映射到 (0,1) 区间 | `nn_math.hpp / .cc` |
| `softmax`                 | 对短向量执行数值稳定的 softmax 归一化                   | `nn_math.hpp / .cc` |
| `logits_to_probabilities` | 将分类结果中的 logits 转换为概率（基于 softmax，原地修改）     | `nn_math.hpp / .cc` |


### 运行时工具与错误检查（Runtime）
| 名称                    | 功能说明                                | 所在文件                      |
| --------------------- | ----------------------------------- | ------------------------- |
| `HBDNN_CHECK_SUCCESS` | 检查 HB-DNN API 调用返回值，失败时打印错误信息并返回错误码 | `runtime.hpp` |
| `HBUCP_CHECK_SUCCESS` | 检查 HB-UCP API 调用返回值，失败时打印错误信息并返回错误码 | `runtime.hpp` |


### 预处理相关（preprocess）
| 函数名                     | 功能说明                                                      | 所在文件                   |
| ----------------------- | --------------------------------------------------------- | ---------------------- |
| `prepare_input_tensor`  | 根据 tensor 属性修正动态 stride，并为每个输入 tensor 分配缓存型 `sysMem`      | `preprocess.hpp / .cc` |
| `prepare_output_tensor` | 按输出 tensor 的 `alignedByteSize` 分配缓存型 `sysMem`             | `preprocess.hpp / .cc` |
| `bgr_to_nv12_tensor`    | 将 BGR 图像转换为 NV12（Y / UV 平面），并写入模型输入 tensor         | `preprocess.hpp / .cc` |
| `write_chw32_to_tensor` | 将 CHW 布局的 float32 数据写入 tensor buffer，并按 stride 正确拷贝  | `preprocess.hpp / .cc` |
| `letterbox_resize`      | 等比例缩放图像并进行对称 padding（letterbox），输出固定尺寸图像                  | `preprocess.hpp / .cc` |


### 后处理相关（postprocess）
| 函数名                              | 功能说明                                                         | 所在文件                    |
| -------------------------------- | ------------------------------------------------------------ | ----------------------- |
| `get_topk_result`                | 从输出 tensor 中提取 Top-K 分类结果（按 score 降序排序）                      | `postprocess.hpp / .cc` |
| `dequantizeTensorS32`            | 将 S32 量化输出 tensor 反量化为 float32（按 stride/scale/zero-point 处理） | `postprocess.hpp / .cc` |
| `yolov5_decode_all_layers`       | 解码 YOLOv5 多层输出为候选 Detection（不包含 NMS）                         | `postprocess.hpp / .cc` |
| `iou`                            | 计算两个检测框（xyxy 格式）的 IoU                                        | `postprocess.hpp / .cc`     |
| `nms_bboxes`                     | 按类别执行非极大值抑制（NMS），输出保留的检测框                                    | `postprocess.hpp / .cc` |
| `scale_letterbox_bboxes_back`    | 将 letterbox 坐标系下的检测框映射回原始图像坐标系                               | `postprocess.hpp / .cc` |
| `scale_keypoints_back_letterbox` | 将 letterbox 坐标系下的关键点映射回原始图像坐标系                               | `postprocess.hpp / .cc` |


### 可视化（visualize）
| 函数名                       | 功能说明                                                      | 所在文件                  |
| ------------------------- | --------------------------------------------------------- | --------------------- |
| `print_topk_results`      | 打印 Top-K 分类结果（可选 label 映射）                      | `visualize.hpp / .cc` |
| `draw_boxes`              | 在图像上绘制检测框、类别名与置信度                           | `visualize.hpp / .cc` |
| `draw_masks`              | 在检测框 ROI 内将分割 mask 以 alpha 混合方式叠加到图像上     | `visualize.hpp / .cc` |
| `draw_contours`           | 从分割 mask 提取轮廓并绘制多边形边界线                       | `visualize.hpp / .cc` |
| `draw_keypoints`          | 绘制姿态关键点（基于阈值过滤并用双圆高亮）                    | `visualize.hpp / .cc` |
| `draw_text`               | 使用 FreeType 将多段文本绘制到图像上（支持 TTF/UTF-8）       | `visualize.hpp / .cc` |
| `draw_polygon_boxes`      | 在图像上绘制闭合多边形框                                    | `visualize.hpp / .cc` |
| `colorize_mask`           | 将分割 label mask（类别 id）转为彩色可视化图                 | `visualize.hpp / .cc` |
| `rgb_to_argb8888`         | 将 RGB(8-bit) 打包为 ARGB8888（alpha=0xFF）                | `visualize.hpp / .cc` |
| `draw_detections_on_disp` | 通过 `sp_display_*` 在硬件显示叠加层绘制检测框与文字         | `visualize.hpp / .cc` |


## API 查阅说明（源码为准）
本模块面向开发者使用，所有函数接口的功能说明、参数含义、返回值约定，均以源码中的函数注释为参考依据，README 不重复描述具体接口细节。

推荐查阅方式：
- 在 第二部分「函数索引」 中找到目标函数；
- 确认其所在文件（如 preprocess.hpp、postprocess.hpp）；
- 打开对应源码文件，直接阅读函数声明处的注释说明。
