简体中文 | [English](./README.md)

# YOLOE 运行说明

本目录提供 `yoloe` sample 在 `RDK X5` 上的 Python 推理入口。

## 文件说明

- `main.py`：统一 Python 推理入口。
- `yoloe_seg.py`：基于 `hbm_runtime` 的 YOLOE 分割封装。
- `run.sh`：一键运行脚本。
- `README.md`：本文档（英文版）。
- `README_cn.md`：本文档（中文版）。

## 快速开始

```bash
cd runtime/python
chmod +x run.sh
./run.sh
```

脚本会在缺少默认模型时自动下载 `yoloe_11s_seg_pf_bayese_640x640_nv12.bin` 到 `../../model/`，并将结果图保存到 `../../test_data/result_seg.jpg`。

s/m/l 共用此入口。通过 `--model-path` 指定模型时，脚本不会下载默认 s 模型。

## 手动运行

```bash
python3 main.py
python3 main.py --model-path ../../model/yoloe_11s_seg_pf_bayese_640x640_nv12.bin
python3 main.py --model-path ../../model/yoloe_11m_seg_pf_bayese_640x640_nv12.bin
python3 main.py --model-path ../../model/yoloe_11l_seg_pf_bayese_640x640_nv12.bin
python3 main.py --img-save-path ../../test_data/result_custom.jpg
python3 main.py --score-thres 0.3 --nms-thres 0.65
```

## 命令行参数

```bash
python3 main.py -h
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-path` | BPU 量化 YOLOE BIN 模型路径。 | `../../model/yoloe_11s_seg_pf_bayese_640x640_nv12.bin` |
| `--test-img` | 测试输入图片路径。 | `../../../../../datasets/coco/assets/bus.jpg` |
| `--label-file` | 可视化使用的类别名称文件路径。 | `../../../../../datasets/yoloe/yoloe_seg_pf_classes.names` |
| `--img-save-path` | 结果图保存路径。 | `../../test_data/result_seg.jpg` |
| `--priority` | 运行时调度优先级。 | `0` |
| `--bpu-cores` | 推理使用的 BPU Core 编号。 | `0` |
| `--classes-num` | 分割类别数量。 | `4585` |
| `--score-thres` | 候选框筛选阈值。 | `0.25` |
| `--nms-thres` | 非极大值抑制 IoU 阈值。 | `0.70` |
| `--strides` | 逗号分隔的检测头 stride 配置。 | `8,16,32` |
| `--reg` | DFL 每边 bin 数量。 | `16` |
| `--mc` | 掩码系数维度。 | `32` |

## 接口说明

### YOLOESegConfig

YOLOE 分割推理的配置数据类。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `model_path` | `str` | *(必填)* | 编译后的 BIN 模型文件路径。 |
| `classes_num` | `int` | `4585` | 分割类别数量。 |
| `score_thres` | `float` | `0.25` | 置信度阈值。 |
| `nms_thres` | `float` | `0.70` | NMS IoU 阈值。 |
| `resize_type` | `int` | `1` | Resize 策略（0：直接缩放，1：letterbox）。 |
| `strides` | `np.ndarray` | `[8, 16, 32]` | 检测头 stride 配置。 |
| `reg` | `int` | `16` | DFL 每边 bin 数量。 |
| `mc` | `int` | `32` | 掩码系数维度。 |

### YOLOESeg

基于 `hbm_runtime` 的 YOLOE 实例分割封装。

| 方法 | 说明 |
| --- | --- |
| `__init__(config)` | 初始化模型并提取运行时元数据。 |
| `set_scheduling_params(priority, bpu_cores)` | 设置 BPU 调度参数。 |
| `pre_process(img, resize_type, image_format)` | 将 BGR 图片转换为 packed NV12 输入。 |
| `forward(input_tensor)` | 在 BPU 上执行推理。 |
| `post_process(outputs, ori_img_w, ori_img_h, score_thres, nms_thres)` | 将原始输出转换为最终分割结果。返回 `(xyxy, score, cls, masks)`。 |
| `predict(img, image_format, resize_type, score_thres, nms_thres)` | 执行完整分割流水线。返回 `(xyxy, score, cls, masks)`。 |
| `__call__(...)` | 函数式调用，等价于 `predict()`。 |
