# YOLO26 Python 推理样例

本样例展示了如何在 RDK X5 平台上使用 `hbm_runtime` 运行 YOLO26 任务模型。

## 环境依赖

本样例没有额外特殊依赖，请确保 RDK X5 的 Python 环境已准备好。

```bash
pip install numpy opencv-python hbm-runtime scipy
```

## 目录结构

```text
.
├── main.py                # 推理入口脚本
├── yolo26_det.py          # 检测封装
├── yolo26_seg.py          # 分割封装
├── yolo26_pose.py         # 姿态封装
├── yolo26_obb.py          # 旋转框封装
├── yolo26_cls.py          # 分类封装
├── run.sh                 # 一键运行脚本
└── README.md              # 使用说明
```

## 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--task` | 任务类型：`detect`、`seg`、`pose`、`cls`、`obb` | `detect` |
| `--model-path` | RDK X5 `.bin` 模型路径 | `../../model/yolo26n_detect_bayese_640x640_nv12.bin` |
| `--test-img` | 测试图片路径 | `../../../../../datasets/coco/assets/bus.jpg` |
| `--label-file` | 标签文件路径，空字符串表示使用任务默认标签 | `""` |
| `--img-save-path` | 结果图保存路径 | `../../test_data/result_detect.jpg` |
| `--priority` | 模型优先级 | `0` |
| `--bpu-cores` | 推理使用的 BPU Core 索引 | `0` |
| `--classes-num` | 检测类任务类别数 | `80` |
| `--score-thres` | 置信度阈值 | `0.25` |
| `--nms-thres` | NMS 阈值 | `0.70` |
| `--strides` | 解码 stride | `8,16,32` |
| `--topk` | 分类任务输出 Top-K 数量 | `5` |
| `--kpt-conf-thres` | 姿态关键点置信度阈值 | `0.50` |
| `--angle-sign` | OBB 角度方向参数 | `1.0` |
| `--angle-offset` | OBB 角度偏移参数 | `0.0` |
| `--regularize` | 是否对 OBB 角度做规范化 | `1` |

## 快速运行

- **一键运行脚本**
  ```bash
  chmod +x run.sh
  ./run.sh
  ```

- **手动运行**
  - 使用默认参数
    ```bash
    python3 main.py
    ```
  - 显式运行检测任务
    ```bash
    python3 main.py \
        --task detect \
        --model-path ../../model/yolo26n_detect_bayese_640x640_nv12.bin \
        --test-img ../../../../../datasets/coco/assets/bus.jpg \
        --img-save-path ../../test_data/result_detect.jpg
    ```

## 任务示例

### 实例分割

```bash
python3 main.py \
    --task seg \
    --model-path ../../model/yolo26n_seg_bayese_640x640_nv12.bin \
    --test-img ../../../../../datasets/coco/assets/bus.jpg \
    --img-save-path ../../test_data/result_seg.jpg
```

### 姿态估计

```bash
python3 main.py \
    --task pose \
    --model-path ../../model/yolo26n_pose_bayese_640x640_nv12.bin \
    --test-img ../../../../../datasets/coco/assets/bus.jpg \
    --img-save-path ../../test_data/result_pose.jpg
```

### 旋转框检测

```bash
python3 main.py \
    --task obb \
    --model-path ../../model/yolo26n_obb_bayese_640x640_nv12.bin \
    --test-img ../../../../../datasets/dotav1/asset/P0009.png \
    --label-file ../../../../../datasets/dotav1/dota_classes.names \
    --img-save-path ../../test_data/result_obb.jpg
```

### 图像分类

```bash
python3 main.py \
    --task cls \
    --model-path ../../model/yolo26n_cls_bayese_224x224_nv12.bin \
    --test-img ../../../../../datasets/imagenet/asset/zebra_cls.jpg \
    --label-file ../../../../../datasets/imagenet/imagenet_classes.names
```

## 接口说明

- **`YOLO26*Config`**: 用于封装每个任务的模型路径和运行参数。
- **`YOLO26*`**: 提供完整推理流程，包括 `pre_process`、`forward`、`post_process` 和 `predict`。

公共预处理、后处理和可视化能力位于 `utils/py_utils/`。
