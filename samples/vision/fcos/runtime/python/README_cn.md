# FCOS 检测 Python 示例

本示例展示如何在 RDK X5 上使用 `hbm_runtime` 运行 FCOS 检测模型。

## 环境依赖

- `RDK OS >= 3.5.0`
- 板端镜像已预装 `hbm_runtime`

## 目录结构

```text
.
├── main.py
├── fcos_det.py
├── run.sh
├── README.md
└── README_cn.md
```

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-path` | FCOS `.bin` 模型路径 | `../../model/fcos_efficientnetb0_detect_512x512_bayese_nv12.bin` |
| `--label-file` | COCO 类别名文件路径 | `../../../../../datasets/coco/coco_classes.names` |
| `--priority` | Runtime 调度优先级 | `0` |
| `--bpu-cores` | BPU 核心编号 | `0` |
| `--test-img` | 测试图片路径 | `../../test_data/bus.jpg` |
| `--img-save-path` | 结果图保存路径 | `../../test_data/result.jpg` |
| `--resize-type` | 缩放策略：0 为直接缩放，1 为 letterbox | `0` |
| `--classes-num` | 检测类别数 | `80` |
| `--conf-thres` | 置信度阈值 | `0.5` |
| `--iou-thres` | NMS 的 IoU 阈值 | `0.6` |
| `--strides` | FCOS 特征层步长 | `8,16,32,64,128` |

## 快速运行

```bash
chmod +x run.sh
./run.sh
```

## 手动运行

- 使用默认参数运行：
  ```bash
  python3 main.py
  ```

- 显式指定参数运行：
  ```bash
  python3 main.py \
    --model-path ../../model/fcos_efficientnetb0_detect_512x512_bayese_nv12.bin \
    --test-img ../../test_data/bus.jpg \
    --img-save-path ../../test_data/result.jpg
  ```

## 接口说明

- `FCOSConfig`：封装模型路径和推理参数。
- `FCOSDetect`：实现 `set_scheduling_params`、`pre_process`、`forward`、`post_process` 和 `predict`。
