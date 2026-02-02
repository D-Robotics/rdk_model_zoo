# YOLOv5x 目标检测示例

本示例展示如何在 BPU 上使用量化后的 Ultralytics YOLOv5x 模型执行图像目标检测。支持前处理、后处理、NMS 以及最终的目标框绘制和结果保存。

## 环境依赖
本样例无特殊环境需求，只需确保安装了pydev中的环境依赖即可。
```bash
pip install -r ../../requirements.txt
```

## 目录结构
```text
.
├── yolov5x.py      # 主推理脚本
└── README.md       # 使用说明
```

## 参数说明

| 参数           | 说明                                                     | 默认值                                      |
|----------------|----------------------------------------------------------|---------------------------------------------|
| `--model-path` | 模型文件路径（.hbm 格式）                                  | `/opt/hobot/model/s600/basic/yolov5x_672x672_nv12.hbm` |
| `--test-img`   | 测试图片路径                                              | `/app/res/assets/kite.jpg`                     |
| `--label-file` | 类别标签路径（每行一个类别）                                | `/app/res/labels/coco_classes.names`           |
| `--img-save-path` | 检测结果图像保存路径                                    | `result.jpg`                                |
| `--priority`  | 模型调度优先级（0~255）                                     | `0`                                         |
| `--bpu-cores` | 使用的 BPU 核心编号列表（如 `--bpu-cores 0 1`）              | `[0]`                                      |
| `--nms-thres`   | 非极大值抑制（NMS）阈值                                    | `0.45`                                    |
| `--score-thres` | 置信度阈值                                                | `0.25`                                    |


## 快速运行
- 运行模型
    - 使用默认参数
        ```bash
        python yolov5.py
        ```
    - 指定参数运行
        ```bash
        python yolov5.py \
            --model-path /opt/hobot/model/s600/basic/yolov5x_672x672_nv12.hbm \
            --test-img /app/res/assets/kite.jpg \
            --label-file /app/res/labels/coco_classes.names \
            --img-save-path result.jpg \
            --priority 0 \
            --bpu-cores 0 \
            --nms-thres 0.45 \
            --score-thres 0.25
        ```
- 查看结果

    运行成功后，会将目标检测框绘制在原图上，并保存到 --img-save-path 指定路径
    ```bash
    [Saved] Result saved to: result.jpg
    ```


## 接口说明
阅读[源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档；
