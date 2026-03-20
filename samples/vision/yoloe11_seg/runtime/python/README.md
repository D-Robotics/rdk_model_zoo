# YOLOe11-Seg 开放词表实例分割示例（Python）

本示例展示如何在 BPU 上使用量化后的 YOLOe11-Seg 开放词表分割模型执行图像实例分割。支持前处理、BPU 推理、后处理（DFL 解码 + MCES 掩码生成 + NMS）以及结果可视化与保存。

> **注意：本示例不支持 S600 平台。** 若在 S600 上运行，程序将输出错误信息并退出。

## 环境依赖

```bash
pip install numpy==1.26.4 opencv-python==4.11.0.86 scipy==1.15.3
```

> 也可直接运行 `run.sh`，脚本会自动检测并安装所需依赖。

## 目录结构

```text
.
├── yoloe11seg.py    # YOLOe11-Seg 推理封装与后处理实现
├── main.py          # 推理程序入口（参数解析与流程控制）
├── run.sh           # 示例一键运行脚本
└── README.md        # 使用说明
```

## 参数说明

| 参数              | 说明                                                         | 默认值                                                                                |
|-------------------|--------------------------------------------------------------|---------------------------------------------------------------------------------------|
| `--model-path`    | 模型文件路径（.hbm 格式）                                    | `/opt/hobot/model/<soc>/basic/yoloe_11s_seg_pf_nashe_640x640_nv12.hbm`               |
| `--test-img`      | 测试图片路径                                                 | `../../test_data/office_desk.jpg`                                                     |
| `--label-file`    | 类别标签路径（每行一个类别）                                 | `../../test_data/coco_extended.names`                                                 |
| `--img-save-path` | 结果图像保存路径                                             | `result.jpg`                                                                          |
| `--priority`      | 模型调度优先级（0~255）                                      | `0`                                                                                   |
| `--bpu-cores`     | 使用的 BPU 核心编号列表（如 `--bpu-cores 0 1`）             | `[0]`                                                                                 |
| `--nms-thres`     | 非极大值抑制（NMS）IoU 阈值                                  | `0.7`                                                                                 |
| `--score-thres`   | 置信度阈值                                                   | `0.25`                                                                                |
| `--no-morph`      | 启用掩码形态学开运算后处理                                   | 默认关闭（开放词表模型保留细节，默认不进行形态学处理）                                |
| `--no-contour`    | 禁用结果图上的轮廓线绘制                                     | `False`（默认启用）                                                                   |

> **注意**：`--model-path` 默认值中的 `<soc>` 会在运行时根据当前设备自动识别（如 `s100`）。S600 平台不支持本模型。

## 快速运行

- 运行模型
    - 使用默认参数
        ```bash
        python main.py
        ```
    - 指定参数运行
        ```bash
        python main.py \
            --model-path /opt/hobot/model/s100/basic/yoloe_11s_seg_pf_nashe_640x640_nv12.hbm \
            --test-img ../../test_data/office_desk.jpg \
            --label-file ../../test_data/coco_extended.names \
            --img-save-path result.jpg \
            --nms-thres 0.7 \
            --score-thres 0.25
        ```
- 查看结果

    运行成功后，会将检测框、实例掩码和轮廓绘制在原图上，并保存到 `--img-save-path` 指定路径：
    ```
    [Saved] Result saved to: result.jpg
    ```

## 接口说明

阅读[源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档；
