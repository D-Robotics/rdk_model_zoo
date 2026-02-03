# YOLOv5x 目标检测示例

本示例展示如何在 BPU 上使用量化后的 YOLOv5x 模型执行图像目标检测。支持前处理、后处理、NMS 以及最终的目标框绘制和结果保存。

## 环境依赖
请先确保系统已安装以下依赖：
```bash
sudo apt update
sudo apt install libgflags-dev
```

## 目录结构
```bash
.
|-- inc                # 头文件目录
|   `-- yolov5.hpp     # YOLOv5 模型封装类声明
|-- src                # 源码目录
|   |-- main.cc        # 推理程序入口（参数解析与流程控制）
|   `-- yolov5.cc      # YOLOv5 推理与后处理实现
|-- CMakeLists.txt     # CMake 构建配置文件
|-- README.md          # C++ 推理示例使用说明
`-- run.sh             # 示例运行脚本
```

## 编译工程
- 配置与编译
    ```bash
    mkdir build && cd build
    cmake ..
    make -j$(nproc)
    ```

## 参数说明
| 参数           | 说明                         | 默认值                                     |
|----------------|------------------------------|--------------------------------------------|
| `--model_path` | 模型文件路径（.hbm 格式）    | `../../model/yolov5x_672x672_nv12.hbm`    |
| `--test-img`    | 测试图片路径                 | `../../../../../datasets/coco/assets/kite.jpg`                 |
| `--label_file`  | 类别标签路径（每行一个类别） | `../../../../../datasets/coco/coco_classes.names`              |
| `--score_thres` | 置信度阈值                   | `0.25`                                     |
| `--nms_thres`   | NMS IoU 阈值                 | `0.45`                                     |


## 快速运行
- 运行模型
    - 使用默认参数
        ```bash
        ./yolov5
        ```
    - 指定参数运行
        ```bash
        ./yolov5 \
            --model_path ../../model/yolov5x_672x672_nv12.hbm \
            --test-img ../../../../../datasets/coco/assets/kite.jpg \
            --label_file ../../../../../datasets/coco/coco_classes.names \
            --score_thres 0.25 \
            --nms_thres 0.45
        ```
- 查看结果

    运行成功后，会将目标检测框绘制在原图上，并保存在 build/result.jpg
    ```bash
    [Saved] Result saved to: result.jpg
    ```

## 接口说明
阅读[源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档；
