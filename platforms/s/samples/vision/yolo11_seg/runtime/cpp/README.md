# YOLO11-Seg 实例分割示例（C++）

本示例展示如何在 BPU 上使用量化后的 Ultralytics YOLO11-Seg 模型执行图像实例分割。支持前处理（Letterbox + NV12）、BPU 推理、后处理（DFL 解码 + MCES 掩码 + NMS）以及结果可视化与保存。

## 环境依赖

```bash
sudo apt install -y libgflags-dev
```

> 也可直接运行 `run.sh`，脚本会自动检测并安装所需依赖。

## 目录结构

```text
.
├── inc/
│   └── yolo11seg.hpp   # Yolo11SegConfig 结构体、模型类及推理接口声明
├── src/
│   ├── yolo11seg.cpp   # 推理实现（init/pre_process/infer/post_process）
│   └── main.cpp        # 推理程序入口（参数解析与流程控制）
├── CMakeLists.txt       # CMake 构建配置
├── run.sh               # 示例一键运行脚本
└── README.md            # 使用说明
```

## 编译工程

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

> 或直接运行 `run.sh`，脚本会自动完成编译与运行。

## 参数说明

| 参数            | 说明                                                         | 默认值                                                                         |
|-----------------|--------------------------------------------------------------|--------------------------------------------------------------------------------|
| `--model_path`  | 模型文件路径（.hbm 格式）                                    | `/opt/hobot/model/<soc>/basic/yolo11n_seg_nashe_640x640_nv12.hbm`              |
| `--test_img`    | 测试图片路径                                                 | `../../../test_data/office_desk.jpg`                                           |
| `--label_file`  | 类别标签路径（每行一个类别）                                 | `../../../test_data/coco_classes.names`                                        |
| `--score_thres` | 置信度阈值                                                   | `0.25`                                                                         |
| `--nms_thres`   | 非极大值抑制（NMS）IoU 阈值                                  | `0.7`                                                                          |
| `--no_morph`    | 禁用掩码形态学开运算后处理                                   | `false`（默认启用）                                                            |

## 快速运行

- 使用 `run.sh` 一键运行（推荐）
    ```bash
    ./run.sh
    ```
- 手动编译后运行
    ```bash
    mkdir -p build && cd build && cmake .. && make -j$(nproc)
    ./yolo11seg \
        --model_path /opt/hobot/model/s100/basic/yolo11n_seg_nashe_640x640_nv12.hbm \
        --test_img ../../../test_data/office_desk.jpg \
        --label_file ../../../test_data/coco_classes.names \
        --score_thres 0.25 \
        --nms_thres 0.7
    ```
- 查看结果

    运行成功后，会将检测框、实例掩码和轮廓绘制在原图上，保存到当前目录的 `result.jpg`：
    ```
    [Saved] Result saved to: result.jpg
    ```

## 接口说明

阅读[源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档；
