# YOLO11 目标检测示例（C++）

本示例展示如何在 BPU 上使用量化后的 Ultralytics YOLO11 模型执行图像目标检测。支持前处理（Letterbox + NV12）、BPU 推理、后处理（DFL 解码 + NMS）以及目标框绘制和结果保存。

## 环境依赖

```bash
sudo apt install -y libgflags-dev
```

> 也可直接运行 `run.sh`，脚本会自动检测并安装所需依赖。

## 目录结构

```text
.
├── inc/
│   └── yolo11.hpp      # YOLO11 配置结构体、模型类及推理接口声明
├── src/
│   ├── yolo11.cpp      # YOLO11 推理实现（init/pre_process/infer/post_process）
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

| 参数            | 说明                                                         | 默认值                                                                      |
|-----------------|--------------------------------------------------------------|-----------------------------------------------------------------------------|
| `--model_path`  | 模型文件路径（.hbm 格式）                                    | `/opt/hobot/model/<soc>/basic/yolo11n_detect_nashe_640x640_nv12.hbm`        |
| `--test_img`    | 测试图片路径                                                 | `../../../test_data/kite.jpg`                                               |
| `--label_file`  | 类别标签路径（每行一个类别）                                 | `../../../test_data/coco_classes.names`                                     |
| `--score_thres` | 置信度阈值                                                   | `0.25`                                                                      |
| `--nms_thres`   | 非极大值抑制（NMS）阈值                                      | `0.45`                                                                      |

> **注意**：C++ 参数名采用 snake_case（如 `--model_path`），与 Python 版本的 kebab-case（`--model-path`）有所不同。

## 快速运行

- 使用 `run.sh` 一键运行（推荐）
    ```bash
    ./run.sh
    ```
- 手动编译后运行
    ```bash
    mkdir -p build && cd build && cmake .. && make -j$(nproc)
    ./yolo11 \
        --model_path /opt/hobot/model/s100/basic/yolo11n_detect_nashe_640x640_nv12.hbm \
        --test_img ../../../test_data/kite.jpg \
        --label_file ../../../test_data/coco_classes.names \
        --score_thres 0.25 \
        --nms_thres 0.45
    ```
- 查看结果

    运行成功后，会将目标检测框绘制在原图上，保存到当前目录的 `result.jpg`：
    ```
    [Saved] Result saved to: result.jpg
    ```

## 接口说明

阅读[源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档；
