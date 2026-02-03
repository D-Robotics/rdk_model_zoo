# ByteTrack Python Runtime

本示例展示如何在 RDK 平台上使用 ByteTrack 算法进行多目标跟踪。示例包含 YOLOv5 检测器的推理和 BYTETracker 的更新逻辑。

## 环境依赖
- RDK 平台
- Python 3.8+
- hbm_runtime
- numpy>=1.24.0
- opencv-python>=4.5.0
- scipy>=1.7.0
- lap>=0.4.0
- Cython>=3.2.4
- cython_bbox>=0.1.5


安装命令：
```bash
pip install "numpy>=1.24.0" "opencv-python>=4.5.0" "scipy>=1.7.0" "lap>=0.4.0" "cython_bbox>=0.1.5" "Cython>=3.2.4" "hbm_runtime"
```

## 目录结构
```text
.
├── tracker/          # ByteTrack 核心算法实现
├── yolov5.py         # YOLOv5 检测器封装
├── bytetrack.py      # 模型封装类（包含 Config 和 ByteTrack）
├── main.py           # 主推理脚本
├── run.sh            # 一键运行脚本
└── README.md         # 使用说明
```

## 参数说明

| 参数           | 说明                                                     | 默认值                                      |
|----------------|----------------------------------------------------------|---------------------------------------------|
| `--model-path` | 检测模型文件路径（.hbm 格式）                               | `../../model/yolov5x_672x672_nv12.hbm` |
| `--input`      | 输入视频路径                                              | `../../test_data/track_test.mp4`            |
| `--output`     | 输出视频路径                                              | `../../test_data/result.mp4`                                |
| `--priority`   | 模型调度优先级（0~255）                                     | `0`                                         |
| `--bpu-cores`  | 使用的 BPU 核心编号列表（如 `--bpu-cores 0 1`）              | `[0]`                                      |
| `--score-thres`| 检测置信度阈值                                            | `0.25`                                      |
| `--track-thresh`| 跟踪置信度阈值                                           | `0.3`                                       |


## 快速运行
- 运行模型
    - 使用默认参数
        ```bash
        bash run.sh
        ```
    - 指定参数运行
        ```bash
        python main.py \
            --model-path ../../model/yolov5x_672x672_nv12.hbm \
            --input ../../test_data/track_test.mp4 \
            --output ../../test_data/result.mp4 \
            --score-thres 0.25 \
            --track-thresh 0.3
        ``` 
- 查看结果

    运行成功后，将在`test_data`目录下生成 `result.mp4`，包含跟踪轨迹和 ID。
    运行效果如图:![result](../../test_data/result.jpg)


## 接口说明
阅读[源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档；
本示例代码提供了详细的注释。为了获取最准确、最新的接口定义，请直接查阅源码中的文档字符串：
- **ByteTrackConfig** 与 **ByteTrack**: 详见 `bytetrack.py`
- **YOLOv5Config** 与 **YoloV5X**: 详见 `yolov5.py`

## 注意事项
 无