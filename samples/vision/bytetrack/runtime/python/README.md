# ByteTrack Python Runtime

本示例展示如何在 RDK 平台上使用 ByteTrack 算法进行多目标跟踪。示例包含 YOLOv5 检测器的推理和 BYTETracker 的更新逻辑。

## 环境依赖
- RDK 平台
- Python 3.8+
- hbm_runtime
- numpy, opencv-python, scipy, lap

安装命令：
```bash
pip install numpy opencv-python scipy lap
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
| `--input`      | 输入视频路径                                              | `../../test_data/test_video.mp4`            |
| `--output`     | 输出视频路径                                              | `result.mp4`                                |
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
            --input ../../test_data/test_video.mp4 \
            --output result.mp4 \
            --score-thres 0.25 \
            --track-thresh 0.3
        ```
- 查看结果

    运行成功后，将在当前目录下生成 `result.mp4`，包含跟踪轨迹和 ID。


## 接口说明

### ByteTrackConfig类

```python
@dataclass
class ByteTrackConfig
```

- 功能

    配置 ByteTrack 及其检测器的参数。继承自 `YOLOv5Config`。

- 参数（新增字段）

    | 字段名           | 类型           | 说明                      |
    | ------------- | ------------ | ----------------------- |
    | `track_thresh`| `float`      | 跟踪置信度阈值               |
    | `track_buffer`| `int`        | 轨迹保留帧数                |
    | `match_thresh`| `float`      | 匹配阈值                   |


### ByteTrack类

#### ByteTrack构造函数

```python
def __init__(self, config: ByteTrackConfig)
```

- 功能

    加载 YOLOv5 检测模型并初始化跟踪器。

- 参数

    | 参数名      | 类型             | 说明   |
    | -------- | -------------- | ---- |
    | `config` | `ByteTrackConfig` | 配置对象 |


#### ByteTrack.predict

```python
def predict(self, img: np.ndarray) -> List[STrack]
```

- 功能

    执行完整的跟踪流程：预处理 → 检测推理（YOLOv5） → 后处理 → 轨迹更新。

- 参数

    | 参数名 | 类型 | 说明 |
    | --- | --- | --- |
    | `img` | `np.ndarray` | 输入图像 (BGR) |

- 返回值

    | 返回值 | 类型 | 说明 |
    | --- | --- | --- |
    | `tracks` | `List[STrack]` | 当前帧的跟踪目标列表 |

## 注意事项
 无