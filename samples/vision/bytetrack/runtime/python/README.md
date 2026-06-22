English | [简体中文](./README_cn.md)

# ByteTrack Python Runtime

This sample demonstrates how to use the ByteTrack algorithm for multi-object tracking on the RDK platform. The sample includes YOLOv5 detector inference and BYTETracker update logic.

## Dependencies

- RDK platform
- Python 3.8+
- hbm_runtime
- numpy>=1.24.0
- opencv-python>=4.5.0
- scipy>=1.7.0
- lap==0.5.12
- cython-bbox==0.1.5

Install command:

```bash
pip install numpy==1.26.4 opencv-python==4.11.0.86 scipy==1.15.3 lap==0.5.12 cython-bbox==0.1.5
```

## Directory Structure

```text
.
├── tracker/          # ByteTrack core algorithm implementation
├── yolov5.py         # YOLOv5 detector wrapper
├── bytetrack.py      # Model wrapper class (includes Config and ByteTrack)
├── main.py           # Main inference script
├── run.sh            # One-click run script
└── README.md         # Usage instructions
```

## Parameters

| Parameter        | Description                                           | Default                                     |
|------------------|-------------------------------------------------------|---------------------------------------------|
| `--model-path`   | Detection model file path (.hbm format)               | `../../model/s100/yolov5x_672x672_nv12.hbm` |
| `--input`        | Input video path                                      | `../../test_data/track_test.mp4`            |
| `--output`       | Output video path                                     | `result.mp4`                                |
| `--priority`     | Model scheduling priority (0~255)                     | `0`                                         |
| `--bpu-cores`    | List of BPU core IDs to use (e.g., `--bpu-cores 0 1`) | `[0]`                                       |
| `--score-thres`  | Detection confidence threshold                        | `0.25`                                      |
| `--track-thresh` | Tracking confidence threshold                         | `0.3`                                       |

## Quick Start

- Run the model:

    - With default parameters:

        ```bash
        bash run.sh
        ```

    - With custom parameters:

        ```bash
        python main.py \
            --model-path ../../model/s100/yolov5x_672x672_nv12.hbm \
            --input ../../test_data/track_test.mp4 \
            --output result.mp4 \
            --score-thres 0.25 \
            --track-thresh 0.3
        ```

- View results:

    After successful execution, `result.mp4` will be generated in the current directory, containing tracking trajectories and IDs.

## Interface Documentation

The sample code provides detailed comments. For the most accurate and up-to-date interface definitions, refer directly to the docstrings in the source code:
- **ByteTrackConfig** and **ByteTrack**: see `bytetrack.py`
- **YOLOv5Config** and **YoloV5X**: see `yolov5.py`

## Notes

The default pipeline tracks only the COCO `person` class. To track other classes, update the class filtering logic in `bytetrack.py`.
