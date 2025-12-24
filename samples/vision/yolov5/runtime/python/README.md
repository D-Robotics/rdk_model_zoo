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

### YOLOv5Config类

```python
@dataclass
class YOLOv5Config
```

- 功能
    用于配置 YOLOv5 模型的初始化、预处理及后处理参数，除模型路径外，其余字段均提供推荐默认值，可直接使用。

- 参数（字段）
    | 字段名           | 类型           | 说明                      |
    | ------------- | ------------ | ----------------------- |
    | `model_path`  | `str`        | `.hbm` 模型文件路径           |
    | `classes_num` | `int`        | 类别数，默认 80               |
    | `resize_type` | `int`        | 缩放方式：0 拉伸，1 等比填充        |
    | `score_thres` | `float`      | 置信度阈值                   |
    | `nms_thres`   | `float`      | NMS 阈值                  |
    | `strides`     | `np.ndarray` | strides，默认 `[8,16,32]`  |
    | `anchors`     | `np.ndarray` | anchors，shape `(3,3,2)` |


- 返回值

    无（配置对象）。

### YoloV5X类

#### YoloV5X构造函数

```python
def __init__(self, config: YOLOv5Config)
```

- 功能

    加载 YOLOv5 模型并读取模型输入输出信息。

- 参数

    | 参数名      | 类型             | 说明   |
    | -------- | -------------- | ---- |
    | `config` | `YOLOv5Config` | 模型配置 |


- 返回值

    无。

#### YoloV5X.set_scheduling_params

```python
def set_scheduling_params(
    self,
    priority: Optional[int] = None,
    bpu_cores: Optional[list] = None
) -> None
```

- 功能

    设置 BPU 推理调度参数。

- 参数

    | 参数名         | 类型               | 说明              |
    | ----------- | ---------------- | --------------- |
    | `priority`  | `Optional[int]`  | 推理优先级 `[0,255]` |
    | `bpu_cores` | `Optional[list]` | 使用的 BPU 核心列表    |


- 返回值

    无。

#### YoloV5X.pre_process

```python
def pre_process(
    self,
    img: np.ndarray,
    resize_type: Optional[int] = None,
    image_format: Optional[str] = "BGR"
) -> Dict[str, Dict[str, np.ndarray]]
```

- 功能

    对输入图像进行预处理，生成模型所需输入 Tensor。

- 参数

    | 参数名            | 类型              | 说明                 |
    | -------------- | --------------- | ------------------ |
    | `img`          | `np.ndarray`    | 输入图像               |
    | `resize_type`  | `Optional[int]` | 缩放策略覆盖             |
    | `image_format` | `str`           | 输入格式，当前仅支持 `"BGR"` |



- 返回值

    | 返回值            | 类型     | 说明                                                |
    | -------------- | ------ | ------------------------------------------------- |
    | `input_tensor` | `dict` | 输入 Tensor 字典 `{model_name: {input_name: tensor}}` |





#### YoloV5X.forward

```python
def forward(
    self,
    input_tensor: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]
```

- 功能

    执行模型推理。

- 参数

    | 参数名            | 类型     | 说明             |
    | -------------- | ------ | -------------- |
    | `input_tensor` | `dict` | 预处理后的输入 Tensor |

- 返回值

    | 返回值       | 类型     | 说明            |
    | --------- | ------ | ------------- |
    | `outputs` | `dict` | 模型原始输出 Tensor |


#### YoloV5X.post_process

```python
def post_process(
    self,
    outputs: Dict[str, Dict[str, np.ndarray]],
    ori_img_w: int,
    ori_img_h: int,
    score_thres: Optional[float] = None,
    nms_thres: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

- 功能

    将模型输出转换为最终目标检测结果。

- 参数

    | 参数名           | 类型                | 说明       |
    | ------------- | ----------------- | -------- |
    | `outputs`     | `dict`            | 推理输出     |
    | `ori_img_w`   | `int`             | 原图宽      |
    | `ori_img_h`   | `int`             | 原图高      |
    | `score_thres` | `Optional[float]` | 置信度阈值覆盖  |
    | `nms_thres`   | `Optional[float]` | NMS 阈值覆盖 |


- 返回值（顺序固定）

    | 返回值       | 类型           | 说明                      |
    | --------- | ------------ | ----------------------- |
    | `boxes`   | `np.ndarray` | `(N,4)`，`[x1,y1,x2,y2]` |
    | `scores`  | `np.ndarray` | `(N,)`                  |
    | `cls_ids` | `np.ndarray` | `(N,)`                  |


#### YoloV5X.predict

```python
def predict(
    self,
    img: np.ndarray,
    image_format: str = "BGR",
    resize_type: Optional[int] = None,
    score_thres: Optional[float] = None,
    nms_thres: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

- 功能

    执行完整推理流程：预处理 → 推理 → 后处理。

- 参数

    同 pre_process / post_process。

- 返回值

    同 post_process。

#### YoloV5X.__call__

```python
def __call__(self, img: np.ndarray, ...) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

- 功能

    可调用接口，等价于 predict()。

- 参数

    同 predict()。

- 返回值

    同 predict()。

## 注意事项
 无
