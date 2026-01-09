# EfficientNet-Lite0 图像分类示例

本示例展示如何在 BPU 上使用量化后的 EfficientNet-Lite0 模型执行图像分类任务。支持前处理、模型推理以及 Top-K 结果提取。

## 环境依赖
本样例无特殊环境需求，只需确保安装了 py_utils 中提到的环境依赖即可。
```bash
pip install numpy opencv-python scipy
```

## 目录结构
```text
.
├── efficientnet.py   # 模型封装类（包含 Config 和 Model）
├── main.py           # 主推理脚本
├── run.sh            # 一键运行脚本
└── README.md         # 使用说明
```

## 参数说明

| 参数           | 说明                                                     | 默认值                                      |
|----------------|----------------------------------------------------------|---------------------------------------------|
| `--model-path` | 模型文件路径（.hbm 格式）                                  | `/opt/hobot/model/s100/basic/efficientnet_lite0_224x224_nv12.hbm` |
| `--test-img`   | 测试图片路径                                              | `../../test_data/Scottish_deerhound.JPEG`                     |
| `--label-file` | 类别标签路径（每行一个类别或字典格式）                                | `../../../../../datasets/imagenet/imagenet1000_clsidx_to_labels.txt`           |
| `--priority`  | 模型调度优先级（0~255）                                     | `0`                                         |
| `--bpu-cores` | 使用的 BPU 核心编号列表（如 `--bpu-cores 0 1`）              | `[0]`                                      |
| `--topk`      | 输出置信度最高的 K 个类别                                    | `5`                                         |
| `--resize-type` | 预处理缩放模式：0-直接缩放，1-保持比例(Letterbox)           | `1`                                         |


## 快速运行
- 运行模型
    - 使用默认参数
        ```bash
        bash run.sh
        ```
    - 指定参数运行
        ```bash
        python main.py \
            --model-path ../../model/efficientnet_lite0_224x224_nv12.hbm \
            --test-img ../../test_data/Scottish_deerhound.JPEG \
            --label-file ../../../../../datasets/imagenet/imagenet1000_clsidx_to_labels.txt \
            --topk 5 \
            --priority 0 \
            --bpu-cores 0
        ```
- 查看结果

    运行成功后，终端将输出 Top-K 的分类预测结果：
    ```bash
    Top-5 Results:
      Scottish deerhound, deerhound: 0.8234
      greyhound: 0.0123
      Saluki, gazelle hound: 0.0056
      whippet: 0.0034
      Irish wolfhound: 0.0021
    ```


## 接口说明

### EfficientNetConfig类

```python
@dataclass
class EfficientNetConfig
```

- 功能

    用于配置 EfficientNet 模型的初始化、预处理及后处理参数，除模型路径外，其余字段均提供推荐默认值。

- 参数（字段）

    | 字段名           | 类型           | 说明                      |
    | ------------- | ------------ | ----------------------- |
    | `model_path`  | `str`        | `.hbm` 模型文件路径           |
    | `num_classes` | `int`        | 类别数，默认 1000               |
    | `topk`        | `int`        | 后处理返回的预测类别数量，默认 5    |
    | `resize_type` | `int`        | 缩放方式：0 直接缩放，1 等比填充        |


- 返回值

    无（配置对象）。

### EfficientNet类

#### EfficientNet构造函数

```python
def __init__(self, config: EfficientNetConfig)
```

- 功能

    加载 EfficientNet 模型并读取模型输入输出信息。

- 参数

    | 参数名      | 类型             | 说明   |
    | -------- | -------------- | ---- |
    | `config` | `EfficientNetConfig` | 模型配置 |


- 返回值

    无。

#### EfficientNet.set_scheduling_params

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

#### EfficientNet.pre_process

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




#### EfficientNet.forward

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


#### EfficientNet.post_process

```python
def post_process(
    self,
    outputs: Dict[str, Dict[str, np.ndarray]],
    topk: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]
```

- 功能

    将模型原始输出转换为置信度和类别索引。

- 参数

    | 参数名           | 类型                | 说明       |
    | ------------- | ----------------- | -------- |
    | `outputs`     | `dict`            | 推理输出     |
    | `topk`        | `Optional[int]`   | Top-K 数量覆盖 |


- 返回值（顺序固定）

    | 返回值       | 类型           | 说明                      |
    | --------- | ------------ | ----------------------- |
    | `topk_probs` | `np.ndarray` | `(K,)`，前 K 个预测的概率 |
    | `topk_indices` | `np.ndarray` | `(K,)`，前 K 个预测的索引 |


#### EfficientNet.predict

```python
def predict(
    self,
    img: np.ndarray,
    image_format: str = "BGR",
    resize_type: Optional[int] = None,
    topk: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]
```

- 功能

    执行完整推理流程：预处理 → 推理 → 后处理。

- 参数

    同 pre_process / post_process。

- 返回值

    同 post_process。

#### EfficientNet.__call__

```python
def __call__(self, img: np.ndarray, ...) -> Tuple[np.ndarray, np.ndarray]
```

- 功能

    可调用接口，等价于 predict()。

- 参数

    同 predict()。

- 返回值

    同 predict()。

## 注意事项
 无