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
| 参数              | 说明                       | 默认值                                     |
| --------------- | --------------------------- | ------------------------------------------ |
| `--model-path`  | 模型文件路径（.hbm 格式）     | `/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm` |
| `--test-img`    | 测试图片路径                 | `/app/res/assets/kite.jpg`                 |
| `--label-file`  | 类别标签文件路径             | `/app/res/labels/coco_classes.names`       |
| `--score-thres` | 置信度阈值 (过滤低分框)      | `0.25`                                     |
| `--nms-thres`   | IoU 阈值 (NMS 非极大值抑制)  | `0.45`                                     |

## 快速运行
- 运行模型
    - 使用默认参数
        ```bash
        ./yolov5
        ```
    - 指定参数运行
        ```bash
        ./yolov5 \
            --model-path /opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm \
            --test-img /app/res/assets/kite.jpg \
            --label-file /app/res/labels/coco_classes.names \
            --score-thres 0.25 \
            --nms-thres 0.45
        ```
- 查看结果

    运行成功后，会将目标检测框绘制在原图上，并保存在 build/result.jpg
    ```bash
    [Saved] Result saved to: result.jpg
    ```

## 接口说明
### 配置结构体 Yolov5Config
- 功能

    用于描述 YOLOv5 后处理（解码、阈值过滤、NMS、坐标映射）所需的配置参数（如 strides / anchors / feature map 尺寸 / 阈值等）。所有字段均提供推荐默认值，适用于标准 YOLOv5 模型；若编译模型的输入分辨率、输出布局、anchors/strides 发生变化，需要同步调整这些字段。

- 参数

    | 参数名            | 类型                               | 说明                       |
    | -------------- | -------------------------------- | ------------------------ |
    | `strides`      | `std::vector<int>`               | 各检测层特征图步长，默认 `{8,16,32}` |
    | `anchors`      | `vector<vector<array<float,2>>>` | 各检测层 Anchor 定义           |
    | `hw_list`      | `vector<pair<int,int>>`          | 各检测层特征图尺寸 `(H, W)`       |
    | `score_thresh` | `float`                          | 置信度阈值，默认 `0.25`          |
    | `nms_thresh`   | `float`                          | NMS IoU 阈值，默认 `0.45`     |
    | `num_classes`  | `int`                            | 目标类别数量，默认 `80`           |
    | `resize_mode`  | `int`                            | 图像缩放方式：`0` 拉伸，`1` 等比填充   |

- 返回值

    无。

### YOLOv5x 类

- 说明：

    该类是模型句柄与张量缓冲区的资源持有者。推理流程通常由外部函数组合完成：pre_process() → infer() → post_process()。
#### 构造函数 YOLOv5x
```c++
YOLOv5x();
```

- 功能

    创建 YOLOv5x 对象（未初始化状态）。构造函数不执行模型加载与内存分配；需要调用 init() 完成模型加载、Tensor 属性查询与 buffer 分配。

- 参数

    无

- 返回值

    无。

#### 初始化接口 init

```c++
int32_t init(const char* model_path);
```

- 功能

    初始化模型资源，包括：

    - 从磁盘加载 hbm 模型

    - 获取模型名称列表并选择第一个模型句柄

    - 查询输入/输出 Tensor 属性

    - 分配输入/输出 Tensor buffer

    - 缓存模型输入分辨率（input_h/input_w）

    init() 不允许重复调用。

- 参数

    | 参数名        | 类型          | 说明 |
    |--------------|---------------|------|
    | `model_path` | `const char*` | 量化模型文件路径（`.hbm`） |


- 返回值

    | 返回值   | 说明    |
    | ----- | ----- |
    | `0`   | 初始化成功 |
    | 非 `0` | 初始化失败 |



#### 析构函数 ~YOLOv5x

```c++
YOLOv5x::~YOLOv5x()；
```

- 功能

    释放输入/输出 Tensor buffer 模型句柄（hbDNNRelease）。即使 init() 失败或部分初始化，也应保证安全释放。

- 参数 / 返回值

    无。


#### 预处理接口 pre_process

```c++
int32_t pre_process(std::vector<hbDNNTensor>& input_tensors,
                    cv::Mat& img,
                    const int input_w,
                    const int input_h,
                    const std::string& image_format = "BGR");

```

- 功能

    对输入图像进行预处理并写入模型输入 Tensor buffer。当前实现流程：

    - 将输入图像按 input_w/input_h 进行 letterbox 缩放

    - 将 BGR 转换为 NV12，并写入 input_tensors[0].sysMem

- 参数

    | 参数名         | 类型                         | 说明 |
    |---------------|------------------------------|------|
    | `input_tensors` | `std::vector<hbDNNTensor>&` | 模型输入 Tensor 列表（将被填充） |
    | `img`         | `cv::Mat&`                   | 输入图像 |
    | `input_w`     | `int`                        | 模型输入宽度（像素） |
    | `input_h`     | `int`                        | 模型输入高度（像素） |
    | `image_format`| `std::string`                | 输入图像格式，当前仅支持 `"BGR"` |

- 返回值

    | 返回值   | 说明 |
    | ----- | -- |
    | `0`   | 成功 |
    | 非 `0` | 失败 |


#### 推理接口 infer

```c++
int32_t infer(std::vector<hbDNNTensor>& output_tensors,
              std::vector<hbDNNTensor>& input_tensors,
              const hbDNNHandle_t dnn_handle,
              hbUCPSchedParam* sched_param = nullptr);
```

- 功能

    提交 BPU 推理任务并等待执行完成。

- 参数

    | 参数名          | 类型                          | 说明 |
    |----------------|-------------------------------|------|
    | `output_tensors` | `std::vector<hbDNNTensor>&` | 输出 Tensor 列表（将被填充） |
    | `input_tensors`  | `std::vector<hbDNNTensor>&` | 输入 Tensor 列表（已预处理并写入 buffer） |
    | `dnn_handle`   | `hbDNNHandle_t`               | 模型句柄 |
    | `sched_param`  | `hbUCPSchedParam*`            | 调度参数；为 `nullptr` 时使用默认参数（默认 `HB_UCP_BPU_CORE_ANY`） |


- 返回值

    | 返回值   | 说明   |
    | ----- | ---- |
    | `0`   | 推理成功 |
    | 非 `0` | 推理失败 |


#### 后处理接口 post_process

```c++
void post_process(std::vector<Detection>& results,
                  std::vector<hbDNNTensor>& output_tensors,
                  const Yolov5Config& config,
                  int orig_img_w,
                  int orig_img_h,
                  int input_w,
                  int input_h);
```

- 功能

    将模型输出 Tensor 转换为最终目标检测结果。当前实现流程：

    - 对每个输出 Tensor 进行反量化为 float（当前使用 S32 反量化路径）

    - 解码得到 bbox 与分类得分，并按 config.score_thresh 过滤

    - 按类别执行 NMS（config.nms_thresh）

    - 将 bbox 从 letterbox 坐标系映射回原始图像尺寸

- 参数

    | 参数名           | 类型                        | 说明 |
    |-----------------|-----------------------------|------|
    | `results`       | `std::vector<Detection>&`   | 输出检测结果 |
    | `output_tensors`| `std::vector<hbDNNTensor>&` | 推理输出 Tensor 列表 |
    | `config`        | `Yolov5Config`              | 后处理配置（anchors/strides/hw_list/阈值/类别数等） |
    | `orig_img_w`    | `int`                       | 原始图像宽度 |
    | `orig_img_h`    | `int`                       | 原始图像高度 |
    | `input_w`       | `int`                       | 模型输入宽度（用于回映射） |
    | `input_h`       | `int`                       | 模型输入高度（用于回映射） |

- 返回值

    无。


## 注意事项
- 无
