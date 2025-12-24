# Model Zoo仓库规范
本文档用于统一 Model Zoo 仓库的目录组织、编码风格、接口规范、注释规则与文档编写规范，以保证不同模型、不同任务、不同语言（C / Python）之间具备 一致的使用体验、良好的复用性与长期可维护性。

本文档适用对象：仓库开发者、仓库维护者

## 目录和文件规范
### 目录规范

本仓库采用按功能与任务分层的目录组织方式，以统一结构、降低接入与维护成本，所有新增文件或目录，必须按照其功能属性放置到对应层级，不允许随意新增顶层目录或跨职责混放。

各级目录功能如下（仅用于举例说明各目录功能，实际内容可能有差异）：

```bash
.
├── datasets                                         # 示例数据 + 数据集下载脚本
│   ├── coco                                         # COCO 数据资源
│   │   ├── samples                                  # 小样本数据（快速跑通用）
│   │   │   ├── annotations                          # 示例标注
│   │   │   └── images                               # 示例图片
│   │   ├── README.md                                # COCO 目录说明
│   │   ├── download_full_coco.sh                    # 下载完整 COCO 数据集脚本
│   │   └── labels.txt                               # COCO 标签文件
│   ├── imagenet                                     # ImageNet 示例数据
│   └── README.md                                    # datasets 目录说明
├── docs                                             # 文档与资源，可添加各接口的链接文档
│   ├── images                                       # 文档图片
│   └── Model_Zoo_Repository_Guidelines.md           # 本仓库中的规范说明
├── samples                                          # 所有模型示例（按任务分类）
│   ├── llm                                          # 大语言模型示例（底线是文档链接）
│   ├── speech                                       # 语音模型示例
│   ├── vision                                       # 视觉模型示例
│   │   └── yolov5                                   # YOLOv5 模型示例
│   │       ├── 3rdparty                             # YOLOv5 特定第三方依赖
│   │       ├── conversion                           # 模型转换脚本（ONNX → HBM）
│   │       ├── evaluator                            # 精度/性能评估
│   │       ├── model                                # 模型文件及下载脚本
│   │       │   └── download_model.sh                # 下载预训练模型
│   │       ├── runtime                              # 推理示例（C++/Python）
│   │       │   ├── cpp                              # C++ 推理实现
│   │       │   │   ├── inc                          # 前后处理头文件
│   │       │   │   │   └── yolov5.hpp
│   │       │   │   ├── src                          # 前后处理与推理实现
│   │       │   │   │   ├── main.cpp                 # 主推理入口
│   │       │   │   │   └── yolov5.cpp
│   │       │   │   ├── CMakeLists.txt               # C++ 构建脚本
│   │       │   │   ├── run.sh                       # 一键运行脚本
│   │       │   │   └── README.md                    # C++ 推理说明
│   │       │   └── python                           # Python 推理实现
│   │       │       ├── README.md                    # Python 推理说明
│   │       │       ├── run.sh                       # 一键运行脚本
│   │       │       ├── main.py                      # 主推理入口
│   │       │       └── yolov5.py
│   │       ├── test_data                            # 测试用数据（图片/标签）
│   │       └── README.md                            # YOLOv5 模型说明文档
│   ├── vla                                          # Vision-Language-Action 模型示例
│   └── vlm                                          # Vision-Language 模型示例
├── tools                                            # 工具脚本（构建/管理/调试）
├── tros                                             # TROS相关链接
├── utils                                            # 通用工具库
│   ├── c_utils                                      # C/C++ 公共工具
│   │   ├── inc                                      # 公共头文件
│   │   └── src                                      # 公共实现
│   ├── py_utils                                     # Python 公共工具
│   └── README.md                                    # utils 模块说明
├── LICENSE                                          # 开源协议
└── README.md                                        # 顶层项目说明（介绍结构、使用方法与示例）
```

适用场景：
- 新增文件或文件夹时；

适用对象：
- Model Zoo 示例开发者
- 仓库维护与评审人员
- 文档与工具贡献者

### 文件规范
#### 规范说明

本规范用于统一 Model Zoo 仓库中关键文件的命名规则，以保证不同模型、不同任务及不同语言实现之间具备一致的可读性与可维护性。

- 本文档中明确规定的文件命名规则，开发者必须严格遵守；

- 对于本文档未覆盖的文件类型，开发者可根据实际需求合理命名；

- 如新增文件类型在多个模型或任务中具有通用性或复用价值，建议同步补充或更新本规范，以保持整体一致性。

#### Sample 源码命名规范

在 Sample 示例中，无论采用 C/C++ 还是 Python 实现，模型相关源码文件需遵循以下统一约定：
- 模型实现源码必须以**模型名称**作为文件名；
- 可执行入口程序，必须统一命名为 main，用于承载示例的主推理流程；
- 一键运行脚本必须统一命名为 run.sh，用于快速启动与验证 Sample；
- C/C++ Sample 中，模型实现文件的 .cc 与 .hpp 文件名必须保持一致；

示例：
```bash
# python 命名
yolov5.py
main.py
run.sh

# C/C++ 命名
yolov5.hpp
yolov5.cc
main.cpp
run.sh
```

#### 模型文件命名规范

模型文件（如 .hbm）需采用统一的命名格式，以便于快速识别其模型类型、输入规格及适用平台。

- 命名格式

    <model_name>_<input_resolution>_<chip_name>.hbm

- 字段说明

    | 字段                | 说明                                                   |
    | ------------------ | ----------------------------------------------------- |
    | `model_name`       | 模型名称，**必须包含模型名、版本信息、模型型号（s,m,l,x）** |
    | `input_resolution` | 模型输入分辨率，如 `640x640`、`672x672`                 |
    | `chip_name`        | 适用芯片型号，字母全部小写，如 `s100`、`s600`            |
    | `.hbm`             | 固定后缀，表示 BPU 模型文件                             |

- 示例
    ```bash
    yolov5x_672x672_s100.hbm
    ```

## 编码规范

### 整体框架规范
为保证不同模型、不同任务在使用方式与工程结构上的一致性，所有 Sample（无论 Python 还是 C/C++）的代码框架至少必须包含三个核心组成部分：模型实现文件（如yolov5.py）、主程序入口（如main.py）、一键运行脚本（如run.sh），分别用于承载模型逻辑、程序执行入口与快速运行验证，相关实现规范将在后续章节中逐项说明。
#### python规范
本节对 Python Sample 的模型代码、main 函数，实际工程示例可参考 samples/vision/yolov5/runtime/python 目录中的实现方式。

##### 模型代码规范
每个模型文件必须包含两个基础类：配置类（Config） 与 模型类（Model）。配置类用于配置模型初始化与推理相关的参数，例如模型路径、阈值、类别数等；模型类则封装了模型的完整运行流程，包括预处理、推理与后处理等操作。通过将参数与逻辑分离，模型结构更清晰，也便于维护与扩展。

- 配置类规范
    - 每个模型必须定义一个独立的配置类 XXXConfig，用于存放模型初始化和运行时参数。
    - 配置类需包含默认值，并且覆盖推荐的典型使用场景，保证默认情况可直接跑通 Sample。

- 模型类规范

    模型类用于封装一次完整推理链路，所有模型（检测、分类、分割、多模态等）都应按照统一结构实现。每个模型类需至少包含以下方法，并遵守对应职责约定：
    - \_\_init\_\_ – 模型初始化
        - 职责：
            - 加载模型；
            - 提取模型 metadata（模型名、输入输出名、输入 shape、量化信息等）；

    - set_scheduling_params – 调度参数设置
        - 职责：
            - 用于设置推理调度相关参数，例如优先级、运行核绑定等；
        - 行为要求（可选）：
            - 入参可为空，若所有参数均为 None，函数应为无副作用；

    - pre_process – 预处理

        - 职责：
            - 将用户输入的数据转换为底层 runtime 所需的输入张量格式；
            - 需对不支持的输入类型/格式抛出 ValueError 或合适异常，而不是静默失败；

        - 返回值
            - 必须为符合hbm_runtime中run接口输入规范，可以直接作为参数传入run方法，进行模型的推理，以便统一对接底层 forward() 接口；

    - forward – 前向推理执行

        - 职责：
            - 调用底层 runtime 执行推理；
            - 入参必须为符合hbm_runtime中run接口输入规范的字典（即 {model_name: {input_name: tensor}}）；
            - 返回值应为hbm_hbruntime中run方法的直接输出；

    - post_process – 后处理
        - 职责：
            - 将 forward 的原始输出张量转换为业务语义结果（检测框、类别、分割 mask、文本结果等）；
        - 参数
            - 输入矩阵必须以形成的形式传入，方便以后多线程扩展；
        - 返回值
            - 必须严格遵守对应任务规范中约定的输出结构与字段含义；

    - predict – 一键推理接口

        - 职责：
            - 将完整推理流程封装为一个高层调用，统一串联：pre_process → forward → post_process
        - 返回值
            - 返回值应与 post_process 保持一致，是推荐给业务层/示例代码调用的主接口。

    - \_\_call\_\_ – 可调用模型实例

        - 职责：
            - 为模型实例提供“函数式调用”能力，语义上等价于 predict；

##### main程序规范

    Main 函数规范用于统一 Sample 示例的程序入口形式，要求通过带默认值的命令行参数完成配置，使示例在零参数情况下即可跑通，同时支持用户按需覆盖参数以实现灵活定制。

    - 参数传递规范

        - 要求：
            - 使用 argparse.ArgumentParser；
            - 参数必须具备“默认可用”能力；
            - 参数名采用 kebab-case：如 --model-path、--score-thres；
            - 每个参数必须包含：type、default、help（一句话清晰描述）；



#### c/c++规范

本节对 C/C++ Sample 的模型代码、main 函数，实际工程示例可参考 samples/vision/yolov5/runtime/cpp 目录中的实现方式。



##### 模型代码规范
C/C++ Sample 的模型代码整体划分为配置结构体、模型类、推理相关函数三部分，分别用于统一管理模型预处理与后处理相关参数、管理模型句柄和Tensor资源以及推理流程。

- 配置结构体规范
    - 每个模型必须定义一个独立的配置结构体，用于存放模型初始化和运行时参数；
    - 结构体需包含默认值，并且覆盖推荐的典型使用场景，保证默认情况可直接跑通 Sample；

- 模型类规范
    模型类用于管理模型运行时资源，在 C/C++ 中通常以一个类实现，模型类至少需要包含以下方法，并遵循明确的职责约束。
    - 构造函数 - 模型对象创建
        - 职责：
            - 创建模型对象本身；
            - 不执行任何重资源操作、不加载模型、不分配 Tensor 内存；
            - 为后续显式调用 init() 做准备；


    - init - 模型资源初始化
        - 职责
            - 加载模型（如通过 Horizon DNN API）；
            - 查询模型元信息；
            - 分配输入 / 输出 tensor 所需的内存；
        - 返回值
            - 返回错误码，0 表示成功，非 0 表示失败；

- 推理相关函数

    - pre_process – 预处理

        - 职责：
            - 将用户输入数据转换为 runtime 所需格式；
            - 对不支持的 image_format 应及时报错，而非静默失败；
            - 模型的input tensor必须以引用方式传递并写入，便于后续多线程/并行扩展；


    - infer – 前向推理执行

        - 职责：
            - 创建推理任务；
            - 提交任务到 BPU 调度器；
            - 等待任务完成；
            - 释放任务句柄；
            - 可通过参数配置底层硬件的调度参数，如sched_param，参数设置缺省值；
            - 输入输出tensor需以形参的形式传入，便于后续多线程/并行扩展；
        - 返回值：
            - 当推理出现底层错误时返回错误码，0为无错误，非0为异常；


    - post_process – 后处理
        - 职责：
            - 将 raw 输出 tensor 转换为业务语义结果；
            - 模型输出tensor需以形参的形式传入，便于后续多线程/并行扩展；

        - 输出结果约定：
            - 必须严格遵守对应任务规范中约定的输出结构与字段含义；

##### main函数规范
Main 程序规范用于统一 C/C++ Sample 的程序入口形式，要求通过带默认值的命令行参数完成配置，使示例在零参数情况下即可跑通，同时支持用户按需覆盖参数以实现灵活定制。

- 参数传递规范

    - 要求：
        - 使用 gflags 作为命令行参数解析库；
        - 所有参数必须具备“默认可用”能力（默认值对应推荐的典型运行场景）；
        - 参数名采用 snake_case：如 --model_path、--score_thres；
        - 每个参数定义必须包含：类型 / default / help（help 需用一句话清晰描述）；


#### 运行脚本规范

一键运行脚本的主要作用是做代码环境的配置、模型下载，编译工作以及运行，真正做到代码一键运行，实际工程示例可参考 samples/vision/yolov5/runtime/cpp以及 samples/vision/yolov5/runtime/python 目录中的实现方式。


一键运行脚本run.sh可以包含如下内容：
- 设备型号读取
- 环境配置
- 模型下载
- 模型编译（仅C/C++）
- 快速运行

### 各类任务规范
本节定义按任务划分的编码规范，用于统一不同任务（如分类、检测、分割、姿态等）的接口形式与代码组织方式，降低用户在不同模型与任务之间切换时的理解与使用成本。

#### 目标检测模型规范
目标检测算法用于在图像中同时完成目标定位与类别识别，是计算机视觉中的基础任务之一。在 ModelZoo 中，所有目标检测模型均以统一的结果输出格式对外提供推理结果，以便于模型之间的对比、替换与集成。因此，目标检测模型在完成推理与后处理后，必须按照以下规范返回检测结果。

##### C++ 输出规范
在 C++ 示例中，目标检测结果以结构体形式表示，用于描述单个检测目标，结构体定义及字段含义如下：

```c++
struct Detection {
    float bbox[4];   // 目标边界框，格式为 [x1, y1, x2, y2]，像素坐标
    float score;     // 目标置信度，取值范围通常为 [0, 1]
    int   class_id;  // 目标类别索引，与模型类别定义对应
};
```

模型应以 std::vector<Detection> 的形式返回所有检测结果，表示当前输入图像中检测到的目标集合。

##### Python 输出规范
在 Python 示例中，目标检测结果以多个 NumPy 数组的形式返回，且返回顺序需保持一致，具体要求如下：

- boxes：np.ndarray，形状为 (N, 4)，表示检测框坐标 [x1, y1, x2, y2]

- scores：np.ndarray，形状为 (N,)，表示对应检测框的置信度

- cls_ids：np.ndarray，形状为 (N,)，表示对应检测框的类别索引

其中 N 表示检测到的目标数量。

返回值的顺序必须严格遵循 (boxes, scores, cls_ids)，以保证不同模型在 Python 侧具有一致的使用方式。

#### 分类模型规范
TODO

#### 分割模型规范
TODO

#### 姿态 / 关键点模型规范
TODO

## 注释规范


### Python 注释规范
本规范用于统一代码中的注释风格与 Docstring 书写方式，确保代码语义清晰、行为可理解、接口可复用，便于长期维护与外部开发者直接基于源码进行使用与二次开发。

-  Docstring 总体要求
    - 所有 类、函数、方法 必须编写 docstring。
    - 统一采用 Doxygen 风格或类 C 注释风格（包含 @brief、@param、@return 等标签）。
    - Docstring 必须简明描述功能、参数含义及返回值结构。
    - 示例
        ```python
        def forward(self, inputs):"""
            @brief Run model inference.
            @param inputs (dict)
                Preprocessed input tensor dictionary.
            @return dict
                Output tensors keyed by output tensor name.
            """
        ```

- 类注释规范

    - 简要说明核心功能；
    - 简要说明使用方式；
    - 示例：
        ```python
        class YoloV5X:"""
            @brief Wrapper class for YOLOv5X inference on HB_HBMRuntime.
            Provides unified preprocessing, inference, and postprocessing interfaces.
            Designed to work with YOLOv5 models compiled into .hbm format.
            """
        ```

- 函数与方法注释规范

    - @brief：一段话概述函数的目的；
    - @param：所有输入参数的说明与类型；
    - @return：返回值结构说明，对于相同任务的返回值，多元组数据必须按固定顺序返回（如检测任务固定为bboxes、classes、scores）；
    - 如有异常情况，需在注释中明确抛出的异常类型；
    - 示例
        ```python
        def set_scheduling_params(self,
                                priority: Optional[int] = None,
                                bpu_cores: Optional[list] = None) -> None:
            """
            @brief Configure inference scheduling parameters.

            @param priority (int, optional)
                Inference priority in range [0, 255].
            @param bpu_cores (list[int], optional)
                BPU core indices used for inference.
            @return None
            """
        ```
- 行内注释规范
    - 行内注释用于解释关键步骤或逻辑，不应冗长，注释使用英文。
    - 示例
        ```python
        # Convert BGR image to NV12 format required by runtime
        y, uv = pre_utils.bgr_to_nv12_planes(resized)
        ```

### C / C++ 注释规范

本规范用于统一 C/C++ Sample 中模型代码、配置结构体及预处理/后处理模块的注释风格，通过明确、规范的注释约定，提升代码的可读性、可维护性与跨团队协作效率。

- 类与结构体注释
    - 必须使用 Doxygen 风格注释；
    - 必须说明类型的用途；
    - 示例：
        ```c++
        /**
         * @class YOLOv5x
        * @brief Wrapper class for YOLOv5x inference using Horizon Robotics DNN APIs.
        *
        * This class encapsulates the complete inference pipeline, including:
        * - Model loading and initialization
        * - Input preprocessing
        * - BPU inference execution
        * - Output postprocessing (decode, thresholding, NMS)
        */
        ```
- 函数方法注释规范
    - @brief：一段话说明用途；
    - @param：参数含义说明，明确标注其输入 / 输出特性([in][out])；
    - @return：返回值结构说明（如有），同一任务类型需保持统一返回格式；
    - 示例：
        ```c++
        /**
         * @brief Preprocess an input image into model input tensor buffers.
        *
        * Current implementation:
        * - Letterbox resize to model input resolution (input_w/input_h)
        * - Convert BGR image to NV12 as required by the compiled model
        * - Write the converted data into input_tensors[0].sysMem
        *
        * @param input_tensors [in,out] Model input tensors to be filled.
        * @param input_w [in] Model input width in pixels.
        * @param input_h [in] Model input height in pixels.
        * @param img [in] Input image (OpenCV Mat).
        * @param image_format [in] Input image format string (only "BGR" is supported).
        *
        * @return int32_t 0 on success, non-zero on failure.
        */
        int32_t pre_process(std::vector<hbDNNTensor>& input_tensors,
                            cv::Mat& img,
                            const int input_w, const int input_h,
                            const std::string& image_format = "BGR");
        ```
- 行内注释规范
    - 行内注释用于解释关键逻辑或原因；
    - 注释统一使用英文。


## 文档规范
文档规范主要约束 ModelZoo 仓库中各级目录下的 README.md 编写形式与内容结构，涵盖仓库说明、数据集说明、模型示例、工具模块及运行时示例等文档。

注意：**对当前目录的修改一定要记得检查是否需要修改上一级目录的READM。**

具体涉及文档如下：

```bash
.
|-- README.md
|-- datasets
|   `-- coco
|       `-- README.md
|-- docs
|   |-- README.md
|   |-- Model_Zoo_Repository_Guidelines.md
|   |-- Python_API_User_Guide.md
|   `-- UCP_User_Guide.md
|-- samples
|   `-- vision
|       `-- yolov5
|           |-- README.md
|           |-- conversion
|           |   `-- README.md
|           |-- evaluator
|           |   `-- README.md
|           |-- model
|           |   `-- README.md
|           `-- runtime
|               |-- cpp
|               |   `-- README.md
|               `-- python
|                   `-- README.md
|-- tros
|   `-- README.md
`-- utils
    |-- c_utils
    |   `-- README.md
    `-- py_utils
        `-- README.md
```

### 顶层README
此文件主要介绍整个 Model Zoo 仓库的整体定位、目录组织方式以及模型使用入口，帮助用户快速理解仓库结构并开始使用 BPU 模型。

- 路径：
    - 仓库顶层目录->README.md

- 规范
    - 仓库整体的简要介绍，包括仓库用途、适用场景以及支持的平台（如 RDKS100、RDKS600）；
    - 仓库顶层目录结构说明，帮助用户快速了解各目录的职责划分及内容组织方式；
    - 快速开始指引，指导用户通过模型列表查找目标模型，并进入对应模型目录阅读其 README.md 以完成快速上手；
    - 模型列表索引，以表格形式汇总仓库中提供的模型，包含模型类别、名称、路径及详情入口，作为查找和使用模型的统一入口。

- 更新场景：
    - 仓库管理员：根据实际需求更新整体目录内容；
    - 普通开发者：当新模型开发完成时，将新增模型加入到模型列表，便于用户快速索引；

- 相关人员
    - 仓库管理员
    - 开发人员

### datasets/coco/README.md

介绍此数据集目录中的目录结构以及完整数据集下载方式，其他数据集说明文件的开发可参考此目录下的README文件。

- 路径：
    - datasets/coco/README.md

- 规范
    - 数据集简要介绍；
    - 本文件夹中目录结构和相关文件的介绍；
    - 完整数据集下载脚本或详细的下载说明；
    - 数据集官方网站或版权声明；

- 更新场景：
    - 开发者：当所需数据集仓库中没有时，将新数据的相关资料参考此处进行准备；

- 相关人员
    - 仓库管理员
    - 开发人员

### docs/README.md

介绍文档目录中各markdown文件的作用，方便用户快速定位所需文件。

- 路径：
    - docs/README.md

- 规范
    - 简要介绍docs目录下各markdown文件的作用；

- 更新场景：
    - 当docs路径下有新增文件时；
    - 当docs路径下的文件功能发生改变时；

- 相关人员
    - 仓库管理员
    - 开发人员

### docs/Model_Zoo_Repository_Guidelines.md

统一 Model Zoo 仓库的目录组织、编码风格、接口规范、注释规则与文档编写规范，以保证不同模型、不同任务、不同语言（C / Python）之间具备 一致的使用体验、良好的复用性与长期可维护性。

- 路径：
    - docs/Model_Zoo_Repository_Guidelines.md

- 规范
    - 对有规范需求的目录、文件、代码、注释、文档等具体要求写明；

- 更新场景：
    - 有新的规范需求；

- 相关人员
    - 仓库管理员
    - 开发人员

### docs/Python_API_User_Guide.md

此文件用于指向BPU python接口的相关说明文档，由仓库管理员维护；

- 路径：
    - docs/Python_API_User_Guide.md

- 规范
    - 写清楚引用文档路径即可；

- 更新场景：
    - 仓库管理员：python api的用户手册位置发生变更，或有其他必要说明时；

- 相关人员
    - 仓库管理员


### docs/UCP_User_Guide.md

此文件用于指向libdnn和libucp接口的相关说明文档，由仓库管理员维护；

- 路径：
    - docs/UCP_User_Guide.md

- 规范
    - 写清楚引用文档路径即可；

- 更新场景：
    - 仓库管理员：libdnn或libucp的用户手册位置发生变更，或有其他必要说明时；

- 相关人员
    - 仓库管理员


### samples/vision/yolov5/README.md

此处README面向模型使用者，用于说明模型功能、整体流程与入口位置，而非重复实现细节，所包含的内容细节可参考如下内容准备：

- 路径：
    - samples/vision/yolov5/README.md

- 规范，需包含如下几部分，且符合相应要求（可参考samples/vision/yolov5/README.md）

    1. 标题与简介
        - 标题格式

            `# <模型名称> 模型说明`

        - 简介内容（必需）

            用 1–2 句话简要介绍这个项目。
    2. 算法介绍（Algorithm Overview）
        - 简要介绍一下模型的算法；
        - 给出模型的官方资料链接；
    3. 目录结构（Directory Structure）
        - 给出当前模型目录的树形结构
        - 每一项一句话说明用途
        - 注释对齐，便于阅读
        - 示例格式：
            ```bash
            .
            |-- conversion        # 模型转换流程说明
            |-- model             # 模型文件与下载脚本
            |-- runtime           # 推理示例（Python / C++）
            |-- evaluator         # 模型评估相关内容
            |-- test_data         # 示例输入或推理结果
            `-- README.md         # 当前模型总览说明
            ```
    4. 快速体验（QuickStart）
        - 内容要求
            - 说明快速开始的方法
            - 介绍run.sh有哪些功能，如：
            - 自动下载模型
            - 自动构建（如 C++）
            - 自动运行示例
        - 结构建议
            ```markdown
            ## 快速体验（QuickStart）

            ### C++
            - 运行方式
            - 指向 runtime/cpp/README.md

            ### Python
            - 运行方式
            - 指向 runtime/python/README.md
            ```
        ⚠️ 不要在此处重复参数说明或代码细节

    5. 模型转换（Model Conversion）

        - 提示已经提供了转换好的模型，普通用户可以跳过；
        - 指向转换文档

    6. 模型推理（Runtime）
        - 说明同时提供 C++ / Python
        - 明确指出详细说明在子 README

    7. 模型评估（可选）

        - 说明评估目录的用途，并指向 evaluator README。

    8. 推理结果
        - 若有示例结果，直接展示

    9. License
    遵循 ModelZoo 顶层 License。


- 更新场景：
    - 新增模型sample；
    - 模型内容发生变更；

- 相关人员
    - 开发人员

### samples/vision/yolov5/conversion

此处README的主要目标是想用户提供模型转换的详细教程，开发者可根据情况，能提供尽可能的提供，目前暂无统一规范，如有相关需求请相关人员及时补充；

- 路径：
    - samples/vision/yolov5/conversion/README.md

### samples/vision/yolov5/evaluator

此处README的主要目标是想用户提供模型评估的详细教程，开发者可根据情况，能提供尽可能的提供，目前暂无统一规范，如有相关需求请相关人员及时补充；

- 路径：
    - samples/vision/yolov5/evaluator/README.md

### samples/vision/yolov5/model/README.md

此目录下可简要介绍模型下载的方式；

- 路径：
    - samples/vision/yolov5/model/README.md

- 规范
    - 写清楚模型下载方式即可；

- 更新场景：
    - 新增模型样例；
    - 样例模型发生改变时；

- 相关人员
    - 普通开发人员；


### samples/vision/yolov5/runtime/python(cpp)/README.md

该 README 用于指导用户在完成 YOLOv5x 目标检测示例的编译（仅c/c++）与运行，说明示例工程的目录组成、可用命令行参数与输出结果位置，并提供该示例对外暴露的核心接口（配置结构体与推理类）说明，便于用户快速上手与二次集成。

- 路径：
    - samples/vision/yolov5/runtime/python/README.md
    - samples/vision/yolov5/runtime/cpp/README.md

- 规范，此处有多想要求，具体要求如下（可参考路径中指出的yolov5相关文档）：

    1. 标题与简介
        - 简介：1–2 句说明“示例做什么 + 输出是什么”，不展开原理；
    2. 环境依赖
        - 说明是否有额外依赖；
        - 给出最小依赖安装命令（如 apt install ...）；
    3. 目录结构
        - 用树形结构展示当前目录内容，注释对齐；
        - 每个条目一句话说明用途；
    4. 编译工程（仅C/C++版本需要）
        - 给出详细的构建步骤（建议统一为 build/ + cmake + make）；
        - 若有 run.sh，可说明其会自动完成构建/运行（但不要重复脚本内容）；
    5. 参数说明
        - 用表格列出：参数 / 说明 / 默认值；
        - 默认值必须与代码一致（路径、阈值等）；
    6. 快速运行
        - 至少包含两种方式：
            - 默认参数运行 ；
            - 指定参数运行 ；
            - 明确输出产物与路径（如 build/result.jpg）；
    7. 接口说明
        - 每个接口固定三段：功能 / 参数 / 返回值；
        - 顺序固定：接口名 → 函数/结构体声明代码块 → 功能 → 参数 → 返回值；
        - 参数使用表格（参数名/类型/说明），返回值用表格（返回值/说明）；
    8. 注意事项（可选）
        - 仅列真实会踩坑的点（如模型路径、运行权限、输出位置）；

- 更新场景：
    - 新增模型样例；
    - 样例模型发生改变时；

- 相关人员
    - 普通开发人员；

### tros
TODO

### utils/c_utils(py_utils)/README.md

此文件主要说明各文件存放什么类别的通用函数，以及提供快速索引表格，供开发者快速查找有无可用接口；

注意：**开发者一般只需维护接口列表，在新增接口后便于用户查找**；

- 路径：
    - utils/c_utils/README.md
    - utils/py_utils/README.md

- 规范
    - 包含目录中各文件的说明；
    - 每个通用函数均添加到第二部分索引表格中；

- 更新场景：
    - 新增模型样例；
    - 样例模型发生改变时；

- 相关人员
    - 普通开发人员；

## 跨平台规范
