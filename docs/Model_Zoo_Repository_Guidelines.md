# Model Zoo仓库规范
本文档用于统一 Model Zoo 仓库的目录组织、编码风格、接口规范、注释规则与文档编写规范，以保证不同模型、不同任务、不同语言（C / Python）之间具备 一致的使用体验、良好的复用性与长期可维护性。

本文档适用对象：仓库开发者、仓库维护者

## 文件和目录规范
### 目录规范
TODO

### 文件规范
TODO

## 编码规范

### 整体框架规范

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


#### 分割模型规范


#### 姿态 / 关键点模型规范


## 注释规范
### C / C++ 注释规范


### Python 注释规范


## 文档规范
文档规范主要约束 ModelZoo 仓库中各级目录下的 README.md 编写形式与内容结构，涵盖仓库说明、数据集说明、模型示例、工具模块及运行时示例等文档。

具体涉及文档如下：

```bash
.
|-- LICENSE
|-- README.md
|-- datasets
|   `-- coco
|       `-- README.md
|-- docs
|   `-- README.md
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

要求：

更新场景：

适用对象：
