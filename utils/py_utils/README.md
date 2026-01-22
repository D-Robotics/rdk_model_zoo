# Utils Module (Python)

本模块提供一组面向开发者的通用工具函数，用于模型推理流程中常见的文件读写、预处理、后处理、数学计算与可视化等操作，便于在不同模型 Sample 中复用。

## 目录结构与文件职责
### 子目录说明
```bash
py_utils
├── __init__.py
├── file_io.py
├── inspect.py
├── nn_math.py
├── preprocess.py
├── postprocess.py
└── visualize.py
```
- py_utils/

    Python 版本的通用工具模块，提供与 C/C++ utils 功能对齐的实现，主要用于 Python Sample 与快速验证场景。

### 文件职责说明

| 文件名            | 说明             | 包含内容                                 |
| ---------------- | ---------------- | --------------------------------------- |
| `file_io.py`     | 文件与资源读写工具 | 图像加载、标签文件读取、词表加载等         |
| `inspect.py`     | 模型与数据检查工具 | Tensor / 模型输出信息打印、调试与辅助检查  |
| `nn_math.py`     | 数学与数值计算工具 | sigmoid / softmax / 数值归一化等         |
| `preprocess.py`  | 推理前数据预处理   | resize、letterbox、输入格式转换等        |
| `postprocess.py` | 推理后结果处理     | 解码、过滤、NMS、坐标映射等               |
| `visualize.py`   | 可视化工具        | 分类 / 检测 / 分割 / 关键点结果渲染       |


## 函数索引（快速查找）
本节用于帮助开发者快速定位所需功能所在的文件。

- 阅读[源码文档说明](../../docs/source_reference/README.md)，根据说明查看源码参考文档；
- 在文档中根据目录结构，找到想查阅的文件，即可查看函数列表及接口说明；
