[English](README.md) | 简体中文

# 推理公共函数


公共函数为 Model Zoo Sample 提供 SDK 会话、模型元数据、图像与张量处理、标签和结果绘制。模型转换脚本也复用其中的数值计算与图像处理函数。

## 目录结构

```text
utils/
├── py_utils/   # Python 推理与数值处理函数
├── c_utils/    # C++ 推理、图像、张量与绘制函数
└── tools/      # 编译、评测与仓库维护工具
```

## 使用方式

从仓库根目录导入 [Python 公共函数](py_utils/README_cn.md)，通过 Sample 的 CMake 工程包含并链接 [C++ 公共函数](c_utils/README_cn.md)。模型特有的预处理与解码放在各 Sample 的任务文件中。

仓库维护及数据准备命令位于 [tools](tools/)。
