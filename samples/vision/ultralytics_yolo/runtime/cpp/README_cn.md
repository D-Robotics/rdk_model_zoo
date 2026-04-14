[English](./README.md) | 简体中文

# Ultralytics YOLO C++ Sample

本目录保存 Ultralytics YOLO 在 RDK X5 上的 C++ 参考运行时。

## 概述

本 sample 当前推荐使用的运行入口是 `runtime/python`。本目录中的 C++ 代码作为
参考实现保留。

## 目录结构

```bash
.
|-- classify/   # 分类参考实现
|-- detect/     # 检测参考实现
|-- pose/       # 姿态参考实现
`-- segment/    # 分割参考实现
```

每个子目录都包含各自的 `main.cc` 和 `CMakeLists.txt`。

## 编译

进入需要查看或构建的任务子目录即可。

以检测为例：

```bash
cd runtime/cpp/detect
mkdir -p build
cd build
cmake ..
make
```

以分类为例：

```bash
cd runtime/cpp/classify
mkdir -p build
cd build
cmake ..
make
```

## 运行

编译完成后，在对应的 `build/` 目录中运行生成的可执行文件即可。

## 说明

- 本目录仅作为参考实现保留。
- 推荐使用的运行入口为 `runtime/python/main.py`。
