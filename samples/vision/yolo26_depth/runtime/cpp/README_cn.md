[English](./README.md) | [简体中文](./README_cn.md)

# C++ 推理

## 环境要求

- 带有 DNN Runtime 头文件和库的 RDK X5 板端环境
- CMake 3.10 或更高版本
- 支持 C++17 的编译器
- OpenCV 开发包

## 编译运行

使用默认路径完成编译和运行：

```bash
bash run.sh
```

显式指定全部路径：

```bash
bash run.sh MODEL.bin INPUT.jpg OUTPUT_DIR
```

脚本在 `runtime/cpp/build` 下配置 CMake，编译 `yolo26_depth`，然后执行一次推理。

## 输出文件

- `depth_native.f32`：原图尺寸、按行存储的 float32 相对深度。
- `depth.png`：深度伪彩色图。
- `overlay.png`：原图与深度可视化叠加图。
- `report.json`：模型名、输入输出尺寸和 BPU 推理延迟。

## 代码接口

`inc/yolo26_depth.hpp` 声明可复用的 `Yolo26Depth` 接口。`src/yolo26_depth.cpp` 实现 NV12 打包、DNN 推理、缓存同步、log-depth 解码和几何还原。

按照[源码文档说明](../../../../../docs/source_reference/README.md)生成接口文档。
