[English](./README.md) | 简体中文

# Ultralytics YOLO C++ Sample

本目录保存 Ultralytics YOLO 在 RDK X5 与 RDK S100/S100P/S600 上的 C++ 参考运行时。

## 概述

通用运行入口仍推荐使用 `runtime/python`。C++ 二进制是板端参考实现，支持两种
BPU 输入协议并在加载模型时自动探测：packed NV12（X5 `.bin`，单 Tensor）与
split Y/UV NV12（S 系列 `.hbm`，两个 UINT8 Tensor）。

输出头协议同样按模型自动选择：

- `detect` 自动识别 YOLO26 direct-LTRB 头（4 通道 box 图）与 YOLO11 系 DFL 头
  （YOLOv5u/v8/v9/yolo11/yolo12/yolov13，64 通道 box 图）；可用
  `--head auto|dfl|ltrb` 覆盖自动识别结果。
- `pose` 与 `segment` 按同一 box 图协议分发，并套用对应的关键点/掩码编码
  （YOLO26 关键点从网格中心直接回归；YOLO11 系使用倍率偏移编码）。
- `classify` 无输出头协议差异。

## 目录结构

```bash
.
|-- classify/   # 分类参考实现
|-- common/     # 共享的解码 / NV12 / benchmark 辅助层
|-- detect/     # 检测参考实现
|-- pose/       # 姿态参考实现
|-- segment/    # 分割参考实现
`-- test/       # common/ 的宿主机单元测试（无需板卡）
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

单图功能验证：

```bash
./build/ultralytics_yolo_detect \
  yolo26n_detect_bayese_640x640_nv12.bin \
  bus.jpg cpp_result.jpg
```

有界端到端性能测试：

```bash
./build/ultralytics_yolo_detect \
  yolo26n_detect_bayese_640x640_nv12.bin \
  bus.jpg cpp_result.jpg \
  --benchmark --warmup 20 --runs 200 --rounds 3 \
  --pipeline-streams 1 --opencv-threads all \
  --score 0.25 --nms 0.7 --no-save \
  --json e2e_cpp.json
```

两路完整流水线并发：

```bash
./build/ultralytics_yolo_detect \
  yolo26n_detect_bayese_640x640_nv12.bin \
  bus.jpg cpp_result.jpg \
  --benchmark --warmup 20 --runs 200 --rounds 3 \
  --pipeline-streams 2 --opencv-threads all \
  --score 0.25 --nms 0.7 --no-save \
  --json e2e_cpp_streams2.json
```

端到端计时从内存中的 BGR 图像开始，到还原至原图坐标的检测结果生成结束。计时
包含 resize/letterbox、BGR 转 NV12、输入拷贝与 cache clean、BPU 执行、输出
cache invalidate、direct-LTRB 解码、分类别 NMS 和坐标还原；不包含模型加载、
图片文件读取、绘制和结果保存。

该端到端命令不限制 CPU affinity，并使用全部在线 CPU 执行 OpenCV 前处理。
`--pipeline-streams` 控制完整的前处理、Runtime 提交和后处理流水线数量；每路拥有
独立的输入输出 Tensor 和 Runtime 上下文。多路吞吐量按所有流水线完成的总帧数除以
共同墙钟时间计算，延迟则按每个请求分别统计。OpenCV CPU 线程数是进程级配置，
与流水线数量以及 Runtime-only 工具的提交线程参数相互独立。

## 宿主机测试

`common/` 中的纯逻辑（解码数学、NV12 平面布局、benchmark JSON）可在任意具备
C++11 编译器的宿主机上构建并运行：

```bash
cd runtime/cpp/test
cmake -S . -B build-host
cmake --build build-host
ctest --test-dir build-host --output-on-failure
```

## 说明

- `detect` 实现了有界端到端 benchmark（单路与双路流水线），覆盖两种输出头协议
  与两种输入协议。
- `classify`、`pose` 和 `segment` 是功能参考实现；它们接受两种输入协议与两种
  输出头协议，但不含 benchmark 框架。
- S 系列二进制链接板端 `dnn` 栈，构建方式与 X5 相同（在目标板上执行）。
- Python 与 C++ 的宿主侧前处理、Runtime 封装和后处理实现不同，性能数据应分别记录。
