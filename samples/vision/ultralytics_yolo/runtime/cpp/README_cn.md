[English](./README.md) | 简体中文

# Ultralytics YOLO C++ Sample

本目录保存 Ultralytics YOLO 在 RDK X5 上的 C++ 参考运行时。

## 概述

通用运行入口仍推荐使用 `runtime/python`。`detect` 可执行程序已经实现经过审阅的
YOLO26 direct-LTRB 输出协议，可在 RDK X5 板端运行。程序会在运行时校验六个
FLOAT32 NHWC 输出头；DFL 输出模型不能直接复用该后处理。

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

## 说明

- `detect` 当前实现 X5 上的 YOLO26 direct-LTRB 检测。
- `classify`、`pose` 和 `segment` 仍为参考实现。
- Python 与 C++ 的宿主侧前处理、Runtime 封装和后处理实现不同，性能数据应分别记录。
