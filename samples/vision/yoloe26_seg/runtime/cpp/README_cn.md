[English](./README.md) | 简体中文

# C++ UCP 推理

在 S100/S100P 上构建，需要已安装 UCP/DNN SDK、OpenCV 开发包、
CMake 和支持 C++17 的编译器。

在本 `runtime/cpp/` 目录执行：

```bash
# SIZE [IMAGE] [OUTPUT]；默认 SIZE=n，使用内置图片，输出 result.jpg。
bash run.sh n
bash run.sh x /path/image.jpg result.jpg

# 下载模型后，也可以直接构建和运行：
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/yoloe26seg /path/yoloe_26n_seg_pf_nashe_640x640_nv12.hbm \
  /path/yoloe_26n_seg_pf.names /path/image.jpg result.jpg 0.25 300 0
```

最后三个可选参数依次为置信度阈值、max_det 和 multi_label（0/1）。
HBM 文件名必须与板型匹配：S100 使用 `nashe`，S100P 使用 `nashm`。
一键脚本会自动选择目标平台并校验 SHA256。

`inc/yoloe26seg.hpp` 提供 `YoloE26Seg::predict` 接口和结果结构体。
`src/yoloe26seg.cpp` 管理 UCP 模型与张量资源，实现前处理和输出解码；
`src/main.cpp` 仅处理图片示例入口和可视化。

输出读取遵循实际字节 stride，并按照 scale/zero-point 元数据反量化，
不执行 DFL 或 NMS。每个推理线程应独立创建模型实例，异常退出时也会释放资源。

默认构建不包含内部 benchmark 程序或测试数据依赖。
模型 Runtime 性能请使用 `hrt_model_exec perf` 测量。
