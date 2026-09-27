[English](./README.md) | 简体中文

# YOLOE-26 分割 C++ 推理

本示例使用 UCP、OpenCV、CMake、C++17 和 gflags，在 S100 或 S100P 上运行
YOLOE-26 无提示词实例分割 HBM 模型。

构建前请安装板端开发依赖：

```bash
sudo apt install -y libgflags-dev libopencv-dev
```

在本 `runtime/cpp/` 目录中下载 n 模型并运行完整示例：

```bash
cd samples/vision/yoloe26_seg/runtime/cpp
bash ../../model/download_model.sh auto n
bash run.sh
```

`run.sh` 会根据脚本自身目录定位资源，因此也可以从其他工作目录通过脚本
绝对路径调用。脚本下载并校验板端模型、构建程序，再通过命名参数启动。
可选参数依次为 `SIZE`、`IMAGE` 和 `OUTPUT`：

```bash
bash run.sh x /path/to/image.jpg /path/to/result.jpg
```

直接构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

程序使用 gflags。当仓库模型目录中已经有 n 模型，且内置 4585 类标签文件和
测试图片存在时，直接执行不带参数的程序即可运行。也可以用以下 snake_case
参数覆盖路径和推理选项：

```bash
./build/yoloe26seg \
  --model_path=/path/yoloe_26n_seg_pf_nashe_640x640_nv12.hbm \
  --model_size=n \
  --label_file=/path/yoloe_26n_seg_pf.names \
  --test_img=/path/image.jpg \
  --output_path=result.jpg \
  --score_thres=0.25 --max_det=300 --multi_label=false
```

模型文件名后缀必须与板型一致：S100 使用 `nashe`，S100P 使用 `nashm`。
运行时会根据 `/sys/class/boardinfo` 校验后缀。

`inc/yoloe26seg.hpp` 提供 `YoloE26SegConfig`、轻量级的 `YoloE26Seg` 资源
管理类，以及分阶段的 `pre_process`、`infer` 和 `post_process` 函数。
`init()` 成功返回 0，初始化错误以非零值返回；析构函数会安全释放初始化中途
已分配的资源。

`post_process()` 返回 `utils/c_utils/inc/model_types.hpp` 中的公共
`InstanceSegResult`。检测框保留原图坐标系下的浮点 xyxy 值；每个 mask 都是
仅包含 0/1 的 `CV_8UC1` 矩阵，并且是裁剪后、整数截断框对应的局部区域。
退化框仍保留空 mask，确保检测框和 mask 按索引对齐。示例入口会在裁剪框内
正确渲染局部 mask，并直接绘制检测框，不依赖共享可选的硬件显示实现。

raw-v1 解码保持十个输出的顺序、张量字节 stride、量化 scale/zero-point 元数据
和确定性的静态 top-K 行为，不执行 IoU NMS。
