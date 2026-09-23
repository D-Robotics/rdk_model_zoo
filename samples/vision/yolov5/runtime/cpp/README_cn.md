# YOLOv5 原生 C++ 运行时

本目录是统一 YOLOv5 样例的原生 C++ 对应实现。X5 HB-DNN 适配器与 S UCP 适配器保持独立，只共享主机可测试的输出头校验和解码代码。原生程序负责参数解析、目标平台输入与推理、三路原始输出解码，再交给独立的 OpenCV visualizer 绘制结果。模型发布事实由 `launcher.py` 通过 `samples.vision.yolov5.runtime.python.model_binding` 解析。

## 目标与资产

| target | 默认资产 | 其他已发布资产 | 输入 | 原生输出契约 |
|---|---|---|---|---|
| `x5` | `yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin`（`n-v7.0`） | 九个 X5 `n/s/m/l/x-v2.0` 与 `s/m/l/x-v7.0` 资产 | 一个 640x640 打包 NV12 tensor | 三路 F32 NHWC，80/40/20、每路 255 通道 |
| `s100` | `s100/yolov5x_672x672_nv12.hbm`（`x-672`） | 无 | 672x672 分离 Y/UV NV12 tensor | 三路按实际 metadata 描述的 head；缓存失效后遵循源 S 反量化 |
| `s600` | `s600/yolov5x_672x672_nv12.hbm`（`x-672`） | 无 | 672x672 分离 Y/UV NV12 tensor | 同 S 契约 |
| `s100p` | 无 | 无 | — | YOLOv5 没有已发布 S100P 资产，因此拒绝 |

外部 `--model-path` 必须同时给出 manifest 中的精确 `--asset-id`。路径本身不能推断模型身份；launcher 会在选择原生程序前校验完整发布行。

## 构建与运行

CMake target 必须显式给出，配置阶段不会读取 sysfs。在具有对应 SDK 的目标环境分别构建：

```bash
cmake -S samples/vision/yolov5/runtime/cpp -B samples/vision/yolov5/runtime/cpp/build/x5 -DYOLOV5_TARGET=x5
cmake --build samples/vision/yolov5/runtime/cpp/build/x5
cmake -S samples/vision/yolov5/runtime/cpp -B samples/vision/yolov5/runtime/cpp/build/s100 -DYOLOV5_TARGET=s100
cmake --build samples/vision/yolov5/runtime/cpp/build/s100
```

主机可以安全执行 `--help`、`--list-models` 和显式 target 的 `--dry-run`，这些模式不读板卡且不需要 SDK。真实运行需要 launcher 的 target 身份门禁以及匹配的板卡：

```bash
samples/vision/yolov5/runtime/cpp/run.sh --target x5 --variant n-v7.0 --dry-run
samples/vision/yolov5/runtime/cpp/run.sh --target x5 --asset-id <精确资产ID> --model-path /absolute/model.bin --test-img /absolute/bus.jpg
```

参数默认值为：`--target auto`，省略 `--variant` 时 X5 为 `n-v7.0`、S 为 `x-672`，给出 `--model-path` 时必须有 `--asset-id`，`--output result.jpg`，`--score-thres 0.25`，`--nms-thres 0.45`，`--priority 0`，`--bpu-core -1`（运行时默认）。`--label-file` 可选，`--binary` 只用于已经构建的可执行文件测试。

## 接口与资源生命周期

原生入口契约是 `yolov5::RuntimeOptions` 与 `run_native`。X5 严格要求一个 packed model、一个输入、三个输出、rank-4 NHWC head 以及 `3 * (5 + classes)` 通道；输入、输出 buffer 和 task 在正常与异常路径均由 RAII 风格清理。S 严格要求一个 packed model、两个输入、三个输出，通过 UCP 使用调用者给出的 `priority` 与 BPU core，并释放所有 UCP tensor 和 task。

三路 head 按实际 metadata 的 stride（8、16、32）匹配，不依据输出顺序或文件名。解码包含 sigmoid、anchor、置信度筛选和按类别 NMS。S 构建接入 SDK 时使用源 `dequantizeTensorS32` 路径；不会根据资产名称猜量化参数。

X5 C++ 源样例族采用与 native 侧一致的 letterbox 处理，而统一 Python 默认是 stretch；这是有意保留的差异，做对照时必须说明。S 源也使用 letterbox。

## 验证状态

主机测试验证数值核心与平台适配器分离、head 唯一性、显式 CMake target、launcher 身份委托以及参数/文档契约。由于本主机没有目标 SDK、板卡或模型资产，原生 SDK 编译、模型推理、板卡身份、渲染和 X5/S 原生 tensor 数值均为 **not-run**。主机测试结果只代表契约和解码检查，不代表板卡性能或精度。
