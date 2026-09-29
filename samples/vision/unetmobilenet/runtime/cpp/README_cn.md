[English](README.md) | 简体中文

# UNetMobileNet C++ 运行时

<a id="supported-boards"></a>
## 适用板卡

S100/S600 各有独立发布 HBM，Python/C++ 使用相同的精确目标选择。X5/S100P 无制品并明确拒绝。原生二进制嵌入显式构建目标，并独立检查本地 S 身份，包括 s100 + board_type s100p/RDK S100P 的细分。主机测试不构成真实 SDK 编译或板测证据。

<a id="dependencies"></a>
## 依赖

需要 C++17 编译器、CMake 3.16+、OpenCV core/imgproc/imgcodecs 开发库，以及匹配的板端 DNN/UCP SDK。头文件沿用 S 源布局：/usr/hobot/include、/usr/include/hobot、/usr/include/hobot/dnn；库位于 /usr/hobot/lib。本示例不再依赖 gflags/fmt。启动器使用 Python 3.10+ 与 PyYAML 解析选择。源未钉住 S SDK 版本，本轮真实 SDK 构建为 not-run。

<a id="build"></a>
## 构建

```bash
# cwd: repository root, on S100; install development dependencies explicitly
sudo apt install build-essential cmake libopencv-dev
bash samples/vision/unetmobilenet/model/download.sh --target s100
bash samples/vision/unetmobilenet/runtime/cpp/run.sh --target s100 --build
# Subsequent run, same binary/model:
bash samples/vision/unetmobilenet/runtime/cpp/run.sh --target s100
```

启动器在 CMake 之前检查板身份、已准备模型和输入文件，不隐式安装或下载。build/s100、build/s600 分离以防误用；CMake 必须指定 UNETMOBILENET_TARGET，不根据未知板身份猜测宏。

<a id="run"></a>
## 运行

```bash
# cwd: repository root; board DNN/UCP headers and libraries already present
cmake -S samples/vision/unetmobilenet/runtime/cpp -B samples/vision/unetmobilenet/runtime/cpp/build/s600 -DUNETMOBILENET_TARGET=s600 -DCMAKE_BUILD_TYPE=Release
cmake --build samples/vision/unetmobilenet/runtime/cpp/build/s600 --parallel 2
bash samples/vision/unetmobilenet/model/download.sh --target s600
bash samples/vision/unetmobilenet/runtime/cpp/run.sh --target s600 --alpha-f 0.5 --img-save-path outputs/unetmobilenet/native.png --mask-save-path outputs/unetmobilenet/native_labels.png --report-path outputs/unetmobilenet/native.json
```

显式构建和准备后，在匹配的已识别板卡上可零参数运行 run.sh。--help/list-models/dry-run 是无需 SDK 的启动器模式。直接运行二进制须提供 --target、--model-path、--test-img；二进制使用字面路径，不执行清单解析。

<a id="parameters"></a>
## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--target` | `auto` | 启动器检测精确板身份；二进制须显式指定 s100/s600 |
| `--asset-id` | `None` | 启动器的精确制品身份；外部 model-path 必须提供 |
| `--model-path` | `None` | 启动器解析 model/<target>/ 下 HBM；二进制须提供路径 |
| `--test-img` | `samples/vision/unetmobilenet/test_data/segmentation.png` | 启动器默认值；二进制须提供路径 |
| `--img-save-path` | `result.jpg` | 原图尺寸叠加图 |
| `--mask-save-path` | `unetmobilenet_mask.png` | 无损 uint8 类别 0..18，须使用 .png |
| `--report-path` | `unetmobilenet_cpp_report.json` | JSON 报告 |
| `--alpha-f` | `0.75` | 原图权重，范围 [0,1] |
| `--priority` | `0` | 调度优先级 0..255 |
| `--bpu-core` | `-1` | 沿用源原生默认任意核心；其他值为 0..3 核心索引 |
| `--build` | `false` | 启动器在运行前显式配置／构建 |
| `--binary` | `None` | 启动器使用的自定义二进制，与 --build 不兼容 |
| `--list-models` | `false` | 启动器仅列出清单 |
| `--dry-run` | `false` | 启动器打印解析命令，不构建／加载 SDK；与 list-models 互斥 |

原生 CLI 还接受历史下划线拼法（--model_path、--test_img、--alpha_f）；启动器仅接受 kebab-case。Python 源默认 bpu-cores [0]，C++ 源使用任意核心，此默认差异保留。--help/-h 打印帮助。输出路径相对于 cwd，已有文件会替换。

<a id="interface-lifecycle"></a>
## 接口与生命周期

ModelRunner 管理 packed/model 句柄、两块输入和一块输出缓冲。构造时先校验目标、数量、形状、stride、容量与分数量化，再分配。析构仅释放已取得的资源；每次 forward 的任务 guard 在 submit/wait/cache 失败时释放任务。runner 不可复制且非线程安全，多线程使用独立实例。可选显式 execution gate 是主机测试入口，不作为 CLI 身份绕过开关。

UnetMobileNetTask 接收 RawRunner 回调。pre_process(image) 返回独立 Y/UV Mat 与本次 ImageContext；forward(prepared) 原样返回拥有独立内存的 RawScores 字节与元数据；post_process(raw, context) 返回原图尺寸 CV_32S ID；predict 组合三阶段。main.cpp 展示完整连接方式。绘图位于 visualization.cpp，资源代码位于 model_runner.cpp，stride／仿射解码位于 tensor_contract.cpp；任务类不负责显示或文件读写。

<a id="results-interpretation"></a>
## 结果解释

成功返回 0，错误返回 2。result.jpg 为源颜色叠加图；PNG mask 保存 uint8 ID，API mask 仍为 int32。读取为整数数组后可与 Python NPY 标签比较。JSON/stdout 记录 target、asset_id、model_path、input_path、publisher_sha256、runtime_version（未查询时为 unknown）、mask_shape、score_shape、score_dtype、scaled、alpha_f、priority、bpu_core 及输出路径。它是单图结果，不是性能或数据集基准。直接最近邻恢复与统一 Python 一致；源 C++ 先恢复到模型输入尺寸，对不能整除输入尺寸的输出可能有差异。
