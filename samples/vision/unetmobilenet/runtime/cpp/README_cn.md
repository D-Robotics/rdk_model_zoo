[English](README.md) | 简体中文

# UNetMobileNet C++ 运行时

<a id="overview"></a>
## C++ 推理

本目录提供C++ 推理所需的程序与操作说明。`main.cpp` 显式构造具名模型并调用 `predict`；阶段计算、张量契约与 SDK 资源管理位于 `segment.cpp`；参数解析、叠加渲染与产物／报告写入位于 `cli.cpp`。

<a id="directory"></a>
## 目录结构

```text
cpp/
├── inc/
│   ├── cli.hpp  # CLI 选项、图像加载、叠加与报告声明
│   └── segment.hpp  # UnetMobileNet 模型、分数契约、板身份
├── src/
│   ├── main.cpp  # 薄入口：解析参数、构造模型、predict、保存
│   ├── cli.cpp  # 参数解析、叠加渲染、图像/mask/报告写入
│   └── segment.cpp  # preprocess/infer/postprocess、NV12 绑定、解码、SDK 句柄
├── tests/  # 自动化测试（分数契约 + 伪 SDK 头的资源清理）
├── CMakeLists.txt  # RDK_TARGET 门控的构建定义
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── launcher.py  # Python 脚本
└── run.sh  # 运行示例
```

<a id="supported-boards"></a>
## 适用板卡

S100/S600 各有独立发布 HBM，Python/C++ 使用相同的精确目标选择。X5/S100P 无制品并明确拒绝。原生二进制嵌入显式构建目标，并独立检查本地 S 身份，包括 s100 + board_type s100p/RDK S100P 的细分。

<a id="dependencies"></a>
## 依赖

需要 C++17 编译器、CMake 3.16+、OpenCV core/imgproc/imgcodecs 开发库，以及匹配的板端 DNN/UCP SDK。头文件路径：/usr/hobot/include、/usr/include/hobot、/usr/include/hobot/dnn；库位于 /usr/hobot/lib。启动器使用 Python 3.10+ 与 PyYAML 解析选择。请使用板卡镜像提供的 S SDK，并在该环境中完成完整构建。

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

启动器在 CMake 之前检查板身份、已准备模型和输入文件，不隐式安装或下载。build/s100、build/s600 分离以防误用；CMake 必须显式指定 s100 或 s600 的 RDK_TARGET（交叉编译时拒绝 auto，因其会读取构建主机的 SoC 身份），对 SoC 字符串归一化校验，其余取值直接报错而非猜测宏。

<a id="run"></a>
## 运行

```bash
# cwd: repository root; board DNN/UCP headers and libraries already present
cmake -S samples/vision/unetmobilenet/runtime/cpp -B samples/vision/unetmobilenet/runtime/cpp/build/s600 -DRDK_TARGET=s600 -DCMAKE_BUILD_TYPE=Release
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
| `--mask-save-path` | `unetmobilenet_mask.png` | 无损 uint8 类别 0..18，须使用.png |
| `--report-path` | `unetmobilenet_cpp_report.json` | JSON 报告 |
| `--alpha-f` | `0.75` | 原图权重，范围 [0,1] |
| `--priority` | `0` | 调度优先级 0..255 |
| `--bpu-core` | `-1` | 任意核心（默认）；其他值为 0..3 核心索引 |
| `--build` | `false` | 启动器在运行前显式配置／构建 |
| `--binary` | `None` | 启动器使用的自定义二进制，与 --build 不兼容 |
| `--list-models` | `false` | 启动器仅列出清单 |
| `--dry-run` | `false` | 启动器打印解析命令，不构建／加载 SDK；与 list-models 互斥 |

二进制接受 kebab-case 与下划线别名（--model_path、--test_img、--alpha_f）；启动器仅接受 kebab-case。--help/-h 打印帮助。输出路径相对于 cwd，已有文件会替换。

<a id="interface-lifecycle"></a>
## 接口与生命周期

`UnetMobileNet`（segment.hpp/segment.cpp）管理 packed/model 句柄、两块分离 NV12 输入缓冲与分数输出缓冲。构造时先以可注入的 gate（默认实现先读 /sys/class/boardinfo 再触碰任何 SDK 调用）核对请求目标与嵌入的构建目标，校验 priority 0..255 与 bpu-core -1 或 0..3，随后在任何分配前校验输入数量、分离 NV12 几何/顺序、行距、字节 stride、输出秩/类型以及分数容量与量化元数据。析构仅释放已取得的资源，构造失败不泄漏。

公开阶段为 preprocess(image) 返回独立 Y/UV Mat 与该次 ImageContext；infer(prepared) 返回拥有独立内存的 RawScores（含填充的字节与描述符，后续调用不会改动）；postprocess(raw, context) 返回原图尺寸 CV_32S 类别 ID；predict(image) 精确组合这三步。score_spec() 暴露绑定后的输出描述符。所有 SDK 返回值均被检查（含 submit/wait/release）；任务在成功路径恰好释放一次，异常路径由任务 guard 兜底。模型不可复制且非线程安全，多线程使用独立实例。

`cli.cpp` 仅负责渲染与报告 IO：parse_options/print_help、load_image、render_overlay（19 色表、alpha 混合）与 save_results（叠加图、uint8 PNG mask、JSON 报告）。main.cpp 构造模型、调用一次 predict 并把结果交给 save_results；CLI 模块不含任何模型或 SDK 逻辑。

<a id="results-interpretation"></a>
## 结果解释

成功返回 0，错误返回 2。result.jpg 为原图上的叠加图；PNG mask 保存 uint8 ID，API mask 仍为 int32。读取为整数数组后可与 Python NPY 标签比较。JSON/stdout 记录 target、asset_id、model_path、input_path、publisher_sha256、runtime_version（未查询时为 unknown）、mask_shape、score_shape、score_dtype、scaled、alpha_f、priority、bpu_core 及输出路径。它是单图结果，不是性能或数据集基准。掩码由分数网格经最近邻整数映射直接恢复到原图尺寸。
