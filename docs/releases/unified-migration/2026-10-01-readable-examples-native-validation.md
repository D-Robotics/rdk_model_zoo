# 可读范例 C++ 主机验证记录（2026-10-01）

分支：`codex/readable-model-examples-20261001`（原生验证起点提交 `c639827f`）。
目标：补齐此前 not-run 的 C++ 主机构建/测试路径，重点覆盖 platforms/
移除所影响的固定提交源码提取（gemma）与迁移后的共享 C++ 工具路径
（`samples/_shared/cpp/c_utils`），并回归 YOLO C++ 主机测试。全部在本机
完成；无 SSH、无权重下载、无真实导出/量化、无板端推理、无 VLA 子模块操作。

## 1. 环境（仅按需安装，未升级无关包）

| 组件 | 来源 | 版本 | 说明 |
| --- | --- | --- | --- |
| cmake/ctest | `pip install --only-binary=:all: cmake`（隔离 venv `rdk_model_zoo/.venv`） | 4.4.3 | 官方二进制 wheel，无源码编译 |
| OpenCV | `HOMEBREW_NO_AUTO_UPDATE=1 brew install opencv`（bottle 倾倒） | 5.0.0 | bottle 二进制；cmake 配置位于 `/opt/homebrew/opt/opencv/lib/cmake/opencv5` |
| nlohmann-json | Homebrew（既有安装） | 3.12.0 | `file_io.hpp` 头依赖；板端由板卡镜像提供同名包 |
| 编译器 | 系统 | Apple clang | `/usr/bin/clang++` |

未安装板端 vendorSDK/工具链；未进行任何源码编译的大型依赖栈构建。

## 2. Gemma 原生主机套件（固定提交提取验证）

目录：`samples/llm/gemma4-e2b/tests/native`。

命令（退出码见括号；原始日志在本地执行目录 `logs/native-gemma-*.txt`）：

```bash
cmake -S samples/llm/gemma4-e2b/tests/native -B /tmp/gemma-native-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/opencv   # exit 0
cmake --build /tmp/gemma-native-build -j 8                                  # exit 0（0 error/warning）
ctest --test-dir /tmp/gemma-native-build --output-on-failure                # exit 0：19/19 通过
```

结果要点：

- **固定提交提取**：configure 阶段 `git archive d2d2a4e0…:platforms/s/samples/
  llm/gemma4-e2b/runtime/cpp` 成功展开到构建目录 `gemma-legacy-pin/cpp/`
  （`gemma4_demo.cpp`、`gemma4_embeddings.cpp`、`gemma4_golden_verify.cpp`
  等），`source_preprocess` 目标以其编译并与统一侧 `runtime/cpp` 在
  `vision_test` 中对照——`vision_contract`（3.47s）与 `vision_source_images`
  均通过。
- **重复 configure**：同一构建目录二次 configure exit 0、无错误——
  `if(NOT EXISTS "${LEGACY}/cpp")` 的重入保护有效，不重复 rename。
- **缺失对象错误路径**：将 CMakeLists 复制到临时目录并把
  `GEMMA_PLATFORMS_PIN` 改为全零 SHA 后 configure **exit 1**，FATAL_ERROR
  原文包含 `Cannot read pinned legacy sources (0000…). Fetch the commit
  first: git fetch origin 0000…`——不存在静默跳过。
- 19 项测试含 vision（统一 vs 固定 legacy 预处理对照）、kv_cache 七项、
  model_io（含注入分配失败的 adopt）、text 全链（engine/tensor/session/
  stages/README 示例）。

## 3. Ultralytics YOLO C++ 主机测试（共享头/别名回归）

目录：`samples/vision/ultralytics_yolo/runtime/cpp/test`。

```bash
cmake -S . -B build-host -DCMAKE_BUILD_TYPE=Release   # exit 0
cmake --build build-host -j 8                         # exit 0
ctest --test-dir build-host --output-on-failure       # exit 0：12/12 通过
```

覆盖共享 `common/`（nv12_geometry、benchmark、dnn_io）、描述符适配器与
任务输出绑定（X5/UCP 窄替身）以及 `test_dnn_io_{x5,ucp}` 自带的
ASan+UBSan 构建（`-fsanitize=address,undefined`，随套件既有选项启用）。
该目录自带声明适用：替身不证明真实 SDK ABI 兼容。验证后在源树内删除了
`build-host/`。

## 4. 迁移后的 `samples/_shared/cpp/c_utils` 主机检查

背景：`platforms/s/utils/c_utils` 以 `git mv` 迁至
`samples/_shared/cpp/c_utils`，resnet/paddle_ocr CMakeLists 的相对路径
`../../../../_shared/cpp/c_utils/{inc,src}` 已核验从
`samples/vision/resnet/runtime/cpp` 恰好解析到迁移后目录（逐文件列出确认）。

主机可验证子集（临时工程 `/tmp/resnet-cutils-hostcheck`，真实 OpenCV 5 +
nlohmann 3.12，`-Wall -Wextra -Werror`）：

- `file_io.cpp` 完整主机编译为静态库（exit 0，0 error）——该单元不使用
  任何 SDK 类型；
- `nn_math.hpp` 头文件内联数学（数值稳定 softmax、sigmoid）经冒烟测试
  验证（softmax 和为 1、单调、sigmoid(0)=0.5；ctest exit 0）。为使头可
  解析，SDK 头替身仅含 `struct hbDNNTensor;` 前向声明——主机路径不定义、
  不读取任何 SDK 对象布局。

明确不做且不宣称的部分：

- `nn_math.cpp`/`postprocess.cpp` 读取 `hbDNNTensor` 布局
  （`properties.quantiType`、`validShape`、scale/zeroPoint 数组）；
  `preprocess.cpp` 引用 `runtime.hpp` 的板端 API；`visualize.cpp` 需要
  OpenCV freetype（contrib）与 `sp_display.h`。按用法臆造这些结构布局即
  是伪 ABI 验证，故不编译。
- **resnet18 生产应用构建为 not-run**：其 CMakeLists 在 configure 阶段
  `file(READ "/sys/class/boardinfo/soc_name")` 生成 SoC 宏，并需要板端
  SDK 头（`/usr/include/hobot/dnn` 等）与链接库（dnn/hbucp）——只能在
  板端环境构建。本记录不宣称 vendorSDK 应用构建通过。

## 5. ASan/UBSan 说明

按“支持处启用、不发明”执行：YOLO `test_dnn_io_{x5,ucp}` 为套件既有
sanitizer 目标并已运行通过；gemma 原生套件未提供 sanitizer 选项，未另行
注入。

## 6. 结论与限制

- platforms/ 移除影响的三条 C++ 主机路径（gemma 固定提交提取、YOLO 共享
  头回归、c_utils 迁移路径与可主机编译子集）全部实测通过，含负路径
  （缺失对象 FATAL_ERROR、重复 configure）。
- 板端 SDK 耦合单元与生产应用构建保持 not-run；如需复现：在匹配板端
  镜像（提供 `hobot/dnn/hb_dnn.h`、libdnn/libhbucp、OpenCV 开发件、
  nlohmann-json 与 `/sys/class/boardinfo/soc_name`）上执行各 sample
  runtime/cpp README 的构建命令。
- OpenCV 5.0.0 主机编译通过本身不构成对板端 OpenCV 版本兼容性的声明。
- 本地执行目录：`local-execution/20261001-readable-model-examples/`
  （`native-report.md` 汇总，`logs/native-*.txt` 为原始日志与退出码）。
