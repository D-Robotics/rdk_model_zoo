[English](README.md) | 简体中文

# PaddleOCR C++ 运行时（S 系列）

<a id="overview"></a>
## C++ 推理

S100 PP-OCRv6 模型对的原生两阶段运行时：DB 文字检测、区域裁剪、
CRNN+CTC 识别，以及左右并排的 JPEG 渲染（左侧为带有序框的原图，
右侧为识别文本）。检测器支持 S16 与 F32 输出；随仓 PP-OCRv6 制品
使用 F32。渲染文本由 FreeType 字体参数配置。

<a id="directory"></a>
## 目录结构

```text
cpp/
├── inc/
│   ├── ocr.hpp   # PaddleOCRDet/PaddleOCRRec/PaddleOCR 类与各阶段自有数据类型
│   └── cli.hpp   # 命令行选项与辅助函数
├── src/
│   ├── ocr.cpp   # 运行时生命周期、检测/识别 preprocess/infer/postprocess 各阶段
│   ├── cli.cpp   # 参数解析、默认值、词典加载、结果渲染
│   └── main.cpp  # 入口：解析选项、predict、打印并保存
├── CMakeLists.txt  # 构建（C++17，显式 RDK_TARGET 板卡选择）
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── run.sh  # 构建并运行示例
```

<a id="supported-boards"></a>
## 适用板卡

| 板卡 | 支持 |
| --- | --- |
| S100 | 支持 |
| S600 | 支持 |
| S100P | 不支持（未发布 PaddleOCR 模型对） |
| X5 | 不支持（未提供 X5 C++ 实现；请使用 Python 运行时） |

<a id="dependencies"></a>
## 依赖

在 RDK S 镜像上构建，需要：CMake 与 C++17 编译器；OpenCV 开发文件；
`polyclipping` 与 FreeType 开发库；`/usr/hobot/include` 下的 Horizon
DNN/UCP 头文件与 `/usr/hobot/lib` 下的库。Debian 系开发镜像可显式安装
缺失头文件（较新镜像的 FreeType 包名为 `libfreetype-dev`，以镜像实际
提供的名称为准）：

```bash
sudo apt install libpolyclipping-dev libfreetype6-dev
```

启动脚本不安装系统包、不修改 SDK、不下载模型。匹配的制品须位于
`/opt/hobot/model/s100/basic`（或显式传入）；缺失时用 Python 入口的
`--prepare` 准备（见[模型准备](../../model/README.md#preparation)）。

<a id="build"></a>
## 构建

启动脚本会自动构建；手动构建（cwd：仓库根目录；成功：build 目录下
生成 `paddle_ocr` 二进制）：

```bash
cmake -S samples/vision/paddle_ocr/runtime/cpp \
      -B samples/vision/paddle_ocr/runtime/cpp/build \
      -DRDK_TARGET=s100 -DCMAKE_BUILD_TYPE=Release
cmake --build samples/vision/paddle_ocr/runtime/cpp/build --parallel
```

`RDK_TARGET` 选择板卡（`s100` 或 `s600`）；在板卡上原生配置时可省略
（`auto` 读取板端 SoC 标识）。交叉编译必须显式传入目标板卡。

<a id="run"></a>
## 运行

在完整检出的任意目录执行（输入：S100 制品对、随仓 S100 测试图与
词典、S 字体；输出：每裁剪一行预测加渲染 JPEG；成功：退出码 0）：

```bash
bash samples/vision/paddle_ocr/runtime/cpp/run.sh
```

启动脚本解析测试图、词典与字体的绝对路径，用户参数转发在其后，
显式参数优先生效：

```bash
bash samples/vision/paddle_ocr/runtime/cpp/run.sh -- \
  --det-model-path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec-model-path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --test-img /data/sign.jpg \
  --vocabulary-path /data/ppocrv6_dict.txt \
  --font-path /data/NotoSansCJK-Regular.ttc \
  --img-save-path /data/sign_result.jpg
```

手动调用直接传路径：

```bash
samples/vision/paddle_ocr/runtime/cpp/build/paddle_ocr \
  --det-model-path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec-model-path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --test-img samples/vision/paddle_ocr/test_data/s100/gt_2322.jpg \
  --vocabulary-path samples/vision/paddle_ocr/test_data/s100/ppocrv6_dict.txt \
  --font-path samples/vision/paddle_ocr/test_data/FangSong.ttf
```

<a id="parameters"></a>
## 参数

`paddle_ocr` 二进制的选项，与 Python 运行时的 kebab-case 命名一致
（启动脚本会用绝对路径填充未显式给出的前三类路径）：

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--det-model-path` | 随 SoC：`/opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm`（S100）或 `s600` 变体 | 检测器 HBM |
| `--rec-model-path` | 随 SoC：`/opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm`（S100）或 `s600` 变体 | 识别器 HBM |
| `--test-img` | （启动脚本：`test_data/s100/gt_2322.jpg`） | BGR 输入图像 |
| `--vocabulary-path` | （启动脚本：`test_data/s100/ppocrv6_dict.txt`） | 识别器词典，每行一条 |
| `--threshold` | `0.5` | 检测器 score map 二值化阈值 |
| `--ratio-prime` | `2.7` | 轮廓膨胀比例 |
| `--font-path` | （启动脚本：S 字体 `FangSong.ttf`） | 渲染用 TTF/TTC 字体 |
| `--img-save-path` | `result.jpg` | 并排渲染输出路径 |
| `--help` / `-h` | — | 打印用法 |

词典逐行原样读入，加载时前置 blank 类、追加末尾空格类，保持含
`{`、`}`、`,` 等标点行在内的 18,710 类 PP-OCRv6 契约。

<a id="interface-lifecycle"></a>
## 接口与生命周期

公开 API 声明于 [`inc/ocr.hpp`](./inc/ocr.hpp)。`main.cpp` 解析选项后
构造 `PaddleOCR model(det_path, rec_path)`——构造函数加载两个 HBM
包、读取张量元数据并分配可复用缓冲——随后调用
`model.predict(image, dictionary, options)` 并渲染返回结果。所有
DNN/UCP 类型都留在 `src/ocr.cpp` 内（私有 `Impl` 结构），头文件只依赖
OpenCV 与标准库。

每个阶段单独暴露，返回的数据由调用方持有：

- 检测器：`OcrDetPrepared preprocess(const cv::Mat&)`（INTER_AREA 缩放
  并打包 NV12 平面），`OcrDetRaw infer(const OcrDetPrepared&)`（以 map
  自身域读出预测图——PP-OCRv6 为 float32，旧版 S16 导出为 int16 +
  scale），`TextDetResult postprocess(const OcrDetRaw&, const cv::Mat&,
  const OcrOptions&)`（阈值、D' = area × ratio_prime / perimeter 的轮廓
  膨胀、最小面积框与矫正裁剪）；
- 识别器：`OcrRecPrepared preprocess(const cv::Mat&)`（RGB、[0, 1]
  float32 CHW 平面，不做 ImageNet 归一化——模型按原始 [0, 1] 裁剪
  标定），`OcrRecRaw infer(const OcrRecPrepared&)`（按 stride 拷出 CTC
  logits），`std::string postprocess(const OcrRecRaw&, const
  std::vector<std::string>& id2token)`（贪心 CTC）；
- `PaddleOCR::predict` 显式组合各阶段：先检测，再对每个裁剪做一次
  识别。未检测到文本区域则完全不调用识别；单个裁剪失败只跳过它自己
  的文本，其余裁剪继续识别。结果保留每条幸存文本的原始裁剪序号，
  并记录每个被跳过裁剪的序号与原因，供 CLI 汇报。

检测器接受两种输出域：float32 预测图（PP-OCRv6 导出），或带非空
scale 的 scale 量化 int16 图（旧版 PP-OCRv3 导出）。其他张量类型、
无 scale 量化的 int16 图、或声明了量化的 float32 图都会作为不支持的
契约被拒绝。识别器始终读取 float32 CTC logits。

错误以 C++ 异常抛出（含 SDK 错误描述）；入口打印后以退出码 2 退出。
所有路径（包括构造中途失败）都由 RAII 释放资源。没有后台线程，进程
执行一次同步的先检测后识别。推理使用 S 系列 `hbDNNInferV2` +
`hbUCPMallocCached` API 族（仅限 S100/S600；与 X5 C++ API 不在调用级
兼容）。

<a id="results-interpretation"></a>
## 结果解释

可执行文件对每个保留裁剪打印一行预测（带原始裁剪序号），并写入
`img_save_path`：左栏为带有序最小面积框的原图，右栏为渲染识别字符串
的白底画布。识别失败的裁剪在 stderr 上带原始序号与原因汇报；可视化
将幸存文本与框按紧凑方式配对。字体缺失或不可读由
可视化工具报告——用 `--font-path` 传入已知 TTF/TTC。成功退出码为 0。
留证时记录板卡身份、制品引用、完整构建/运行命令、打印的预测与渲染
图像。S600 结果需在 S600 板卡上运行获取，不能用 S100 运行替代其结果。
