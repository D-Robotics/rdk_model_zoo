[English](./README.md) | 简体中文

# PaddleOCR C++ 运行时（S 系列）

审计过的 S100 PP-OCRv6 模型对的 canonical 原生两阶段运行时：DB 文字
检测、区域裁剪、CRNN+CTC 识别，以及左右并排的 JPEG 渲染（左侧为带
有序框的原图，右侧为识别文本）。它同时保留源交付中面向旧兼容制品的
S16 检测器输出路径与随仓 PP-OCRv6 制品使用的 F32 路径，以及原有
FreeType 字体选项与默认值。

<a id="supported-boards"></a>
## 适用板卡

| 板卡 | 状态 |
| --- | --- |
| S100 | supported-verified（2026-09-17 构建并运行；渲染输出像素与源基线一致） |
| S600 | supported-not-run（同一源码与 SoC 探测；板卡连接不可用） |
| S100P | not-supported（无经审计的 PaddleOCR 模型对；构建默认值不构成支持） |
| X5 | not-supported（审计基线中无 X5 C++ 源码；请使用 Python 运行时） |

<a id="dependencies"></a>
## 依赖

在 RDK S 镜像上构建，需要：CMake 与 C++17 编译器；OpenCV 开发文件；
`gflags`、`polyclipping` 与 FreeType 开发库；`/usr/hobot/include` 下的
Horizon DNN/UCP 头文件与 `/usr/hobot/lib` 下的库。Debian 系开发镜像可
显式安装缺失头文件（较新镜像的 FreeType 包名为 `libfreetype-dev`，
以镜像实际提供的名称为准）：

```bash
sudo apt install libgflags-dev libpolyclipping-dev libfreetype6-dev
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
      -DCMAKE_BUILD_TYPE=Release
cmake --build samples/vision/paddle_ocr/runtime/cpp/build --parallel
```

<a id="run"></a>
## 运行

在完整检出的任意目录执行（输入：S100 制品对、随仓 S100 测试图与
词典、S 字体；输出：每裁剪一行预测加渲染 JPEG；成功：退出码 0）：

```bash
bash samples/vision/paddle_ocr/runtime/cpp/run.sh
```

启动脚本解析 canonical 测试图、词典与字体的绝对路径，用户参数转发在
其后，显式参数优先生效：

```bash
bash samples/vision/paddle_ocr/runtime/cpp/run.sh -- \
  --det_model_path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec_model_path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --test_image /data/sign.jpg \
  --label_file /data/ppocrv6_dict.txt \
  --font_path /data/NotoSansCJK-Regular.ttc \
  --img_save_path /data/sign_result.jpg
```

手动调用直接传路径：

```bash
samples/vision/paddle_ocr/runtime/cpp/build/paddle_ocr \
  --det_model_path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec_model_path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --test_image samples/vision/paddle_ocr/test_data/s100/gt_2322.jpg \
  --label_file samples/vision/paddle_ocr/test_data/s100/ppocrv6_dict.txt \
  --font_path platforms/s/samples/vision/paddle_ocr/test_data/FangSong.ttf
```

<a id="parameters"></a>
## 参数

`paddle_ocr` 二进制的原生 gflags（启动脚本会用 canonical 绝对路径
覆盖前四项；默认值为审计源值）：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--det_model_path` | string | `/opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` | 检测器 HBM |
| `--rec_model_path` | string | `/opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` | 识别器 HBM |
| `--test_image` | string | （启动脚本：`test_data/s100/gt_2322.jpg`） | BGR 输入图像 |
| `--label_file` | string | （启动脚本：`test_data/s100/ppocrv6_dict.txt`） | 识别器词典，每行一条 |
| `--threshold` | double | `0.5` | 检测器 score map 二值化阈值 |
| `--ratio_prime` | double | `2.7` | 轮廓膨胀比例 |
| `--font_path` | string | （启动脚本：S 字体 `FangSong.ttf`） | 渲染用 TTF/TTC 字体 |
| `--img_save_path` | string | `result.jpg` | 并排渲染输出路径 |

词典逐行读入，加载时前置 blank 类、追加末尾空格类，保持含标点行在
内的 18,710 类 PP-OCRv6 契约。

<a id="interface-lifecycle"></a>
## 接口与生命周期

公开 API 声明于 [`inc/paddle_ocr.hpp`](./inc/paddle_ocr.hpp)：
`PaddleOCRDet::init` / `pre_process_det` / `infer` / `post_process_det`
及对应的 `PaddleOCRRec`/CTC 函数，供应用显式组合各阶段。
`src/paddle_ocr.cpp` 集中维护模型元数据提取、按 stride 的张量访问、
NV12 准备、S16/F32 检测器输出处理、几何与 CTC 解码。重活发生在构造
之后而非构造函数中；DNN/UCP 资源在作用域退出时释放；没有后台线程，
进程执行一次同步的先检测后识别。推理使用 S 系列
`hbDNNInferV2` + `hbUCPMallocCached` API 族，与 X5 C++ API 不在调用级
兼容——这正是本目录不存在 X5 C++ 移植的原因。

<a id="results-interpretation"></a>
## 结果解释

可执行文件对每个保留裁剪打印一行预测，并写入 `img_save_path`：左栏
为带有序最小面积框的原图，右栏为渲染识别字符串的白底画布。字体缺失
或不可读由可视化工具报告——用 `--font_path` 传入已知 TTF/TTC。成功
退出码为 0。留证时记录板卡身份、制品引用、完整构建/运行命令、打印
的预测与渲染图像。S600 在板卡连接恢复前保持 `not-run`，不能用
S100 运行替代其结果。
