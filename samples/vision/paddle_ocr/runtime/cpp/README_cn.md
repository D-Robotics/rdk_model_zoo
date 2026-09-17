[English](./README.md) | 简体中文

# PaddleOCR C++ 运行时

本目录提供已审计 S100 PP-OCRv6 组合的维护版 C++ 两阶段运行时。程序先
用 DB 检测文字区域，再裁剪区域并用 CRNN+CTC 识别，最后生成左右并排的
JPEG：左侧显示原图和检测框，右侧显示识别文字。C++ 实现保留原源码的
S16 检测输出处理分支（适用于仍使用该输出协议的旧制品）以及已登记
PP-OCRv6 制品使用的 F32 分支，也保留原来的 FreeType 字体参数和默认值。

Python 运行时是两个已审计组合的通用库入口：X5 使用 PP-OCRv3，检测输入
是打包 NV12、识别输出为 97 类；S100 使用 PP-OCRv6，检测输入是分离的
NV12、识别输出使用 18,710 类字典。本目录的 C++ 实现是 S 系列 C++ 代码，
不是 X5 C++ 移植。兼容构建仍保留 S100P/S600 的原 SOC 默认分支，但本次
集成没有新增这些目标的板端证据。

## 准备板端环境

请在 RDK S100 系统上准备已有的 DNN/HB UCP、OpenCV、gflags、polyclipping
和 FreeType 开发文件。运行脚本不会安装依赖，也不会下载模型。Debian 系
统缺少头文件时可以显式安装：

```bash
sudo apt install libgflags-dev libpolyclipping-dev libfreetype6-dev
```

较新的系统可能将 FreeType 包命名为 `libfreetype-dev`，请按系统实际包名
准备。还需要 `/usr/hobot` 下匹配的 RDK SDK 库以及
`/usr/hobot/include` 下的 C++ 头文件。

如果模型还没有放在 `/opt/hobot/model/s100/basic`，可在完整 checkout 中显式
准备已发布的 S100 制品：

```bash
python samples/vision/paddle_ocr/runtime/python/main.py --prepare \
  --target s100 \
  --det-asset-id s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec-asset-id s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --model-dir /opt/hobot/model/s100/basic
```

该命令通过已有 release manifest 使用 URL，并打印实际文件摘要，是本 sample
唯一可以联网准备模型的入口。规范的测试图片和字典已复制到 canonical
目录。默认字体保留旧路径；运行脚本会自动解析仓库已有的 S 字体文件。

## 编译与运行

在完整 checkout 的任意目录执行：

```bash
bash samples/vision/paddle_ocr/runtime/cpp/run.sh
```

脚本会在 `runtime/cpp/build` 配置 CMake、编译 `paddle_ocr`，再使用
canonical 目录中的 S100 测试图片、字典和字体的绝对路径执行。脚本把用户
参数放在默认参数之后，因此显式参数会覆盖默认值：

```bash
bash samples/vision/paddle_ocr/runtime/cpp/run.sh -- \
  --det_model_path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec_model_path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --test_image /data/sign.jpg \
  --label_file /data/ppocrv6_dict.txt \
  --font_path /data/NotoSansCJK-Regular.ttc \
  --img_save_path /data/sign_result.jpg
```

手动编译和运行：

```bash
cmake -S samples/vision/paddle_ocr/runtime/cpp \
      -B samples/vision/paddle_ocr/runtime/cpp/build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build samples/vision/paddle_ocr/runtime/cpp/build --parallel
samples/vision/paddle_ocr/runtime/cpp/build/paddle_ocr \
  --det_model_path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec_model_path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --test_image samples/vision/paddle_ocr/test_data/s100/gt_2322.jpg \
  --label_file samples/vision/paddle_ocr/test_data/s100/ppocrv6_dict.txt \
  --font_path platforms/s/samples/vision/paddle_ocr/test_data/FangSong.ttf
```

程序参数为 `--det_model_path`、`--rec_model_path`、`--test_image`、
`--label_file`、`--threshold`（默认 `0.5`）、`--ratio_prime`（默认 `2.7`）、
`--font_path` 和 `--img_save_path`（默认 `result.jpg`）。字典逐行原样读取，
在开头加入 CTC blank，在末尾加入一个空格类，保持 PP-OCRv6 的 18,710 类
协议，包括包含标点符号的字典行。

## 结果与库调用

程序会为每个保留的裁剪区域打印预测，并把结果写到 `img_save_path`。左侧
是带有按顺序排列的最小外接框的原图，右侧是白色画布和识别文字。如果字体
缺失或不可读，已有可视化工具会报告错误；请通过 `--font_path` 传入可用的
TTF/TTC 文件。

原生 API 声明在 [`inc/paddle_ocr.hpp`](./inc/paddle_ocr.hpp)。两个 wrapper
提供 `PaddleOCRDet::init`、`pre_process_det`、`infer`、`post_process_det`，
以及对应的 `PaddleOCRRec`/CTC 函数，供需要自行组合阶段的应用使用。
`src/paddle_ocr.cpp` 维护模型元数据、按 stride 读取 tensor、NV12 预处理、
S16/F32 检测输出处理、几何处理和 CTC 解码，并沿用已有 C++ 工具接口。

嵌入式应用通常也可以使用 Python 库：

```python
import cv2
from samples.vision.paddle_ocr.runtime.python.model_binding import resolve_pair
from samples.vision.paddle_ocr.runtime.python.model_runner import create_stage_runners
from samples.vision.paddle_ocr.runtime.python.pipeline import OCRPipeline

pair = resolve_pair(
    "s100",
    det_asset_id="s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm",
    rec_asset_id="s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm",
    det_model_path="/opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm",
    rec_model_path="/opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm",
)
detector, recognizer = create_stage_runners(pair, priority=0, bpu_cores=[0])
result = OCRPipeline(pair, detector, recognizer).predict(cv2.imread("sign.jpg"))
print(result.texts)
```

## 转换与评估

完整且按目标分开的转换命令见
[`conversion/README_cn.md`](../../conversion/README_cn.md)。X5 PP-OCRv3 使用
`hb_mapper`/`bayes-e` 配方，S100 PP-OCRv6 使用 `hb_compile`/`nash-e` 配方。
二者的字典、模型图、输出名和 NV12 协议均保持独立。转换说明特别标出了
S100 必须保留末尾 `Dequantize`、使检测输出为 F32 的要求。

可把程序输出的 JSON 与用户提供的标注 JSONL 交给
[`evaluator/evaluate.py`](../../evaluator/evaluate.py)。评估器只根据输入记
录报告 IoU 匹配的检测计数和匹配区域上的识别一致性，不会把示例图片变成
准确率声明；记录格式和空 GT 行为见
[`evaluator/README_cn.md`](../../evaluator/README_cn.md)。

## 故障排查

* 找不到 `hbm_runtime` 或 DNN 初始化失败：请在匹配的 RDK 系统执行，并检
  查两个模型文件是否存在。Python 的 help/list/dry-run 仍可在没有板端 SDK
  时运行。
* 模型元数据或 tensor 名不匹配：不要混用 X5 PP-OCRv3 与 S100 PP-OCRv6
  制品，使用同一组合的检测/识别限定引用。
* 没有检测框：检查图片路径、检测阈值和模型输出是否为预期 F32 图；C++
  `--threshold` 默认是 `0.5`。
* 文字乱码：使用同一 S100 PP-OCRv6 识别模型配套的字典；S100 字典不能替
  换为 X5 字符表。
* CMake 找不到 `polyclipping`、gflags 或 FreeType：在编译系统显式安装匹
  配的开发头文件后重新运行 CMake；运行脚本不会修改系统。

Python 主机检查和迁移证据见上级 [`README_cn.md`](../../README_cn.md)。板端
准确率和性能需要带标注数据的目标特定测量，本 sample 的确定性流程对照不
会推导出这两项指标。
