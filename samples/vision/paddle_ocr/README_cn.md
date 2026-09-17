[English](./README.md) | 简体中文

# PaddleOCR 文本检测与识别

本 sample 提供完整的两阶段 OCR 流程：DB 检测器先定位文字区域，随后对每个
裁剪图使用 CRNN+CTC 识别。维护版 Python 运行时包含两组已经审计的模型组
合，请按目标板选择，并始终把同一组合的检测模型、识别模型和字典放在一起：

| 板卡 | 模型组合 | 检测输入 | 识别输出 | 已发布引用 |
| --- | --- | --- | --- | --- |
| RDK X5 | PP-OCRv3 英文 | 打包 NV12 `[1,960,640,1]` U8 | F32 `[1,40,97,1]`，固定 96 字符表加 blank | `x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin` + `x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin` |
| RDK S100 | PP-OCRv6 | 分离的 `x_y [1,640,640,1]` 和 `x_uv [1,320,320,2]` U8 | F32 `[1,40,18710]`，仓库 UTF-8 字典加 blank/空格 | `s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` + `s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` |

最后一列的字符串是 X5 和 S [release manifest](../../../platforms/x5/docs/release/models.yaml)
中已有行的限定引用，也是 S [release manifest](../../../platforms/s/docs/release/models.yaml)
中已有行的限定引用，不是新的独立 asset ID。当前没有匹配的已审计 S100P 或
S600 OCR 组合，因此 sample 不会从文件名推断这些板卡的支持。

## 准备环境和模型

Python 板端推理需要提供匹配 `hbm_runtime` 的 RDK 系统、Python 3.10 或更新
版本、NumPy、OpenCV-Python、PyYAML 和 `pyclipper`。`hbm_runtime` 只在正式
推理时导入；`--help`、`--list-models`、`--dry-run` 不需要板端 SDK。模型文
件必须预先存在，并且两个路径必须与精确的限定引用对应。

模型准备是显式操作，会使用现有 release manifest 中的 URL 并打印实际
SHA-256；普通推理不会下载：

```bash
python samples/vision/paddle_ocr/runtime/python/main.py --prepare \
  --target x5 \
  --det-asset-id x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin \
  --rec-asset-id x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin \
  --model-dir /tmp/rdk-models
```

准备 S100 时将 `--target` 和两条引用替换为上表中的 S100 组合。重新生成
模型的转换流程见 [`conversion/README_cn.md`](./conversion/README_cn.md)，其
中 X5 PP-OCRv3 的 `hb_mapper` 与 S100 PP-OCRv6 的 `hb_compile` 配方分开维护。

## 运行 Python sample

在板端执行前先查看可用组合：

```bash
python samples/vision/paddle_ocr/runtime/python/main.py --list-models --target auto
python samples/vision/paddle_ocr/runtime/python/main.py --dry-run --target x5
```

准备好 X5 模型后，一条完整命令如下：

```bash
python samples/vision/paddle_ocr/runtime/python/main.py \
  --target x5 \
  --det-asset-id x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin \
  --rec-asset-id x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin \
  --det-model-path /tmp/rdk-models/en_PP-OCRv3_det_640x640_nv12.bin \
  --rec-model-path /tmp/rdk-models/en_PP-OCRv3_rec_48x320_rgb.bin \
  --test-img samples/vision/paddle_ocr/test_data/x5/paddleocr_test.jpg \
  --output-format json \
  --json-output /tmp/paddleocr-x5.json
```

S100 运行时替换目标、引用、模型路径和测试图片：

```bash
python samples/vision/paddle_ocr/runtime/python/main.py \
  --target s100 \
  --det-asset-id s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec-asset-id s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --det-model-path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec-model-path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --test-img samples/vision/paddle_ocr/test_data/s100/gt_2322.jpg \
  --output-format json
```

`--priority` 默认是 `0`，`--bpu-cores` 默认是 `0`；板端应用需要其他调度时
可以同时传入这两个参数。`--vocabulary-path` 可指定替换的 S100 UTF-8 字典，
但文件字节必须通过已审计摘要校验。入口会拒绝不完整路径、混合目标引用、
元数据不匹配、非 F32 输出以及无法识别的执行目标板。

## 查看结果

文本模式逐项打印识别文字和有序多边形框。JSON 包含 `target`、`image_shape`、
检测/识别资产引用和对齐的 `boxes`/`texts`，例如：

```json
{
  "target": "x5",
  "boxes": [[[20, 30], [180, 30], [180, 70], [20, 70]]],
  "texts": ["RDK"]
}
```

框和文字保留检测顺序，返回数组由结果对象独立持有；检测没有框时不会调用
识别。检测输出按已观察的分数图直接阈值处理，不加入未经确认的激活，也不
由此宣称准确率。X5 检测缩放使用 linear，S100 使用 area，掩码缩放和识别缩
放均使用 linear。轮廓扩张、最小面积过滤、裁剪旋转和退化裁剪行为按目标分
开保留。

S 系列 C++ 程序会生成左右并排 JPEG，同时保留原有 S16/F32 检测输出处理和
FreeType 字体渲染：

```bash
bash samples/vision/paddle_ocr/runtime/cpp/run.sh
```

C++ 的依赖、模型、参数、API 和排障说明见
[`runtime/cpp/README_cn.md`](./runtime/cpp/README_cn.md)。C++ 默认路径保留旧
的 SOC 行为，本轮新增集成证据覆盖 S100。

## 使用 Python 库

完整 checkout 中可以直接导入 package 模块，模块不会修改 `sys.path`；只有
直接执行脚本的入口会加入 checkout 根目录。用 `resolve_pair` 绑定精确资产，
再组合两个惰性 runner：

```python
import cv2

from samples.vision.paddle_ocr.runtime.python.model_binding import resolve_pair
from samples.vision.paddle_ocr.runtime.python.model_runner import create_stage_runners
from samples.vision.paddle_ocr.runtime.python.pipeline import OCRPipeline

pair = resolve_pair(
    "x5",
    det_asset_id="x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin",
    rec_asset_id="x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin",
    det_model_path="/tmp/rdk-models/en_PP-OCRv3_det_640x640_nv12.bin",
    rec_model_path="/tmp/rdk-models/en_PP-OCRv3_rec_48x320_rgb.bin",
)
detector, recognizer = create_stage_runners(pair, priority=0, bpu_cores=[0])
image = cv2.imread("samples/vision/paddle_ocr/test_data/x5/paddleocr_test.jpg")
result = OCRPipeline(pair, detector, recognizer).predict(image)
print(result.texts)
```

主要库模块如下：

* `model_binding.py` 保存有限的目标契约、manifest 引用、字典身份和运行时元
  数据检查。
* `tensor_io.py` 准备目标相关的打包/分离 NV12，以及共用的 RGB float32 NCHW
  识别输入。
* `model_runner.py` 惰性加载单个阶段、校验物理 tensor 并应用调度参数。
* `geometry.py`、`decode.py` 和 `pipeline.py` 负责目标相关后处理、CTC 最佳
  路径解码、裁剪和有序结果归属。

## 评估带标注记录

为每张图片保存一条预测记录并加入 `image` 标识，再按照
[`evaluator/README_cn.md`](./evaluator/README_cn.md) 与标注 JSONL 对比：

```bash
python samples/vision/paddle_ocr/evaluator/evaluate.py \
  --ground-truth /data/labels.jsonl \
  --predictions /data/predictions.jsonl \
  --iou-threshold 0.5 \
  --output /data/paddleocr-evaluation.json
```

评估器按 IoU 确定性匹配框，并报告检测计数以及匹配区域上的识别一致性。空
GT 记录合法；没有匹配区域时识别状态为 `not_run`。只有具备完整标注数据集并
完成目标板测量，才可以报告数据集准确率或性能。

## 故障排查

* **没有可用组合/引用混用：**运行 `--list-models`，从同一组合中取得两条引
  用并使用对应字典。
* **模型文件不存在：**使用 `--prepare` 或自行复制制品，然后显式传入两个路
  径；推理入口不会获取缺失文件。
* **运行时元数据不匹配：**使用工具链的 model-info 命令检查制品。X5 必须是
  打包 NV12 和 97 类；S100 必须是分离 NV12 和 18,710 类。
* **缺少 `pyclipper`：**在板端 Python 环境中安装后再处理非空检测结果；
  help/list/dry-run 不需要它。
* **没有检测框：**确认图片和检测制品，然后检查已观察的 `0.5` 阈值。修改阈
  值是运行参数，不是新的准确率结果。
* **文字乱码：**使用识别模型配套字典。S100 的 blank/字典行/末尾空格顺序必
  须保持，不能换成 X5 字符表。

## 旧入口映射与验证

旧 Python 入口仍可作为兼容 shim 使用，并转发到本 canonical 实现。旧 S C++
路径的头文件、编译单元和 launcher 也转发到这里。主要映射为：

| 旧入口 | canonical 实现 | 保留契约 |
| --- | --- | --- |
| X5 `PaddleOCR.pre_process` | `tensor_io.prepare_detection` | 打包 NV12、linear 缩放 |
| S `PaddleOCRDet.pre_process` | `tensor_io.prepare_detection` | 分离 NV12、area 缩放 |
| X5/S 识别预处理 | `tensor_io.prepare_recognition` | RGB F32 NCHW 操作顺序 |
| 旧 forward 方法 | `RuntimeStageRunner` | 一个惰性、带元数据绑定的阶段 runtime |
| 检测膨胀/裁剪 helper | `geometry.py` + `pipeline.py` | 目标相关过滤、顺序和旋转 |
| 旧 CTC converter | `decode.py` | blank 重置、repeat 合并、字典顺序 |
| 旧 CLI | `runtime/python/main.py` | 显式资产、JSON 输出、无隐式下载 |

运行主机检查：

```bash
python -m unittest discover -s samples/vision/paddle_ocr/tests -p 'test_*.py' -v
```

当前证据包括 X5 8GB、X5 4GB 和 S100 的 Python 阶段/输入/输出/生命周期精确
检查，也包括旧 Python wrapper。准确率和性能需要带标注数据及目标特定测量，
当前尚未测量。转换和协议证据见
[`OCR 源码审计`](../../../docs/releases/unified-migration/p2-ocr-source-audit.md)。
