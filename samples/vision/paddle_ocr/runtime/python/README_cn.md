# PaddleOCR Python 运行时

`main.py` 是 X5 PP-OCRv3 与 S100 PP-OCRv6 模型对的 canonical 入口。它
解析完整引用的检测器/识别器对，对照实际硬件校验执行目标，懒加载两个
阶段，并输出有序的框与文本。

<a id="environment"></a>
## 环境

- 与目标匹配、自带 `hbm_runtime` 的板卡镜像（X5 镜像对应 `.bin`，
  RDK S 镜像对应 `.hbm`）；仅在执行时导入。
- Python 3.10+、NumPy、OpenCV-Python、PyYAML 必备；检测器返回至少
  一个框时另需 `pyclipper`。`--help`、`--list-models`、`--dry-run`
  在不带任何板端 SDK 包的普通主机即可运行。
- 两个模型路径提前准备（见[模型准备](../../model/README.md#preparation)）；
  推理不联网。
- 以下命令 cwd 均为完整检出的仓库根目录。包模块使用绝对完整检出导入，
  不修改 `sys.path`；只有 `main.py` 支持从任意目录直接脚本调用。

<a id="usage"></a>
## 用法

检查模式（无需 SDK 与模型；成功：退出码 0）：

```bash
python3 samples/vision/paddle_ocr/runtime/python/main.py --help
python3 samples/vision/paddle_ocr/runtime/python/main.py --list-models --target auto
python3 samples/vision/paddle_ocr/runtime/python/main.py --dry-run --target x5
```

完整 X5 推理（cwd：仓库根目录；成功：退出码 0，stdout 输出 JSON）：

```bash
python3 samples/vision/paddle_ocr/runtime/python/main.py \
  --target x5 \
  --det-asset-id x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin \
  --rec-asset-id x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin \
  --det-model-path /tmp/rdk-models/en_PP-OCRv3_det_640x640_nv12.bin \
  --rec-model-path /tmp/rdk-models/en_PP-OCRv3_rec_48x320_rgb.bin \
  --test-img samples/vision/paddle_ocr/test_data/x5/paddleocr_test.jpg \
  --output-format json \
  --json-output /tmp/paddleocr-x5.json
```

完整 S100 推理（target、引用、路径、测试图作为一个整体一起切换）：

```bash
python3 samples/vision/paddle_ocr/runtime/python/main.py \
  --target s100 \
  --det-asset-id s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec-asset-id s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --det-model-path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec-model-path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --test-img samples/vision/paddle_ocr/test_data/s100/gt_2322.jpg \
  --output-format json
```

`--prepare` 配合 `--model-dir` 是显式取模操作（见
[模型准备](../../model/README.md#preparation)）。入口拒绝不完整的路径
组合、跨 target 的引用、运行时元数据不匹配、非 F32 输出以及无法识别
的执行目标——绝不按文件名猜测。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | choice | auto | 执行目标：`auto`、`x5`、`s100`、`s100p`、`s600`；真实执行要求与探测硬件精确匹配 |
| `--det-asset-id` | string | null | 检测器完整引用 `group:sample:filename`，来自 `--list-models` |
| `--rec-asset-id` | string | null | 同一模型对的识别器完整引用 |
| `--det-model-path` | string | null | 已存在的本地检测器制品；绝不隐式下载；缺省时按 `model/<filename>` 查找 |
| `--rec-model-path` | string | null | 已存在的本地识别器制品；须与检测器路径成对提供 |
| `--vocabulary-path` | string | null | 可选的 S100 词典替换；仅在命中审计摘要时接受 |
| `--test-img` | string | null | BGR 输入图像；缺省时使用所解析模型对的测试图 |
| `--output-format` | choice | text | `text` 或 `json` 结果渲染 |
| `--json-output` | string | null | 同时把 JSON 推理结果写入该路径 |
| `--priority` | int | 0 | 运行时调度优先级（0-255） |
| `--bpu-cores` | int list | [0] | 运行时 BPU 核心索引 |
| `--model-dir` | string | null | 显式 `--prepare` 操作的目标目录 |
| `--list-models` | flag | false | 无 SDK、OpenCV、pyclipper 列出清单支撑的模型对 |
| `--dry-run` | flag | false | 解析模型对并打印静态契约，不加载模型 |
| `--prepare` | flag | false | 显式取回所选清单资产到本地路径 |

`--list-models`、`--dry-run`、`--prepare` 互斥。上表默认值由 Q3 检查器
对照 `build_parser()` 机器校验。

<a id="results"></a>
## 结果

文本输出逐行打印识别字符串及其有序多边形框。JSON 包含 `target`、
`image_shape`、`detector_asset`、`recognizer_asset` 与对齐的
`boxes`/`texts`（示例见 [sample README](../../README.md#expected-results)）；
`--json-output` 把同一对象写入文件。框与文本保持检测器顺序；返回数组
归结果所有；检测器输出为空时跳过识别。模型输出语义保持观测到的
score-map/CTC 策略——不插入未经验证的激活。`s100p` 与 `s600` 没有
经审计的 PaddleOCR 模型对，在模型对解析处即被拒绝。

<a id="integration-example"></a>
## 集成示例

从完整检出直接组合两个阶段（绝对导入；输入全部有定义；不改
`sys.path`）：

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
image = cv2.imread("samples/vision/paddle_ocr/test_data/s100/gt_2322.jpg")
result = OCRPipeline(pair, detector, recognizer).predict(image)
print(result.texts)
```

`resolve_pair` 绑定精确资产并校验 target/引用/词典身份；
`create_stage_runners` 返回共享调度参数的两个懒加载阶段运行器；
`OCRPipeline.predict(image)` 串联检测 → 裁剪 → 识别并返回结果对象
（`texts`、`boxes`）。

<a id="stage-io"></a>
## 阶段 I/O

本 sample 是仓库的多阶段参照：每个阶段有显式数据契约，pipeline 以
可读顺序组合它们。

| 阶段 | 输入 | 输出 | 契约 |
| --- | --- | --- | --- |
| `tensor_io.prepare_detection`（检测预处理） | BGR 图像 | packed NV12 `[1,960,640,1]` U8（X5，linear 缩放）或 split `x_y [1,640,640,1]` + `x_uv [1,320,320,2]` U8（S100，area 缩放） | 无 CLI、无下载、无文件写入 |
| 检测 `forward` | 上述 NV12 张量 | 原始 F32 score map（如 `[1,1,640,640]`） | 懒加载 `HB_HBMRuntime`；元数据对照 binding 校验；此处不解码 |
| 检测 `post_process` | 原始 score map、本次调用的几何 context | 阈值化轮廓 → 有序多边形框与裁剪 | 阈值 `0.5`；膨胀与最小面积过滤按 target 本地规则；context 只携带本次调用的缩放信息 |
| `tensor_io.prepare_recognition`（识别预处理） | BGR 裁剪 | RGB F32 NCHW `[1,3,48,320]`，值域 `[0,1]`，linear 缩放 | 与检测预处理同纪律 |
| 识别 `forward` | 上述 RGB 张量 | F32 `[1,40,C]`（X5 `C=97`，S100 `C=18710`） | 懒加载第二阶段；校验物理张量 |
| 识别 `post_process` | 原始 `[1,40,C]`、模型对词典 | 解码字符串，每裁剪一个 | CTC 最佳路径：blank 重置、重复折叠、按词典顺序；无文件 I/O |
| `OCRPipeline.predict` | BGR 图像 | 对齐 `boxes`/`texts` 的结果 | 恰好串联上述阶段；无第二套算法 |

阶段错误归属也是契约的一部分：检测阶段失败归检测器（制品、元数据、
target），识别阶段失败归识别器——pipeline 不混淆两者。零检测是合法
结果，直接跳过识别。每次调用的几何信息保存在 prepared context 中，
绝不放入会被下次调用覆盖的实例字段，因此不同尺寸图像交错处理不会
互相污染（由主机测试覆盖）。

<a id="troubleshooting"></a>
## 故障排查

- **无审计模型对/混用引用：**运行 `--list-models`；从同一行取两条
  引用及该模型对的词典。
- **模型文件不存在：**运行 `--prepare` 或复制既有制品后显式传入两个
  路径；推理路径绝不联网取回。
- **运行时元数据不匹配：**用工具链的 model-info 命令检查制品。X5
  要求 packed NV12 与 97 类；S100 要求 split NV12 与 18,710 类。
- **缺 `pyclipper`：**在板端 Python 环境安装后再处理非空检测结果；
  help/list/dry-run 不需要。
- **无框输出：**检查图像与检测器制品，再尝试观测到的 `0.5` 阈值；
  阈值调整是运行选择，不是新的精度结果。
- **乱码：**使用与识别器配对的词典。S100 的 blank/行/空格顺序不可
  改变；不能使用 X5 字母表。
