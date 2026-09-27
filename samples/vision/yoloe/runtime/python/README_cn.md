# YOLOE Python 运行时

<a id="environment"></a>
## 环境

Python 3.10+，NumPy/OpenCV/SciPy/PyYAML；只有实际模型执行才导入板端 `hbm_runtime`。系统版本与未经验证事项见 [主说明](../../README_cn.md#prerequisites)。S 公开量化模型不能直接运行。

<a id="usage"></a>
## 使用

```bash
# cwd: repository root
bash samples/vision/yoloe/model/download.sh --target x5 --variant 11s
python3 samples/vision/yoloe/runtime/python/main.py --target x5 --variant 11s
```

缺省无参数运行会检测当前板卡，推导默认变体；X5 需先下载原制品，S 会给出浮点制品缺口。自定义 X5 命令如下：

```bash
# cwd: repository root; first prepare 11m using model/download.sh --target x5 --variant 11m
python3 samples/vision/yoloe/runtime/python/main.py --target x5 --variant 11m --score-thres 0.35 --resize-type 0
```

成功为 rc=0，JSON 含检测数/类别/分数/结果图路径；rc=2 表示错误。单纯 dry-run 成功不证明 SDK 兼容。

<a id="parameters"></a>
## 参数

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | str | `auto` | 执行目标；auto 读取本地身份 |
| `--variant` | str | `null` | 11s/m/l 或 26n/s/m/l/x；按目标推导缺省 |
| `--asset-id` | str | `null` | 精确源发布 ID |
| `--model-path` | str | `null` | 原发布文件或另行转换浮点文件 |
| `--local-float-sha256` | str | `null` | 本地浮点转换文件哈希；必须配合路径 |
| `--test-img` | str | `samples/vision/yoloe/test_data/office_desk.jpg` | 输入 BGR 图片 |
| `--label-file` | str | `samples/vision/yoloe/test_data/classes.names` | 哈希固定的 4585 类词表 |
| `--img-save-path` | str | `samples/vision/yoloe/test_data/result.jpg` | 叠加框与掩码的输出图片 |
| `--score-thres` | float | `0.25` | 严格大于此 sigmoid 概率 |
| `--nms-thres` | float | `null` | 11 缺省 0.7；26 禁止设置 |
| `--resize-type` | int | `1` | 11：0 拉伸/1 letterbox；26 只允许 1 |
| `--no-morph` | flag | `false` | 关闭 S11 命令行默认的形态学开运算 |
| `--no-contour` | flag | `false` | 关闭掩码轮廓线 |
| `--max-det` | int | `300` | 26 的最大候选数 1..8400；11 不应用截断 |
| `--multi-label` | flag | `false` | 仅 26，允许同一 anchor 多类别 |
| `--priority` | int | `0` | 调度优先级 0..255 |
| `--bpu-cores` | int | `[0]` | 非负 BPU 核编号；合法板端范围由 SDK 判断 |
| `--list-models` | flag | `false` | 列出发布矩阵，不加载 SDK |
| `--dry-run` | flag | `false` | 解析并输出选择，不加载模型；须显式 target |

X5 11 保留源逻辑：先将置信阈值夹紧至 `[1e-6,1-1e-6]`，再转为 logit；S11/26 直接使用给定阈值。

<a id="results"></a>
## 结果

`Result.boxes` 为 float32 `[N,4]` 原图连续 xyxy 像素坐标，裁剪到 `[0,W]/[0,H]`；`scores` 是 `[N]` float32 sigmoid 概率；`class_ids` 是 `[N]` int64 固定词表 ID，不能直接用作 COCO 类别 ID。`masks` 在 X5 为 bool `[N,H,W]`（`mask_layout="full"`），在 S 为 N 个 uint8 0/1 ROI（`mask_layout="roi"`），坐标截断成整数后截取，保留空 ROI 对齐。返回数据独立拥有内存。 S11 保留精确零轴 ROI 形状，并将 Lanczos 过冲归一为 0/1，不改变前景范围。

CLI 保存彩色叠加图，默认 `test_data/result.jpg`，不会保存原始张量或把模型推理当作精度报告。

<a id="integration-example"></a>
## 集成示例

前提：在 X5 上显式下载 11s 模型。下述代码在 sample 测试中使用真实绑定与合成 SDK 输出执行，不代表板测。

```python
# cwd: repository root; on X5 after the explicit model/download.sh step
from samples.vision.yoloe.runtime.python.model_binding import resolve_selection, SAMPLE_DIR
from samples.vision.yoloe.runtime.python.model_runner import build_runner
from samples.vision.yoloe.runtime.python.visualization import load_inputs
from samples.vision.yoloe.runtime.python.yoloe import YOLOE, Config
selection = resolve_selection("x5", variant="11s")
runner = build_runner(selection)
image, labels = load_inputs(SAMPLE_DIR / "test_data/office_desk.jpg", SAMPLE_DIR / "test_data/classes.names")
task = YOLOE(selection, Config(), runner=runner)
prepared = task.pre_process(image)
raw = task.forward(prepared.tensors)
result = task.post_process(raw, prepared.context)
print(result.boxes.shape, result.mask_layout)
# task.predict(image) composes exactly the same three stages.
```

<a id="stage-io"></a>
## 三阶段接口

`YOLOE(selection, Config(), runner=...)` 构造任务，Config 冻结。pre_process 接受非空 uint8 BGR HWC；返回 `Prepared.tensors` 和本次 `context`。X5 发送一维 packed NV12，共 614400 字节；S 发送 Y `[1,640,640,1]`、UV `[1,320,320,2]`。11 使用截断尺寸/127 填充（拉伸使用最近邻），26 使用四舍五入尺寸/114 填充。

forward 只调用一次 runner，保留 raw float32，不做激活或反量化；输出是借用的 `RawOutputs`，必须在下一次 SDK 调用前消费，或由调用者复制。每 stride 8/16/32 为 cls 4585、box 64（11）或 4（26）、mces 32，另有 NHWC `[1,160,160,32]` proto。实际输出按完整形状唯一绑定，不依赖名字/枚举顺序。

post_process 必须收到匹配的 context。11 使用 DFL 与 NMS；X5 在低分辨率裁剪 mask 概率后两次线性插值，S 使用 ROI 二值掩码流程。26 在 640 尺寸插值 logits 后二值化，去 padding 并最近邻还原。框按实际整数 resize 的横纵比例还原，这修正了源代码理想缩放带来的取整误差。predict 只串联三阶段；不缓存上一张图，不承诺 SDK 并发安全。

库 Config 的 do_morph 缺省 False，沿用 S11 库接口；CLI 在 S11 上缺省 True，沿用源命令行。调度使用 `runner.set_scheduling_params`。

X5 保留源中两种 RGB 形状描述符 `[1,3,640,640]` 与 `[1,640,640,3]`，两者实际仍发送 614400 字节 packed NV12。共享绑定器只为明确声明的 YOLOE-11 协议启用 NHWC RGB 描述符，不放宽其他 sample 的输入契约。

<a id="troubleshooting"></a>
## 排障

| 现象 | 原因与处理 |
| --- | --- |
| `Published S YOLOE outputs are quantized` | 原 S HBM 不能用于浮点入口；保留 Dequantize 输出节点后另行转换，记录新哈希并验证 SDK 描述符。 |
| `Local float SHA-256 mismatch` | 本地文件与指定摘要不同，核对产物，不能复用原发布摘要。 |
| `No unique YOLOE asset` | target/variant/asset-id 冲突或无资产；用 --list-models 检查，不改名冒充目标。 |
| `Vocabulary checksum mismatch` | 恢复配套词表，禁止通过重排 label 改类别。 |
| `YOLOE requires the ten declared NHWC float32 outputs` | 编译协议、布局或输出精度不符；检查转换，不能直接 cast 整数。 |

自行生成浮点制品请先阅读[转换准备说明](../../conversion/README_cn.md)；校准完成、编译成功和输出精度验证分别记录。
