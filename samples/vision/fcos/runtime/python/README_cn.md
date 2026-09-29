# Python 运行时 — FCOS

<a id="environment"></a>
## 环境

- 板端：提供 `hbm_runtime` 的 RDK X5 镜像；主机 help/list/dry-run 不导入该 SDK。
- 主机：Python 3.10+，依赖为 `requirements-host.txt` 中的 `numpy`、`opencv-python`、`PyYAML`。
- runner 只有在 target 身份、精确 asset 身份和模型文件检查通过后才加载 SDK。

<a id="usage"></a>
## 使用

```bash
# cwd：仓库根目录；先用 model/download.sh 准备制品
python3 samples/vision/fcos/runtime/python/main.py --target x5 --asset-id x5:fcos:fcos_efficientnetb0_detect_512x512_bayese_nv12.bin
# 成功：退出码 0，板端输出 JSON 并写入 test_data/result.jpg

# cwd：仓库根目录；显式选择 B3
python3 samples/vision/fcos/runtime/python/main.py \
  --target x5 --variant efficientnetb3 \
  --asset-id x5:fcos:fcos_efficientnetb3_detect_896x896_bayese_nv12.bin \
  --test-img samples/vision/fcos/test_data/bus.jpg --img-save-path /tmp/fcos-b3.jpg
# 成功：退出码 0 且 /tmp/fcos-b3.jpg 存在
```

`run.sh` 只委托 `main.py`，不会下载或安装依赖。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | str | `auto` | 执行为 `x5`；`auto` 只用于列出模型 |
| `--asset-id` | str | `None` | 精确 `x5:fcos:<filename>` 身份 |
| `--variant` | str | `None` | 省略时选择 B0；给 asset ID 时可选 B0/B2/B3 |
| `--model-path` | str | `None` | 外部文件；必须同时给精确 asset ID |
| `--test-img` | str | `samples/vision/fcos/test_data/bus.jpg` | BGR 图片 |
| `--label-file` | str | `datasets/coco/coco_classes.names` | 可选 COCO 标签 |
| `--img-save-path` | str | `samples/vision/fcos/test_data/result.jpg` | 标注 JPEG |
| `--resize-type` | int | `None` | 源默认 0 直接 resize；1 为 letterbox，并逆向恢复 padding/scale |
| `--classes-num` | int | `80` | 源 FCOS 类数；binding 固定 80 |
| `--conf-thres` | float | `0.5` | FCOS confidence 阈值 |
| `--iou-thres` | float | `0.6` | OpenCV NMS IoU 阈值 |
| `--priority` | int | `0` | 运行时 priority |
| `--bpu-cores` | int 列表 | `[0]` | BPU core |
| `--list-models` | flag | `false` | 不加载 SDK 列出三个制品 |
| `--dry-run` | flag | `false` | 不加载 SDK 解析选择和协议 |

<a id="results"></a>
## 结果

标准输出 JSON 字段为 `asset_id`、`boxes`、`scores`、`class_ids`、`result_path`。`boxes` 是原始 BGR 图像中的 float32 `[x1,y1,x2,y2]` 像素坐标并已裁剪，`scores` 是源 FCOS confidence，`class_ids` 是拥有内存的 int32、从 0 开始类别 ID；结果图绘制框和可用标签。

<a id="integration-example"></a>
## 集成示例

前提：用 `model/download.sh` 准备 B0 精确制品，`bus.jpg` 已随 sample 提供。

```python
import cv2
import numpy as np
from samples.vision.fcos.runtime.python.fcos import FCOSTask
from samples.vision.fcos.runtime.python.model_binding import resolve_selection
from samples.vision.fcos.runtime.python.model_runner import RuntimeModelRunner

selection = resolve_selection("x5", asset_id="x5:fcos:fcos_efficientnetb0_detect_512x512_bayese_nv12.bin")
runner = RuntimeModelRunner(selection)
binding = runner.load()  # 仅板端；主机测试注入 fake runtime
task = FCOSTask(runner, binding)
image = cv2.imread("samples/vision/fcos/test_data/bus.jpg", cv2.IMREAD_COLOR)
result = task.predict(np.asarray(image))
print(result.boxes, result.scores, result.class_ids)
```

<a id="stage-io"></a>
## 三阶段 I/O

- `pre_process`：BGR `uint8 (H,W,3)` → contiguous packed NV12 `uint8 (1.5*input_h*input_w,)` 和冻结 `ImageContext`。
- `forward`：packed tensor → 15 个原始数组；shape 为 `(1,input_h/stride,input_w/stride,{80,4,1})`；不做激活、反量化、NMS 或文件 I/O。原始输出映射按精确名字集合匹配——板端 `run()` 返回 dict 的键顺序可能与 `metadata.output_names` 不同（X5 证据 2026-09-24）——但缺失/多余的名字，或任何 shape/dtype/非有限值与绑定不符，都会拒绝，且调用方数组身份保持不变。
- `post_process`：原始数组和 context → `sqrt(sigmoid(cls_max)*sigmoid(center))`、stride 框解码、源 OpenCV NMS 和原图结果。direct resize 使用独立的宽高比例；letterbox 会减去冻结的上/左 padding，再按实际整数 resized 宽高反向缩放并裁剪。
- 量化：FCOS 遵循固定源的 `dequantize_outputs` 路径；即使 runtime 数组是 F32，只要观察到 SCALE descriptor 也会应用它。每个输出必须有可检查的 descriptor，缺失或未知 descriptor 会拒绝，不能只按 dtype 选择 raw-F32 路径。
- `predict` 每次严格按上述三阶段串联一次。

<a id="troubleshooting"></a>
## 故障排查

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| `--model-path requires the exact --asset-id reference` | 外部身份不完整 | 使用 `--list-models` 输出的完整引用 |
| `Local execution requires recognized board identity` | 在主机或错误板卡执行真实 runner | 主机使用 help/list/dry-run；推理使用匹配的 X5 |
| `Expected exactly one FCOS output with shape ...` | 制品 metadata 与分辨率选择不符 | 选择对应 variant/asset，不要从文件名猜协议 |
| `Output names must match the binding exactly; missing=..., unexpected=...` | runtime 返回的输出集合与绑定 metadata 不一致（仅 dict 顺序不同会被接受） | 重新准备与 `--list-models` 一致的发布制品；不要手改 metadata |
