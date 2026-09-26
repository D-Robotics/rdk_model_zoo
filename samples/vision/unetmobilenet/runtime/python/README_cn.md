[English](README.md) | 简体中文

# UNetMobileNet Python 运行时

<a id="environment"></a>
## 环境

S100/S600 上的 Python 3.10+ 及匹配的板端 hbm_runtime。源环境使用 NumPy 1.26.4/OpenCV 4.11.0.86；统一代码还通过 PyYAML 读取清单，不再需要源工具模块的 SciPy 导入。通用依赖命令为 `python3 -m pip install numpy opencv-python PyYAML`。源未钉住 S OS/SDK 最低版本，本轮未验证真实兼容性。help/list/dry-run 不加载 SDK。

<a id="usage"></a>
## 使用

```bash
# cwd: repository root; run on the selected S100 board
bash samples/vision/unetmobilenet/model/download.sh --target s100
python3 samples/vision/unetmobilenet/runtime/python/main.py --target s100
# For S600, use --target s600 for BOTH preparation and inference.
# Host-only selection inspection, no SDK/model/download:
python3 samples/vision/unetmobilenet/runtime/python/main.py --dry-run --target s600
```

```bash
# cwd: repository root, recognized S board, model already prepared
python3 samples/vision/unetmobilenet/runtime/python/main.py
python3 samples/vision/unetmobilenet/runtime/python/main.py --target s100 --alpha-f 0.5 --img-save-path outputs/unetmobilenet/overlay.png --mask-save-path outputs/unetmobilenet/labels.npy --report-path outputs/unetmobilenet/report.json
```

run.sh 将 cwd 设为仓库根并转发参数，不安装／下载。成功返回 0，错误返回 2。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | choice | `auto` | 精确板身份或显式准备目标 |
| `--asset-id` | str | `None` | 按目标区分的精确发布身份 |
| `--model-path` | Path | `None` | 默认 model/<target>/ 下 HBM；外部副本须提供 asset-id |
| `--test-img` | Path | `samples/vision/unetmobilenet/test_data/segmentation.png` | 按示例位置解析的绝对默认路径 |
| `--img-save-path` | Path | `result.jpg` | 原图尺寸叠加图 |
| `--mask-save-path` | Path | `unetmobilenet_mask.npy` | 原图尺寸 int32 标签，须使用 .npy |
| `--report-path` | Path | `unetmobilenet_report.json` | JSON 报告 |
| `--alpha-f` | float | `0.75` | 原图权重，范围 [0,1] |
| `--priority` | int | `0` | 沿用源默认优先级，范围 0..255 |
| `--bpu-cores` | int list | `[0]` | 沿用源核心列表，索引非负 |
| `--list-models` | flag | `false` | 不加载 SDK，仅列清单 |
| `--dry-run` | flag | `false` | 选择／配置检查，与 list-models 互斥 |

--help/-h 打印帮助。自定义相对路径使用 cwd，已有输出会替换。主机 dry-run 需要显式 target 或精确 asset-id；真实板端 auto 不把未知／S100P 身份默认为 S100。

<a id="results"></a>
## 结果

NPY mask 为原图尺寸 int32 类别 0..18，result.jpg 在相同尺寸叠加源颜色。JSON/stdout 记录 target、asset_id、model_path、input_path、publisher_sha256、runtime_version、metadata、mask_shape、class_ids、alpha_f、img_save_path、mask_save_path；不计算 mIoU、置信度或延迟。

<a id="integration-example"></a>
## 集成示例

```python
# cwd: repository root; on S100 after explicit model preparation
import cv2
from samples.vision.unetmobilenet.runtime.python.model_binding import resolve_selection, SAMPLE_DIR
from samples.vision.unetmobilenet.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.unetmobilenet.runtime.python.unetmobilenet import UnetMobileNetTask
from samples.vision.unetmobilenet.runtime.python.visualization import render_overlay

image = cv2.imread(str(SAMPLE_DIR / "test_data/segmentation.png"))
runner = RuntimeModelRunner(resolve_selection("s100"))
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = UnetMobileNetTask(runner, binding)
prepared = task.pre_process(image)
raw = task.forward(prepared.tensors)
mask = task.post_process(raw, prepared.context)
mask_again = task.predict(image)
overlay = render_overlay(image, mask, alpha_f=0.75)
print(mask.shape, mask.dtype)  # original image height/width, int32
```
与归档 UnetMobileNet.predict API 不同，task.predict 返回类别 ID，须显式调用 render_overlay。runner 管理 SDK 生命周期和调度，binding 管理制品／元数据契约。任务不缓存共享的“上次图片尺寸”，请保留每次 PreparedInput 的 context；不保证 SDK 并发安全。

<a id="stage-io"></a>
## 阶段 I/O

pre_process 接收非空 BGR uint8 HWC，以 INTER_AREA 拉伸到 2048×1024，产生 Y uint8 [1,1024,2048,1]、UV uint8 [1,512,1024,2]，并逐次保存不可变的原图尺寸；不做 CPU 归一化或 letterbox。forward 原样返回 [1,H,W,19] int32/F32。post_process 校验绑定的几何／dtype 和有限数值。显式 NONE int32 直接比较，避免浮点舍入；SCALE 校验正 scale／offset 后以 float64 仿射解码，修正“整数 argmax 总保序”的源假设。F32 不重复反量化；精确平局取最小类别 ID。类别图直接以 INTER_NEAREST 恢复原图尺寸，不绘图或读写文件。缺失整数量化元数据时明确拒绝。

<a id="troubleshooting"></a>
## 故障排查

缺模型：为相同目标显式下载。板身份未知：显式 target 仅用于主机检查，实际执行仍检查身份。输出通道／dtype／量化不匹配：检查真实元数据，不通过改名或重解释模型规避。颜色与 Cityscapes 标准调色板不同，因为保留了源 rdk_colors。alpha_f 是原图而非 mask 的权重。原图尺寸 mask 不等同于另一个 X5 UNet 的固定 512×512 输出。
