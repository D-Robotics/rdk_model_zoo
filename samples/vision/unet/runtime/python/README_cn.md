[English](README.md) | 简体中文

# UNet Python 运行时

<a id="environment"></a>
## 环境

RDK X5、OS 3.5.0+、Python 3.10+、镜像配套 hbm_runtime；不要用 PyPI 同名包替代。主机 help/list/dry-run 无需 SDK 或模型。

```bash
# cwd: repository root; general Python dependencies only
python3 -m pip install numpy opencv-python PyYAML
python3 samples/vision/unet/runtime/python/main.py --help
```

<a id="usage"></a>
## 使用

以下命令在仓库根执行。run.sh 只是转发参数并切换到仓库根，不下载模型；推理成功返回 0，错误返回 2。
```bash
# cwd: repository root; explicitly prepare the selected artifact first
bash samples/vision/unet/model/download.sh --target x5 --variant resnet18
python3 samples/vision/unet/runtime/python/main.py
python3 samples/vision/unet/runtime/python/main.py --target x5 --variant resnet18 --test-img samples/vision/unet/test_data/2007_000033.jpg --alpha 0.4 --mask-save-path outputs/unet/mask.png --img-save-path outputs/unet/overlay.png --report-path outputs/unet/report.json
python3 samples/vision/unet/runtime/python/main.py --dry-run --target x5 --variant resnet34
```

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | choice | `auto` | auto 解析为 x5；实际执行核对本机身份 |
| `--variant` | choice | `None` | 默认 resnet18，或由精确 asset-id 推导 |
| `--asset-id` | str | `None` | x5:unet:<filename>，必须与 variant 一致 |
| `--model-path` | Path | `None` | 默认 model/ 下所选文件；外部副本必须给 asset-id |
| `--test-img` | Path | `samples/vision/unet/test_data/2007_000033.jpg` | 默认解析为 sample 绝对路径，BGR 图片 |
| `--mask-save-path` | Path | `unet_mask.png` | 类别 ID PNG |
| `--img-save-path` | Path | `unet_result.png` | 彩色叠加图 |
| `--report-path` | Path | `unet_runtime_report.json` | JSON 报告 |
| `--priority` | int | `None` | 0..255，省略则保持 SDK 默认 |
| `--bpu-core` | int | `None` | 非负 SDK 核编号；省略保持默认 |
| `--alpha` | float | `0.55` | 叠加颜色权重 [0,1] |
| `--dry-run` | flag | `false` | 仅解析，不加载/下载模型 |
| `--list-models` | flag | `false` | 只读清单，与 dry-run 互斥 |

`--help` / `-h` 输出帮助。相对输出路径基于 cwd；同名结果会被替换。Dry-run 成功不证明文件、实际 metadata 或 SDK 已通过校验。

<a id="results"></a>
## 结果

掩码固定为 512×512 uint8、VOC ID 0..20；彩色图同分辨率，以 `alpha` 混合输入缩放图和 VOC 调色板。不自动还原原图尺寸。JSON 包含 target、variant、asset_id、runtime_version、model_path、image_path、metadata、mask_shape、classes_present、elapsed_ms 和输出路径。计时覆盖三阶段，不是纯 BPU benchmark。

<a id="integration-example"></a>
## 集成

```python
# cwd: repository root; on X5 after explicit model preparation
import cv2
from samples.vision.unet.runtime.python.model_binding import resolve_selection, SAMPLE_DIR
from samples.vision.unet.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.unet.runtime.python.unet import UNetTask

image = cv2.imread(str(SAMPLE_DIR / "test_data/2007_000033.jpg"))
runner = RuntimeModelRunner(resolve_selection("x5", variant="resnet18"))
binding = runner.load()
task = UNetTask(runner, binding)
prepared = task.pre_process(image)
raw = task.forward(prepared.tensors)
mask = task.post_process(raw)
mask_again = task.predict(image)
print(mask.shape, mask.dtype)  # (512, 512), uint8
```
模型类只负责阶段；binding 负责制品/张量契约，runner 懒加载 SDK，visualization.py 负责调色板。调度通过 runner.set_scheduling_params 设置，不承诺 SDK 并发安全。旧 UNetConfig/UNet 接口留在源快照；统一接口用显式 selection/binding。

<a id="stage-io"></a>
## 阶段 I/O / Stage IO

前处理接受非空 BGR uint8 HWC，INTER_LINEAR 直接缩放至 512×512，再转换为连续 packed NV12 uint8 `(1,768,512,1)`。metadata 可表达逻辑 NCHW `(1,3,512,512)`、NHWC `(1,512,512,3)` 或物理 packed 形状，dtype 必须 NV12。每次返回独立冻结 context，记录原始尺寸。

forward 仅返回经 runner 校验的原始 logits，不做反量化或 argmax。post_process 接受 `(1,21,512,512)` 或 `(1,512,512,21)`；整数需有效 SCALE 参数，float32 不重复反量化，即使带遗留 descriptor。随后按类别 argmax，平局取最小 ID。输出为模型分辨率，因此后处理无需 context；没有 softmax、自动拉回原图或文件 IO。

<a id="troubleshooting"></a>
## 排查

模型缺失：先显式下载。哈希不符：核对发布文件，不绕过校验；自行编译文件使用 evaluator 的 --model，定制文件名同时给 --backbone。S100/S100P/S600 没有本样例制品，不能静默回退 X5。图片必须非空 uint8 三通道；metadata 不符需核对导出/制品，不更改类别数或尺寸来强行运行。
