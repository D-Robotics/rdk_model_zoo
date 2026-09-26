[English](README.md) | 简体中文

# PP-LiteSeg Python 运行时

<a id="environment"></a>
## 环境

RDK X5 OS 3.5.0+、Python 3.10+、板端提供的 hbm_runtime。在仓库根执行 `python3 -m pip install numpy opencv-python PyYAML` 安装通用依赖；不要安装来源不明的同名 SDK 包。主机 help/list/dry-run 不需要 SDK 或模型。

<a id="usage"></a>
## 使用

```bash
# cwd: repository root; prepare explicitly, then run on RDK X5
bash samples/vision/pp_liteseg/model/download.sh --target x5
python3 samples/vision/pp_liteseg/runtime/python/main.py
# Host-only inspection, no SDK/model/download required:
python3 samples/vision/pp_liteseg/runtime/python/main.py --dry-run --target x5
```

```bash
# cwd: repository root; custom paths, same explicitly prepared asset
python3 samples/vision/pp_liteseg/runtime/python/main.py --target x5 --test-img samples/vision/pp_liteseg/test_data/test.jpg --output outputs/pp_liteseg/custom.png --alpha 0.4
```

成功返回 0，校验或加载错误返回 2。run.sh 转发相同参数并将 cwd 设为仓库根；推理不触发下载。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | choice | `auto` | auto 解析为 x5，实际执行检查板身份 |
| `--asset-id` | str | `None` | 精确发布身份 |
| `--model-path` | Path | `None` | 默认 model/ 下 BIN，外部路径须提供 asset-id |
| `--test-img` | Path | `samples/vision/pp_liteseg/test_data/street.png` | 按示例位置解析的绝对默认路径 |
| `--output` | Path | `outputs/pp_liteseg/result.jpg` | 三面板图像 |
| `--mask-save-path` | Path | `outputs/pp_liteseg/labels.npy` | int32 类别数组，必须使用 .npy 扩展名 |
| `--report-path` | Path | `outputs/pp_liteseg/result.json` | JSON 记录 |
| `--alpha` | float | `0.55` | 叠加权重，范围 [0,1] |
| `--input-width` | int | `1024` | 固定编译宽度 |
| `--input-height` | int | `512` | 固定编译高度 |
| `--priority` | int | `None` | 0..255；省略则保留 SDK 默认值 |
| `--bpu-cores` | int list | `None` | 非负 SDK 核心索引；省略则保留默认值 |
| `--list-models` | flag | `false` | 仅列出清单 |
| `--dry-run` | flag | `false` | 仅解析选择，与 list-models 互斥 |

`--help` / `-h` 打印帮助。输出路径相对于 cwd，已有输出会被替换。dry-run 检查选择和 CLI 配置，不校验制品字节、实际元数据或板端兼容性。

<a id="results"></a>
## 结果

3078×548 图像包含三个 1024×512 面板、分隔条及 36 像素标题栏。labels.npy 保存 512×1024 int32 类别 0..18。JSON/stdout 字段为 target、asset_id、model_path、input_path、publisher_sha256、runtime_version、metadata、class_ids、class_names、mask_shape、output_shape、output、mask_save_path。不返回置信度、原图尺寸 mask、延迟或 mIoU。

<a id="integration-example"></a>
## 集成示例

```python
# cwd: repository root; on X5 after explicit model preparation
import cv2
from samples.vision.pp_liteseg.runtime.python.model_binding import resolve_selection, SAMPLE_DIR
from samples.vision.pp_liteseg.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.pp_liteseg.runtime.python.pp_liteseg import PPLiteSegTask

image = cv2.imread(str(SAMPLE_DIR / "test_data/street.png"))
runner = RuntimeModelRunner(resolve_selection("x5"))
binding = runner.load()
task = PPLiteSegTask(runner, binding)
prepared = task.pre_process(image)
raw = task.forward(prepared.tensors)
mask = task.post_process(raw)
mask_again = task.predict(image)
print(mask.shape, mask.dtype)  # (512, 1024), int32
```

任务类只负责阶段逻辑；binding 负责制品与张量契约，共享 runner 负责 SDK 加载和调度，visualization.py 负责绘图。调度调用 runner.set_scheduling_params，不保证 SDK 并发安全。旧 PPLiteSeg/PPLiteSegConfig API 保留于源快照。

<a id="stage-io"></a>
## 阶段 I/O

pre_process 接收非空 HWC BGR uint8，以 INTER_LINEAR 拉伸到 1024×512，再打包为连续 NV12 uint8 `(768,1024)`；不做 CPU 归一化或 letterbox。返回 context 独立携带原图尺寸，不写入任务可变状态。forward 原样返回 int32 `(1,512,1024,1)`。post_process 校验类别 0..18 并返回拥有独立内存的 int32 `(512,1024)`；不做 softmax、argmax、反量化或尺寸恢复。binding 接受 NV12 逻辑 NCHW/NHWC 或物理 packed 元数据；logits 导出、错误 dtype/尺寸会明确拒绝。

<a id="troubleshooting"></a>
## 故障排查

缺少 BIN：先显式准备。缺少 SDK：使用 X5 板端镜像，不在主机伪装运行环境。S 系列无制品，不能回退至 X5。输出元数据不匹配：检查实际部署图，不要给已经解码的类别图再加 argmax。保持 1024×512 几何。图片缺失：使用随附的 street.png，不是 street.jpg。外部文件须提供精确 asset-id，并独立保留文件来源记录。
