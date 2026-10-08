# MODNet Python runtime

<a id="overview"></a>
## Python 推理

使用 MODNet 生成人像 alpha matte，通过 CLI 保存结果或合成背景。

<a id="directory"></a>
## 目录结构

```text
python/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── cli.py  # 参数、模型选择与结果交付
├── main.py  # 命令行入口：构造模型并调用 predict
├── modnet.py  # 模型初始化与推理阶段
└── run.sh  # 运行示例
```

<a id="environment"></a>
## 环境

在带有 `hbm_runtime` 的 RDK X5 系统镜像上使用 Python 3、NumPy 和 OpenCV。runtime 在模型文件和板卡检查通过后才懒加载板端 SDK；`--help`、`--list-models`、`--dry-run` 不加载 SDK。绑定模型必须暴露 float32 `(1,3,512,512)` 输入和 float32 `(1,1,512,512)` 输出。

<a id="usage"></a>
## 使用

将精确外部制品放到 `samples/vision/modnet/model/modnet_512x512_rgb.bin` 后，在仓库根目录运行：

```bash
python3 -m samples.vision.modnet.runtime.python.main --target x5 \
  --asset-id x5:modnet:modnet_512x512_rgb.bin
```

成功判断为退出码 `0`、JSON 含 `matte_path` 且写出 uint8 matte。默认背景可读时会合成 `test_data/result.png`。`bash samples/vision/modnet/runtime/python/run.sh --target x5 --asset-id x5:modnet:modnet_512x512_rgb.bin` 等价。

<a id="parameters"></a>
## 参数

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--target` | `auto` | 解析唯一发布目标 X5 |
| `--asset-id` | `null` | 精确手工身份，外部模型路径必需 |
| `--model-path` | `null` | 已存在外部模型路径；不下载 |
| `--test-img` | `samples/vision/modnet/test_data/person.jpg` | BGR 输入图像 |
| `--bg-img` | `samples/vision/modnet/test_data/bg.jpg` | 可选 BGR 背景；文件缺失则跳过合成 |
| `--matte-save-path` | `samples/vision/modnet/test_data/matte.png` | uint8 灰度输出 |
| `--img-save-path` | `samples/vision/modnet/test_data/result.png` | 合成图输出 |
| `--priority` | `0` | runtime 调度优先级 |
| `--bpu-cores` | `[0]` | BPU 核索引 |
| `--ref-size` | `512` | 固定编译输入尺寸；其他值拒绝 |
| `--list-models` | `false` | 不加载 SDK，列出 manifest 事实 |
| `--dry-run` | `false` | 不加载 SDK 或模型，打印契约 |

`--list-models` 与 `--dry-run` 互斥。用户或 runtime 错误返回 `2`。

<a id="results"></a>
## 结果

`MODNetMatting.postprocess` 返回原图几何的 owned uint8 灰度 matte。可选合成使用源线性 alpha 公式并写出 BGR 图像。raw forward 结果保持 float32 `[0,1]`，runner 不保存也不再归一化。

<a id="integration-example"></a>
## 集成示例

准备好手工模型和本地图片后，下面完整示例定义所有变量，并将显式阶段与 `predict` 对照：

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.modnet.runtime.python.cli import resolve_selection
from samples.vision.modnet.runtime.python.modnet import MODNetMatting
from samples.vision.modnet.runtime.python.cli import composite

target = "x5"
asset_id = "x5:modnet:modnet_512x512_rgb.bin"
model_path = Path("samples/vision/modnet/model/modnet_512x512_rgb.bin")
image_path = Path("samples/vision/modnet/test_data/person.jpg")
background_path = Path("samples/vision/modnet/test_data/bg.jpg")
selection = resolve_selection(target, asset_id=asset_id, model_path=model_path)
task = MODNetMatting(selection)
image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
prepared = task.preprocess(image)
raw_matte = task.infer(prepared.tensors)
explicit_matte = task.postprocess(raw_matte, prepared.context)
composed_result = composite(image, explicit_matte, cv2.imread(str(background_path)))
assert np.array_equal(explicit_matte, task.predict(image))
print(explicit_matte.shape, composed_result.shape)
```

<a id="stage-io"></a>
## 三阶段 I/O

- `preprocess(image)` 校验 BGR HWC，执行 BGR→RGB、`(pixel-127.5)/127.5`、长边 512 resize 和居中 zero padding，返回含 `tensors, context` 的 `PreparedInput`。
- `infer(tensors)` 校验绑定 tensor，返回 owned raw float32 `(1,1,512,512)` matte。
- `postprocess(raw, context)` 将源 `[0,1]` matte 转 uint8、去 padding 并恢复原图几何。
- `predict(image)` 串联三个阶段；geometry 放在该次调用的冻结 context 中，不放入可变 task 字段。
- 既有的 `pre_process`、`forward`、`post_process` 名称保留为 `preprocess`、`infer`、`postprocess` 的可导入薄别名——同一实现，两个名字。

<a id="troubleshooting"></a>
## 故障排查

- 没有精确 `--asset-id x5:modnet:modnet_512x512_rgb.bin` 的模型路径会被拒绝。
- 手工模型缺失或输入不可读时，在板端推理前返回 `2`。
- metadata 名称、shape 或 dtype 错误会被拒绝；不会用 runtime cast 掩盖不匹配。
- `--ref-size` 必须保持 512，因为没有发布其他编译配置。
