# Python 运行时

`main.py` 是 canonical 用户入口。它从发布 Manifest 解析一个完整制品引用，
检查检测到的板卡，延迟加载 `hbm_runtime`，然后执行一条
`ClassificationTask` 流程。模型准备必须显式完成；运行时不会下载模型或安装包。

## 准备和运行

请使用已经包含匹配 `hbm_runtime`、NumPy 和 OpenCV-Python 的板端 Python
环境。在仓库根目录先准备制品：

```bash
bash samples/vision/resnet/model/download.sh x5
```

然后在匹配的 X5 板卡执行：

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file platforms/x5/datasets/imagenet/imagenet_classes.names
```

S100 或 S600 使用相应的 `s:resnet18:<target>/...` 引用和 model 目录。
`--list-models` 会列出已发布引用；`--dry-run` 可在不访问板卡、不加载模型、
不下载的情况下解析选择。

## 命令参数

| 参数 | 行为 |
| --- | --- |
| `--target` | `auto`、`x5`、`s100`、`s100p` 或 `s600`；真实运行目标必须匹配检测到的板卡 |
| `--asset-id` | Manifest 中的完整 `group:sample:filename` 引用 |
| `--model-path` | 已存在的 `.bin` 或 `.hbm`，必须与 `--asset-id` 配对 |
| `--test-img` / `--label-file` | BGR 输入图片和 ImageNet 标签 |
| `--top-k` / `--topk` | 结果数量，默认 5 |
| `--resize-type` | 0 直接拉伸，1 使用 BGR 127 填充的 letterbox |
| `--priority` / `--bpu-cores` | 运行时调度，默认 0 和 `[0]` |
| `--img-save-path` | 可选标注图输出路径 |
| `--list-models` / `--dry-run` | 不依赖 SDK 的查看模式 |

canonical API 接收一张 BGR `uint8` 图片：

```python
from samples.vision.resnet.runtime.python.classification import ClassificationTask
from samples.vision.resnet.runtime.python.model_binding import bind_model, resolve_selection
from samples.vision.resnet.runtime.python.model_runner import RuntimeModelRunner

selection = resolve_selection(
    "x5",
    asset_id="x5:resnet:resnet18_224x224_nv12.bin",
    model_path="samples/vision/resnet/model/resnet18_224x224_nv12.bin",
)
runner = RuntimeModelRunner(selection)
binding = runner.load()
task = ClassificationTask(runner, binding, top_k=5)
result = task.predict(image)
print(result.class_ids, result.scores, result.labels)
```

任务返回 `ClassificationResult(class_ids, scores, labels)`。X5 输入为 packed
`(1,336,224,1)` uint8 NV12。S100/S600 保持两个数组：Y `(1,224,224,1)`、
UV `(1,112,112,2)`。运行前会按选择的契约校验张量名、输出名、形状和 F32 类型。

源 wrapper 对一个返回 score 向量执行 softmax。由于现有 X5 转换资料不能证明
`prob` 是在图内还是 wrapper 中归一化，canonical 将这个行为保留为
`legacy_softmax`，不会静默更改输出语义。

## 兼容适配器

原有导入路径仍然有效，并且是薄适配器：

```python
from platforms.x5.samples.vision.resnet.runtime.python.resnet import ResNet, ResNetConfig
from platforms.s.samples.vision.resnet18.runtime.python.resnet18 import Resnet18, Resnet18Config
```

X5 适配器保留嵌套 `{model_name: {input_name: tensor}}` 输入以及
`(topk_idx, topk_prob, topk_labels)` 返回值。S18 适配器保留嵌套 Y/UV 输入以及
`(class_id, probability)` 列表。它们的 `pre_process`、`forward`、`post_process`
分别委托 `tensor_io.prepare_nv12`、`RuntimeModelRunner` 和
`classification.topk_from_scores`。`runtime=`/`runtime_factory=` 仅是主机
fixture 接口；正常板端使用会延迟构造已安装的 SDK。

## 代码流程和故障排查

```text
main.py
  -> model_binding.resolve_selection
  -> platforms.require_execution_target
  -> model_runner.RuntimeModelRunner.load
  -> model_binding.bind_model
  -> classification.ClassificationTask.predict
       -> tensor_io.prepare_nv12
       -> runtime.run
       -> classification.topk_from_scores
```

`model_path requires --asset-id` 表示不能从文件名安全选择输入协议。
`Cannot identify this board` 表示无法自动读取板卡身份；显式目标可帮助 dry-run，
但本身不是硬件证据。形状或类型错误表示制品与 X5 packed 或 S split 契约不匹配。
如果输出不同，请先比较 raw F32 输出、图片、缩放方式、Top-K 和制品，再改变
score 处理。

在仓库根目录执行主机契约和适配器测试：

```bash
python3 -m unittest discover -s samples/vision/resnet/tests -v
```
