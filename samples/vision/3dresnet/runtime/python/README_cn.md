[English](./README.md) | 简体中文

# Python Runtime — R3D-18

<a id="environment"></a>
## 环境

- 板端执行：RDK S100 和匹配的 `hbm_runtime` Python 环境；source 没有记录板端镜像/runtime 版本，板端状态为 not-run。
- 主机验证：仓库 `.venv`、Python 3.14.7、`numpy` 和 `PyYAML`（`requirements-host.txt` 列出主机依赖）。
- runtime 接收已经准备好的 NumPy 片段，不导入视频解码器，不读取视频帧，不做 resize，也不做像素归一化。
- `--help`、`--list-models` 和显式 `--dry-run` 不需要 SDK，不会构造 `hbm_runtime`。

<a id="usage"></a>
## 使用

先准备模型。在仓库根目录执行显式的板端命令：

```bash
# cwd：仓库根目录；前置：model/download.sh 已放置 HBM
python3 samples/vision/3dresnet/runtime/python/main.py \
  --target s100 \
  --asset-id s:3dresnet:s100/r3d_18.hbm
# 预期：退出码 0 并输出 predictions JSON；板卡身份必须解析为 S100
```

在 runtime 目录也可以执行：

```bash
# cwd：samples/vision/3dresnet/runtime/python
bash run.sh --target s100 --asset-id s:3dresnet:s100/r3d_18.hbm
```

`run.sh` 只是 CLI 委托，不会下载模型。主机 fixture pipeline 的验证命令为：

```bash
# cwd：仓库根目录
.venv/bin/python -m unittest discover -s samples/vision/3dresnet/tests -v
# 预期：全部发现的测试通过，OK；这不等于板端验证
```

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `--target` | `auto`/`s100` | `auto` | 目标板卡；`auto` 只在解析准备/执行目标时检测身份 |
| `--asset-id` | string | `null` | 精确 manifest reference；外部 `--model-path` 必须提供 |
| `--model-path` | path | `null` | 外部 HBM 路径；省略时使用选中制品的 sample 默认路径 |
| `--test-clip` | path | `samples/vision/3dresnet/test_data/video0.npy` | 准备好的 `.npy` 输入 |
| `--label-file` | path | `samples/vision/3dresnet/test_data/kinetics_classnames.json` | 400 条 source 标签映射 |
| `--top-k` | integer | `5` | 返回数量，范围 1 到 400 |
| `--priority` | integer | `0` | runtime 调度优先级，范围 0 到 255 |
| `--bpu-cores` | 一个或多个 integer | `[0]` | 非负 BPU core 编号 |
| `--list-models` | flag | `false` | 不加载模型，仅打印精确 manifest 制品 |
| `--dry-run` | flag | `false` | 不加载 SDK，仅打印解析后的路径和 tensor 契约 |

外部路径必须配对 `--asset-id s:3dresnet:s100/r3d_18.hbm`；runtime 不从文件名猜身份。主机 `--dry-run` 必须显式给 `--target s100`。

<a id="results"></a>
## 结果

CLI 成功时向 stdout 输出一个 JSON 对象：

```json
{
  "asset_id": "s:3dresnet:s100/r3d_18.hbm",
  "target": "s100",
  "clip": ".../test_data/video0.npy",
  "predictions": [
    {"class_id": 5, "score": "float32 probability", "label": "archery"}
  ]
}
```

`predictions` 恰好包含 `--top-k` 条按 source 兼容 softmax 概率降序排列的结果。`class_id` 在 `[0,399]`；`score` 是 float32 softmax 值；`label` 是去除字面双引号后的 source JSON 名称。CLI 不写输出文件。

<a id="integration-example"></a>
## 集成示例

目录名 `3dresnet` 不能出现在 `from ... import ...` 语句中。请从仓库根目录运行下面示例，使用完整 package name 的 `importlib.import_module`，不需要注入 runtime 目录到 `sys.path`：

```python
import importlib
import sys
from pathlib import Path

import numpy as np

repo = Path.cwd()
binding_mod = importlib.import_module("samples.vision.3dresnet.runtime.python.model_binding")
runner_mod = importlib.import_module("samples.vision.3dresnet.runtime.python.model_runner")
task_mod = importlib.import_module("samples.vision.3dresnet.runtime.python.classification")
labels_mod = importlib.import_module("samples.vision.3dresnet.runtime.python.labels")

selection = binding_mod.resolve_selection(
    "s100",
    asset_id="s:3dresnet:s100/r3d_18.hbm",
    model_path=repo / "samples/vision/3dresnet/model/s100/r3d_18.hbm",
)
runner = runner_mod.RuntimeModelRunner(selection)
binding = runner.load()
labels = labels_mod.load_labels(repo / "samples/vision/3dresnet/test_data/kinetics_classnames.json")
task = task_mod.VideoClassificationTask(runner, binding, top_k=5, labels=labels)
clip = np.load(repo / "samples/vision/3dresnet/test_data/video0.npy", allow_pickle=False)

prepared = task.pre_process(clip)
raw_outputs = task.forward(prepared.tensors)
explicit_result = task.post_process(raw_outputs)
composed_result = task.predict(clip)
assert np.array_equal(explicit_result.class_ids, composed_result.class_ids)
assert np.array_equal(explicit_result.scores, composed_result.scores)
assert explicit_result.labels == composed_result.labels
```

<a id="stage-io"></a>
## 四阶段 API I/O

| 阶段 | 契约 |
| --- | --- |
| `pre_process(clip)` | 输入精确 shape `(1,3,16,112,112)` 的 NumPy numeric clip；返回使用 runtime 实际 input name 的 `PreparedInput.tensors`，值 cast 为 contiguous float32，并返回本调用的 `VideoContext`。 |
| `forward(tensors)` | 校验实际 input name、五维 shape、F32 finite 值和唯一实际 output name；返回 raw F32 score，不做 softmax 或文件 I/O。 |
| `post_process(outputs)` | 校验实际 output name、400-score shape、F32 和 finite；委托 `samples._shared.classification.topk_from_scores` 执行 softmax/Top-K。 |
| `predict(clip)` | 按顺序串联三个阶段并返回 `ClassificationResult`；不把 context 存入 task 状态。 |

输入片段已经是 RGB 且已经归一化；task 不做视频/图像解码、resize 或归一化。

<a id="troubleshooting"></a>
## 故障排查

- **模型路径被拒绝：** 外部路径必须提供精确 asset ID：`s:3dresnet:s100/r3d_18.hbm`。
- **板卡身份被拒绝：** `--target s100` 只选择发布制品，不构成硬件证据；执行仍需检测到 S100。
- **模型缺失：** 在仓库根目录执行 `bash samples/vision/3dresnet/model/download.sh s100`。
- **片段 shape 错误：** 使用精确 shape `(1,3,16,112,112)` 的准备好的 `.npy`，runtime 不 reshape 或解码视频。
- **tensor name 不匹配：** binding 读取 runtime 的唯一实际名称；缺少、多余或重排 tensor 都会拒绝，不会编造 `input` 或 `output` 名称。
