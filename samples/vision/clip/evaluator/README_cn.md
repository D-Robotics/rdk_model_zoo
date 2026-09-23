[English](./README.md) | 简体中文

# 模型评估 — CLIP 图文匹配

本文记录源验证路径。源没有发布 benchmark 表，因此不虚构延迟或精度数值。本轮没有执行模型或板端运行。

<a id="dataset"></a>
## 数据集

默认验证输入为 `samples/vision/clip/test_data/dog.jpg`，prompt 为 `a diagram` 和 `a dog`。源 BPE 词表是真实的 `platforms/x5/samples/vision/clip/runtime/python/bpe_simple_vocab_16e6.txt.gz`（已保留到 unified runtime）。没有发布更大的评估数据集或准备脚本。

```text
# cwd：仓库根目录
samples/vision/clip/test_data/dog.jpg
samples/vision/clip/runtime/python/bpe_simple_vocab_16e6.txt.gz
# prompts：a diagram,a dog
```

<a id="environment"></a>
## 环境

- 目标：RDK X5；图像 encoder 通过板端 `hbm_runtime`，文本 encoder 通过 CPU `onnxruntime`。
- 主机测试：Python 3.14.7、NumPy、OpenCV、`ftfy==6.3.1`、`regex==2026.9.10`；注入 runtime fixture，不要求主机 ONNX Runtime。
- Legacy 对照源目录：`platforms/x5/samples/vision/clip/runtime/python`。复用源 `SimpleTokenizer` 和 BPE 文件；不调用 legacy 主入口，避免覆盖 `test_data/inference.png`。

<a id="command"></a>
## 评估命令

源验证入口是 `runtime/python/main.py` 和 `run.sh`。下面的显式命令运行当前 sample 并写入用户选择的图片，只作文档记录，本轮未执行。

```bash
# cwd：仓库根目录；前置：X5 两个模型制品已准备
python3 samples/vision/clip/runtime/python/main.py \
  --target x5 \
  --test-img samples/vision/clip/test_data/dog.jpg \
  --texts "a diagram,a dog" \
  --img-save-path /tmp/clip-eval/inference.png
# 预期：JSON scores/order 和 /tmp/clip-eval/inference.png；没有发布数值 benchmark
```

同板 legacy/unified raw 对照使用真实 legacy 目录和唯一 UTC 输出目录。流程通过 `CLIPMatcher` 复用源 BPE，不调用 legacy 主入口，也不覆盖 `test_data/inference.png`。它保存完整 image/text raw 数组并断言对拍。本轮未执行。

```bash
# cwd：仓库根目录；前置：X5 板卡、精确模型对，且板端两个 runtime 可用
PYTHONPATH="$PWD:$PWD/platforms/x5/samples/vision/clip/runtime/python" python3 - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np

from samples._shared.platforms import require_execution_target
require_execution_target("x5")  # before legacy imports or either SDK factory
from clip_retrieval import CLIPConfig, CLIPMatcher
from samples.vision.clip.runtime.python.matching import CLIPTask
from samples.vision.clip.runtime.python.model_binding import resolve_selection
from samples.vision.clip.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.clip.runtime.python.tokenization import PromptTokenizer

repo = Path.cwd()
image_path = repo / "samples/vision/clip/test_data/dog.jpg"
image_model_path = repo / "samples/vision/clip/model/img_encoder.bin"
text_model_path = repo / "samples/vision/clip/model/text_encoder.onnx"
image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
texts = ["a diagram", "a dog"]
if image is None or not image_model_path.is_file() or not text_model_path.is_file():
    raise RuntimeError("prepare the image and exact X5 model pair before running")

run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
raw_dir = repo / "evaluator-output" / f"clip-raw-{run_id}"
raw_dir.mkdir(parents=True, exist_ok=False)

legacy = CLIPMatcher(CLIPConfig(str(image_model_path), str(text_model_path)))
legacy_inputs = legacy.pre_process(image)
legacy_image_nested = legacy.image_model.run(legacy_inputs)
legacy_image_raw = np.asarray(legacy_image_nested[legacy.image_model_name][legacy.output_names[0]])
legacy_text_raw = np.asarray(legacy.text_session.run(
    [legacy.text_output_name], {legacy.text_input_name: legacy.tokenize(texts)}
)[0])
np.save(raw_dir / "legacy_image.npy", legacy_image_raw, allow_pickle=False)
np.save(raw_dir / "legacy_text.npy", legacy_text_raw, allow_pickle=False)

selection = resolve_selection("x5")
runner = RuntimeModelRunner(selection)
binding = runner.load()
task = CLIPTask(runner, binding, PromptTokenizer())
prepared = task.pre_process(image, texts)
unified_raw = task.forward(prepared.tensors)
np.save(raw_dir / "unified_image.npy", unified_raw["image_feature"], allow_pickle=False)
np.save(raw_dir / "unified_text.npy", unified_raw["text_features"], allow_pickle=False)

np.testing.assert_array_equal(legacy_inputs[legacy.image_model_name][legacy.input_names[0]],
                              prepared.tensors["image"])
np.testing.assert_array_equal(legacy.tokenize(texts), prepared.tensors["texts"])
for reference, candidate in ((legacy_image_raw, unified_raw["image_feature"]),
                             (legacy_text_raw, unified_raw["text_features"])):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise AssertionError((reference.shape, reference.dtype, candidate.shape, candidate.dtype))
    np.testing.assert_allclose(reference, candidate, rtol=0.0, atol=1e-5)
print({"raw_dir": str(raw_dir), "comparison": "passed", "run_id": run_id})
PY
# 预期：唯一 UTC 目录中的四个完整 raw .npy 数组；断言失败时退出非零
```

| 参数 | 类型 | 示例默认值 | 说明 |
| --- | --- | --- | --- |
| `target` | str | `x5` | 唯一发布 CLIP target。 |
| `texts` | list[str] | `a diagram`、`a dog` | 使用保留 BPE 词表编码的源 prompt。 |
| `run_id` | UTC 字符串 | 每次生成 | 防止 raw 数组覆盖。 |
| `raw_dir` | path | `evaluator-output/clip-raw-<run_id>` | 完整 legacy/unified 图像和文本数组的唯一目录。 |

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| Cosine similarity | 每个文本特征与图像特征点积除以两者 L2 范数及 `1e-12` 稳定项。 | 一张图、N 个 prompt、float32 特征；分数保持 prompt 顺序。 |
| Rank order | 对 cosine 分数执行降序 `argsort`。 | 保留源 NumPy argsort 次序，不保证不同版本下精确平局的排序次序。 |
| Legacy parity | 在 cosine 转换前对比图像/文本 encoder raw 数组。 | 同板、同输入、同源 BPE；前处理张量相同，shape 和 F32 dtype 必须精确一致，`rtol=0`、`atol=1e-5`。 |

本 sample 没有发布延迟、top-k、检索或分类指标。

<a id="outputs"></a>
## 输出

runtime 将标注图片按 `--img-save-path` 精确写出，并打印 JSON scores/order。对照流程在唯一 UTC 目录下写 `legacy_image.npy`、`legacy_text.npy`、`unified_image.npy`、`unified_text.npy`。本轮没有新 raw 数组或 benchmark 结果。

<a id="reference-results"></a>
## 参考结果

源 evaluator 没有公开数值 benchmark 表。源验证的定性预期是：对于 `dog.jpg`，`a dog` 的分数应高于 `a diagram`。本轮状态为 `not-run`，不声明数值。

| 参考项 | 值 | 条件 | 来源 |
| --- | --- | --- | --- |
| Dog prompt 排序 | `a dog` 排在 `a diagram` 之前 | X5 制品对、源 BPE、`dog.jpg`、cosine 排序 | `platforms/x5/samples/vision/clip/evaluator/README.md` |
| Published benchmark | 未提供 | 源无延迟/精度表 | source evaluator README |

<a id="boundaries"></a>
## 边界

- 没有独立数据集评估器或公开 benchmark。
- 文本 encoder 是 CPU ONNX，板端需要 ONNX Runtime；主机 fixture 测试不证明板端可用。
- Legacy 对照只比较 raw 数组，不调用 legacy 主入口，也不覆盖受版本控制的可视化图。
- 本 sample 只覆盖图文相似度，不包含 C++、训练或文本生成任务。

## 许可

评估文档遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。
