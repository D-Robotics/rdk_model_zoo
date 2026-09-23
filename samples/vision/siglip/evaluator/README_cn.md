[English](./README.md) | 简体中文

# 模型评估 — SigLIP 视觉特征

本文所有数值表都是固定 S sample 和发布 benchmark 中的历史源记录，用于保留来源和可比性；不是本轮重新跑出的 benchmark。本轮未使用评估脚本、数据集下载、板卡或 HBM 下载。

<a id="dataset"></a>
## 数据集

历史 `pooler_output` 零样本分类记录使用 ImageNet-1k validation（50,000 张）。历史 `last_hidden_state` 语义一致性记录使用 COCO2014 validation（5,000 张）。源资料没有发布准备脚本、精确压缩包版本、目录结构或评估实现。

```text
# cwd：仓库根目录
# 准备：未提供；不要从本文推断下载命令。
# 预期源数据布局：由评估负责人提供的 ImageNet-1k val 和 COCO2014 val
```

<a id="environment"></a>
## 环境

- 历史测量：RDK S100 和 S100P，CPU/BPU 设置见下文。
- 当前对照流程：同一块板卡、板端 `hbm_runtime`、runtime 依赖（`numpy`、`opencv-python`、`PyYAML`）和固定源 runtime `platforms/s/samples/vision/siglip/runtime/python`。
- 复用：unified `model_binding.py`、`model_runner.py`、`tensor_io.py`、`embedding.py`；固定源 runtime 的 legacy `SigLIPConfig`/`SigLIP`。
- 当前板端及 runtime 版本：未核验。

<a id="command"></a>
## 评估命令

仓库没有评估脚本。下面是可复制的同板 raw output 对照流程，仅作文档记录，本轮未执行。它使用现有源 runtime `platforms/s/samples/vision/siglip/runtime/python`，把完整数组写入唯一 run 目录，不做 pooling、归一化、反量化或分数转换。

```bash
# cwd：仓库根目录；前置：板端已准备一个精确 HBM
PYTHONPATH="$PWD:$PWD/platforms/s:$PWD/platforms/s/samples/vision/siglip/runtime/python" python3 - <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import cv2
import numpy as np

from siglip import SigLIP, SigLIPConfig
from samples.vision.siglip.runtime.python.model_binding import resolve_selection
from samples.vision.siglip.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.siglip.runtime.python.embedding import SigLIPTask

repo = Path.cwd()
model_path = repo / "samples/vision/siglip/model/s100/bpu-siglip-base-patch16-224.hbm"
image_path = repo / "samples/vision/siglip/test_data/dog.jpg"
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
raw_dir = repo / "evaluator-output" / f"siglip-raw-{run_id}"
raw_dir.mkdir(parents=True, exist_ok=False)
image = cv2.imread(str(image_path))
if image is None or not model_path.is_file():
    raise RuntimeError("prepare the image and exact HBM before running")

target = "s100"
variant = "base-patch16-224"
submodel = "pooler_output"
image_size = 224
priority = 0
bpu_cores = [0]

legacy = SigLIP(SigLIPConfig(str(model_path), image_size=image_size, submodel=submodel))
legacy.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)
legacy_inputs = legacy.pre_process(image)
legacy_raw_nested = legacy.forward(legacy_inputs)
legacy_raw = np.asarray(legacy_raw_nested[submodel]["_output_0"])
np.save(raw_dir / "legacy.npy", legacy_raw, allow_pickle=False)

selection = resolve_selection(target, variant=variant, model_path=model_path,
                              asset_id="s:siglip:s100/bpu-siglip-base-patch16-224.hbm",
                              submodel=submodel, image_size=image_size)
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)
task = SigLIPTask(runner, binding)
prepared = task.pre_process(image)
unified_raw_mapping = task.forward(prepared.tensors)
unified_raw = task.post_process(unified_raw_mapping)
np.save(raw_dir / "unified.npy", unified_raw, allow_pickle=False)

if legacy_raw.shape != unified_raw.shape or legacy_raw.dtype != unified_raw.dtype:
    raise AssertionError((legacy_raw.shape, legacy_raw.dtype, unified_raw.shape, unified_raw.dtype))
if np.issubdtype(legacy_raw.dtype, np.floating):
    np.testing.assert_allclose(legacy_raw, unified_raw, rtol=0.0, atol=1e-5)
else:
    np.testing.assert_array_equal(legacy_raw, unified_raw)
print({"legacy": str(raw_dir / "legacy.npy"), "unified": str(raw_dir / "unified.npy"), "comparison": "passed", "run_id": run_id})
PY
# 预期：两个完整 .npy raw 数组和通过对照结果行；断言失败时退出非零；本流程未执行
```

| 参数 | 类型 | 示例默认值 | 说明 |
| --- | --- | --- | --- |
| `target` | str | 示例为 `s100` | 显式板卡 target；第二个支持目标需在 `s100p` 重复。 |
| `variant` | str | 示例为 `base-patch16-224` | 八个发布 variant 之一。 |
| `submodel` | str | 示例为 `pooler_output` | 每次对照一个固定打包子模型。 |
| `image_size` | int | 示例为 `224` | 必须等于所选 variant。 |
| `raw_dir` | path | `evaluator-output/siglip-raw-<UTC 微秒 run id>` | 完整 legacy/unified 数组的唯一落盘目录。 |

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| `pooler_output` 延迟 | 全局特征子模型的一次 BPU `perf` 测量。 | 单线程；输入分辨率和输出 shape 见表；使用下述板端 CPU/BPU 设置。 |
| `last_hidden_state` 延迟 | patch 特征子模型的一次 BPU `perf` 测量。 | 单线程；输入分辨率和输出 shape 见表；使用下述板端 CPU/BPU 设置。 |
| TOP1/TOP5 | 由全局嵌入得到的 ImageNet 零样本分类准确率。 | ImageNet-1k val，50,000 张；浮点和 BPU 路径都使用 RGB `(127,127,127)` letterbox。 |
| Cosine Similarity | patch 特征相对参照的平均/最小~最大及 1% low 相似度。 | COCO2014 val，5,000 张；相同 RGB letterbox。 |
| MSE | patch 特征相对参照的平均/最小~最大及 1% low 均方误差。 | COCO2014 val，5,000 张；相同 RGB letterbox。 |

历史板端设置：

- S100：CPU `6 x A78AE @ 1.5GHz`，BPU `1 x Nash-E @ 1.0GHz`。
- S100P：CPU `6 x A78AE @ 2.0GHz`，BPU `1 x Nash-M @ 1.5GHz`。
- 源资料记录了 CPU policy 0/4 和 BPU `28108000.bpu` 的 performance governor 命令；本轮未执行。

### 历史 `pooler_output` 性能（本轮未复测）

| Model Name | Input Size | Embedding Size | Params total / vision | RDK S100 | RDK S100P |
|---|---|---|---|---|---|
| siglip-base-patch16-224 | `(1,3,224,224)` | `(1,1,768)` | `0.2 B / 0.09 B` | 26.8 ms | 18.8 ms |
| siglip-base-patch16-384 | `(1,3,384,384)` | `(1,1,768)` | `0.2 B / 0.09 B` | 46.7 ms | 32.3 ms |
| siglip-base-patch16-512 | `(1,3,512,512)` | `(1,1,768)` | `0.2 B / 0.09 B` | 81.7 ms | 55.8 ms |
| siglip-large-patch16-256 | `(1,3,256,256)` | `(1,1,1024)` | `0.7 B / 0.32 B` | 68.8 ms | 47.2 ms |
| siglip-large-patch16-384 | `(1,3,384,384)` | `(1,1,1024)` | `0.7 B / 0.32 B` | 132.5 ms | 91.4 ms |
| siglip-so400m-patch14-224 | `(1,3,224,224)` | `(1,1,1152)` | `0.9 B / 0.43 B` | 89.8 ms | 62.2 ms |
| siglip-so400m-patch14-384 | `(1,3,384,384)` | `(1,1,1152)` | `0.9 B / 0.43 B` | 255.7 ms | 175.5 ms |
| siglip-so400m-patch16-256-i18n | `(1,3,256,256)` | `(1,1,1152)` | `1.0 B / 0.43 B` | 89.6 ms | 61.9 ms |

### 历史 `last_hidden_state` 性能（本轮未复测）

| Model Name | Input Size | Embedding Size | Params total / vision | RDK S100 | RDK S100P |
|---|---|---|---|---|---|
| siglip-base-patch16-224 | `(1,3,224,224)` | `(1,196,768)` | `0.2 B / 0.09 B` | 26.0 ms | 18.3 ms |
| siglip-base-patch16-384 | `(1,3,384,384)` | `(1,576,768)` | `0.2 B / 0.09 B` | 45.9 ms | 31.7 ms |
| siglip-base-patch16-512 | `(1,3,512,512)` | `(1,1024,768)` | `0.2 B / 0.09 B` | 80.8 ms | 55.3 ms |
| siglip-large-patch16-256 | `(1,3,256,256)` | `(1,256,1024)` | `0.7 B / 0.32 B` | 67.6 ms | 46.5 ms |
| siglip-large-patch16-384 | `(1,3,384,384)` | `(1,576,1024)` | `0.7 B / 0.32 B` | 131.3 ms | 90.5 ms |
| siglip-so400m-patch14-224 | `(1,3,224,224)` | `(1,256,1152)` | `0.9 B / 0.43 B` | 88.6 ms | 61.4 ms |
| siglip-so400m-patch14-384 | `(1,3,384,384)` | `(1,729,1152)` | `0.9 B / 0.43 B` | 254.2 ms | 174.5 ms |
| siglip-so400m-patch16-256-i18n | `(1,3,256,256)` | `(1,256,1152)` | `1.0 B / 0.43 B` | 88.3 ms | 61.1 ms |

<a id="outputs"></a>
## 输出

对照流程将完整 raw 数组写入唯一的 `evaluator-output/siglip-raw-<UTC 微秒 run id>/legacy.npy` 和 `unified.npy`，不会用缩减摘要替代数组。未来评估可在旁边增加 JSON 记录，但目前没有此类结果。必须先相等 shape 和 dtype；整数 raw 必须完全相等，浮点 raw 允许 `rtol=0`、`atol=1e-5`，且断言必须通过。

<a id="reference-results"></a>
## 参考结果

以下两张历史表保留源中的全部行和列。本轮状态为 `not-run`。来源：`platforms/s/samples/vision/siglip/evaluator/README.md`，并由 `platforms/s/docs/release/benchmarks.yaml` 佐证。

### 历史 `pooler_output` 零样本分类（本轮未复测）

| Model Name | PyTorch TOP1 / TOP5 | BPU TOP1 / TOP5 |
|---|---|---|
| siglip-base-patch16-224 | 0.7123 / 0.9143 | 0.7118 / 0.9144 |
| siglip-base-patch16-384 | 0.7411 / 0.9318 | 0.7418 / 0.9319 |
| siglip-base-patch16-512 | 0.7490 / 0.9343 | 0.7482 / 0.9340 |
| siglip-large-patch16-256 | 0.7490 / 0.9238 | 0.7490 / 0.9242 |
| siglip-large-patch16-384 | 0.7584 / 0.9252 | 0.7595 / 0.9256 |
| siglip-so400m-patch14-224 | 0.7659 / 0.9361 | 0.7651 / 0.9357 |
| siglip-so400m-patch14-384 | 0.7872 / 0.9433 | 0.7893 / 0.9447 |
| siglip-so400m-patch16-256-i18n | 0.7678 / 0.9395 | 0.7668 / 0.9397 |

### 历史 `last_hidden_state` 语义一致性（本轮未复测）

| Model Name | Cosine Similarity mean (min ~ max), 1% low | MSE mean (min ~ max), 1% low |
|---|---|---|
| siglip-base-patch16-224 | 0.991 (0.951 ~ 0.997), 0.980 | 0.087 (0.024 ~ 0.471), 0.039 |
| siglip-base-patch16-384 | 0.989 (0.960 ~ 0.997), 0.977 | 0.113 (0.029 ~ 0.409), 0.050 |
| siglip-base-patch16-512 | 0.987 (0.956 ~ 0.995), 0.974 | 0.142 (0.045 ~ 0.507), 0.067 |
| siglip-large-patch16-256 | 0.990 (0.933 ~ 0.997), 0.974 | 0.069 (0.018 ~ 0.497), 0.024 |
| siglip-large-patch16-384 | 0.985 (0.900 ~ 0.995), 0.965 | 0.111 (0.034 ~ 0.775), 0.048 |
| siglip-so400m-patch14-224 | 0.984 (0.850 ~ 0.995), 0.961 | 0.104 (0.028 ~ 1.038), 0.041 |
| siglip-so400m-patch14-384 | 0.980 (0.859 ~ 0.993), 0.957 | 0.140 (0.040 ~ 1.093), 0.059 |
| siglip-so400m-patch16-256-i18n | 0.984 (0.878 ~ 0.996), 0.959 | 0.082 (0.018 ~ 0.570), 0.030 |

<a id="boundaries"></a>
## 边界

- 本目录没有评估实现或数据集准备脚本；对照只是板端流程文档，目前为 not-run。
- 四张表是历史记录，不能证明当前制品身份、runtime 版本或板端可复现性。
- 本 sample 仅评估视觉特征编码器，不覆盖文本编码器、文本 tokenizer、图文分数、校准配方或 C++ 评估器。

## 许可

评估文档和对照 helper 遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。保留源贡献者署名：Cauchy @吴超。
