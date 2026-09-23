[English](./README.md) | 简体中文

# R3D-18 评测记录

<a id="dataset"></a>
## 数据集

随附功能输入是预处理后的 `test_data/video0.npy`，shape `(1,3,16,112,112)`、dtype float32，内容为 16 帧射箭样例。`test_data/kinetics_classnames.json` 是 CLI 使用的 400 条 Kinetics name-to-id 映射；读取后按 source 行为去除字面双引号并转为 id-to-name 标签。

本目录没有完整 Kinetics-400 数据集、视频解码器、抽帧代码或数据集下载命令。该片段原始获取和预处理命令未记录。

<a id="environment"></a>
## 环境

- 统一主机检查：仓库 `.venv`、Python 3.14.7、NumPy 和 PyYAML；fixture tests 不需要 HBM 或板卡。
- 板端功能检查：RDK S100、匹配的 `hbm_runtime`、准备好的 `model/s100/r3d_18.hbm` 和可识别的 S100 身份。
- source 性能表使用 `hrt_model_exec` 生成；完整命令、镜像、runtime 版本和 raw 输出文件均未保留，因此只是历史 source 记录，不是复现实测。

<a id="command"></a>
## 评测命令

统一主机契约检查：

```bash
# cwd：仓库根目录
.venv/bin/python -m unittest discover -s samples/vision/3dresnet/tests -v
# 预期：全部发现的测试通过，OK；只检查 source 兼容的主机行为
```

统一 S100 功能 smoke 命令：

```bash
# cwd：仓库根目录；前置：显式下载模型和 S100 板卡
bash samples/vision/3dresnet/model/download.sh s100
python3 samples/vision/3dresnet/runtime/python/main.py \
  --target s100 \
  --asset-id s:3dresnet:s100/r3d_18.hbm \
  --model-path samples/vision/3dresnet/model/s100/r3d_18.hbm \
  --test-clip samples/vision/3dresnet/test_data/video0.npy \
  --label-file samples/vision/3dresnet/test_data/kinetics_classnames.json \
  --top-k 5 --priority 0 --bpu-cores 0
# 预期：退出码 0 和包含 5 条 predictions 的 JSON；本轮未执行该命令
```

旧 source 只说明使用 `hrt_model_exec` 做性能测试，没有保留完整可复制命令或输入/输出文件参数，因此这里不杜撰替代命令。

如果未来需要迁移对照，可以使用下面的自包含 legacy/unified 基线：两边使用同一个 S100 模型和 `video0.npy`。它在导入 source wrapper 或加载 SDK 前先做实际 S100 身份门禁，在同一个唯一 UTC 目录保存两边的输入和 raw 输出，并比较 metadata、raw 数值和 source Top-K。本段只是主机编写的 recipe/fixture，本迁移**未执行**，不构成板端结果：

```bash
# cwd：仓库根目录；前置：已准备 HBM、存在 source tree 且当前是可识别的 S100 板卡
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="/tmp/3dresnet-baseline/${RUN_ID}"
mkdir -p "${OUT_DIR}"
OUT_DIR="${OUT_DIR}" python3 - <<'PY'
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
import numpy as np

repo = Path.cwd()
out = Path(os.environ["OUT_DIR"])
target = "s100"
asset_id = "s:3dresnet:s100/r3d_18.hbm"
model_path = repo / "samples/vision/3dresnet/model/s100/r3d_18.hbm"
clip_path = repo / "samples/vision/3dresnet/test_data/video0.npy"

# 这是唯一的板卡门禁，必须先于 source 导入和 SDK 使用。
platforms = importlib.import_module("samples._shared.platforms")
platforms.require_execution_target(target)
if not model_path.is_file():
    raise FileNotFoundError(model_path)

# 身份确认后才导入未修改的 source wrapper；它要求 platforms/s 在 sys.path
# 中并会导入 hbm_runtime。
sys.path.insert(0, str(repo / "platforms/s"))
legacy_spec = importlib.util.spec_from_file_location(
    "source_r3d18_resnet3d",
    repo / "platforms/s/samples/vision/3dresnet/runtime/python/resnet3d.py",
)
if legacy_spec is None or legacy_spec.loader is None:
    raise RuntimeError("cannot load source R3D-18 wrapper")
legacy_mod = importlib.util.module_from_spec(legacy_spec)
sys.modules[legacy_spec.name] = legacy_mod
legacy_spec.loader.exec_module(legacy_mod)

# 统一实现使用完整 repository package 名称导入；不插入 runtime 目录，也不
# 使用裸模块名。
binding = importlib.import_module("samples.vision.3dresnet.runtime.python.model_binding")
runner_mod = importlib.import_module("samples.vision.3dresnet.runtime.python.model_runner")
task_mod = importlib.import_module("samples.vision.3dresnet.runtime.python.classification")
labels_mod = importlib.import_module("samples.vision.3dresnet.runtime.python.labels")
selection = binding.resolve_selection(
    target, asset_id=asset_id, model_path=model_path)

# 两条路径使用同一模型和同一个 source-preprocessed fixture。
legacy = legacy_mod.ResNet3D(legacy_mod.ResNet3DConfig(str(model_path)))
legacy.set_scheduling_params(priority=0, bpu_cores=[0])
runner = runner_mod.RuntimeModelRunner(selection)
bound = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
clip = np.load(clip_path, allow_pickle=False)
task = task_mod.VideoClassificationTask(
    runner, bound, top_k=5,
    labels=labels_mod.load_labels(repo / "samples/vision/3dresnet/test_data/kinetics_classnames.json"))
source_input = legacy.pre_process(clip)[legacy.model_name][legacy.input_name]
unified_prepared = task.pre_process(clip)
unified_input = unified_prepared.tensors[bound.input_name]
if source_input.shape != unified_input.shape or source_input.dtype != unified_input.dtype:
    raise AssertionError("legacy/unified input shape or dtype differs")
if not np.array_equal(source_input, unified_input):
    raise AssertionError("legacy/unified prepared input differs")
np.save(out / "source_input.npy", source_input, allow_pickle=False)
np.save(out / "unified_input.npy", unified_input, allow_pickle=False)

source_raw_nested = legacy.forward({legacy.model_name: {legacy.input_name: source_input}})
source_raw = np.asarray(source_raw_nested[legacy.model_name][legacy.output_name])
unified_raw_map = runner(unified_prepared.tensors)
unified_raw = np.asarray(unified_raw_map[bound.output_name])
if source_raw.shape != unified_raw.shape or source_raw.dtype != unified_raw.dtype:
    raise AssertionError("legacy/unified raw shape or dtype differs")
if not np.isfinite(source_raw).all() or not np.isfinite(unified_raw).all():
    raise AssertionError("legacy/unified raw output is not finite")
np.save(out / "source_raw.npy", source_raw, allow_pickle=False)
np.save(out / "unified_raw.npy", unified_raw, allow_pickle=False)
raw_close = bool(np.allclose(source_raw, unified_raw, rtol=0.0, atol=1e-5))

source_topk = legacy.post_process(source_raw_nested, top_k=5)
unified_result = task.post_process(unified_raw_map)
source_ids = [int(item[0]) for item in source_topk]
source_scores = [float(item[1]) for item in source_topk]
unified_ids = [int(item) for item in unified_result.class_ids]
unified_scores = [float(item) for item in unified_result.scores]
ids_equal = source_ids == unified_ids
scores_equal = bool(np.allclose(source_scores, unified_scores, rtol=0.0, atol=1e-6))
passed = raw_close and ids_equal and scores_equal

metadata = {
    "target": target, "asset_id": asset_id, "model_path": str(model_path),
    "source_input": {"shape": list(source_input.shape), "dtype": str(source_input.dtype)},
    "unified_input": {"shape": list(unified_input.shape), "dtype": str(unified_input.dtype)},
    "source_raw": {"name": legacy.output_name, "shape": list(source_raw.shape), "dtype": str(source_raw.dtype)},
    "unified_raw": {"name": bound.output_name, "shape": list(unified_raw.shape), "dtype": str(unified_raw.dtype)},
    "raw_allclose": raw_close, "raw_atol": 1e-5, "raw_rtol": 0.0,
    "source_topk_ids": source_ids, "unified_topk_ids": unified_ids,
    "source_topk_scores": source_scores, "unified_topk_scores": unified_scores,
    "score_atol": 1e-6, "score_rtol": 0.0,
    "ids_equal": ids_equal, "scores_equal": scores_equal,
    "passed": passed, "id_mismatch_requires_review": not ids_equal,
}
(out / "comparison.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps({"out_dir": str(out), **metadata}, indent=2))
if not passed:
    raise AssertionError("Migration comparison failed; inspect full raw arrays and comparison.json. No automatic tie exemption.")
PY
# 预期：唯一 UTC 目录中有 source_input.npy、unified_input.npy、source_raw.npy、
# unified_raw.npy、comparison.json，并在 stdout 显示 shape/dtype 和容差。
# 这是可复制的主机 recipe/fixture，不是当前板端结果
```

该 recipe 要求输入和输出 shape/dtype 完全一致，raw 使用 `allclose(atol=1e-5, rtol=0)`，source Top-K score 使用 `atol=1e-6, rtol=0`。Top-K ID 必须一致。任何 ID 不一致都失败并保留 comparison.json 供独立复核，包括精确平局；不按近似平局或边界容差自动放行。这些都是数值比较，不能称为板端 raw output 的 bitwise 相等。source 的 `archery` 展示仍只是参考，本 recipe 不测量它。

<a id="metrics"></a>
## 指标

- **Top-1：** 数值稳定 softmax 后按概率降序排列的第一个 class ID。
- **Top-K：** 前 `K` 个 class ID 和 float32 概率，`K` 由 `--top-k` 控制，默认 5。
- **功能标签检查：** source 参考将 `video0.npy` 的 Top-1 记为 `archery`；不是当前板端结果。
- **性能：** 下表完整保留 source thread-performance 记录。“Total Latency”和“Average Latency”单位是毫秒，FPS 是吞吐率。数字不是本轮测量，不应当作为当前 S100 证据。

| 线程数 | 帧数 | 总耗时 (ms) | 平均耗时 (ms) | FPS |
| --- | --- | --- | --- | --- |
| 1 | 100 | 18267.76 | 182.68 | 5.47 |
| 2 | 100 | 18291.76 | 182.93 | 10.82 |
| 4 | 100 | 18501.06 | 185.03 | 21.07 |
| 8 | 100 | 24743.56 | 249.19 | 30.74 |

source 还记录 BPU 占用约 5.2%、ION 内存约 91.9 MB、读带宽 533、写带宽 304。单位和测试环境记录不完整，这些只作为 source 备注保留。

<a id="outputs"></a>
## 输出

统一功能命令将 JSON 报告写到 stdout，不创建结果文件。每条 prediction 包含 `class_id`、`score` 和 `label`；CLI 不保存 raw model output。主机测试输出是 unittest 日志。以下截图保留 source 的射箭帧和 Top-5 展示：

![Archery frame](../test_data/readme_img/image-4.png)
![Top-5 result](../test_data/readme_img/image-5.png)

<a id="reference-results"></a>
## 参考结果

| 参考项 | 状态 | 来源 |
| --- | --- | --- |
| `video0.npy` Top-1 `archery` | source 功能参考；板端 not-run | source evaluator README 和截图 |
| 四行 thread-performance 表 | source 历史记录；未复现 | source evaluator README |
| BPU/ION/带宽备注 | source 历史记录；未复现 | source evaluator README 和截图 |

不作 raw output bitwise 对照、数据集精度或当前 FPS 声明。

<a id="boundaries"></a>
## 边界

- 本 sample 没有完整数据集 evaluator 实现。
- 本迁移没有当前板端执行记录、raw output 归档或唯一 UTC 结果目录。
- 没有完整 source `hrt_model_exec` 命令，因此仅凭仓库内容无法复现旧性能记录。
- 主机测试覆盖预处理、finite/shape/dtype 防护、动态 tensor 名、source softmax/Top-K 对照、labels、CLI gate 和 mock download 委托；不覆盖 HBM 执行。

以下 source 附加性能指标截图继续保留：

![Additional metrics](../test_data/readme_img/image-6.png)
