[English](README.md) | 简体中文

# 分类模型板端对比工具

`b3_classification_compare.py` 对比固定参考实现与 Sample 实现，覆盖 13 个 X5 分类模型变体：ConvNeXt atto；EdgeNeXt base/small/x_small/xx_small；FasterNet s/t0/t1/t2；FastViT s12/sa12/t12/t8。每次调用在 X5 8GB 或 4GB 上运行一个样例、一个开发板和一个变体。

参考源码闭包从固定的 Git 树读取，并加载与其匹配的工具模块。参考与 Sample 实现使用相同图像字节、缩放类型、Top-K 和调度配置执行预处理、推理及后处理。

参考闭包验证将入口模块及其导入的 `utils.py_utils` 模块（`__init__`、`file_io`、`preprocess`、`visualize`）与固定 Git blob 对比。加载器临时绑定这些确切模块，随后恢复此前导入，并在 `source_closure` 中记录哈希。对象缺失或字节不一致时返回 2。在同一个 Python 解释器内应串行运行对比。

## 最小依赖

- **板端**：匹配的 `hbm_runtime` Python 环境（兼容 Python 3.10；哈希计算复用 `utils/py_utils/assets.py::sha256_file`），以及 numpy、OpenCV、scipy（固定源码导入 `scipy.special.softmax`）。板端需部署完整仓库检出，从仓库根目录运行。
- **主机**（仅测试）：Python 3.10+、numpy、OpenCV、scipy、PyYAML。无需 `hbm_runtime`；行为测试注入模拟 SDK runtime，不表示能在主机上执行板端推理。

## 使用方法

先为各样例明确准备资源（工具不会下载）：

```bash
bash samples/vision/convnext/model/download.sh x5          # atto
bash samples/vision/edgenext/model/download.sh x5 base      # or small/x_small/xx_small
bash samples/vision/fasternet/model/download.sh x5 s        # or t0/t1/t2
bash samples/vision/fastvit/model/download.sh x5 s12        # or sa12/t12/t8
```

然后在所选开发板的仓库根目录运行，每次调用覆盖矩阵中的一个组合，使用新的证据目录：

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=/tmp/b3-convnext-atto-x5-8g-$STAMP
python3 tools/board_validation/b3_classification_compare.py \
  --sample convnext --target x5 --variant atto \
  --output-dir "$OUT" > "$OUT.stdout" 2> "$OUT.stderr"
echo $? > "$OUT.rc"
```

省略 `--variant` 时遵循样例自身的默认选择语义（atto/base/s/s12）；组合使用 `--model-path` 与确切的 `--asset-id`，可指向其他位置已准备的资源。`--top-k`（默认 5）、`--resize-type`（0/1，默认 1）、`--priority` 与 `--bpu-cores`（默认分别为 0 / `[0]`）对两侧一致生效。按变体和开发板重复执行；8GB/4GB 区别通过开发板身份及 `/proc/meminfo` 的 MemTotal 记录。

## 输出

新的 `--output-dir` 内包含 `comparison.json` 和每个捕获数组对应的一个 `.npy` 文件（成功运行时共 13 个文件）：

- 身份信息：UTC 起止时间、argv/cwd、Git head/branch/工作区修改状态；工具、执行的固定源码模块、实际绑定且通过固定版本验证的 `utils.py_utils` 依赖文件（实测与固定版本哈希、各文件的 `matches_pin`）、`source_closure` 验证记录、共享模块与样例 runtime 源码的 SHA-256；开发板身份文件、`/etc/os-release`、MemTotal；SDK 模块文件/版本；模型发布方与实测 SHA-256（`verify_asset_file`；清单中的发布方哈希为 `null`，输出中保持 `null`）；图像与标签的实测 SHA-256。
- 执行信息：通过 `utils/py_utils/runtime_meta.py::metadata_evidence` 获取两侧 SDK 元数据（不使用 `asdict`，板端 `QuantParams` 不允许复制）；全部预处理输入、原始输出、Top-K 和前 8 项证据数组及其 shape/dtype/SHA-256；失败时记录真实异常和返回码。
- 对比信息：非空、有限的 uint8 输入必须具有**完全一致的字节**（旧版 `(1, 3H/2, W, 1)` 视图与统一平坦缓冲区允许形状不同，但字节必须相同）；非空原始输出的形状和类型一致、数值有限，并记录完整差异（`raw_abs_diff_*.npy` 及 max/mean/nonzero/argmax，只报告而不作断言）；Top-K 数量符合请求、ID **唯一**、ID 序列一致、各 ID 分数绝对差 ≤ 1e-5，且**标签字符串一致**（固定版本的 `load_imagenet_labels` 与统一的 `load_labels` 是独立实现，标签加载器的静默变化无法隐藏）；在前 (k+1) 项窗口内出现完全相同分数时，记录两侧逐 ID 的前 8 项证据，**不会**放宽为通过。

退出码：`0` 表示所有检查通过；`1` 表示对比完成但有检查失败（保留数组）；`2` 表示执行错误，包括固定源码闭包与固定版本不匹配（保留错误证据）。stdout 输出一条可机器读取的 JSON 摘要，包含检查、并列标记、两侧 Top-K 和证据路径。

## 适用范围

使用此工具检查上述四个 X5 样例对固定图像的输入、原始输出和 Top-K 一致性。部署包含参考提交的检出，并保留完整输出目录。数据集精度和延迟按各 Sample 的评测流程执行。

## 测试

仅主机运行的模拟 runtime 行为测试，无需板端 SDK：

```bash
python3 -m unittest discover -s tools/board_validation/tests -v
```
