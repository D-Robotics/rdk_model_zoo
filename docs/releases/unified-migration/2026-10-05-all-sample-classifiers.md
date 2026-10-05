# 全部本仓 Sample 可读 Runtime — classifiers 批次报告

> 批次时点记录：最终代码已在本地提交，本文保留执行时的计数和状态。
> 全 51 Sample 的最终源码/主机验收结果见 [Codex 独立验收](2026-10-05-all-sample-codex-review.md)。

日期：2026-10-05
基线：`c1510ede652d30d83eeb691ff0e90b3f735a70a4`（分支 `codex/readable-model-examples-20261001`）
执行器：本地 Claude Code 2.1.276 + GLM `glm-5.3[1m]`；主机 Python 为 `rdk_model_zoo/.venv/bin/python`（3.14.7，NumPy 2.5.3 / OpenCV 4.14.0）。
设计依据：`docs/superpowers/specs/2026-10-05-all-sample-readable-runtime-design.md`；范例为 ResNet `classify.py`（`main.py`/`classify.py`/`cli.py` 三件套）。
本报告只覆盖本批次二十一个分类样例；代码与测试未提交，由 Codex 按路径评估提交。外部证据日志目录：`local-execution/20261005-all-sample-readable-runtime/classifiers/`（baseline-/failing-first-/final4- 日志、checker- 日志、cli-snapshot-before.json / cli-verify-after.json、generate_classifiers.py、progress.json）。开始时工作树内已有其他批次并行修改（vision_tasks、detection_tracking 等），本批次未触碰。

> **批次快照声明**：本报告是本批次的执行快照（含同日
> [classifier-alias-fix](2026-10-05-all-sample-classifier-alias-fix.md) 后的状态）；
> 测试计数与结论均为批次时点记录，**最终状态以 Codex 终审报告为准**，本文不构成
> 行为验收结论。

## 统一改动模式

二十一个样例原先的 `runtime/python/classification.py` 只是共享
`ClassificationTask` 的转出口，`main.py` 在 `_run` 内自行组 runner/binding/task。
本批次按 ResNet 范例为每个样例新增三个本地文件并重写入口（生成工具
`generate_classifiers.py` 放在外部执行目录；每个样例的 CLI 事实——description、
`--variant` 块、`--label-file` 块、`--test-img` help、list 标题、标签常量——一律
从 HEAD 版 `main.py` 逐字节提取，不重打字）：

* **`classify.py`（新增）**：本地具名模型类（如 `ConvNeXtClassifier`），完整主线
  可见：构造时经 sample runner 解析并加载绑定（板卡身份、SDK、张量元数据校验
  都在 `runner.load()`），`preprocess` 读图（路径或 BGR 数组）并按绑定契约打包
  NV12（X5 packed / S 系 split 由 binding 决定，不硬编码平台），`infer` 恰好
  一次模型调用，`postprocess` 执行声明的 output_transform 与
  output_score_policy 后做稳定 Top-K，`predict` 类内串联三步。旧拼写
  `pre_process`/`forward`/`post_process` 为薄兼容委托：`infer` 同时接受
  `PreparedInput` 与裸 tensors 映射，`forward` 只委托 `infer`（本批首版
  `forward` 曾自行解包直调 runner 形成第二条路径，已按
  [classifier-alias-fix 记录](2026-10-05-all-sample-classifier-alias-fix.md)
  于同日修正并补 21 份委托测试）；`__call__`、`set_scheduling_params` 转发
  保留。可复用部分不复制：NV12 打包在 `samples/_shared/tensor_io.py`，Top-K
  数学与输出提取在 `samples/_shared/classification.py`，量化变换在
  `samples/_shared/quantization.py`。`predict` 不打印、不绘图、不写文件。
  本类是新 API：labels 做严格类数校验（序列必须恰好覆盖 class_count，映射键
  必须落在范围内），错误信息带具体计数；不新增 custom_selection（21 个样例的
  model_binding 均无该入口，保持"自训练导出不凭空扩展"）。
* **`cli.py`（新增）**：参数声明、model-free 的 `--list-models`/`--dry-run`、
  结果打印与标注图保存。CLI 面（旗标、默认值、help 文案、互斥组、vit 的
  `--model-variant` 别名、mobilenetv1/2/3 无 `--variant`）与旧 main 逐字节
  等价：生成后以 `verify-cli` 对 21 个样例逐一比对改前/改后的 `--help` stdout、
  `--list-models --target auto` stdout、`--dry-run --target <t>` stdout 与
  `build_parser()` 全部默认值 JSON，21/21 identical（`cli-snapshot-before.json`
  vs `cli-verify-after.json`）。
* **`main.py`（重写为薄入口，正文 92 行；九个带许可证头的样例另含原 13 行
  Apache 头，共 105 行）**：解析参数 → 处理 list/dry-run → 显式
  `resolve_selection` + 文件存在性 + `require_execution_target` 板卡校验 →
  懒导入并显式构造本地分类器（top_k/labels/resize_type）→
  `set_scheduling_params` → `model.predict(args.test_img)` → 展示与可选
  `--img-save-path`。`build_parser` 自 main 再导出（contract checker 从 main
  导入它）；错误面（BindingError/FileNotFoundError/OSError/RuntimeError/
  ValueError → stderr + exit 2）与返回码不变。
* **`classification.py`（未改动）**：共享 `ClassificationTask` 的旧导入路径与
  构造签名原样保留——它是**兼容出口**，公共面仍是旧名
  （`pre_process`/`forward`/`post_process` + `predict`），不带 canonical 方法，
  也不假装有；canonical 主线只在新 `classify.py` 本地类。既有调用方（hgnetv2
  evaluator、各既有测试、README 兼容说明）不受影响。
* **README/README_cn（runtime/python 级，锚定式编辑）**：标题后段落改为
  classify.py/<类名> 描述；集成示例改为最小 `Classifier(selection)` +
  `predict(路径)` 用法（原示例中的 selection 表达式按样例逐字节保留，asset-id/
  model-path/variant 事实不变），并说明旧拼写为薄别名、共享
  `ClassificationTask` 仍可从 `classification.py` 导入；Stage I/O 表阶段名改为
  规范名（括注旧名）。参数表未动（Q3 检查器 21/21 通过）。vit 的两份 runtime
  README 仍各只有一个 ```python 块（其测试断言恰一个并实际执行该块）。

新增测试统一为每样例一个 `tests/test_predict_entry.py`（15 项，ResNet
`test_predict_entry.py` 的移植）：`predict` 与显式三阶段逐字段一致（双协议样例
在 x5 与 s100 各跑一遍）、旧拼写路由到同一实现、runner 每图恰好调用一次、输入
数组不被原地修改、连续不同尺寸图像结果/几何不串扰、输出形状不匹配经
`predict` 传播 BindingError 且恰在执行时失败、路径与数组双输入、缺失路径报错
含路径、非法输入类型 TypeError、序列/映射标签解析与计数错误、top_k 越界。
全部使用注入 runtime 的 `RuntimeModelRunner`（无板端 SDK、不伪造板端通过）；
vit 用 s100 fixture（10 类、split NV12），双协议样例覆盖 packed 与 split。
每个样例先运行记录失败（`failing-first-<sample>.log`：`ModuleNotFoundError:
...classify`），实现后转绿。

## 逐样例状态

除下列分项外，公共事实：入口均为 `runtime/python/main.py`（薄入口 + 本地
`cli.py`）；兼容接口均为 `pre_process`/`forward`/`post_process` 别名 + `__call__`
+ `set_scheduling_params` + 旧 `classification.ClassificationTask` 导入；每样例
新增 15 项测试（先失败后通过）；contract checker 0 violations；CLI 面与基线
逐字节一致；板端推理 not-run。

### samples/vision/convnext

* 模型类：`classify.py ConvNeXtClassifier`；实际后端：`RuntimeModelRunner`（X5
  atto 制品）或注入 runner；softmax 策略、letterbox 线性插值默认不变。
* 测试：基线 28 OK → 终态 43 OK。边界：S 系无 ConvNeXt 资产为显式错误。

### samples/vision/edgenext

* 模型类：`classify.py EdgeNeXtClassifier`；实际后端：X5（base/small/x_small/
  xx_small 制品）。测试：26 → 41。边界：S 系无资产。

### samples/vision/efficientformer

* 模型类：`classify.py EfficientFormerClassifier`；实际后端：X5（l1/l3）。
  测试：25 → 40。边界：S 系无资产。

### samples/vision/efficientformerv2

* 模型类：`classify.py EfficientFormerV2Classifier`；实际后端：X5（s0/s1/s2）。
  测试：26 → 41。边界：S 系无资产。

### samples/vision/efficientnet

* 模型类：`classify.py EfficientNetClassifier`；实际后端：X5（B2/B3/B4 packed
  NV12）与 S 系（lite0..lite4 split NV12，224/240/260/300/380 几何）。
* 测试：28 → 43（双协议各验证 `predict` 与三阶段一致）。边界：跨平台制品不
  复用，绑定校验不变。

### samples/vision/efficientvit

* 模型类：`classify.py EfficientViTClassifier`；实际后端：X5（m5）。测试：
  27 → 42。边界：S 系无资产。

### samples/vision/fasternet

* 模型类：`classify.py FasterNetClassifier`；实际后端：X5（s/t0/t1/t2）。
  测试：28 → 43。边界：S 系无资产。

### samples/vision/fastvit

* 模型类：`classify.py FastViTClassifier`；实际后端：X5（s12/sa12/t12/t8）。
  测试：28 → 43。边界：S 系无资产。

### samples/vision/googlenet

* 模型类：`classify.py GoogLeNetClassifier`；实际后端：X5（googlenet 单变体）。
* 测试：10 → 25。边界：与 legacy 源的字节级比对测试原样保留（走
  `classification.ClassificationTask`，不受本批改动影响）。

### samples/vision/hgnetv2

* 模型类：`classify.py HGNetV2Classifier`；实际后端：X5（b0..b4）。
* 测试：16 → 31。边界：`evaluator/eval.py` 继续使用共享 `ClassificationTask`
  与注入 runner（其测试不变，绿）；direct-resize（resize_type 0）默认保留。

### samples/vision/mobilenetv1

* 模型类：`classify.py MobileNetV1Classifier`；实际后端：X5 与 S 系（s600
  等制品）。CLI 无 `--variant` 旗标的原状保留（单变体样例）。
* 测试：17 → 32（双协议）。边界：无 variant 选择面即不新增。

### samples/vision/mobilenetv2

* 模型类：`classify.py MobileNetV2Classifier`；实际后端：X5 与 S 系。无
  `--variant` 原状。测试：24 → 39（含既有 cpp launcher identity 测试）。
  边界：C++ runtime 与 launcher 未动。

### samples/vision/mobilenetv3

* 模型类：`classify.py MobileNetV3Classifier`；实际后端：X5 与 S 系。无
  `--variant` 原状。测试：17 → 32。

### samples/vision/mobilenetv4

* 模型类：`classify.py MobileNetV4Classifier`；实际后端：X5 与 S 系（small/
  medium）。测试：20 → 35。边界：`test_conversion_readme_shapes` 等既有测试
  原样通过。

### samples/vision/mobileone

* 模型类：`classify.py MobileOneClassifier`；实际后端：X5（s0..s4）。
  测试：10 → 25。边界：S 系无资产。

### samples/vision/repghost

* 模型类：`classify.py RepGhostClassifier`；实际后端：X5（100/111/130/150/200）。
  测试：8 → 23。边界：S 系无资产。

### samples/vision/repvgg

* 模型类：`classify.py RepVGGClassifier`；实际后端：X5（a0/a1/a2/b0/b1g2/
  b1g4）。测试：10 → 25。边界：S 系无资产。

### samples/vision/repvit

* 模型类：`classify.py RepViTClassifier`；实际后端：X5（m0_9/m1_0/m1_1）。
  测试：10 → 25。边界：S 系无资产。

### samples/vision/resnext

* 模型类：`classify.py ResNeXtClassifier`；实际后端：X5（50_32x4d）。
  测试：10 → 25。边界：S 系无资产。

### samples/vision/vargconvnet

* 模型类：`classify.py VargConvNetClassifier`；实际后端：X5（vargconvnet）。
  测试：10 → 25。边界：S 系无资产。

### samples/vision/vit

* 模型类：`classify.py ViTClassifier`；实际后端：S100（int8/int16 HBM；split
  NV12、nearest direct-resize 默认、10 类 CIFAR-10、`--model-variant` 别名）。
* 特有适配（测试缝隙换到新架构、语义不变）：① `test_vit.py` 的
  `test_execution_target_mismatch_stops_before_runtime` 原先 patch
  `main._run`；改为断言板卡不匹配时 `main` 返回 2 且 `classify` 模块从未被
  导入（先从 sys.modules 弹出以保证与测试顺序无关）——薄入口在板卡校验处
  停止，构造模型发生在其后。② README 集成示例执行测试的 patch 目标由仅
  `model_runner.RuntimeModelRunner` 扩展为同时 patch `classify.
  RuntimeModelRunner`（示例改为构造 `ViTClassifier`；仍恰一个 ```python 块，
  期望 labels deer / [4,5,2,6,9] 不变）。③ 样例根 README 两语的"新集成使用
  ClassificationTask"改为指向 `ViTClassifier`（共享流程仍可导入）。
* 测试：13 → 28。边界：X5/S100P/S600 无资产为显式错误；量化 raw 输出仍被
  拒绝。

## 汇总

| 样例 | 基线 | 终态 | 新增 | contract | CLI 面 |
| --- | --- | --- | --- | --- | --- |
| convnext | 28 OK | 43 OK | test_predict_entry×15（红绿已录） | 0 violations | 4 探针 identical |
| edgenext | 26 OK | 41 OK | 同上 | 0 violations | identical |
| efficientformer | 25 OK | 40 OK | 同上 | 0 violations | identical |
| efficientformerv2 | 26 OK | 41 OK | 同上 | 0 violations | identical |
| efficientnet | 28 OK | 43 OK | 同上（双协议） | 0 violations | identical |
| efficientvit | 27 OK | 42 OK | 同上 | 0 violations | identical |
| fasternet | 28 OK | 43 OK | 同上 | 0 violations | identical |
| fastvit | 28 OK | 43 OK | 同上 | 0 violations | identical |
| googlenet | 10 OK | 25 OK | 同上 | 0 violations | identical |
| hgnetv2 | 16 OK | 31 OK | 同上 | 0 violations | identical |
| mobilenetv1 | 17 OK | 32 OK | 同上（双协议） | 0 violations | identical |
| mobilenetv2 | 24 OK | 39 OK | 同上（双协议） | 0 violations | identical |
| mobilenetv3 | 17 OK | 32 OK | 同上（双协议） | 0 violations | identical |
| mobilenetv4 | 20 OK | 35 OK | 同上（双协议） | 0 violations | identical |
| mobileone | 10 OK | 25 OK | 同上 | 0 violations | identical |
| repghost | 8 OK | 23 OK | 同上 | 0 violations | identical |
| repvgg | 10 OK | 25 OK | 同上 | 0 violations | identical |
| repvit | 10 OK | 25 OK | 同上 | 0 violations | identical |
| resnext | 10 OK | 25 OK | 同上 | 0 violations | identical |
| vargconvnet | 10 OK | 25 OK | 同上 | 0 violations | identical |
| vit | 13 OK | 28 OK | 同上（s100 fixture） | 0 violations | identical |

验证命令（均自仓库根、每样例独立进程，日志在批次外部目录）：

* 基线与终态：`$VENV -m unittest discover -s samples/vision/<sample>/tests -v`
  → `baseline-<sample>.log`（全 21 exit 0）/ `final4-<sample>.log`（全 21
  exit 0）。
* 红绿循环：`-p test_predict_entry.py` 实现前运行 → `failing-first-<sample>.log`
  （21/21 exit 1，`ModuleNotFoundError ...classify`）。
* 契约检查：`$VENV tools/sample_contract/check.py --sample samples/vision/<sample>`
  → `checker-<sample>.log`（21/21 exit 0，0 violations，0 exemptions；另跑过一次
  `--scope migration` 全仓 52 samples 0 violations，当时含其他批次在途修改）。
* CLI 面：`--help` / `--list-models --target auto` / `--dry-run --target <t>` /
  `build_parser()` 默认值 JSON，改前后各 21 份快照逐一相同。

每个样例以独立进程 discover，未与其他样例混跑；未运行其他批次或全仓回归
（其他执行器正在并行修改）。批次内修改文件共 129 个（每样例：改
README.md/README_cn.md/main.py，增 classify.py/cli.py/test_predict_entry.py；
vit 另有样例根 README×2 与 test_vit.py 缝隙更新），全部位于上述 21 个样例
目录（`git status` 核对，无共享模块/其他批次文件改动；本报告为本批次唯一的
docs 写入）。

## 未执行（not-run）

* 板端 SDK 链接与推理、真实权重下载、真实 ONNX 导出、OE/Mapper/HMCT 编译、
  量化精度与数据集评测；主机注入 runner 的通过不构成以上证据。
* mobilenetv2 的 C++ runtime/launcher 与各样例 conversion 配方未动，仅既有
  主机测试回归通过。
* Catalog build/check、全仓 shared 测试与干净 checkout 入口复现属于最终集成
  阶段，不在本批次范围。
