# 全部本仓 Sample 可读 Runtime — multistage 批次报告

日期：2026-10-05
执行分支：`codex/readable-model-examples-20261001`；批次开始时 HEAD `c1510ede652d30d83eeb691ff0e90b3f735a70a4`，本批次六个样例在 `c1510ede..HEAD`（`9bc2fdc0ef39fca550e8f2ed9e752ba348adf021`）间无提交，因此 HEAD 即本批次六个样例的改前基线。
执行器：本地 Claude Code 2.1.276 + GLM `glm-5.3[1m]`；主机 Python 为 `rdk_model_zoo/.venv/bin/python`（3.14.7）。
设计依据：`docs/superpowers/specs/2026-10-05-all-sample-readable-runtime-design.md`、`docs/superpowers/plans/2026-10-05-all-sample-readable-runtime.md`（Batch 4）、`docs/sample-standards/inference-contract.md` §3 多阶段条款；范例为 ResNet `main.py`/`classify.py`/`cli.py` 三件套与 `samples/vision/paddle_ocr/runtime/python/pipeline.py`。
本报告只覆盖 multistage 批次六个样例；代码与测试未提交，由 Codex 按路径评估提交。外部证据目录：`local-execution/20261005-all-sample-readable-runtime/multistage/`（baseline_/failingfirst_/final_ 日志、cli-equivalence.json、cli_equiv_check.py、progress.json）。工作树内其他批次（classifiers、vision_tasks、detection_tracking、speech_policy_native）并行修改，本批次未触碰其文件，也未修改 `samples/_shared/`、根指南、plan、Catalog。

## 统一改动模式

六个样例的 `main.py` 全部改为薄入口：解析参数 → 处理 model-free 模式（list/dry-run/prepare 等，经由新本地 `cli.py`）→ 解析 selection/pair → 文件存在性与 `require_execution_target` 板卡校验 → **入口内显式构造** runner/binding/模型对象 → `predict()` → 展示/保存（经由 `cli.py`）。`build_parser` 自 main 再导出（contract checker 与既有测试从 main 导入）。旧入口的下划线助手（`_run`/`_prepare`/`_resolve_pair_from_args`）删除或迁入 `cli.py` 为公开函数，受影响的测试改为经由新入口面。模型文件中规范拼写 `preprocess`/`infer`/`postprocess` 成为主线，旧 `pre_process`/`forward`/`post_process` 为薄兼容委托（单实现，无第二套流程）。

CLI 事实等价性以 `cli_equiv_check.py` 机器验证：对每个样例从 `git show HEAD:<main.py>` 与工作树分别加载入口，比对 `--help` 全文、`parse_args([])` 全部默认值、`--list-models` 与 `--dry-run` 的 stdout/stderr/退出码，6 样例 × 4 项 = 24/24 逐字节一致（paraformer 的 argparse description 一度随 `__doc__` 变化，已钉回原文后全等；证据 `cli-equivalence.json`）。

## 逐样例

### samples/vision/clip（X5 图文匹配，15→19 测试）

* 入口：`main.py`（薄）+ 新 `cli.py`（build_parser、run_list_models、run_dry_run、parse_prompts、read_bgr_image、print_match_result、save_annotated_image）。`main.build_parser` 再导出；`model_binding.list_available_assets` 导入路径保留。
* 模型类：`matching.py::CLIPTask` —— `preprocess`（BGR→RGB bicubic/中心 crop/F32 + 真实 BPE tokens，per-call context）、`infer`（恰好一次成对 runner 调用）、`postprocess`（cosine + 1e-12 稳定项 + 降序 argsort）、`predict` 串联；旧 `pre_process`/`forward`/`post_process` 为薄别名。tokens、文本模型 runner（onnxruntime CPU）、normalize 语义原样保留。
* 实际后端：`model_runner.py::RuntimeModelRunner` —— BPU `hbm_runtime` 图像编码器 + CPU ONNX 文本编码器（X5 已发布对），懒导入；`PromptTokenizer` 加载捆绑 BPE 词表。
* 兼容面：旧三步拼写、`MatchResult(scores, order)`、JSON 输出字段（含 `image_saved`）与默认 `--img-save-path` 行为、错误面（ImportError/OSError/ValueError/RuntimeError → exit 2）不变。
* 测试：新增 `tests/test_readable_stages.py`（4 项，failing-first 见 `failingfirst_samples_vision_clip.log`：规范名缺失、无 cli 模块，4/4 失败）；`test_cli.py` 的 wrong-board 测试改为 patch `matching.CLIPTask`。final：19/19 OK（`final_samples_vision_clip.log`）。

### samples/vision/siglip（S100/S100P 视觉特征，19→23 测试）

* 入口：`main.py`（薄）+ 新 `cli.py`（build_parser、run_list_models、run_dry_run、read_bgr_image、summarize_result、print_summary、save_feature_tensor）。
* 模型类：`embedding.py::SigLIPTask` —— `preprocess`（RGB F32 NCHW [-1,1]，per-call padding context）、`infer`（恰好一次 runner 调用）、`postprocess`（原生存档校验 + owned copy，无 softmax/dequant）、`predict` 串联；旧拼写薄别名。
* 实际后端：`model_runner.py::RuntimeModelRunner` —— S 系 packed HBM（`pooler_output`/`last_hidden_state` 两个子模型均校验），懒 `hbm_runtime`，双模型调度参数。
* 兼容面：原生 dtype/shape 输出策略、`--output-file` 精确路径语义、错误面不变。
* 测试：新增 `tests/test_readable_stages.py`（4 项，failing-first 4/4）；`test_siglip.py` 的 identity/size 错误前置测试改为 patch `embedding.SigLIPTask`。final：23/23 OK（`final_samples_vision_siglip.log`）。

### samples/vision/efficient_sam（X5/S 双模型分割，19→23 测试）

* 入口：`main.py`（薄）+ 新 `cli.py`（build_parser、run_list_models、run_dry_run/dry_run_report、selection_report、read_bgr_image、save_outputs、print_report）。
* 模型类：`pipeline.py` 重写 —— 本地 `EfficientSAMEncoder`/`EfficientSAMDecoder` 为共享 `EncoderStage`/`DecoderStage` 的规范拼写视图（`preprocess`/`infer`/`postprocess` 薄别名到继承实现，单实现），`EfficientSAMPipeline` 保持 `SAMPipeline` 子类（共享测试 `test_sam_stages.py` 的子类/绑定检查仍绿），`predict` 在本地写明 encoder 三步 → decoder 三步的真实编排（`encode_image`/`decode_masks` 单阶段助手），失败经共享 `StageError` 归属 stage 且不执行后续 stage。SAM 数学（归一化、box 校验、mask resize/阈值）全部复用 `samples/_shared/sam_tensor_io.py`/`sam_binding.py`，未复写。
* 实际后端：`model_runner.py::RuntimeModelRunner`（共享 `sam_runner`，encoder/decoder 双模型懒加载；X5 `.bin` / S `.hbm`）。
* 兼容面：`EfficientSAMPipeline(runner, binding)` 构造与跨样例绑定拒绝、`predict(image)` 固定 prompt（运行时 box → ValueError，`predict(image, box=...)` → TypeError，共享测试钉住）、结果 dict（mask/iou/mask_index/low_res_masks）不变。
* 测试：新增 `tests/test_readable_stages.py`（4 项，failing-first 3 error + 1 pass：规范 stage API 与 encode/decode 助手缺失）；final 23/23 OK；共享回归 `samples/_shared/tests -p test_sam_stages.py` 11/11 OK（`final_samples_vision_efficient_sam.log`、`final_shared_sam_stages.log`）。

### samples/vision/mobile_sam（X5/S 双模型分割，17→21 测试）

* 与 efficient_sam 同模式：本地 `MobileSAMEncoder`/`MobileSAMDecoder` 规范视图 + `MobileSAMPipeline.predict` 显式编排；**点/框 prompt API 原样保留**：`predict(self, image, *, box=DEFAULT_BOX)`（默认 `(185,120,380,445)`），自定义 box 逐值进入 decoder 张量、缺省时用共享默认框（新测试钉住）；`main.parse_box` 再导出（既有测试使用）。X5 `[1,4,1,1]`/S `[1,4]` box 形状仍由实际 metadata 决定。
* 实际后端：共享 `sam_runner`（encoder `normalized_images` ImageNet mean/std + decoder box prompt，懒加载）。
* 测试：新增 4 项（failing-first 3 error + 1 pass）；final 21/21 OK；共享 SAM 回归 11/11 OK（`final_samples_vision_mobile_sam.log`）。

### samples/vision/paddle_ocr（X5/S100 两阶段 OCR，44→46 测试）

* `pipeline.py` **未改动**：本仓多阶段参照实现，`OCRPipeline` 每个实际模型的公开三步（`prepare_detection`/`forward_detection`/`postprocess_detection`、`prepare_recognition`/`forward_recognition`/`decode_recognition`，`run_detection`/`run_recognition` 单阶段串联）与 `predict` 的 检测→裁剪→识别 显式编排、零检测短路、逐 crop 错误归属已符合契约（inference-contract §6 参照），不重排阶段。
* 入口：`main.py` 重写为薄入口 —— model-free 三模式与 `_pair_record`/`_contract_record`、取图、渲染、JSON 写出移入新 `cli.py`（`_prepare`→`run_prepare`、`_resolve_pair_from_args`→`resolve_pair_from_args` 公开化）；执行路径在 main 内显式 `create_stage_runners` + `OCRPipeline(...).predict(image)`。`main` 再导出 `build_parser`/`BindingError`/`list_available_pairs`/`resolve_pair`。
* 实际后端：`model_runner.py::create_stage_runners` → `RuntimeStageRunner` 懒加载目标 `hbm_runtime`（X5 packed NV12/97 类，S100 split NV12/18710 类）。
* 兼容面：CLI 旗标/默认值/返回码、`--prepare` 唯一联网操作语义、pyclipper 懒依赖不变；`tests/test_main.py` 中两处下划线入口调用改经 `cli` 模块（行为断言不变）。
* 测试：新增 `tests/test_readable_entry.py`（cli 再导出 + 注入 runner 的执行路径：构造参数透传、恰好一次检测调用、零框短路不触发识别、JSON 载荷字段）。注意：该项在旧入口上亦可能通过，作为新结构回归锁；failing-first 未单独取证（实现先于该测试文件完成，报告如实记录）。final 46/46 OK（`final_samples_vision_paddle_ocr.log`）。

### samples/speech/paraformer（S100 三模型 ASR，82→87 测试，skipped=12 不变）

* `stages.py`：每个阶段类新增规范拼写 `preprocess`/`infer`/`postprocess` 为实现本体，旧 `pre_process`/`forward`/`post_process` 为薄别名；`attributed` 装饰器增加 `label` 参数，**StageError 的 operation 措辞保持既有字符串**（`pre_process`/`forward`/`post_process`），既有报告与测试看到的消息逐字节不变（CIF 仍为 `cif`/`integrate`）。
* `pipeline.py`：`predict` 改为调用规范拼写，编排不变 —— encoder → predictor → CPU `cif_numpy`（真实 T 传入）→ 零 token 跳过 decoder（timing 为 `None`）→ decoder；timings/`StageError`/链式异常/features 输入 API/feature_length ∈ [1,400] 前置校验全部保留；CIF 与 decoder 仍在 pipeline 显式编排，未藏入任何 stage。
* `application.py` 拆为 `prepare`（输出目录必须新建、输入证据 digest、items、vocabulary digest 校验、报告骨架）+ `run`（逐语句 `frontend.pre_process` + `bundle.pipeline.predict`、digest 复验、result/failed 写出）+ `record_failure`；`execute` 保留为同一编排的兼容组合。证据顺序不变：模型文件 digest 先于加载；构造期失败仍写 `failed.json`。
* 入口：`main.py` 薄入口 + 新 `cli.py`（build_parser、normalize_args、resolve_selections_for、print_resolution）；main 内显式 `require_execution_target`（preprocess-only 不触 SDK）→ `application.prepare` → 显式构造 `ParaformerFrontend` 与 `load_runtime` bundle（懒导入，host-test patch 面保持）→ `application.run`。argparse description 钉回原文后 `--help` 与 HEAD 逐字节一致。
* 实际后端：`runtime.py::load_runtime` → 三个 `NamedArrayRunner`（s100 encoder/predictor/decoder HBM，懒 `hbm_runtime`，共享板卡身份与资产文件校验）。
* 测试：新增 `tests/test_readable_stages.py`（5 项，failing-first 4 失败：规范拼写缺失、无 cli 模块、main 不显式构造 bundle；error-wording 锁定项在旧码即通过）；既有 `test_cli.py`/`test_stage_api.py`/`test_pipeline.py` 无需改动全部通过。final 87/87（12 skip 为既有板端/环境依赖跳过，数目与基线一致；`final_samples_speech_paraformer.log`）。

## 验证汇总（主机）

命令与退出码（Python：`rdk_model_zoo/.venv/bin/python`，每样例独立 discover 进程）：

| 命令 | 退出码 | 结果 | 日志 |
| --- | --- | --- | --- |
| `-m unittest discover -s samples/vision/clip/tests` | 0 | 19 tests OK | final_samples_vision_clip.log |
| `-m unittest discover -s samples/vision/siglip/tests` | 0 | 23 tests OK | final_samples_vision_siglip.log |
| `-m unittest discover -s samples/vision/efficient_sam/tests` | 0 | 23 tests OK | final_samples_vision_efficient_sam.log |
| `-m unittest discover -s samples/vision/mobile_sam/tests` | 0 | 21 tests OK | final_samples_vision_mobile_sam.log |
| `-m unittest discover -s samples/vision/paddle_ocr/tests` | 0 | 46 tests OK | final_samples_vision_paddle_ocr.log |
| `-m unittest discover -s samples/speech/paraformer/tests` | 0 | 87 tests OK (skipped=12) | final_samples_speech_paraformer.log |
| `-m unittest discover -s samples/_shared/tests -p test_sam_stages.py` | 0 | 11 tests OK（SAM 共享回归） | final_shared_sam_stages.log |
| `tools/sample_contract/check.py --sample <每个样例>` | 0×6 | 0 violations，各 1 处 R-STAGE-PURITY policy skip（CLI layer，既有策略） | 未见独立日志，逐样例运行于终端 |
| `cli_equiv_check.py`（HEAD vs 工作树，6 样例 × help/defaults/list/dry-run） | 0 | 24/24 逐字节一致 | cli-equivalence.json |
| SDK-free 冒烟：六个入口 `--help`；`--dry-run --target <t>` | 全部 OK | 主机可执行，无 SDK/模型加载 | 终端输出 |

failing-first 证据：clip 4/4 失败、siglip 4/4、efficient_sam 3 error、mobile_sam 3 error、paraformer 4 失败（见 failingfirst_*.log）；paddle_ocr 未单独取证（见上文说明）。

## 未执行边界（not-run）

* 板端推理、真实权重下载、真实 ONNX 导出、OE/Mapper/HMCT 编译：全部 **not-run**（主机 fixture 通过不构成板端证据；模型二进制不在树内）。
* Paraformer 前端环境（Torch/FunASR）主机回归：未新建环境运行（12 项既有 skip 维持原语义）。
* 全仓回归、Catalog build/check、干净 checkout 入口复现：属最终集成批次，本批次按约束不执行。

## 兼容性声明

六个样例的旧导入路径、构造参数、结果结构、CLI 旗标/默认值/返回码（成功 0 / 用户错误 2）保持；规范拼写为新增主线，旧拼写为同一实现的薄别名（paraformer 的 StageError operation 措辞因此保持旧字符串，属有意兼容）。SAM 样例继续满足 `samples/_shared/tests/test_sam_stages.py` 的子类与 prompt 约束。未新增任何"统一伪协议"、未扩展自训练导出支持、未改动共享模块与编译配方事实。
