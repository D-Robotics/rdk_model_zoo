# 可读模型范例迁移说明（2026-09-30 方案，2026-10-01 实施）

分支：`codex/readable-model-examples-20261001`（基线 `bfbe6aaf`）。
架构说明见 [docs/architecture/model-examples.md](../architecture/model-examples.md)。
本文逐条记录旧文件、旧类/函数、参数到新接口的映射，保留的转接，以及明确的
不兼容项。既有 X5/S 迁移映射
（[x5-s-migration-map](../releases/unified-migration/x5-s-migration-map.md)）
不受影响。

## 1. 公共层

| 旧入口 | 新入口 | 兼容状态 |
| --- | --- | --- |
| `samples/_shared/model_runner.RuntimeModelRunner`（自行做目标检查、SDK 导入、构造） | 生产加载路径改经 `samples/_shared/runtime.RuntimeSession`（身份检查 → SDK 导入 → 构造） | 公开 API 不变；`runtime`/`runtime_factory` 注入缝不变（host 测试语义不变） |
| `model_runner.RuntimeUnavailableError` | 同名类现在就是 `samples/_shared/runtime.py` 中会话抛出的那个类（此前 model_runner 内重复定义并被导入名遮蔽，已修复为单一类型） | `except model_runner.RuntimeUnavailableError` 现在能捕获会话失败 |
| `samples/_shared/classification._extract_output` | 公开为 `extract_score_tensor`；私有名保留为别名 | 单一实现 |
| `cls_binding.ModelSelection` | 新增带默认值的 `custom` 字段；`bind_model` 对 `custom=True` 的选择只跳过 Manifest 重新枚举，张量/类别数/目标检查全部保留 | 追加字段，既有构造不受影响 |

## 2. ResNet

| 旧入口 | 新入口 | 兼容状态 |
| --- | --- | --- |
| `runtime/python/main.py`（入口内联执行分类流程） | 同名文件瘦身为薄入口：`cli.build_parser()` 委托 + `ResNetClassifier` 构造 + `predict` + 展示辅助 | CLI 参数与模式不变；`build_parser`、`_load_labels` 仍可从 `main` 导入（契约检查器依赖前者） |
| 入口内选项声明 / `_list_models` / `_dry_run` / `_save_visualization` | `runtime/python/cli.py`：`build_parser`、`run_list_models`、`run_dry_run`、`default_labels`、`read_bgr_image`、`print_classification_result`、`save_result_image` | 私有名（`_` 前缀）移入 `cli.py` 并转公开命名；无外部调用方依赖旧私有名 |
| 共享 `ClassificationTask` 流程 | 不变；`classification.py` 再导出面保持 | 库调用方零改动 |
| （新增）可读模型类 | `runtime/python/classify.py:ResNetClassifier(selection, *, top_k=5, labels=None, resize_type=None, runner=None)`；`preprocess`/`infer`/`postprocess`/`predict`；`pre_process`/`forward`/`post_process` 为薄别名；`predict(source)` 接受路径或 BGR 数组 | 新入口 |
| （新增）自定义模型选择 | `model_binding.custom_selection(model_path, target, *, input_height, input_width, class_count, …)` | 无需 Manifest 注册；绑定仍校验实际张量 |

行为变化（明确声明）：

1. `--label-file` 默认值由固定路径改为自动规则：显式给出时必须与类别数一致
   （长度不等 → 具体报错，不再静默用类别 ID 顶替缺失项）；未给出时仅 1000 类
   模型默认加载内置 ImageNet 标签，自定义类别数保留类别 ID。官方模型无参数
   运行行为与旧版一致。
2. 自定义类别数模型不再可能套用 ImageNet 标签（旧入口总是加载默认标签文件）。

不兼容项：无公开 CLI/库不兼容；依赖 `main._dry_run` 等私有名的脚本需改用
`cli` 模块（仓内无此类调用方）。

## 3. Ultralytics YOLO

| 旧入口 | 新入口 | 兼容状态 |
| --- | --- | --- |
| `runtime/python/yolo_detect.py:YoloDetect/YoloDetectConfig` | 实现移至 `runtime/python/detect.py`（可读主线：初始化、`preprocess`、`infer`、`postprocess`、`predict`）；`yolo_detect.py` 为再导出 | 同一类对象；`yolo_dispatch` 按名加载 `yolo_detect` 不变 |
| `YoloDetect.pre_process/forward/post_process` | 薄别名指向 `preprocess/infer/postprocess` | 单一实现；`YoloV10Detect` 子类与 `_predict_task` 兼容 |
| `YoloDetect.predict(img, …)` | `predict(source, image_format="BGR", score_thres=None, nms_thres=None)`：`source` 额外接受本地图片路径 | 数组语义不变；路径读取错误指明路径；输入数组不原地修改 |
| `yolo26_obb.YOLO26OBB` 借用 `YoloDetect.pre_process/forward` | 改为借用 `YoloDetect.preprocess/infer` 及别名 | 外部行为不变 |
| `runtime/python/main.py`（入口内联渲染与执行） | 薄入口：`yolo_cli` 委托 + `create_runtime_model` 构造 + `predict`（`_run_task` 内可见）+ `present_result` 展示 | CLI 参数与模式不变；`build_parser`、`load_labels`、`run_inference` 仍可从 `main` 导入（`run_inference` 委托 `_run_task`，测试补丁目标不变） |
| 入口内 `build_parser`/`print_model_listing`/`describe_plan`/`print_dry_run`/`ensure_model`/`load_labels`/渲染 | `runtime/python/yolo_cli.py`（平铺导入风格与补丁目标保持不变） | 仓内测试 `from main import …` 继续成立 |
| `ModelRunner.from_selection`（自行做目标检查、SDK 导入、构造） | 生产路径改经共享 `RuntimeSession`；失败包装为 `RunnerError` 并保留原因 | `runtime_loader` 注入缝不变；注入路径不走会话 |
| 其他任务类（`yolo26_det/seg/pose/cls`、`yolo_v10detect`、`yolo_seg/pose/cls`） | 不变 | 原有能力未减少 |

不兼容项：无。`run_inference` 返回值维持旧约定（无返回值），调用方不受影响。

## 4. 验证与 not-run

主机验证（本地，`rdk_model_zoo/.venv`；任务 2/3 完成时、platforms/ 移除前）：

- `samples/vision/resnet/tests` 67 项通过（52 基线 + 15 新增行为测试）。
- `samples/vision/ultralytics_yolo/tests` 156 项通过（143 基线 + 13 新增）。
- `tools/sample_contract/check.py --sample samples/vision/{resnet,ultralytics_yolo}`
  0 violations。
- 共享 runner 消费方回归：lprnet 23、mobilenetv3 17、himloco 23 通过。

主机验证（platforms/ 移除后，最终态）：

- `samples/vision/resnet/tests` 63 项（移除 4 项 shim 行为测试）、
  `samples/vision/ultralytics_yolo/tests` 147 项（移除旧 CLI/适配器/shell 等
  shim 面测试）、`samples/_shared/tests`（显式排除 `test_vla_integration.py`）
  167 项全部通过。
- 约 30 个 sample 的历史源码基线测试、yolov5/bytetrack/fcos/lprnet/modnet/
  yoloworld 评估器测试、B3 工具 30 项全部通过（legacy 侧经固定提交物化）。
- `tools/sample_contract/check.py --scope migration`：51 samples 0 violations。
- `npm --prefix tools/catalog-publisher run check`：130 项测试通过，catalog
  可复现（`catalog-v1.0.0-8457fa691252f928`，57 families / 812 benchmarks）。
- 干净 checkout（临时 `git worktree`，未初始化 VLA 子模块）：两个范例入口
  `--help`/`--list-models`/`--dry-run` 与上述回归全部通过；catalog 用本地
  node_modules 生成后 YOLO 目录对照测试通过。
- 环境限制（非本次改动引入）：`samples/vision/yoloe/tests/test_conversion.py`
  需要 onnx 包，本 venv 未安装；gemma CMake configure 需 cmake，本机未安装。
- `samples/_shared/tests` 计数口径：任务 1 时的 158/168 曾意外包含 2 个 VLA
  测试（范围口径问题，未触碰子模块），此后显式排除。

not-run（本轮明确未执行，不以主机测试替代）：真实 ONNX 导出、OE/Mapper/
HMCT 量化编译、板端 `hbm_runtime` 推理与精度/性能验证。历史 README 中的
量化配方按 2026-09-28 用户约定视为可信源材料，未重跑。

<a id="platforms-removal"></a>
## 5. 历史 platforms/ 目录移除（2026-10-01）

用户已授权移除。活动依赖先迁移，目录随后从活动树删除；历史内容不换目录
保存，以既有 Git 引用保留。

### 5.1 固定引用

| 引用 | 值 | 说明 |
| --- | --- | --- |
| platforms/ 整树固定提交 | `d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d` | 触碰 platforms/ 的最后一个提交（2026-09-28），其树与移除前工作区逐字节一致；可从 `origin/develop` 历史到达 |
| X3 Catalog 数据固定提交 | `6fcef2b87c12435e11fbd7327ea70d4efd917b1c` | 最后一个触碰 `platforms/x3/release` 的提交（2026-09-14）；`tools/catalog-publisher/sources.json` 以 `commit` 模式读取该树（`release/` 清单与 `VERSION`），x3 变体链接改用 `blob/<该提交>/platforms/x3/...` |
| 交付分支 | `rdk_x5`、`rdk_s`、`rdk_x3` | 平台交付线仍持有各自完整历史与文档（X5 指南、S Python API/UCP、社区资源等），见根 README 链接 |

全新克隆须知：默认完整克隆（非 shallow）即包含上述对象；shallow/部分克隆需
先 `git fetch origin d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d`（Catalog 另需
`git fetch origin 6fcef2b87c12435e11fbd7327ea70d4efd917b1c`），否则相关测试与
Catalog 构建按设计报错并指明精确 fetch 命令，不会静默丢数据。

### 5.2 依赖迁移清单

| 依赖 | 迁移方式 |
| --- | --- |
| X3 Catalog 来源 | `sources.json` x3 改 `commit` 模式（`sources.ts` 新增最小 commit 支持：40 位 SHA 校验、对象类型校验、缺失对象报错并给出 fetch 命令）；`platforms/registry.json` 交叉检查测试随注册表一起移除 |
| 测试读取历史源码（约 30 个 sample 的基线对照测试） | 新增 `samples/_shared/legacy_platforms.py`：`legacy_path`/`legacy_tree` 从固定提交物化到进程临时目录（sha256 夹具继续校验同字节）；测试侧经 `samples/_shared/tests/legacy_platforms.py` 再导出 |
| YOLO 数值基线（`source_reference.load_source`） | base 指向 platforms 时自动改经固定提交读取 |
| 过渡兼容 shim 的行为测试（resnet `test_integration` 4 项、YOLO 旧 CLI/pose/seg/yolo26 适配器与 shell 编译检查） | 随 shim 一并移除——被测的兼容面不复存在；转发目标的统一行为仍由既有测试覆盖 |
| yolov5/bytetrack 评估器与 B3 板端对照工具 | legacy 侧改从固定提交物化执行；证据 `code_sha256`/`observed_file` 以 `platforms/<相对路径>` 树名记录，保持可比性；B3 的 pin 逐字节校验语义不变 |
| C++ 共享工具 `platforms/s/utils/c_utils` | `git mv` 至 `samples/_shared/cpp/c_utils`（活动依赖，保留在活动树）；resnet/paddle_ocr C++ 构建路径与文档同步更新 |
| PaddleOCR 字体 `FangSong.ttf` | `git mv` 至 `samples/vision/paddle_ocr/test_data/` |
| gemma C++ 主机对照测试 | CMake configure 阶段 `git archive` 固定提交提取 legacy 预处理源码到构建目录（缺失对象时 FATAL_ERROR 并给出 fetch 命令）；本轮本机无 cmake，configure 未运行（not-run） |
| 文档导航（根 README/AGENTS/CLAUDE/datasets/samples 索引/各 evaluator·model·conversion README/skills） | 链接去链为“固定提交 + 路径”指引；涉及可执行路径的文档追加双语 pinned 访问说明；历史报告（docs/releases、superpowers、adr、SOURCE_MAP、evidence）原文不动 |
| `_shared` 清单覆盖测试 | 原有断言（清单不得指向 platforms/ 下）保持不变，继续守护活动清单 |

### 5.3 保留与不兼容说明

- 移除的是过渡兼容入口与冻结快照，不是能力删除：统一树 51 个 sample 覆盖
  已迁移能力；未迁移能力的权威仍在交付分支。
- 旧导入路径（如 `platforms/x5/samples/vision/resnet/runtime/python/resnet.py`
  的 `ResNet`）不再存在于活动树——这是本次移除的明确不兼容项；统一导入见
  第 2、3 节映射。
- Catalog x3 变体的 `source_ref` 由 `main` 变为固定提交 SHA，因此 catalog
  版本哈希变化（`catalog-v1.0.0-8457fa691252f928`，57 families / 812
  benchmarks 与基线一致，无数据丢失）；这不构成发布。
