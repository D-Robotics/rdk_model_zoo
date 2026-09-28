# 只读影响面分析：共用 DFL 解码逻辑的真实消费者与验证范围

## 0. 操作记录

- 已读 Skill：`/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/.coordination/20260928-skills-behavior/AG07/skill/SKILL.md` 及 `references/context-policy.md`、`references/repository-rules.md`、`references/evidence-contract.md`。本任务对应 SKILL.md 第 3 步“公共 utils 先列调用方与平台影响”与 repository-rules“公共工具变更通过 import/include/callsite 搜索扩大影响面，而不是只审改动文件”。
- 目标仓库事实：`REPO_ROOT = rdk-b7-board-integration`，存在 `docs/sample-standards/readme-contract.md` 与 `inference-contract.md`（skill 第 2 步要求列为最高层适用条款）。无 shell 工具，git ref/HEAD/dirty 状态无法按 context-policy 记录，如实声明为未取证。
- 未读 `fixture.json`（任务未引用它）。全程仅使用 Read/Glob/Grep，未修改任何文件。

## 1. “共用 DFL 解码”的真实定位：一处 C++ 共用头 + 两条 Python 链（相互镜像）

| 实现 | 位置 | 关键符号 |
|---|---|---|
| C++ 共用头 | `samples/vision/ultralytics_yolo/runtime/cpp/common/decode.h` | `decode_box_dfl` (:88)、`kDflBins=16` (:38)、`box_decode_from_channels` (:48)、`raw_logit_threshold` (:63)；头注释 :17-23 明示"mirrors runtime/python/rdk_yolo_utils/postprocess.py，保持 C++/Python contract-compatible" |
| Python 契约式解码 | `samples/vision/ultralytics_yolo/runtime/python/decode.py` | `decode_dfl` (:291)、`_dfl_offsets` (:124)、`softmax` (:46)、`sigmoid` (:35)、`_decode_heads` (:199，DFL/LTRB 共用核心)、`_apply_nms` (:187, 延迟导入 postprocess.NMS :194) |
| Python source 等价链 | `samples/vision/ultralytics_yolo/runtime/python/rdk_yolo_utils/postprocess.py` | `decode_boxes` (:339，`reshape(-1,4,16)` 硬编码 16 bins，:369-370 softmax+weights 期望)、`filter_classification` (:214)、`decode_layer`/`decode_outputs` (:469/:523) |

修改前必须先声明改动落在哪一条（或是否要求三方数值同步），三者任一改动都会打破镜像承诺。

## 2. 真实受影响消费者（按 callsite/import 证据）

### 2.1 C++ `decode.h` —— 含一个跨 sample 消费者（已核实非副本）

**样例内：**
- `runtime/cpp/detect/main.cc:51,421`、`runtime/cpp/segment/main.cc:84,506`、`runtime/cpp/pose/main.cc:88,491`（三任务主程序均调用 `yolo::decode_box_dfl`）
- 自测 `runtime/cpp/test/test_decode.cc:53,63,72`

**跨 sample（真实，非重复代码）：**
- `samples/vision/yoloe/runtime/cpp/common/e11_decode.h:7` `#include "common/decode.h"`，:85 调用 `yolo::decode_box_dfl`
- 解析证据：yoloe 自身 `runtime/cpp/common/` 无 `decode.h` 副本（已 Glob 确认），`yoloe/runtime/cpp/CMakeLists.txt:8,14` 将 `ultralytics_yolo/runtime/cpp` 作为 `YOLO_CPP` 加入 include 路径
- 传递链：`postprocess.h:4` → `src/yoloe.cpp:3`（主程序）；yoloe 测试 `tests/test_e11_decode.cc:2`、`tests/decode_probe.cc:3`、`tests/CMakeLists.txt:7,10-12` 同样挂 `YOLO_CPP`

### 2.2 Python `decode.py`（`decode_dfl` 契约链）

- 唯一运行时调用点：`yolo_detect.py:34` 导入，`YoloDetect.post_process` :166 调用
- 继承扩散：`yolo_v10detect.py:51` `class YoloV10Detect(YoloDetect)`（NMS-free DFL 路径同走此段）
- 共享助手消费者（改 `sigmoid`/`_decode_heads`/`DecodeError` 时受波及，但不过 `decode_dfl`）：`yolo26_det.py:30-33`（`decode_ltrb`）、`segmentation_decode.py:19`、`pose_decode.py:25`、`obb_decode.py:8`（均导入 `sigmoid`）
- 运行入口：`runtime/python/main.py`（经 `yolo_dispatch`/family_registry 分发到上述类）
- evaluator（直接消费运行时类）：`evaluator/eval_yolo_det.py:53-54`（YoloDetect+YoloV10Detect）、`eval_yolo_seg.py:52`、`eval_yolo_pose.py:50`

### 2.3 Python `rdk_yolo_utils/postprocess.py` source 等价链

- ultralytics 样例内：`segmentation_decode.py:20-22` 与 `pose_decode.py:18-20`（seg/pose 的 box 解码用 `post.decode_boxes`，DFL 分支 `weights=np.arange(reg_bins)`）；`decode.py:194` 复用其 `NMS`
- **跨 sample**：`samples/vision/yoloe/runtime/python/decode.py:7-9,23-33`（X5 YOLOE-11 PF 的 `decode_x5` 用 `post.filter_classification + post.decode_boxes`）；yoloe `model_binding.py:10-14` 直接导入 ultralytics 的 `DFLSegmentationContract/LTRBSegmentationContract`，:37 `PF11Contract(DFLSegmentationContract)` —— 契约层也复用
- 历史证据脚本（`docs/releases/unified-migration/evidence/*/compare_*.py`）import 此链，属已提交证据的可复现依赖，不是运行时消费者，不设门禁但重命名会破坏其可复现性

### 2.4 平台布局 shim（X5 与 S 线均真实消费共用代码）

- X5：`platforms/x5/samples/vision/ultralytics_yolo/runtime/python/ultralytics_yolo_det.py:82-89,108` 用 importlib 从 canonical 树加载 `yolo_detect.py` 并继承 `YoloDetect`；seg/pose/cls 同模式。evaluator 包装器 `eval_Ultralytics_YOLO_Detect_YUV420SP.py:142-155` 经 `runpy` 转发 canonical evaluator
- S 线：`platforms/s/samples/vision/ultralytics_yolo/runtime/python/yolo_detect.py:85,112` 同为 shim（`_find_sample_root` + `spec_from_file_location`，`class YoloDetect(_BaseModel)`）
- 结论：一次共用解码改动会同时进入 X5 与 S 线的平台布局运行时与评估链；S100P/S600 的支持范围仍须按各 sample/artifact 分别判断，不能从 S 线聚合推导（context-policy）

### 2.5 明确**不是**共用代码消费者（避免误报，但属一致性核对对象）

- `platforms/x5/samples/vision/yoloe/runtime/python/yoloe_seg.py:128` `decode_seg_layer_dfl`（X5 独立实现；`samples/vision/yoloe/tests/test_source_equivalence.py:140` 对其做 source 等价测试）
- `platforms/x3/demos/detect/YOLOv8/YOLOv8_Detect.py:194-195`、`YOLOv10/YOLOv10_Detect.py:194-195`、`Instance_Segmentation/YOLOv8-Seg/YOLOv8_Seg.py:208-209`（legacy 内联 DFL 期望）
- `platforms/s/samples/vision/yolo11_pose|yolo11_seg|yoloe11_seg`（仅 docstring 声明 16-bin DFL，独立实现，无 `from samples.…decode` 导入）
- 若改动改变数值行为，这些独立实现会出现“同模型不同结果”的一致性偏差，应列入回归观察而非直接修改范围

## 3. 应覆盖的验证范围（按 evidence-contract 分层，本次全部未执行）

**static（本次已完成的部分：消费者清单、镜像关系、构建证据；待改动后补：diff 审阅）**
- 三方镜像一致性审阅：`decode.py` ↔ `postprocess.py` ↔ `decode.h` 的 softmax 数值稳定、bin 期望、raw-logit 阈值（`raw_logit_threshold` 的 0/1 退化边界，decode.py:240-251 与 decode.h:63-67 必须同步）
- 文档条款：按 `docs/sample-standards/readme-contract.md` 列出将触及的 README 章节——`samples/vision/ultralytics_yolo/README(_cn).md`、`samples/vision/yoloe/README(_cn).md` 与 `yoloe/runtime/cpp/README(_cn).md`（其 C++ 解码依赖共用头）；MZ-DOC-01/MZ-DOC-02 逐项“文件↔契约章节↔证据”
- MZ-DET-01：空结果仍保持 `boxes(0,4)/scores(0,)/cls_ids(0,)` shape（decode.py `_empty` :54-57）

**host（可执行，本次 not-run）**
- Python：`tests/test_forward_purity.py`（:99 decode_dfl 数值）、`test_detection_binding.py`（:94-110）、`test_ltrb_binding.py`（:93-200，含 reg_bins=1 边界与大 logit 类别次序）、`test_v10_stages.py`、`test_runtime_contract.py`、`test_segmentation_binding.py`、`test_pose_binding.py`、`test_yolo26.py:152,236`（dispatch 走 LTRB、DFL 输出被拒——LTRB 不回归的守卫）、`test_platform_assets.py:29`（X5 保留 DFL、S 线 NMS-free 的资产约定）、`test_obb_binding/test_obb_reference`（`sigmoid` 被复用）
- 跨语言契约：`tests/test_cpp_contract.py:31`（断言 C++ 源含 `decode_box_dfl`）
- C++：`runtime/cpp/test/test_decode.cc`；yoloe 侧 `tests/CMakeLists.txt` 构建的 `test_e11_decode`（跨 sample 回归的核心证据）
- 覆盖点提醒：`decode.py` 支持 `reg_bins` 参数化而 `postprocess.py:369` 硬编码 16、`decode.h` 固定 `kDflBins=16`——参数化行为改动需用非 16 bins 用例划清各自边界

**board（not-run + 原因：本环境无板卡/模型/量化执行授权）**
- evaluator COCO 精度回归、X5/S 线 shim 端到端、yoloe C++ E11 上板数值。不得写成 board-verified；精度比较须先定义数据集、基线与容差

## 4. 限制

- 无 shell/执行工具：git ref/HEAD/dirty 未取证；所有 host/board 检查均未运行，上文仅为应覆盖清单
- 静态搜索已覆盖显式 import、CMake include、importlib 动态加载与 runpy 包装器；不排除未识别的动态路径，但列出的每个消费者均有 `路径:行号` 证据
- `docs/releases/.../evidence/` 下历史脚本依赖共用模块路径，其可复现性属次要影响，已注明不计门禁

## 5. 未覆盖与遗留

- 独立实现（2.5 节）与共用实现的一致性无自动化守卫，改动后建议记录为遗留项
- 本报告为只读分析交付；按 SKILL.md 第 10 步，实际修改应由独立 `rdk-model-zoo-review` 复审，本次不涉及
