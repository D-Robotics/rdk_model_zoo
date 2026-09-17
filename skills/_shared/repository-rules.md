<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Model Zoo 开发与评审规则索引

这不是第二套独立仓库标准。先读取目标版本规范，并为本次适用条款记录“来源路径/章节、作用域、是否强制、是否变更”。下表提炼已读取规范与真实交付风险；标为工程检查的项目用于发现缺陷，不能伪装成上游原文。

## 适用性

`required`：适用的明确强制条款或交付规格。`platform`：经过目标平台代码/文档验证的限制。`convention`：同类参考。`proposal`：尚未批准的新约定。只有前两类以及有实证的正确性问题可作为强制结论依据；惯例建议与强制缺陷分列。

规范中分类、分割、姿态输出接口尚有空缺时，不自行发明全仓唯一 tuple/dict；为本次实现写明接口并申请维护者确认。目标可能是 X5、S100/S100P/S600、X3 或 legacy；维护源 `rdk_x5` 的模板和示例不能覆盖目标 ref 的实际交付。HBM、BIN 等后缀只按目标 metadata/README 解释，不按后缀批量推断平台或兼容性。明确类别要求优先于样例自行减少承诺：例如 robotics 规定双 runtime 时，不能仅写一句 Python-only 就豁免。

## 核心检查

| ID | 检查与适用范围 | 证据 |
|---|---|---|
| MZ-CTX-01 | 平台、模型变体、运行语言和目标 ref 相容；不将 `rdk_x5` 维护源当成用户目标 | 用户约束、target 身份、README、Manifest、模型 metadata |
| MZ-LAYOUT-01 | 新增规范化 sample 按职责使用 conversion/evaluator/model/runtime/test_data；不无理由增加顶层目录 | 完整文件列表、目标目录规范 |
| MZ-ENTRY-01 | 规范化入口 main.py/main.cc/main.cpp，启动脚本 run.sh，模型实现按模型或任务命名 | 源码、脚本、CMake |
| MZ-PY-01 | 独立 Config 与 Model，参数/模型逻辑分离；pre_process/forward/post_process/predict 等遵守目标 runtime 协议 | 模型源码、适用 Python 规范 |
| MZ-CLI-01 | Python argparse、kebab-case、合理默认、type/default/help；布尔 action 等语言特例按实际语义判断 | main.py、CLI 文档 |
| MZ-CPP-01 | 适用 C++ 规范要求配置结构、显式 init、错误返回、任务/张量生命周期；C++ gflags snake_case 不照搬 Python 参数形式 | 头文件、实现、主入口 |
| MZ-DET-01 | 检测 Python 返回 boxes(N,4), scores(N,), cls_ids(N,)；C++ 目标 Detection 约定；空结果也保持 shape | post_process、边界测试 |
| MZ-DOC-01 | 公共 Python module/class/function Google-style docstrings；C++ 公共接口 Doxygen；文档描述真实语义 | 变更代码与文档 |
| MZ-REUSE-01 | 新增 helper 前查 utils；复用时核对行为、shape、dtype，不能以“复用”引入语义回归 | 工具定义、全部受影响调用点 |
| MZ-ASSET-01 | 模型获取方式、默认路径、下载脚本和 Manifest 一致；manual 明确条件；可信哈希缺失如实披露 | model 文档/脚本、Manifest |
| MZ-CONV-01 | conversion 保留可复现导出、平台配置与工具环境；训练权重/大模型/日志不随意入库 | conversion 文件与说明 |
| MZ-EVAL-01 | evaluator 的指标、数据、模型阶段、阈值、计时边界和运行方式可识别 | evaluator、基线、日志 |
| MZ-INDEX-01 | sample 增删改名或平台能力变化同步目标分支已存在/明确要求的根及分类索引 | README 双语、清单、路径 |
| MZ-DOC-02 | 双语、run.sh/main 默认参数、输出目录和目录树语义一致；没有实现的不宣称支持 | 全链交叉对照 |

## 技术正确性检查（工程检查，不自动等同格式规则）

图像检查 resize/letterbox 逆变换、RGB/BGR/NV12、归一化责任、量化/反量化、输出次序、NMS/阈值、类别标签。不要依据模型名就假定所有 YOLO 输出相同。

C++ 检查错误分支释放、等待完成前数据有效性、cache/同步操作、shape 与对齐元数据；不能无依据把某分支固定对齐字节数写成全平台常量。

音频检查采样率、声道、格式、分块/状态和 token 解码。多模型 pipeline 逐个绑定 encoder/decoder/tokenizer/host 模型的来源及输入输出。机器人/ACT/VLA 检查观测历史、关节次序、缩放、动作限幅和状态重置；离线数值一致不验证闭环稳定，不自动发送执行器指令。

公共工具变更通过 import/include/callsite 搜索扩大影响面，而不是只审改动文件。跨平台移动应检查源 ref 与目标 ref 差异；不能直接要求 X5、S100/S100P/S600、X3 或 legacy 实现等同维护源实现。

## 文件与文档例外

不要用“一切生成文件都禁止”误报受维护的文档资源、必要小样本、金标或专用任务配置。大模型、缓存、编译目录、调试 dump、用户私有数据和临时报告不应混入 sample；具体例外须有职责说明。

没有分类 README，且根 README 是该分支唯一索引时，不强制新建分类 README。存在目录不证明功能完成；没有 C++ 不自动是缺陷，先确认该类别规则与交付承诺。相反，文档承诺 C++ 而代码缺失必须指出。

## 严重程度与置信度分开

`blocking`：明确不能运行、错误平台/产物、丢失数据、危险副作用、必需交付缺失或可证明的严重正确性问题。`major`：实质不完整、回归风险或违反适用维护规范。`minor`：不影响正确性的可读性或惯例改进。每条另列 `confirmed` 或 `needs-verification`；纯猜测不列为已确认阻塞。

历史问题不抹去，但与本次引入、回归、因修改而暴露的问题分列。未改动的历史注释不足不应淹没一个文档修复 PR。无交付规格时写 `No delivery specification available`，不把自己的偏好当成承诺。

## 来源

- [适用条款原文：目录、编码、任务、注释、文档](https://github.com/D-Robotics/rdk_model_zoo/blob/529aece791b8f6a21cf93a14e1caa3edbeb11995/docs/Model_Zoo_Repository_Guidelines.md)
- [现有 Review 的双轴与证据原则](https://github.com/maxma615/skills/blob/d45931b688722fd9895558cec860b6762a91c1d9/skills/rdk-model-zoo-demo-review/SKILL.md)

旧 review 的工作区名字、X3 迁移假设、静态平台例外不继承为当前规则。目标 ref 的 README、Manifest 和实际代码优先；不能因当前维护源默认分支而改变用户指定平台。新规则 ID 是本包索引，不是上游已有规范编号。
