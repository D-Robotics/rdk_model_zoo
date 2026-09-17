---
name: rdk-model-zoo
description: "Use when asking about ready-made RDK Model Zoo models, matching branches, downloads, sample execution, or published benchmarks. 触发词：现成模型、跑示例、模型目录、帧率查询。Do not use as the primary skill for PR review, repository development, custom quantization, or fresh performance measurement."
version: "1.1.0"
license: Apache-2.0
metadata:
  author: "RDK Model Zoo maintainers"
  content-license: "CC-BY-4.0"
  pack: "rdk-model-zoo"
  data-classification: "public"
  workspace-router-handoff: "availability-gated"
---

# RDK Model Zoo

## Purpose

帮助使用者找到并使用目标版本的现成模型。保留现有同名入口的职责，将仓库开发与工具链任务交给专属技能，不在此维护硬编码模型性能表。

## When to use

适用：哪个分支、有哪些现成模型、在哪里下载、如何启动 sample、查询已发布指标。已有用户输入优先使用。

不适用：PR/未提交修改评审用 `rdk-model-zoo-review`；仓库贡献用 `rdk-model-zoo-develop`；接入自有产物用 `rdk-model-zoo-integrate`；真正量化走既有工具链；重新测量用 `rdk-model-zoo-validate` 组织。

## Instructions

1. 读取 [context-policy.md](references/context-policy.md)。区分 `SKILL_ROOT` 与目标 `REPO_ROOT`；`rdk_x5` 只是本 Pack 的维护源，不是默认目标。按用户约束和目标 ref 核实 X5、S100/S100P/S600、X3 或 legacy，再建立仓库/版本、板卡、sample、模型变体、任务和 runtime 上下文。仅浏览模型列表不要求用户先提供板卡日志；执行前必须核实目标环境，冲突时不切换分支。
2. 有目标工作区时读取该版本 README 与模型 Manifest。可用以下只读脚本，不安装依赖、不执行 sample：
   ```bash
   python3 "$SKILL_ROOT/scripts/read_catalog.py" --repo "$REPO_ROOT" --model "$MODEL_QUERY"
   ```
   `MODEL_QUERY` 可省略 `--model` 来列出全部。没有 Manifest 时阅读真实目录/README；多个 Manifest 时显式选择。返回空集不能解释为板卡不支持。
3. 按请求选一个实际存在的样例，核对精确模型资产、任务、下载方式、输入输出与运行语言。`availability: manual` 说明自行提供条件；`sha256: null` 保持未知。多模型 pipeline 不强制只有一个 BPU 文件。
4. 查询性能只转述绑定原始版本和条件的记录；保留 qualifier、缺失字段和 source。不同平台/变体不能代填，无对应实测就明确说未提供。
5. 运行前先读 runtime README、main、run.sh、下载脚本和所需公共函数。记录 cwd/argv、输入、输出及安装/下载/写系统路径的副作用。优先使用已编译模型；不要为一次 smoke test 自动安装 OE。
6. 获授权后按文档和源码匹配的命令执行。记录退出码、日志、本次新生成输出，并检查任务语义；存在历史截图不是运行证据。没有硬件时输出经静态核对的运行计划，保持 not-run。
7. 自有模型需要转换时读 [toolchain-handoff.md](references/toolchain-handoff.md)，先确定精确子任务和可用 router。LLM/VLM 聊天部署、机器人闭环应用、ROS 节点开发分别选择实际可用领域 Skill/官方文档，不发明不存在的技能。
8. 加载/路径问题先对照 sample；运行环境或工具链失败按证据交接。不要因为低 FPS 就断定所有 ONNX 都只在 CPU 上执行，也不要承诺固定速度。

## Workspace router availability gate

Apply these handoffs only after the target platform/version and an actual toolchain task are established; the maintenance branch is not a platform selector.

- For X5, check whether `x5-router` is available in the current session. If unavailable, do not hand off: use `rdk-pack-installer` to install `OE Tool Chain (X5)` within the authorized installation scope.
- For a matching S-series toolchain, check whether `horizon-router` is available in the current session. If unavailable, do not hand off: use `rdk-pack-installer` to install `OE Tool Chain (S)` within the authorized installation scope.

Explain the workspace writes and reuse existing authorization. If installation is not authorized, the installer is absent, or the target is X3/legacy requiring a different version, record the missing capability and consult the target version's documentation instead. After an approved installation, restart or reload the Agent session, check availability again, and retry the scoped handoff; naming a router does not prove it is loaded. Do not install a toolchain merely to browse or run an already compiled sample.

## Output

输出：选定模型/版本与来源、适用条件、工作目录和准确命令、必要副作用、实际执行结果或阻断、[分范围证据](references/evidence-contract.md)。区分“仓库发布声明”和“本次实测”。

## Safety

默认只读检索；运行以用户授权和已说明副作用为边界。禁止替用户上传私有模型、自动覆盖模型、改系统配置或发真实机器人动作。工具缺失时报告，不假装调用。未知数据保持未知。
