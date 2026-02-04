# Repository Documentation Overview

本目录用于存放与 BPU Sample 示例仓库（Model Zoo） 相关的用户文档、接口说明及规范性文档，旨在帮助开发者快速理解仓库结构、核心组件、接口来源以及统一的开发与维护规范。

## 目录结构说明

```text
.
├── images/
├── BPU_Python_API_UserGuide.md
├── Model_Zoo_Repository_Guidelines.md
├── README.md
└── UCP_UserGuide.md
```

## 各文件/目录说明
### images

用于存放文档中使用的 示意图、流程图或截图资源，主要服务于各类 UserGuide 文档的说明与展示，不包含可执行代码。

### Model_Zoo_Repository_Guidelines.md

D-Robotics Model Zoo 仓库规范文档，该文档用于统一说明 Model Zoo 仓库的目录和文件规范、编码规范、注释规范以及文档规范，以保证不同模型、不同任务、不同语言（C / Python）在同一仓库中具备 一致的使用体验、良好的复用性与长期可维护性。

### BPU_Python_API_UserGuide.md

BPU Python API 使用说明文档，该文档用于说明本仓库中 Python Sample 所使用的 BPU Python 接口背景与来源，
其中涉及的 Python 接口主要基于官方提供的 hbm_runtime Python 包，用于在 Python 环境中完成 BPU 模型加载、推理执行等功能。

### UCP_UserGuide.md

UCP/DNN 相关接口说明文档，用于说明本仓库 Sample 中涉及的 hb_ucp和hb_dnn相关接口的定位与使用背景。当在示例代码中涉及底层资源管理、内存分配或模型推理等相关内容时，可通过该文档快速了解应参考的官方接口说明位置。

### README.md

当前目录的说明文档。

### 使用建议

初次使用本仓库时，建议先阅读本 README，了解各文档的职责划分：

- 整体规范与仓库约定，请参考 Model_Zoo_Repository_Guidelines.md;

- 在使用 Python Sample 时，重点参考 BPU_Python_API_UserGuide.md;

- 在使用底层能力或 C/C++ Sample 时，可结合 UCP_UserGuide.md 阅读;
