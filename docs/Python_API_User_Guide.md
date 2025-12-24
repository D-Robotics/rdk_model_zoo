## 关于本文档说明

本仓库中涉及的 **BPU Python 接口**，主要来源于地瓜机器人（D-Robotics）官方提供的
**`hbm_runtime` Python 包**。

`hbm_runtime` 是对底层 **BPU 核心运行库 `hb_dnn` 与 `hb_ucp`** 的 Python 层封装，
用于在 Python 环境中完成 BPU 模型加载、推理执行、资源管理以及相关数据处理等核心功能，
从而简化 BPU 能力在 Python 场景下的使用流程。

对应的官方接口文档如下：

- **`hbm_runtime` BPU Python API 使用文档**
  👉 [hbm_runtime Python API 文档](https://developer.d-robotics.cc/rdk_doc/rdk_s/Algorithm_Application/python-api)
