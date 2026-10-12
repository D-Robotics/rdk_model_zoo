# BPU Python 接口

Sample 使用板端的 `hbm_runtime` Python 包加载模型、执行推理并管理资源。
它封装了底层 BPU 运行库 `hb_dnn` 与 `hb_ucp` 的调用。

模型调用示例见 `samples/<领域>/<模型>/runtime/python/`；接口参数和返回值参阅
[hbm_runtime Python API 文档](https://developer.d-robotics.cc/rdk_s_doc/Algorithm_Application/python-api)。
