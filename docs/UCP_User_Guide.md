# BPU C/C++ 接口

Sample 使用以下运行库加载模型、管理张量并执行推理：

- `hb_dnn`：模型加载、推理执行和张量管理。
- `hb_ucp`：内存分配、任务调度和计算资源管理。

各模型的调用示例见 `samples/<领域>/<模型>/runtime/cpp/`。接口参数和返回值参阅：

- [DNN API Overview](https://toolchain.d-robotics.cc/guide/ucp/runtime/bpu_sdk_api/bpu_sdk_api_overview.html)
- [UCP API Overview](https://toolchain.d-robotics.cc/guide/ucp/ucp_api_reference/ucp_api_overview.html)
