## 关于 BPU Sample 中使用的核心库说明

本仓库为 **BPU（Brain Processing Unit）Sample 示例仓库**，用于演示和验证 BPU 相关能力在实际工程中的基本使用方式。

在本仓库的示例代码中，主要会使用到以下两类核心运行库：

- **`hb_dnn`**：
  用于模型的加载、推理执行以及张量数据管理等，是 BPU 推理流程中的核心接口库。

- **`hb_ucp`**：
  用于底层资源管理、内存分配、任务调度等通用计算能力相关功能，为 BPU 推理运行提供基础支撑。

---

### 接口文档说明

本仓库中的 Sample 代码主要用于展示 **接口的典型调用流程和使用示例**，并不会对所有接口的参数和行为进行完整说明。

对应的官方接口文档如下：

- **BPU（`hb_dnn`）接口文档**
  👉 [DNN API Overview](http://j6.doc.oe.hobot.cc/guide/ucp/runtime/bpu_sdk_api/bpu_sdk_api_overview.html)

- **UCP（`hb_ucp`）接口文档**
  👉 [UCP API Overview](http://j6.doc.oe.hobot.cc/guide/ucp/ucp_api_reference/ucp_api_overview.html)


如需进一步了解 BPU DNN 和 UCP 的完整能力，请优先查阅上述官方文档。
