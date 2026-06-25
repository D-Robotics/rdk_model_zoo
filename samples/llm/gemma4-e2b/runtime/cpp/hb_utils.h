#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/dnn/hb_dnn_status.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_status.h"
#include "hobot/hb_ucp_sys.h"

#define HBDNN_CHECK(call, ctx)                                                     \
  do {                                                                             \
    int32_t _rc = (call);                                                          \
    if (_rc != 0) {                                                                \
      throw std::runtime_error(std::string("DNN error ") + std::to_string(_rc) +   \
                               ": " + hbDNNGetErrorDesc(_rc) + " (" + (ctx) + ")"); \
    }                                                                              \
  } while (0)

#define HBUCP_CHECK(call, ctx)                                                     \
  do {                                                                             \
    int32_t _rc = (call);                                                          \
    if (_rc != 0) {                                                                \
      throw std::runtime_error(std::string("UCP error ") + std::to_string(_rc) +   \
                               ": " + hbUCPGetErrorDesc(_rc) + " (" + (ctx) + ")"); \
    }                                                                              \
  } while (0)

inline uint32_t Align64(uint32_t w) {
  return (w + 63U) & ~63U;
}

inline int32_t ElementSize(int32_t type) {
  switch (type) {
    case HB_DNN_TENSOR_TYPE_BOOL8:
    case HB_DNN_TENSOR_TYPE_S8:
    case HB_DNN_TENSOR_TYPE_U8:
      return 1;
    case HB_DNN_TENSOR_TYPE_F16:
    case HB_DNN_TENSOR_TYPE_S16:
    case HB_DNN_TENSOR_TYPE_U16:
      return 2;
    case HB_DNN_TENSOR_TYPE_F32:
    case HB_DNN_TENSOR_TYPE_S32:
    case HB_DNN_TENSOR_TYPE_U32:
      return 4;
    case HB_DNN_TENSOR_TYPE_F64:
    case HB_DNN_TENSOR_TYPE_S64:
    case HB_DNN_TENSOR_TYPE_U64:
      return 8;
    default:
      throw std::runtime_error("unsupported tensor type " + std::to_string(type));
  }
}

inline int64_t ProdSize(const int32_t* dim, int ndim) {
  int64_t p = 1;
  for (int i = 0; i < ndim; ++i) {
    p *= dim[i];
  }
  return p;
}

// OE basic_sample / alignment_rule: copy valid data into BPU buffer using stride padding.
inline void CopyWithStridePadding(void* dst, const void* src,
                                  const hbDNNTensorProperties& props) {
  const auto& shape = props.validShape;
  const int ndim = shape.numDimensions;
  std::vector<uint32_t> dim(static_cast<size_t>(ndim));
  for (int i = 0; i < ndim; ++i) {
    dim[static_cast<size_t>(i)] = static_cast<uint32_t>(shape.dimensionSize[i]);
  }
  const int elem = ElementSize(props.tensorType);
  const int64_t valid_bytes = ProdSize(shape.dimensionSize, ndim) * elem;

  if (valid_bytes == props.alignedByteSize) {
    std::memcpy(dst, src, static_cast<size_t>(valid_bytes));
    return;
  }

  // Recursive row copy (matches HB_HBMRuntime::add_padding_core).
  std::function<void(void*, const void*, int)> rec;
  rec = [&](void* out, const void* in, int d) {
    if (d == ndim - 1) {
      std::memcpy(out, in, static_cast<size_t>(dim[static_cast<size_t>(d)] * elem));
      return;
    }
    const int64_t sub =
        ProdSize(shape.dimensionSize + d + 1, ndim - d - 1) * elem;
    auto* out_c = static_cast<char*>(out);
    auto* in_c = static_cast<const char*>(in);
    for (uint32_t i = 0; i < dim[static_cast<size_t>(d)]; ++i) {
      rec(out_c + props.stride[d] * static_cast<int64_t>(i),
          in_c + sub * static_cast<int64_t>(i), d + 1);
    }
  };
  rec(dst, src, 0);
}

inline void ZeroTensorMem(hbDNNTensor& tensor) {
  if (tensor.sysMem.virAddr && tensor.properties.alignedByteSize > 0) {
    std::memset(tensor.sysMem.virAddr, 0,
                static_cast<size_t>(tensor.properties.alignedByteSize));
  }
}

inline void WriteInputTensor(hbDNNTensor& tensor, const void* src) {
  ZeroTensorMem(tensor);
  CopyWithStridePadding(tensor.sysMem.virAddr, src, tensor.properties);
}

inline void FlushClean(hbUCPSysMem& mem) {
  HBUCP_CHECK(hbUCPMemFlush(&mem, HB_SYS_MEM_CACHE_CLEAN), "flush clean");
}

inline void FlushInvalidate(hbUCPSysMem& mem) {
  HBUCP_CHECK(hbUCPMemFlush(&mem, HB_SYS_MEM_CACHE_INVALIDATE), "flush invalidate");
}

inline hbDNNTensor MakeTensor(hbDNNHandle_t handle, bool is_input, int index) {
  hbDNNTensor tensor{};
  if (is_input) {
    HBDNN_CHECK(hbDNNGetInputTensorProperties(&tensor.properties, handle, index),
                "get input tensor props");
  } else {
    HBDNN_CHECK(hbDNNGetOutputTensorProperties(&tensor.properties, handle, index),
                "get output tensor props");
  }
  const int64_t bytes = tensor.properties.alignedByteSize;
  HBUCP_CHECK(hbUCPMallocCached(&tensor.sysMem, bytes, 0), "malloc tensor");
  ZeroTensorMem(tensor);
  return tensor;
}

inline void FreeTensors(std::vector<hbDNNTensor>& tensors) {
  for (auto& t : tensors) {
    if (t.sysMem.virAddr != nullptr) {
      hbUCPFree(&t.sysMem);
      t.sysMem.virAddr = nullptr;
    }
  }
  tensors.clear();
}

// OE model_inference flow: hbDNNInferV2 -> hbUCPSubmitTask -> hbUCPWaitTaskDone
// -> hbDNNGetTaskOutputTensorProperties -> hbUCPReleaseTask
// Flushes ALL input tensors before inference and ALL output tensors after.
inline void RunInfer(hbDNNHandle_t handle, std::vector<hbDNNTensor>& inputs,
                     std::vector<hbDNNTensor>& outputs) {
  for (auto& in : inputs) {
    FlushClean(in.sysMem);
  }

  hbUCPTaskHandle_t task = nullptr;
  HBDNN_CHECK(hbDNNInferV2(&task, outputs.data(), inputs.data(), handle), "infer");

  hbUCPSchedParam sched{};
  HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
  sched.backend = HB_UCP_BPU_CORE_ANY;
  HBUCP_CHECK(hbUCPSubmitTask(task, &sched), "submit");
  HBUCP_CHECK(hbUCPWaitTaskDone(task, 0), "wait");

  for (size_t i = 0; i < outputs.size(); ++i) {
    FlushInvalidate(outputs[i].sysMem);
    HBDNN_CHECK(hbDNNGetTaskOutputTensorProperties(&outputs[i].properties, task, 0,
                                                   static_cast<int32_t>(i)),
                "get task output props");
  }

  HBUCP_CHECK(hbUCPReleaseTask(task), "release task");
}

// Selective-flush variant: only flush the input tensors at the given indices.
// This skips flushing KV cache tensors (which BPU owns and haven't changed on CPU),
// saving ~30 cache flush operations per decode step.
inline void RunInferSelective(hbDNNHandle_t handle,
                              std::vector<hbDNNTensor>& inputs,
                              std::vector<hbDNNTensor>& outputs,
                              const std::vector<int>& flush_input_indices,
                              const std::vector<int>& flush_output_indices = {}) {
  for (int idx : flush_input_indices) {
    if (idx >= 0 && idx < static_cast<int>(inputs.size())) {
      FlushClean(inputs[static_cast<size_t>(idx)].sysMem);
    }
  }

  hbUCPTaskHandle_t task = nullptr;
  HBDNN_CHECK(hbDNNInferV2(&task, outputs.data(), inputs.data(), handle), "infer");

  hbUCPSchedParam sched{};
  HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
  sched.backend = HB_UCP_BPU_CORE_ANY;
  HBUCP_CHECK(hbUCPSubmitTask(task, &sched), "submit");
  HBUCP_CHECK(hbUCPWaitTaskDone(task, 0), "wait");

  if (flush_output_indices.empty()) {
    for (size_t i = 0; i < outputs.size(); ++i) {
      FlushInvalidate(outputs[i].sysMem);
      HBDNN_CHECK(hbDNNGetTaskOutputTensorProperties(&outputs[i].properties, task, 0,
                                                     static_cast<int32_t>(i)),
                  "get task output props");
    }
  } else {
    for (int idx : flush_output_indices) {
      if (idx >= 0 && idx < static_cast<int>(outputs.size())) {
        FlushInvalidate(outputs[static_cast<size_t>(idx)].sysMem);
        HBDNN_CHECK(hbDNNGetTaskOutputTensorProperties(
                        &outputs[static_cast<size_t>(idx)].properties, task, 0, idx),
                    "get task output props");
      }
    }
  }

  HBUCP_CHECK(hbUCPReleaseTask(task), "release task");
}

inline const int16_t* LogitsRowPtr(const hbDNNTensor& logits, int seq_idx) {
  const auto& props = logits.properties;
  const int ndim = props.validShape.numDimensions;
  const int64_t row_stride_elems =
      props.stride[ndim - 2] / ElementSize(props.tensorType);
  const auto* base = static_cast<const int16_t*>(logits.sysMem.virAddr);
  return base + seq_idx * row_stride_elems;
}
