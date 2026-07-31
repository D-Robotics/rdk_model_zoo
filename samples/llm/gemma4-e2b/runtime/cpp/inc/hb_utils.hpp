/**
 * @file hb_utils.hpp
 * @brief Lightweight helpers around the Horizon BPU (libdnn / libucp) C API.
 *
 * Provides:
 *   - HBDNN_CHECK / HBUCP_CHECK macros that turn non-zero return codes into
 *     `std::runtime_error` with a human-readable description.
 *   - ElementSize / ProdSize / stride-aware copy helpers used by the prefill
 *     and decode paths to populate BPU input tensors.
 *
 * @note Header-only; safe to include from any translation unit.
 */
#pragma once

#include <dlfcn.h>

#include <cstdlib>
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

/**
 * @brief Mirror the optional DNN V3 inference parameter ABI.
 *
 * The compatibility layout allows the sample to select V2 or V3 inference
 * without requiring newer SDK headers on every supported board image.
 */
struct DNNInferV3ParamCompat {
  bool enable_pre_submit{false};
  bool enable_poll{false};
};

using DNNInferV3Fn = int32_t (*)(hbUCPTaskHandle_t*, hbDNNTensor*,
                                const hbDNNTensor*, hbDNNHandle_t,
                                const DNNInferV3ParamCompat*);

inline int32_t DNNInfer(hbUCPTaskHandle_t* task, hbDNNTensor* outputs,
                        const hbDNNTensor* inputs, hbDNNHandle_t handle) {
#if defined(SOC_S600)
  if (std::getenv("HB_DNN_USER_DEFINED_L2M_SIZES") == nullptr) {
    setenv("HB_DNN_USER_DEFINED_L2M_SIZES", "6:6:6:6", 0);
  }
#endif
  const char* use_v3 = std::getenv("GEMMA4_USE_DNN_V3");
  if (use_v3 != nullptr && std::strcmp(use_v3, "1") == 0) {
    static const auto infer_v3 = reinterpret_cast<DNNInferV3Fn>(
        dlsym(RTLD_DEFAULT, "hbDNNInferV3"));
    if (infer_v3 != nullptr) {
      const DNNInferV3ParamCompat params{};
      return infer_v3(task, outputs, inputs, handle, &params);
    }
  }
  return hbDNNInferV2(task, outputs, inputs, handle);
}

inline uint64_t DNNBpuBackend(hbDNNHandle_t handle) {
#if defined(SOC_S600)
  int32_t core_count = 0;
  HBDNN_CHECK(hbDNNGetCompileBpuCoreNum(&core_count, handle),
              "get compile BPU core count");
  if (core_count < 1 || core_count > 4) {
    throw std::runtime_error("unsupported BPU core count " +
                             std::to_string(core_count));
  }
  uint64_t backend = 0;
  for (int32_t core = 0; core < core_count; ++core) {
    backend |= HB_UCP_BPU_CORE_0 << core;
  }
  return backend;
#else
  static_cast<void>(handle);
  return HB_UCP_BPU_CORE_ANY;
#endif
}

// OE model_inference flow: hbDNNInferV2 (or opt-in V3) -> hbUCPSubmitTask
// -> hbUCPWaitTaskDone -> hbDNNGetTaskOutputTensorProperties -> hbUCPReleaseTask
// Flushes ALL input tensors before inference and ALL output tensors after.
inline void RunInfer(hbDNNHandle_t handle, std::vector<hbDNNTensor>& inputs,
                     std::vector<hbDNNTensor>& outputs) {
  for (auto& in : inputs) {
    FlushClean(in.sysMem);
  }

  hbUCPTaskHandle_t task = nullptr;
  HBDNN_CHECK(DNNInfer(&task, outputs.data(), inputs.data(), handle), "infer");

  hbUCPSchedParam sched{};
  HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
  sched.backend = DNNBpuBackend(handle);
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
  HBDNN_CHECK(DNNInfer(&task, outputs.data(), inputs.data(), handle), "infer");

  hbUCPSchedParam sched{};
  HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
  sched.backend = DNNBpuBackend(handle);
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
