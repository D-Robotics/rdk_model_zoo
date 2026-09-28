// Production TextEngine + tensor helpers against a host SDK double that
// speaks the fixed-export descriptor contract. No vendor SDK, weights, BPU,
// or board: the double fills int16 logits and S8 KV rows with known
// patterns, so generation results, cache transport, and descriptor
// rejections are checked end to end.
#include "gemma4_text_engine.hpp"
#include "hb_utils.hpp"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using gemma4::kCacheLen;
using gemma4::kChunkSize;
using gemma4::kHeadDims;
using gemma4::kHiddenSize;
using gemma4::kNumKvLayers;
using gemma4::kVocabSize;

std::map<void *, int64_t> g_buffers;
std::map<const void *, int> g_models;  // Packed models only.
std::map<const void *, const void *> g_tasks;  // Task handle -> subgraph handle.
int g_infer_calls = 0, g_release_calls = 0;

// Descriptor mutation for the rejection matrix; "" means contract-valid.
std::string g_mutate;
bool g_infer_fails = false;
bool g_drift_logits_type = false;
bool g_drift_kv_rows = false;
// Set to the expected number of right-aligned resident rows before the one
// decode step that must observe them; the check consumes it once.
int g_expect_cache_rows = 0;

const char *hbDNNGetErrorDesc(int) { return "fixture dnn error"; }
const char *hbUCPGetErrorDesc(int) { return "fixture ucp error"; }

int SubgraphSeq(const void *handle) {
  const char *name = static_cast<const char *>(handle);
  const bool prefill = name != nullptr && name[0] == 'p';
  if (prefill && g_mutate == "prefill_seq") return kChunkSize / 2;
  return prefill ? kChunkSize : 1;
}

int KvLayer(int input_index) {
  return input_index < 5 + kNumKvLayers ? input_index - 5
                                        : input_index - 5 - kNumKvLayers;
}

// Contract-valid descriptor for one binding, unless mutated.
hbDNNTensorProperties BuildProps(const void *handle, bool input, int index) {
  const int seq = SubgraphSeq(handle);
  hbDNNTensorProperties properties{};
  properties.quantiType = NONE;
  if (input && index == 0) {  // inputs_embeds F32 [seq, hidden]
    properties.tensorType = HB_DNN_TENSOR_TYPE_F32;
    if (g_mutate == "embed_dtype") properties.tensorType = HB_DNN_TENSOR_TYPE_F16;
    properties.validShape.numDimensions = 2;
    properties.validShape.dimensionSize[0] = seq;
    properties.validShape.dimensionSize[1] =
        g_mutate == "hidden" ? 2048 : kHiddenSize;
    properties.stride[1] = ElementSize(properties.tensorType);
    properties.stride[0] =
        static_cast<int64_t>(properties.validShape.dimensionSize[1]) *
        properties.stride[1];
  } else if (input && index == 1) {  // token ids S64 [1, seq]
    properties.tensorType = g_mutate == "token_dtype" ? HB_DNN_TENSOR_TYPE_S32
                                                      : HB_DNN_TENSOR_TYPE_S64;
    properties.validShape.numDimensions = 2;
    properties.validShape.dimensionSize[0] = 1;
    properties.validShape.dimensionSize[1] = seq;
    properties.stride[1] = ElementSize(properties.tensorType);
    properties.stride[0] = static_cast<int64_t>(seq) * properties.stride[1];
  } else if (input && index == 2) {  // position ids S32 [seq]
    properties.tensorType = HB_DNN_TENSOR_TYPE_S32;
    properties.validShape.numDimensions = 1;
    properties.validShape.dimensionSize[0] = seq;
    properties.stride[0] = 4;
  } else if (input && (index == 3 || index == 4)) {  // masks S16 [seq, cache]
    properties.tensorType = HB_DNN_TENSOR_TYPE_S16;
    properties.validShape.numDimensions = 2;
    properties.validShape.dimensionSize[0] = seq;
    properties.validShape.dimensionSize[1] =
        g_mutate == "mask_cols" ? 2048 : kCacheLen;
    properties.stride[1] = 2;
    properties.stride[0] =
        static_cast<int64_t>(properties.validShape.dimensionSize[1]) * 2;
  } else if (input) {  // KV cache inputs S8 [cache_len, 1, head_dim]
    const int head =
        g_mutate == "kv_head" && index == 5 ? 128 : kHeadDims[KvLayer(index)];
    properties.tensorType = HB_DNN_TENSOR_TYPE_S8;
    properties.validShape.numDimensions = 3;
    properties.validShape.dimensionSize[0] = kCacheLen;
    properties.validShape.dimensionSize[1] = 1;
    properties.validShape.dimensionSize[2] = head;
    properties.stride[2] = 1;
    if (g_mutate == "kv_row_pad" && index == 5) {
      // Internal row padding breaks rolling; the singleton axis keeps a
      // padded stride so the canonical matrix is not contiguous.
      properties.stride[1] = head + 8;
      properties.stride[0] = head + 8;
    } else {
      // Contiguous [cache_len, 1, head_dim]: the singleton contributes nothing.
      properties.stride[1] = head;
      properties.stride[0] = head;
    }
  } else if (index == 0) {  // logits S16 [seq, vocab]
    properties.tensorType = g_mutate == "logit_dtype" ? HB_DNN_TENSOR_TYPE_S32
                                                      : HB_DNN_TENSOR_TYPE_S16;
    properties.validShape.numDimensions = 2;
    properties.validShape.dimensionSize[0] = seq;
    properties.validShape.dimensionSize[1] =
        g_mutate == "vocab" ? kVocabSize / 2 : kVocabSize;
    properties.stride[1] = ElementSize(properties.tensorType);
    properties.stride[0] =
        static_cast<int64_t>(properties.validShape.dimensionSize[1]) *
        properties.stride[1];
  } else {  // KV outputs S8 [seq, head_dim]
    const int layer =
        index < 1 + kNumKvLayers ? index - 1 : index - 1 - kNumKvLayers;
    properties.tensorType = HB_DNN_TENSOR_TYPE_S8;
    properties.validShape.numDimensions = 2;
    properties.validShape.dimensionSize[0] =
        g_mutate == "kv_rows" && index == 1 ? 128 : seq;
    properties.validShape.dimensionSize[1] = kHeadDims[layer];
    properties.stride[1] = 1;
    properties.stride[0] = kHeadDims[layer];
  }
  properties.alignedByteSize =
      properties.stride[0] * properties.validShape.dimensionSize[0];
  if (g_mutate == "quanti" && input && index == 0) properties.quantiType = SCALE;
  if (g_mutate == "overlap" && input && index == 0)
    properties.stride[0] = 4;  // Rows would overlap inside the allocation.
  if (g_mutate == "unknown_type" && input && index == 0)
    properties.tensorType = 99;
  if (g_mutate == "zero_seq" && input && index <= 4) {
    // A zero dimension with a positive allocation must be rejected by the
    // shape contract itself (GEMMA-TEXT-R1), not by the size check.
    properties.validShape.dimensionSize[0] = 0;
    properties.alignedByteSize = properties.stride[0] * kChunkSize;
  }
  return properties;
}

int hbDNNInitializeFromFiles(hbDNNPackedHandle_t *out, const char **, int) {
  static char model_slot = 0;  // Stable stand-in for one packed model.
  *out = &model_slot;
  ++g_models[*out];
  return 0;
}
int hbDNNRelease(hbDNNPackedHandle_t handle) {
  assert(g_models.erase(handle) == 1);
  return 0;
}
int hbDNNGetModelHandle(hbDNNHandle_t *out, hbDNNPackedHandle_t,
                        const char *name) {
  // Subgraph handles are borrowed; the string literal is a stable identity.
  *out = static_cast<hbDNNHandle_t>(const_cast<char *>(name));
  return 0;
}
int hbDNNGetInputCount(int *count, hbDNNHandle_t) {
  *count = 5 + 2 * kNumKvLayers;
  return 0;
}
int hbDNNGetOutputCount(int *count, hbDNNHandle_t) {
  *count = 1 + 2 * kNumKvLayers;
  return 0;
}
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *properties,
                                  hbDNNHandle_t handle, int index) {
  *properties = BuildProps(handle, true, index);
  return 0;
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *properties,
                                   hbDNNHandle_t handle, int index) {
  *properties = BuildProps(handle, false, index);
  return 0;
}
int hbDNNGetCompileBpuCoreNum(int32_t *count, hbDNNHandle_t) {
  *count = 1;
  return 0;
}

int hbUCPMallocCached(hbUCPSysMem *mem, int64_t bytes, int) {
  mem->virAddr = std::malloc(static_cast<size_t>(bytes));
  assert(mem->virAddr);
  g_buffers[mem->virAddr] = bytes;
  return 0;
}
int hbUCPFree(hbUCPSysMem *mem) {
  assert(g_buffers.erase(mem->virAddr) == 1);
  std::free(mem->virAddr);
  mem->virAddr = nullptr;
  return 0;
}
int hbUCPMemFlush(hbUCPSysMem *, int) { return 0; }

// Source greedy semantics under the fixture: logits row r argmaxes to
// 100 + r; KV rows carry a per-layer, per-side byte pattern.
int8_t KvByte(bool value_side, int layer, int row) {
  return static_cast<int8_t>((1 + row + layer + (value_side ? 50 : 0)) & 0x7f);
}

int hbDNNInferV2(hbUCPTaskHandle_t *task, hbDNNTensor *outputs,
                 const hbDNNTensor *inputs, hbDNNHandle_t handle) {
  ++g_infer_calls;
  if (g_infer_fails) {
    *task = nullptr;
    return -7;
  }
  *task = handle;
  g_tasks[handle] = handle;
  const int seq = SubgraphSeq(handle);
  const bool prefill = static_cast<const char *>(handle)[0] == 'p';

  // The engine must have carried the prepared embedding row through the
  // descriptor strides into input 0: prompt row 11 for prefill, the
  // decoded token (104, then repeated 100s) for decode.
  float first_embed = 0.f;
  std::memcpy(&first_embed, inputs[0].sysMem.virAddr, sizeof(first_embed));
  assert(prefill ? first_embed == 11.f
                 : (first_embed == 104.f || first_embed == 100.f));

  if (!prefill && g_expect_cache_rows > 0) {
    // The shared cache holds the chunk's appended rows right-aligned,
    // proving the prefill KV outputs reached the borrowed decode inputs.
    // Rolling shifts rows on every step, so this checks the first step only.
    const auto *keys = static_cast<const int8_t *>(inputs[5].sysMem.virAddr);
    const auto *values = static_cast<const int8_t *>(inputs[20].sysMem.virAddr);
    const int64_t base =
        static_cast<int64_t>(kCacheLen - g_expect_cache_rows) * kHeadDims[0];
    assert(keys[base] == KvByte(false, 0, 0));
    assert(values[base] == KvByte(true, 0, 0));
    g_expect_cache_rows = 0;
  }

  auto *logits = static_cast<int16_t *>(outputs[0].sysMem.virAddr);
  const int64_t row_stride = outputs[0].properties.stride[0] /
                             ElementSize(outputs[0].properties.tensorType);
  for (int row = 0; row < seq; ++row)
    logits[static_cast<int64_t>(row) * row_stride + 100 + row] =
        static_cast<int16_t>(40 + row);
  for (int index = 1; index < 1 + 2 * kNumKvLayers; ++index) {
    const int layer =
        index < 1 + kNumKvLayers ? index - 1 : index - 1 - kNumKvLayers;
    auto *rows = static_cast<int8_t *>(outputs[index].sysMem.virAddr);
    const int64_t stride = outputs[index].properties.stride[0];
    for (int row = 0; row < seq; ++row)
      std::memset(rows + static_cast<int64_t>(row) * stride,
                  KvByte(index >= 1 + kNumKvLayers, layer, row),
                  static_cast<size_t>(kHeadDims[layer]));
  }
  return 0;
}

int hbUCPSubmitTask(hbUCPTaskHandle_t, hbUCPSchedParam *) { return 0; }
int hbUCPWaitTaskDone(hbUCPTaskHandle_t, int) { return 0; }
int hbUCPReleaseTask(hbUCPTaskHandle_t task) {
  ++g_release_calls;
  assert(g_tasks.erase(task) == 1);
  return 0;
}

int hbDNNGetTaskOutputTensorProperties(hbDNNTensorProperties *properties,
                                       hbUCPTaskHandle_t task, int, int index) {
  assert(g_tasks.count(task) == 1);
  *properties = BuildProps(task, false, index);
  if (g_drift_logits_type && index == 0) {
    properties->tensorType = HB_DNN_TENSOR_TYPE_S32;
    properties->stride[1] = 4;
    properties->stride[0] = static_cast<int64_t>(kVocabSize) * 4;
    properties->alignedByteSize =
        static_cast<int64_t>(SubgraphSeq(task)) * kVocabSize * 4;
  }
  if (g_drift_kv_rows && index == 1) {
    properties->validShape.dimensionSize[0] = 128;
    properties->alignedByteSize = 128 * kHeadDims[0];
  }
  return 0;
}

template <class F> bool Throws(F action) {
  try {
    action();
  } catch (const std::exception &) {
    return true;
  }
  return false;
}

void ExpectNoLeak() {
  assert(g_buffers.empty());
  assert(g_models.empty());
  assert(g_tasks.empty());
}

namespace gemma4 {
// Embedding double: every row is the token id repeated, so the fixture can
// verify the transport into inputs_embeds.
TokenEmbeddings::TokenEmbeddings(const std::string &) {}
void TokenEmbeddings::Lookup(const std::vector<int64_t> &ids, float *out) const {
  for (size_t i = 0; i < ids.size(); ++i)
    for (int c = 0; c < kHiddenSize; ++c)
      out[i * kHiddenSize + c] = static_cast<float>(ids[i]);
}
const float *TokenEmbeddings::GetRow(int64_t) const { std::abort(); }
std::vector<float>
TokenEmbeddings::BuildPromptHidden(const std::vector<int64_t> &,
                                   const std::vector<float> &) const {
  std::abort();
}
}  // namespace gemma4

int main() {
  using gemma4::TextEngine;

  // ---- normal construction and generation through the fixed contract ----
  {
    TextEngine engine("fixture.hbm", "unused");
    g_expect_cache_rows = 5;  // Verified by the first decode step.
    const auto generated = engine.Generate({11, 22, 33, 44, 55}, 3);
    // Prefill argmax row 4 -> 104; each decode step argmaxes row 0 -> 100.
    assert((generated ==
            std::vector<int64_t>{11, 22, 33, 44, 55, 104, 100, 100}));
    assert(engine.ProcessedTokens() == 7);  // Five prompt + two decode steps.
  }
  ExpectNoLeak();

  // Streaming with an early stop keeps the returned prefix consistent.
  {
    TextEngine engine("fixture.hbm", "unused");
    std::vector<int64_t> seen;
    const auto out = engine.GenerateStream(
        {11, 22, 33, 44, 55}, 5, [&](int64_t token) {
          seen.push_back(token);
          return seen.size() < 2;
        });
    assert(seen.size() == 2);
    assert(out.size() == 7 && out.back() == seen.back());
  }
  ExpectNoLeak();

  // ---- inference failure surfaces without leaking the task or buffers ----
  {
    g_infer_fails = true;
    TextEngine engine("fixture.hbm", "unused");
    assert(Throws([&] { engine.Generate({11, 22, 33, 44, 55}, 2); }));
    g_infer_fails = false;
  }
  ExpectNoLeak();

  // ---- refreshed output descriptors are revalidated before use ----
  {
    g_drift_logits_type = true;
    TextEngine engine("fixture.hbm", "unused");
    assert(Throws([&] { engine.Generate({11, 22, 33, 44, 55}, 1); }));
    g_drift_logits_type = false;
  }
  ExpectNoLeak();
  {
    g_drift_kv_rows = true;
    TextEngine engine("fixture.hbm", "unused");
    assert(Throws([&] { engine.Generate({11, 22, 33, 44, 55}, 1); }));
    g_drift_kv_rows = false;
  }
  ExpectNoLeak();

  // ---- constructor rejection matrix: incompatible exports never allocate ----
  for (const char *mutation :
       {"embed_dtype", "token_dtype", "prefill_seq", "hidden", "mask_cols",
        "kv_head", "kv_row_pad", "kv_rows", "logit_dtype", "vocab", "quanti",
        "overlap", "unknown_type", "zero_seq"}) {
    g_mutate = mutation;
    assert(Throws([] { TextEngine engine("fixture.hbm", "unused"); }));
    ExpectNoLeak();
  }
  g_mutate.clear();

  // ---- a valid construction still benchmarks after all rejections ----
  {
    TextEngine engine("fixture.hbm", "unused");
    const auto bench = engine.Benchmark({11, 22, 33, 44, 55}, 2, 1);
    assert(bench.decode_steps == 1);  // The timed loop runs max_new_tokens - 1.
    assert(bench.tokens_per_sec >= 0.0);
  }
  ExpectNoLeak();

  assert(g_infer_calls == g_release_calls + 1);  // The failed infer owns no task.
  std::cout << "text tensor flow: generation, cache transport, drift and "
               "descriptor rejections passed\n";
  return 0;
}
