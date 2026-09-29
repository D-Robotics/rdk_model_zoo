// Host SDK double shared by the Gemma Text stage/session tests. NOT vendor
// ABI, NOT a model: implements the fixed-export descriptor contract and a
// deterministic inference pattern — logits row r argmaxes to 100 + r, KV
// rows carry a per-layer/per-side byte pattern. Single translation unit per
// test binary. Text fixture knobs live in text_fixture::state().
#pragma once

#include "gemma4_text_engine.hpp"
#include "hb_utils.hpp"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace text_fixture {

using gemma4::kCacheLen;
using gemma4::kChunkSize;
using gemma4::kHeadDims;
using gemma4::kHiddenSize;
using gemma4::kNumKvLayers;
using gemma4::kVocabSize;

struct State {
  std::map<void *, int64_t> buffers;
  std::map<const void *, int> models;            // Packed models only.
  std::map<const void *, const void *> tasks;    // Task -> subgraph handle.
  int infer_calls = 0;
  int release_calls = 0;
  std::string mutate;                            // Descriptor mutation; ""=valid.
  bool fail_infer = false;
  bool drift_logits_type = false;
  bool drift_kv_rows = false;
  int expect_cache_rows = 0;                     // Consumed by the first decode.
  float last_first_embed = 0.f;                  // First float of inputs[0].
  int64_t last_prompt_inputs = 0;                // inputs[0] byte count written.
};

inline State &state() {
  static State s;
  return s;
}

inline void Reset() { state() = State{}; }

// Lazily registered stand-in packed model; re-registers after a Reset.
inline hbDNNPackedHandle_t Packed() {
  static hbDNNPackedHandle_t packed = nullptr;
  if (packed == nullptr || state().models.count(packed) == 0) {
    hbDNNInitializeFromFiles(&packed, nullptr, 0);
  }
  return packed;
}

inline int SubgraphSeq(const void *handle) {
  const char *name = static_cast<const char *>(handle);
  const bool prefill = name != nullptr && name[0] == 'p';
  if (prefill && state().mutate == "prefill_seq") return kChunkSize / 2;
  return prefill ? kChunkSize : 1;
}

inline int KvLayer(int input_index) {
  return input_index < 5 + kNumKvLayers
             ? input_index - 5
             : input_index - 5 - kNumKvLayers;
}

// Contract-valid descriptor for one binding, unless mutated.
inline hbDNNTensorProperties BuildProps(const void *handle, bool input,
                                        int index) {
  const int seq = SubgraphSeq(handle);
  hbDNNTensorProperties properties{};
  properties.quantiType = NONE;
  if (input && index == 0) {  // inputs_embeds F32 [seq, hidden]
    properties.tensorType = HB_DNN_TENSOR_TYPE_F32;
    if (state().mutate == "embed_dtype")
      properties.tensorType = HB_DNN_TENSOR_TYPE_F16;
    properties.validShape.numDimensions = 2;
    properties.validShape.dimensionSize[0] = seq;
    properties.validShape.dimensionSize[1] =
        state().mutate == "hidden" ? 2048 : kHiddenSize;
    properties.stride[1] = ElementSize(properties.tensorType);
    properties.stride[0] =
        static_cast<int64_t>(properties.validShape.dimensionSize[1]) *
        properties.stride[1];
  } else if (input && index == 1) {  // token ids S64 [1, seq]
    properties.tensorType = state().mutate == "token_dtype"
                                ? HB_DNN_TENSOR_TYPE_S32
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
        state().mutate == "mask_cols" ? 2048 : kCacheLen;
    properties.stride[1] = 2;
    properties.stride[0] =
        static_cast<int64_t>(properties.validShape.dimensionSize[1]) * 2;
  } else if (input) {  // KV cache inputs S8 [cache_len, 1, head_dim]
    const int head = state().mutate == "kv_head" && index == 5
                         ? 128
                         : kHeadDims[KvLayer(index)];
    properties.tensorType = HB_DNN_TENSOR_TYPE_S8;
    properties.validShape.numDimensions = 3;
    properties.validShape.dimensionSize[0] = kCacheLen;
    properties.validShape.dimensionSize[1] = 1;
    properties.validShape.dimensionSize[2] = head;
    properties.stride[2] = 1;
    if (state().mutate == "kv_row_pad" && index == 5) {
      properties.stride[1] = head + 8;
      properties.stride[0] = head + 8;
    } else {
      properties.stride[1] = head;
      properties.stride[0] = head;
    }
  } else if (index == 0) {  // logits S16 [seq, vocab]
    properties.tensorType = state().mutate == "logit_dtype"
                                ? HB_DNN_TENSOR_TYPE_S32
                                : HB_DNN_TENSOR_TYPE_S16;
    properties.validShape.numDimensions = 2;
    properties.validShape.dimensionSize[0] = seq;
    properties.validShape.dimensionSize[1] =
        state().mutate == "vocab" ? kVocabSize / 2 : kVocabSize;
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
        state().mutate == "kv_rows" && index == 1 ? 128 : seq;
    properties.validShape.dimensionSize[1] = kHeadDims[layer];
    properties.stride[1] = 1;
    properties.stride[0] = kHeadDims[layer];
  }
  properties.alignedByteSize =
      properties.stride[0] * properties.validShape.dimensionSize[0];
  if (state().mutate == "quanti" && input && index == 0)
    properties.quantiType = SCALE;
  if (state().mutate == "overlap" && input && index == 0)
    properties.stride[0] = 4;
  if (state().mutate == "unknown_type" && input && index == 0)
    properties.tensorType = 99;
  if (state().mutate == "zero_seq" && input && index <= 4) {
    properties.validShape.dimensionSize[0] = 0;
    properties.alignedByteSize = properties.stride[0] * kChunkSize;
  }
  return properties;
}

// Source greedy semantics under the double: row r argmaxes to 100 + r.
inline int8_t KvByte(bool value_side, int layer, int row) {
  return static_cast<int8_t>((1 + row + layer + (value_side ? 50 : 0)) & 0x7f);
}

}  // namespace text_fixture

const char *hbDNNGetErrorDesc(int) { return "fixture dnn error"; }
const char *hbUCPGetErrorDesc(int) { return "fixture ucp error"; }

int hbDNNInitializeFromFiles(hbDNNPackedHandle_t *out, const char **, int) {
  static char model_slot = 0;  // Stable stand-in for one packed model.
  *out = &model_slot;
  ++text_fixture::state().models[*out];
  return 0;
}
int hbDNNRelease(hbDNNPackedHandle_t handle) {
  assert(text_fixture::state().models.erase(handle) == 1);
  return 0;
}
int hbDNNGetModelHandle(hbDNNHandle_t *out, hbDNNPackedHandle_t,
                        const char *name) {
  *out = static_cast<hbDNNHandle_t>(const_cast<char *>(name));
  return 0;
}
int hbDNNGetInputCount(int *count, hbDNNHandle_t) {
  *count = 5 + 2 * text_fixture::kNumKvLayers;
  return 0;
}
int hbDNNGetOutputCount(int *count, hbDNNHandle_t) {
  *count = 1 + 2 * text_fixture::kNumKvLayers;
  return 0;
}
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *properties,
                                  hbDNNHandle_t handle, int index) {
  *properties = text_fixture::BuildProps(handle, true, index);
  return 0;
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *properties,
                                   hbDNNHandle_t handle, int index) {
  *properties = text_fixture::BuildProps(handle, false, index);
  return 0;
}
int hbDNNGetCompileBpuCoreNum(int32_t *count, hbDNNHandle_t) {
  *count = 1;
  return 0;
}

int hbUCPMallocCached(hbUCPSysMem *mem, int64_t bytes, int) {
  mem->virAddr = std::malloc(static_cast<size_t>(bytes));
  assert(mem->virAddr != nullptr);
  text_fixture::state().buffers[mem->virAddr] = bytes;
  return 0;
}
int hbUCPFree(hbUCPSysMem *mem) {
  assert(text_fixture::state().buffers.erase(mem->virAddr) == 1);
  std::free(mem->virAddr);
  mem->virAddr = nullptr;
  return 0;
}
int hbUCPMemFlush(hbUCPSysMem *, int) { return 0; }

int hbDNNInferV2(hbUCPTaskHandle_t *task, hbDNNTensor *outputs,
                 const hbDNNTensor *inputs, hbDNNHandle_t handle) {
  auto &fixture = text_fixture::state();
  ++fixture.infer_calls;
  if (fixture.fail_infer) {
    *task = nullptr;
    return -7;
  }
  *task = handle;
  fixture.tasks[handle] = handle;
  const int seq = text_fixture::SubgraphSeq(handle);
  const bool prefill = static_cast<const char *>(handle)[0] == 'p';

  // Record what the pipeline actually carried into inputs_embeds.
  std::memcpy(&fixture.last_first_embed, inputs[0].sysMem.virAddr,
              sizeof(fixture.last_first_embed));

  if (!prefill && fixture.expect_cache_rows > 0) {
    // The shared cache holds the chunk's appended rows right-aligned; the
    // check consumes itself (rolling shifts rows on every later step).
    const auto *keys = static_cast<const int8_t *>(inputs[5].sysMem.virAddr);
    const auto *values = static_cast<const int8_t *>(inputs[20].sysMem.virAddr);
    const int64_t base =
        static_cast<int64_t>(text_fixture::kCacheLen - fixture.expect_cache_rows) *
        text_fixture::kHeadDims[0];
    assert(keys[base] == text_fixture::KvByte(false, 0, 0));
    assert(values[base] == text_fixture::KvByte(true, 0, 0));
    fixture.expect_cache_rows = 0;
  }

  auto *logits = static_cast<int16_t *>(outputs[0].sysMem.virAddr);
  const int64_t row_stride = outputs[0].properties.stride[0] /
                             ElementSize(outputs[0].properties.tensorType);
  for (int row = 0; row < seq; ++row)
    logits[static_cast<int64_t>(row) * row_stride + 100 + row] =
        static_cast<int16_t>(40 + row);
  for (int index = 1; index < 1 + 2 * text_fixture::kNumKvLayers; ++index) {
    const int layer = index < 1 + text_fixture::kNumKvLayers
                          ? index - 1
                          : index - 1 - text_fixture::kNumKvLayers;
    auto *rows = static_cast<int8_t *>(outputs[index].sysMem.virAddr);
    const int64_t stride = outputs[index].properties.stride[0];
    for (int row = 0; row < seq; ++row)
      std::memset(rows + static_cast<int64_t>(row) * stride,
                  text_fixture::KvByte(index >= 1 + text_fixture::kNumKvLayers,
                                       layer, row),
                  static_cast<size_t>(text_fixture::kHeadDims[layer]));
  }
  return 0;
}

int hbUCPSubmitTask(hbUCPTaskHandle_t, hbUCPSchedParam *) { return 0; }
int hbUCPWaitTaskDone(hbUCPTaskHandle_t, int) { return 0; }
int hbUCPReleaseTask(hbUCPTaskHandle_t task) {
  auto &fixture = text_fixture::state();
  ++fixture.release_calls;
  assert(fixture.tasks.erase(task) == 1);
  return 0;
}

int hbDNNGetTaskOutputTensorProperties(hbDNNTensorProperties *properties,
                                       hbUCPTaskHandle_t task, int, int index) {
  auto &fixture = text_fixture::state();
  assert(fixture.tasks.count(task) == 1);
  *properties = text_fixture::BuildProps(task, false, index);
  if (fixture.drift_logits_type && index == 0) {
    properties->tensorType = HB_DNN_TENSOR_TYPE_S32;
    properties->stride[1] = 4;
    properties->stride[0] = static_cast<int64_t>(text_fixture::kVocabSize) * 4;
    properties->alignedByteSize =
        static_cast<int64_t>(text_fixture::SubgraphSeq(task)) *
        text_fixture::kVocabSize * 4;
  }
  if (fixture.drift_kv_rows && index == 1) {
    properties->validShape.dimensionSize[0] = 128;
    properties->alignedByteSize = 128 * text_fixture::kHeadDims[0];
  }
  return 0;
}

namespace gemma4 {
// Embedding double: every row repeats its token id, so tests can verify the
// exact transport into inputs_embeds.
TokenEmbeddings::TokenEmbeddings(const std::string &) {}
void TokenEmbeddings::Lookup(const std::vector<int64_t> &ids,
                                    float *out) const {
  for (size_t i = 0; i < ids.size(); ++i)
    for (int c = 0; c < kHiddenSize; ++c)
      out[i * kHiddenSize + c] = static_cast<float>(ids[i]);
}
const float *TokenEmbeddings::GetRow(int64_t) const {
  static const float row[kHiddenSize] = {};  // Unused by these tests.
  return row;
}
std::vector<float>
TokenEmbeddings::BuildPromptHidden(const std::vector<int64_t> &ids,
                                   const std::vector<float> &vision) const {
  // Rows repeat the token id; image soft-token slots carry the vision rows.
  std::vector<float> hidden(ids.size() * kHiddenSize, 0.f);
  size_t vision_row = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] == kImageTokenId && vision_row * kHiddenSize < vision.size()) {
      std::copy(vision.begin() + static_cast<long>(vision_row) * kHiddenSize,
                vision.begin() +
                    static_cast<long>(vision_row + 1) * kHiddenSize,
                hidden.begin() + static_cast<long>(i) * kHiddenSize);
      ++vision_row;
    } else {
      for (int c = 0; c < kHiddenSize; ++c)
        hidden[i * kHiddenSize + c] = static_cast<float>(ids[i]);
    }
  }
  return hidden;
}
}  // namespace gemma4
