// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: MIT
/**
 * @file gemma4_text_transport.cpp
 * @brief Stage 2 implementation: Text subgraph transport over the BPU SDK.
 *
 * The descriptor roles, validation ordering, binding rules and task
 * lifecycle are moved from the former TextEngine; the accepted tensor
 * contract helpers own the physical checks.
 */

#include "gemma4_text_transport.hpp"

#include <stdexcept>
#include <vector>

#include "gemma4_config.hpp"
#include "hb_utils.hpp"

namespace gemma4 {
namespace {

// Fixed-export binding of a Text input slot.
TextTensorRole InputRole(int index) {
  switch (index) {
    case 0:
      return TextTensorRole::kInputsEmbeds;
    case 1:
      return TextTensorRole::kTokenIds;
    case 2:
      return TextTensorRole::kPositionIds;
    case 3:
      return TextTensorRole::kFullMask;
    case 4:
      return TextTensorRole::kSlidingMask;
    default:
      return TextTensorRole::kKvInput;
  }
}

// KV inputs are 15 keys (5..19) followed by 15 values (20..34).
int InputLayer(int index) {
  return index < kKvInputStart + kNumKvLayers
             ? index - kKvInputStart
             : index - kKvInputStart - kNumKvLayers;
}

TextTensorRole OutputRole(int index) {
  return index == kLogitsOutputIndex ? TextTensorRole::kLogits
                                     : TextTensorRole::kKvOutput;
}

// KV outputs are 15 keys (1..15) followed by 15 values (16..30).
int OutputLayer(int index) {
  return index < 1 + kNumKvLayers ? index - 1 : index - 1 - kNumKvLayers;
}

}  // namespace

ModelIo InitTextSubgraph(hbDNNPackedHandle_t packed, const char *name,
                         int seq_len) {
  ModelIo io;
  HBDNN_CHECK(hbDNNGetModelHandle(&io.handle, packed, name), name);

  if (!io.handle) throw std::runtime_error("null text subgraph handle");
  int input_count = 0;
  HBDNN_CHECK(hbDNNGetInputCount(&input_count, io.handle), "input count");
  if (input_count != 5 + 2 * kNumKvLayers)
    throw std::runtime_error("text subgraph requires 35 inputs");

  int output_count = 0;
  HBDNN_CHECK(hbDNNGetOutputCount(&output_count, io.handle), "output count");
  if (output_count != 1 + 2 * kNumKvLayers)
    throw std::runtime_error("text subgraph requires 31 outputs");

  // Validate every descriptor against the fixed-export contract before
  // allocating, so an incompatible export is rejected without acquiring
  // buffers its bindings can never address.
  std::vector<hbDNNTensorProperties> input_properties;
  input_properties.reserve(static_cast<size_t>(input_count));
  for (int i = 0; i < input_count; ++i) {
    hbDNNTensorProperties properties{};
    HBDNN_CHECK(hbDNNGetInputTensorProperties(&properties, io.handle, i),
                "get input tensor props");
    ValidateTextTensor(properties, InputRole(i), seq_len, InputLayer(i));
    input_properties.push_back(properties);
  }
  std::vector<hbDNNTensorProperties> output_properties;
  output_properties.reserve(static_cast<size_t>(output_count));
  for (int i = 0; i < output_count; ++i) {
    hbDNNTensorProperties properties{};
    HBDNN_CHECK(hbDNNGetOutputTensorProperties(&properties, io.handle, i),
                "get output tensor props");
    ValidateTextTensor(properties, OutputRole(i), seq_len, OutputLayer(i));
    output_properties.push_back(properties);
  }

  io.inputs.reserve(static_cast<size_t>(input_count));
  for (auto &properties : input_properties) {
    io.AddInput(AllocateTensor(properties));
  }
  io.outputs.reserve(static_cast<size_t>(output_count));
  for (auto &properties : output_properties) {
    io.AddOutput(AllocateTensor(properties));
  }

  io.seq_len = seq_len;
  return io;
}

void BindKvCache(ModelIo &prefill, ModelIo &decode, KvCache &cache) {
  std::vector<int64_t> k_bytes(kNumKvLayers);
  std::vector<int64_t> v_bytes(kNumKvLayers);
  for (int i = 0; i < kNumKvLayers; ++i) {
    // One shared cache buffer backs both subgraphs, so their descriptors
    // (already validated S8 [kCacheLen, head_dim] matrices) must also
    // reserve identical room.
    if (prefill.inputs[5 + i].properties.alignedByteSize !=
            decode.inputs[5 + i].properties.alignedByteSize ||
        prefill.inputs[20 + i].properties.alignedByteSize !=
            decode.inputs[20 + i].properties.alignedByteSize) {
      throw std::runtime_error(
          "prefill/decode KV inputs disagree on the cache allocation size");
    }
    k_bytes[i] = decode.inputs[5 + i].properties.alignedByteSize;
    v_bytes[i] = decode.inputs[20 + i].properties.alignedByteSize;
  }
  cache.Allocate(k_bytes, v_bytes);

  // Borrowed slots are tracked independently from the cache owner.
  for (int i = 0; i < kNumKvLayers; ++i) {
    prefill.BindBorrowedInput(5 + i, cache.KMem(i), k_bytes[i]);
    prefill.BindBorrowedInput(20 + i, cache.VMem(i), v_bytes[i]);
    decode.BindBorrowedInput(5 + i, cache.KMem(i), k_bytes[i]);
    decode.BindBorrowedInput(20 + i, cache.VMem(i), v_bytes[i]);
  }
}

void WriteBatchInputs(ModelIo &io, const TextBatchInputs &batch) {
  const int seq_len = io.seq_len;
  // The stage boundary can check what the former raw-buffer path assumed:
  // a prepared batch must match the subgraph shape before any write.
  if (batch.token_ids.size() != static_cast<size_t>(seq_len) ||
      batch.positions.size() != static_cast<size_t>(seq_len) ||
      batch.hidden.size() != static_cast<size_t>(seq_len) * kHiddenSize ||
      batch.full_mask_q.size() != static_cast<size_t>(seq_len) * kCacheLen ||
      batch.slide_mask_q.size() != static_cast<size_t>(seq_len) * kCacheLen)
    throw std::invalid_argument(
        "prepared batch does not match the subgraph shape");
  WriteTextInput(io.inputs[0], batch.hidden.data(),
                 static_cast<int64_t>(seq_len) * kHiddenSize,
                 TextTensorRole::kInputsEmbeds, seq_len, io.InputCapacity(0));
  WriteTextInput(io.inputs[1], batch.token_ids.data(), seq_len,
                 TextTensorRole::kTokenIds, seq_len, io.InputCapacity(1));
  WriteTextInput(io.inputs[2], batch.positions.data(), seq_len,
                 TextTensorRole::kPositionIds, seq_len, io.InputCapacity(2));
  WriteTextInput(io.inputs[3], batch.full_mask_q.data(),
                 static_cast<int64_t>(seq_len) * kCacheLen,
                 TextTensorRole::kFullMask, seq_len, io.InputCapacity(3));
  WriteTextInput(io.inputs[4], batch.slide_mask_q.data(),
                 static_cast<int64_t>(seq_len) * kCacheLen,
                 TextTensorRole::kSlidingMask, seq_len, io.InputCapacity(4));
}

void RunSubgraphInference(ModelIo &io) {
  // KV rows are rolled into the cache on CPU after every inference, so all
  // inputs must be cleaned before the BPU reads the cache again.
  static const std::vector<int> flush_in = TextInputFlushIndices();
  // Flush ALL outputs — logits (0) and the KV outputs (1..30) are read on CPU.
  RunInferSelective(io.handle, io.inputs, io.outputs, flush_in);
}

TextKvOutputSet CollectKvOutputs(const ModelIo &io, int rows) {
  // The export carries one row per subgraph position; a chunk appends only
  // the leading rows.
  if (rows <= 0 || rows > io.seq_len)
    throw std::runtime_error("KV append row count is outside the subgraph shape");
  TextKvOutputSet set;
  for (int i = 0; i < kNumKvLayers; ++i) {
    // Refreshed descriptors are revalidated against the allocation this
    // engine owns before any row is consumed.
    set.keys[i] = TextKvOutputRows(io.outputs[1 + i], i, io.seq_len,
                                   io.OutputCapacity(1 + i));
    set.values[i] = TextKvOutputRows(io.outputs[16 + i], i, io.seq_len,
                                     io.OutputCapacity(16 + i));
    // RollAppendLayer advances K and V sources with one shared row stride.
    if (set.values[i].row_stride != set.keys[i].row_stride) {
      throw std::runtime_error(
          "K/V output row strides differ; the cache append reads both sides "
          "with one shared stride");
    }
  }
  return set;
}

}  // namespace gemma4
