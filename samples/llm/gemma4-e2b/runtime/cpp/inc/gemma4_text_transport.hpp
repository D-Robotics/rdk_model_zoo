// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: MIT
/**
 * @file gemma4_text_transport.hpp
 * @brief Stage 2 of the Text pipeline: raw SDK inference and transport.
 *
 * Owns everything that touches SDK tensor memory for the Text subgraphs:
 * descriptor validation and allocation, zero-copy KV binding, strided writes
 * of prepared inputs, the selective-flush inference/task lifecycle, and
 * post-inference KV output collection. No decoding, no session state, no
 * user-facing IO.
 */
#pragma once

#include <array>
#include <cstdint>

#include "hobot/dnn/hb_dnn.h"

#include "gemma4_kv_cache.hpp"
#include "gemma4_model_io.hpp"
#include "gemma4_text_inputs.hpp"
#include "gemma4_text_tensor.hpp"

namespace gemma4 {

/**
 * @brief Validate and allocate one Text subgraph's tensors.
 *
 * Enforces the fixed-export descriptor contract (35 inputs / 31 outputs,
 * pinned dtypes/shapes/strides, prefill seq = kChunkSize, decode seq = 1)
 * before any allocation. Throws on an incompatible export.
 */
ModelIo InitTextSubgraph(hbDNNPackedHandle_t packed, const char *name,
                         int seq_len);

/**
 * @brief Borrow one KvCache's buffers as both subgraphs' KV input slots.
 *
 * Requires the subgraphs to agree on every cache allocation size; records
 * the borrowed capacity. Only call before inference; successful reallocation
 * of the cache invalidates the aliases.
 */
void BindKvCache(ModelIo &prefill, ModelIo &decode, KvCache &cache);

/**
 * @brief Write one prepared batch into the subgraph's BPU input buffers.
 *
 * Strided writes through the accepted descriptor contract; the batch's
 * element counts must match the subgraph's sequence length.
 */
void WriteBatchInputs(ModelIo &io, const TextBatchInputs &batch);

/**
 * @brief Run one inference: flush CPU-modified inputs, submit, wait, refresh
 * all output descriptors.
 *
 * KV rows are rolled into the cache on CPU after every inference, so all 35
 * inputs are flushed before the BPU reads them again (source behavior).
 * Throws on SDK errors; the task is released on every failure path.
 */
void RunSubgraphInference(ModelIo &io);

/// Validated per-layer KV output rows (pointers borrowed from the tensors).
struct TextKvOutputSet {
  std::array<TextKvRows, kNumKvLayers> keys;
  std::array<TextKvRows, kNumKvLayers> values;
};

/**
 * @brief Revalidate the refreshed KV output descriptors against the owned
 * allocations and gather per-layer rows for the cache append.
 *
 * @param io Subgraph whose inference just completed.
 * @param rows Leading rows the caller will append (chunk_valid or 1); the
 *        descriptors themselves must carry a row per subgraph position.
 */
TextKvOutputSet CollectKvOutputs(const ModelIo &io, int rows);

}  // namespace gemma4
