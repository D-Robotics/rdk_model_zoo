// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: MIT
#include "gemma4_text_tensor.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemma4_config.hpp"
#include "hb_utils.hpp"

namespace gemma4 {
namespace {

constexpr int kMaxDimensions =
    sizeof(hbDNNShape::dimensionSize) / sizeof(hbDNNShape::dimensionSize[0]);

const char *RoleName(TextTensorRole role) {
  switch (role) {
    case TextTensorRole::kInputsEmbeds:
      return "inputs_embeds";
    case TextTensorRole::kTokenIds:
      return "token_ids";
    case TextTensorRole::kPositionIds:
      return "position_ids";
    case TextTensorRole::kFullMask:
      return "full_mask";
    case TextTensorRole::kSlidingMask:
      return "sliding_mask";
    case TextTensorRole::kLogits:
      return "logits";
    case TextTensorRole::kKvInput:
      return "kv_input";
    case TextTensorRole::kKvOutput:
      return "kv_output";
  }
  return "text tensor";
}

[[noreturn]] void Reject(TextTensorRole role, const char *what) {
  throw std::invalid_argument("Invalid Text tensor binding " +
                              std::string(RoleName(role)) + ": " + what);
}

// The fixed export pins one storage type per binding; the CPU read/write
// paths produce and consume exactly these widths.
int32_t ExpectedType(TextTensorRole role) {
  switch (role) {
    case TextTensorRole::kInputsEmbeds:
      return HB_DNN_TENSOR_TYPE_F32;
    case TextTensorRole::kTokenIds:
      return HB_DNN_TENSOR_TYPE_S64;
    case TextTensorRole::kPositionIds:
      return HB_DNN_TENSOR_TYPE_S32;
    case TextTensorRole::kFullMask:
    case TextTensorRole::kSlidingMask:
    case TextTensorRole::kLogits:
      return HB_DNN_TENSOR_TYPE_S16;
    case TextTensorRole::kKvInput:
    case TextTensorRole::kKvOutput:
      return HB_DNN_TENSOR_TYPE_S8;
  }
  Reject(role, "unknown binding");
}

int StorageBytes(TextTensorRole role, int32_t type) {
  if (type != ExpectedType(role)) {
    // A different or unknown type is rejected, never reinterpreted at a
    // guessed width or as float.
    Reject(role, "storage type differs from the fixed export");
  }
  try {
    return ElementSize(type);
  } catch (const std::runtime_error &) {
    Reject(role, "unknown storage type");
  }
}

// Collapse singleton axes: a one-sized axis carries a single index and does
// not change the physical element order, so fixed-export shapes may be
// declared with or without them (the source graph declares token ids as
// [1, seq] and the KV caches as [cache_len, 1, head_dim]).
struct CanonicalShape {
  int num_dimensions = 0;
  int32_t dimension[kMaxDimensions] = {};
  int64_t stride[kMaxDimensions] = {};
  int physical_axis[kMaxDimensions] = {};  // canonical index -> SDK axis
};

CanonicalShape Canonicalize(TextTensorRole role,
                            const hbDNNTensorProperties &properties) {
  const auto &shape = properties.validShape;
  if (shape.numDimensions < 1 || shape.numDimensions > kMaxDimensions)
    Reject(role, "rank is outside the contract");
  CanonicalShape canonical;
  for (int axis = 0; axis < shape.numDimensions; ++axis) {
    if (shape.dimensionSize[axis] < 0)
      Reject(role, "negative dimension");
    if (shape.dimensionSize[axis] == 1) continue;
    canonical.dimension[canonical.num_dimensions] = shape.dimensionSize[axis];
    canonical.stride[canonical.num_dimensions] = properties.stride[axis];
    canonical.physical_axis[canonical.num_dimensions] = axis;
    ++canonical.num_dimensions;
  }
  if (canonical.num_dimensions == 0) {
    // A fully singleton descriptor still addresses exactly one element.
    canonical.dimension[0] = 1;
    canonical.stride[0] = properties.stride[0];
    canonical.physical_axis[0] = 0;
    canonical.num_dimensions = 1;
  }
  return canonical;
}

int64_t FlattenedElements(const CanonicalShape &shape) {
  int64_t elements = 1;
  for (int axis = 0; axis < shape.num_dimensions; ++axis) {
    if (elements > std::numeric_limits<int64_t>::max() / shape.dimension[axis])
      return -1;  // Signals an impossible element count.
    elements *= shape.dimension[axis];
  }
  return elements;
}

// Require the canonical shape to be the semantic [rows, cols] matrix: rank
// one (a fully collapsed one-row matrix, which is how a decode [1, cols]
// descriptor canonicalizes) or rank two with an exact leading row count.
void ExpectMatrix(TextTensorRole role, const CanonicalShape &shape, int rows,
                  int cols) {
  if (shape.num_dimensions > 2)
    Reject(role, "rank differs from the fixed semantic shape");
  if (shape.dimension[shape.num_dimensions - 1] != cols)
    Reject(role, "shape differs from the fixed semantic shape");
  if (FlattenedElements(shape) != static_cast<int64_t>(rows) * cols)
    Reject(role, "shape differs from the fixed semantic shape");
  if (shape.num_dimensions == 2 &&
      shape.dimension[0] != static_cast<int32_t>(rows))
    Reject(role, "row count differs from the fixed semantic shape");
}

// True when the canonical strides describe a dense matrix with no internal
// padding (only trailing allocation padding may follow).
bool DenseMatrix(const CanonicalShape &shape, int64_t element) {
  int64_t inner = element;
  for (int axis = shape.num_dimensions - 1; axis >= 0; --axis) {
    if (shape.stride[axis] != inner) return false;
    inner *= shape.dimension[axis];
  }
  return true;
}

// Element-aligned, nonoverlapping strides whose every addressed byte lands
// inside the declared allocation and, when tracked, the original capacity.
void CheckStrides(TextTensorRole role, const CanonicalShape &shape,
                  int64_t element, int64_t aligned_byte_size,
                  int64_t capacity) {
  int64_t span = element;
  for (int axis = shape.num_dimensions - 1; axis >= 0; --axis) {
    const int64_t stride = shape.stride[axis];
    const int64_t steps = shape.dimension[axis] - 1;
    if (stride <= 0 || stride % element || (steps > 0 && stride < span) ||
        span > aligned_byte_size ||
        (steps > 0 && stride > (aligned_byte_size - span) / steps))
      // The division guard above bounds span, so the addition cannot overflow.
      Reject(role, "byte strides overlap or exceed the allocation");
    span += steps * stride;
  }
  if (capacity > 0 && aligned_byte_size > capacity)
    Reject(role, "allocation exceeds the original buffer capacity");
}

}  // namespace

void ValidateTextTensor(const hbDNNTensorProperties &properties,
                        TextTensorRole role, int seq_len, int layer,
                        int64_t capacity) {
  if (properties.alignedByteSize <= 0 || capacity < 0)
    Reject(role, "allocation size or capacity is invalid");
  if (properties.quantiType != NONE)
    Reject(role, "carries quantization metadata; mask/logit/KV quantization "
                 "is a CPU-side source algorithm, not a descriptor flag");

  const CanonicalShape shape = Canonicalize(role, properties);
  const int element = StorageBytes(role, properties.tensorType);

  switch (role) {
    case TextTensorRole::kInputsEmbeds:
      ExpectMatrix(role, shape, seq_len, kHiddenSize);
      break;
    case TextTensorRole::kTokenIds:
    case TextTensorRole::kPositionIds:
      ExpectMatrix(role, shape, 1, seq_len);
      break;
    case TextTensorRole::kFullMask:
    case TextTensorRole::kSlidingMask:
      ExpectMatrix(role, shape, seq_len, kCacheLen);
      break;
    case TextTensorRole::kLogits:
      ExpectMatrix(role, shape, seq_len, kVocabSize);
      break;
    case TextTensorRole::kKvInput:
      // The cache input spans the whole physical cache, not one chunk.
      if (layer < 0 || layer >= kNumKvLayers)
        Reject(role, "layer index is outside the cache");
      ExpectMatrix(role, shape, kCacheLen, kHeadDims[layer]);
      break;
    case TextTensorRole::kKvOutput:
      if (layer < 0 || layer >= kNumKvLayers)
        Reject(role, "layer index is outside the cache");
      ExpectMatrix(role, shape, seq_len, kHeadDims[layer]);
      break;
  }

  CheckStrides(role, shape, element, properties.alignedByteSize, capacity);

  if (role == TextTensorRole::kKvInput) {
    // KvCache owns raw dense [cache_len, head_dim] matrices; only trailing
    // allocation padding is allowed, never internal row padding.
    if (!DenseMatrix(shape, element))
      Reject(role, "must be a dense S8 matrix with no internal padding");
  }
  if (role == TextTensorRole::kKvOutput) {
    // Rows move one row stride at a time; columns are read compactly.
    if (shape.stride[shape.num_dimensions - 1] != element)
      Reject(role, "must address rows of contiguous S8 elements");
  }
}

void WriteTextInput(hbDNNTensor &tensor, const void *source,
                    int64_t source_elements, TextTensorRole role, int seq_len,
                    int64_t capacity) {
  if (role == TextTensorRole::kLogits || role == TextTensorRole::kKvInput ||
      role == TextTensorRole::kKvOutput)
    Reject(role, "is not a CPU-written input binding");
  ValidateTextTensor(tensor.properties, role, seq_len, 0, capacity);
  if (!tensor.sysMem.virAddr || !source || source_elements <= 0)
    Reject(role, "buffer or prepared source is unavailable");
  const CanonicalShape shape = Canonicalize(role, tensor.properties);
  if (FlattenedElements(shape) != source_elements)
    Reject(role, "prepared element count differs from the descriptor");
  // The strided writer copies compactly along the last axis, so element gaps
  // cannot be honored and are rejected instead of writing past their slots.
  const int element = StorageBytes(role, tensor.properties.tensorType);
  if (shape.stride[shape.num_dimensions - 1] != element)
    Reject(role, "innermost stride must equal the element width");
  WriteInputTensor(tensor, source);
}

TextKvRows TextKvOutputRows(const hbDNNTensor &tensor, int layer, int seq_rows,
                            int64_t capacity) {
  ValidateTextTensor(tensor.properties, TextTensorRole::kKvOutput, seq_rows,
                     layer, capacity);
  if (!tensor.sysMem.virAddr)
    Reject(TextTensorRole::kKvOutput, "output buffer is unavailable");
  const CanonicalShape shape =
      Canonicalize(TextTensorRole::kKvOutput, tensor.properties);
  TextKvRows rows;
  rows.data = static_cast<const int8_t *>(tensor.sysMem.virAddr);
  // A collapsed single-row descriptor has no row axis; its one row still
  // spans the full head dimension, which keeps the append stride valid.
  const int64_t dense_row =
      static_cast<int64_t>(shape.dimension[shape.num_dimensions - 1]) *
      ElementSize(tensor.properties.tensorType);
  const int64_t declared = tensor.properties.stride[shape.physical_axis[0]];
  rows.row_stride = declared > dense_row ? declared : dense_row;
  return rows;
}

int64_t ArgmaxTextLogits(const hbDNNTensor &tensor, int seq_idx, int seq_len,
                         int64_t capacity) {
  ValidateTextTensor(tensor.properties, TextTensorRole::kLogits, seq_len, 0,
                     capacity);
  if (!tensor.sysMem.virAddr)
    Reject(TextTensorRole::kLogits, "output buffer is unavailable");
  if (seq_idx < 0 || seq_idx >= seq_len)
    Reject(TextTensorRole::kLogits, "sequence row index is outside the shape");
  const CanonicalShape shape =
      Canonicalize(TextTensorRole::kLogits, tensor.properties);

  const auto &properties = tensor.properties;
  // Rank one means a single collapsed row (decode); its only axis is vocab,
  // and seq_idx is necessarily zero there.
  const int64_t row_stride = shape.num_dimensions == 2
                                 ? properties.stride[shape.physical_axis[0]]
                                 : 0;
  const auto *base = static_cast<const unsigned char *>(tensor.sysMem.virAddr) +
                     static_cast<int64_t>(seq_idx) * row_stride;
  const int64_t column_stride =
      properties.stride[shape.physical_axis[shape.num_dimensions - 1]];
  int best = 0;
  float best_score = -1e30f;
  // Source greedy semantics: scale the int16 storage by kLogitScale and keep
  // the first maximum. The descriptor is validated S16, so no reinterpret.
  for (int i = 0; i < kVocabSize; ++i) {
    int16_t stored = 0;
    std::memcpy(&stored, base + static_cast<int64_t>(i) * column_stride,
                sizeof(stored));
    const float score = static_cast<float>(stored) * kLogitScale;
    if (score > best_score) {
      best_score = score;
      best = i;
    }
  }
  return best;
}

}  // namespace gemma4
