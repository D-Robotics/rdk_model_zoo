/**
 * @file gemma4_text_engine.cpp
 * @brief Text LLM engine for Gemma4-E2B (prefill + decode + KV cache).
 *
 * One coherent engine executes the whole Text pipeline: CPU input
 * preparation (stage 1), raw SDK transport over the fixed-export descriptor
 * contract (stage 2), and the explicit output decoding plus KV/session
 * update orchestration (stage 3) for chunked prefill, greedy decode, and
 * reusable conversation prefixes. Session policy decisions come from
 * gemma4_text_session; this file executes them.
 *
 * @note TextEngine instances are not thread-safe.
 */

#include "gemma4_text_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemma4_config.hpp"
#include "hb_utils.hpp"

namespace gemma4 {

// ---- Stage 1: prepared per-call CPU inputs ----
// The algorithms preserve the source TextEngine input preparation verbatim;
// the implicit debug print was removed (diagnostics are a caller-installed
// engine sink concern) and the fixed-context geometry/extent contract is
// validated before any buffer access.

namespace {

// The mask layout indexes a full seq_len window starting at chunk_start
// inside the fixed kCacheLen-token context. With
// `0 <= chunk_valid <= seq_len`, `1 <= seq_len <= kCacheLen` and
// `chunk_start <= kCacheLen - seq_len`, every column index provably stays in
// [0, kCacheLen): cache_col_start = kCacheLen - (seq_len - chunk_valid) -
// total_seen is nonnegative and the last attended column is at most
// kCacheLen - 1. Anything else is rejected before any write instead of
// silently changing attention.
void ValidateMaskGeometry(int chunk_start, int chunk_valid, int seq_len) {
  if (seq_len <= 0 || seq_len > kCacheLen || chunk_start < 0 ||
      chunk_valid < 0 || chunk_valid > seq_len ||
      chunk_start > kCacheLen - seq_len) {
    throw std::invalid_argument(
        "mask geometry is outside the fixed 4096-token context window");
  }
}

// Image token ids → pad for PLE token-identity (token input and the embed
// lookup base), exactly as the source prepared them.
std::vector<int64_t> PleSubstituted(const std::vector<int64_t> &ids) {
  std::vector<int64_t> substituted = ids;
  for (auto &id : substituted) {
    if (id == kImageTokenId) {
      id = kPadTokenId;
    }
  }
  return substituted;
}

}  // namespace

void BuildFullMask(float *mask, int chunk_start, int chunk_valid,
                   int seq_len) {
  ValidateMaskGeometry(chunk_start, chunk_valid, seq_len);
  if (mask == nullptr) {
    throw std::invalid_argument("mask output buffer is null");
  }

  const int total = seq_len * kCacheLen;
  std::fill(mask, mask + total, kMaskValue);

  const int total_seen = std::min(chunk_start + chunk_valid, kCacheLen);
  const int valid_total = std::min(total_seen, kCacheLen);
  const int cache_start_abs = total_seen - valid_total;
  const int current_pad = seq_len - chunk_valid;
  const int cache_col_start = kCacheLen - current_pad - valid_total;

  for (int r = 0; r < seq_len; ++r) {
    int query_abs = chunk_start + r;
    if (query_abs >= total_seen) {
      query_abs = total_seen - 1;
    }
    const int allowed_start = cache_start_abs;
    const int allowed_end = query_abs;
    if (allowed_end < allowed_start) continue;
    const int start_col = cache_col_start + (allowed_start - cache_start_abs);
    const int end_col = cache_col_start + (allowed_end - cache_start_abs);
    float *row = mask + static_cast<size_t>(r) * kCacheLen;
    for (int c = start_col; c <= end_col; ++c) {
      row[c] = 0.f;
    }
  }
}

void BuildSlidingMask(float *mask, int chunk_start, int chunk_valid,
                      int seq_len) {
  ValidateMaskGeometry(chunk_start, chunk_valid, seq_len);
  if (mask == nullptr) {
    throw std::invalid_argument("mask output buffer is null");
  }

  const int total = seq_len * kCacheLen;
  std::fill(mask, mask + total, kMaskValue);

  const int total_seen = std::min(chunk_start + chunk_valid, kCacheLen);
  const int valid_total = std::min(total_seen, kCacheLen);
  const int cache_start_abs = total_seen - valid_total;
  const int current_pad = seq_len - chunk_valid;
  const int cache_col_start = kCacheLen - current_pad - valid_total;

  for (int r = 0; r < seq_len; ++r) {
    int query_abs = chunk_start + r;
    if (query_abs >= total_seen) {
      query_abs = total_seen - 1;
    }
    int allowed_start = cache_start_abs;
    allowed_start = std::max(allowed_start, query_abs - kSlidingWindow + 1);
    const int allowed_end = query_abs;
    if (allowed_end < allowed_start) continue;
    const int start_col = cache_col_start + (allowed_start - cache_start_abs);
    const int end_col = cache_col_start + (allowed_end - cache_start_abs);
    float *row = mask + static_cast<size_t>(r) * kCacheLen;
    for (int c = start_col; c <= end_col; ++c) {
      row[c] = 0.f;
    }
  }
}

void QuantizeMask(const float *mask_f32, int16_t *mask_i16, int rows,
                  int cols) {
  // rows*cols cannot overflow int64 for int operands; reject negative
  // dimensions and null buffers instead of looping on them.
  if (rows < 0 || cols < 0) {
    throw std::invalid_argument("mask quantization count is negative");
  }
  const int64_t total = static_cast<int64_t>(rows) * cols;
  if (total != 0 && (mask_f32 == nullptr || mask_i16 == nullptr)) {
    throw std::invalid_argument("mask quantization buffers are null");
  }
  for (int64_t i = 0; i < total; ++i) {
    const float v =
        std::max(-32768.f, std::min(32767.f, std::round(mask_f32[i])));
    mask_i16[i] = static_cast<int16_t>(v);
  }
}

TextBatchInputs PrepareBatchInputs(const TokenEmbeddings &embeddings,
                                   const std::vector<int64_t> &token_ids,
                                   int chunk_start, int chunk_valid,
                                   const std::vector<float> *prebuilt_hidden,
                                   int seq_len) {
  // Validate the whole per-call contract before any allocation, lookup or
  // write: geometry, token count and the hidden extent this stage indexes.
  ValidateMaskGeometry(chunk_start, chunk_valid, seq_len);
  if (static_cast<int>(token_ids.size()) != chunk_valid) {
    throw std::invalid_argument(
        "prepared token count differs from the valid chunk length");
  }
  if (prebuilt_hidden != nullptr) {
    const int64_t needed_rows = static_cast<int64_t>(chunk_start) + chunk_valid;
    if (needed_rows > static_cast<int64_t>(std::numeric_limits<int>::max()) /
                          kHiddenSize ||
        prebuilt_hidden->size() <
            static_cast<size_t>(needed_rows) * kHiddenSize) {
      throw std::invalid_argument(
          "prebuilt hidden is smaller than the prompt rows it must cover");
    }
  }

  TextBatchInputs batch;
  batch.token_ids.resize(static_cast<size_t>(seq_len), 0);
  batch.hidden.assign(static_cast<size_t>(seq_len) * kHiddenSize, 0.f);
  batch.positions.resize(static_cast<size_t>(seq_len), 0);
  batch.full_mask_q.resize(static_cast<size_t>(seq_len) * kCacheLen);
  batch.slide_mask_q.resize(static_cast<size_t>(seq_len) * kCacheLen);

  // PLE substitution over the padded id row: pad ids keep the pad embedding
  // while the image slots are substituted before the lookup.
  std::vector<int64_t> padded = token_ids;
  padded.resize(static_cast<size_t>(seq_len), 0);
  const std::vector<int64_t> ple_padded = PleSubstituted(padded);

  // Hidden buffer: pad embedding lookup first; with prebuilt hidden the
  // first chunk_valid rows are then overwritten (raw vision at image slots).
  embeddings.Lookup(ple_padded, batch.hidden.data());
  if (prebuilt_hidden != nullptr) {
    for (int i = 0; i < chunk_valid; ++i) {
      const float *src = prebuilt_hidden->data() +
                         static_cast<int64_t>(chunk_start + i) * kHiddenSize;
      float *dst = batch.hidden.data() + static_cast<size_t>(i) * kHiddenSize;
      std::copy(src, src + kHiddenSize, dst);
    }
  }

  for (int i = 0; i < seq_len; ++i) {
    batch.token_ids[static_cast<size_t>(i)] = ple_padded[static_cast<size_t>(i)];
    batch.positions[static_cast<size_t>(i)] =
        i < chunk_valid ? chunk_start + i
                        : chunk_start + std::max(chunk_valid - 1, 0);
  }

  std::vector<float> full_mask(static_cast<size_t>(seq_len) * kCacheLen);
  std::vector<float> slide_mask(static_cast<size_t>(seq_len) * kCacheLen);
  BuildFullMask(full_mask.data(), chunk_start, chunk_valid, seq_len);
  BuildSlidingMask(slide_mask.data(), chunk_start, chunk_valid, seq_len);
  QuantizeMask(full_mask.data(), batch.full_mask_q.data(), seq_len, kCacheLen);
  QuantizeMask(slide_mask.data(), batch.slide_mask_q.data(), seq_len,
               kCacheLen);
  return batch;
}

TextBatchInputs PrepareDecodeInputs(const TokenEmbeddings &embeddings,
                                    int64_t token_id, int pos) {
  // A decode step occupies exactly one context row: the same window rule as
  // the mask geometry with seq_len = 1.
  if (pos < 0 || pos >= kCacheLen) {
    throw std::invalid_argument(
        "decode position is outside the fixed 4096-token context");
  }
  TextBatchInputs batch;
  batch.token_ids = {token_id};
  batch.hidden.assign(kHiddenSize, 0.f);
  batch.positions = {static_cast<int32_t>(pos)};
  batch.full_mask_q.resize(kCacheLen);
  batch.slide_mask_q.resize(kCacheLen);

  embeddings.Lookup(std::vector<int64_t>{token_id}, batch.hidden.data());

  std::vector<float> full_mask(kCacheLen);
  std::vector<float> slide_mask(kCacheLen);
  BuildFullMask(full_mask.data(), pos, 1, 1);
  BuildSlidingMask(slide_mask.data(), pos, 1, 1);
  QuantizeMask(full_mask.data(), batch.full_mask_q.data(), 1, kCacheLen);
  QuantizeMask(slide_mask.data(), batch.slide_mask_q.data(), 1, kCacheLen);
  return batch;
}

// ---- Fixed Text export descriptor contract and strided physical IO ----

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
    // Reject nonpositive dimensions before any arithmetic consumes them.
    if (shape.dimensionSize[axis] <= 0)
      Reject(role, "nonpositive dimension");
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
    // Defensive regardless of callers: Canonicalize rejects nonpositive
    // dimensions first, and division by zero would be undefined behavior.
    if (shape.dimension[axis] <= 0 ||
        elements > std::numeric_limits<int64_t>::max() / shape.dimension[axis])
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

// ---- Stage 2: raw SDK inference and transport ----
// The descriptor roles, validation ordering, binding rules and task
// lifecycle preserve the source TextEngine; the accepted tensor contract
// helpers above own the physical checks.

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

// ---- Stage 3: orchestration (decode + KV update + session) ----

TextEngine::TextEngine(const std::string& text_hbm, const std::string& embed_path)
    : embeddings_(embed_path) {
  try {
    const char* path = text_hbm.c_str();
    const char* paths[] = {path};

    auto t0 = std::chrono::steady_clock::now();
    HBDNN_CHECK(hbDNNInitializeFromFiles(&packed_, paths, 1), "load hbm");
    if (!packed_) throw std::runtime_error("load hbm returned null packed model");
    auto t1 = std::chrono::steady_clock::now();
    load_ms_ =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    prefill_ = InitTextSubgraph(packed_, "prefill", kChunkSize);
    decode_ = InitTextSubgraph(packed_, "decode", 1);
    BindKvCache(prefill_, decode_, kv_);
  } catch (...) {
    prefill_.Clear();
    decode_.Clear();
    if (packed_) hbDNNRelease(packed_);
    packed_ = nullptr;
    throw;
  }
}

TextEngine::~TextEngine() {
  prefill_.Clear();
  decode_.Clear();
  if (packed_) hbDNNRelease(packed_);
}

void TextEngine::EmitDebug(const std::string& message) {
  if (debug_sink_) {
    debug_sink_(message);
  }
}

bool TextEngine::IsEos(int64_t token_id) {
  return token_id == kEosTokenId || token_id == kTurnEndTokenId;
}

// leap_llm mask algorithm: right-aligned cache layout (see
// BuildFullMask/BuildSlidingMask for the per-row window). Prepared per call
// by stage 1.

void TextEngine::AppendKvChunk(const TextKvOutputSet& rows, int chunk_start,
                               int chunk_valid) {
  const int8_t* k_outs[kNumKvLayers];
  const int8_t* v_outs[kNumKvLayers];
  int64_t row_strides[kNumKvLayers];
  for (int i = 0; i < kNumKvLayers; ++i) {
    k_outs[i] = rows.keys[i].data;
    v_outs[i] = rows.values[i].data;
    row_strides[i] = rows.keys[i].row_stride;
  }
  kv_.AppendPrefillChunk(k_outs, v_outs, row_strides, chunk_start, chunk_valid);
}

void TextEngine::AppendKvStep(const TextKvOutputSet& rows, int pos) {
  const int8_t* k_outs[kNumKvLayers];
  const int8_t* v_outs[kNumKvLayers];
  int64_t row_strides[kNumKvLayers];
  for (int i = 0; i < kNumKvLayers; ++i) {
    k_outs[i] = rows.keys[i].data;
    v_outs[i] = rows.values[i].data;
    row_strides[i] = rows.keys[i].row_stride;
  }
  kv_.AppendDecodeStep(k_outs, v_outs, row_strides, pos);
}

void TextEngine::RunPrefillChunk(const std::vector<int64_t>& chunk,
                                 int chunk_start,
                                 const std::vector<float>* prebuilt_hidden) {
  const int chunk_valid = static_cast<int>(chunk.size());
  // Stage 1: prepared per-call context (embeddings, positions, masks). The
  // vector form carries its own extent, which stage 1 validates against the
  // prompt rows it indexes.
  const TextBatchInputs batch =
      PrepareBatchInputs(embeddings_, chunk, chunk_start, chunk_valid,
                         prebuilt_hidden, prefill_.seq_len);
  if (prebuilt_hidden != nullptr) {
    EmitDebug("FillCommonInputs: using prebuilt_hidden, chunk_start=" +
              std::to_string(chunk_start) + " chunk_valid=" +
              std::to_string(chunk_valid));
  }
  // Stage 2: strided write + one selective-flush inference.
  WriteBatchInputs(prefill_, batch);
  RunSubgraphInference(prefill_);
  // Stage 3: append the validated KV output rows, then the caller advances
  // the session state.
  AppendKvChunk(CollectKvOutputs(prefill_, chunk_valid), chunk_start,
                chunk_valid);
}

void TextEngine::PrefillSuffix(const std::vector<int64_t>& ids, int start,
                               const std::vector<float>* hidden) {
  int offset = start;
  while (offset < static_cast<int>(ids.size())) {
    const int remain = static_cast<int>(ids.size()) - offset;
    const int take = std::min(kChunkSize, remain);
    std::vector<int64_t> chunk(ids.begin() + offset,
                               ids.begin() + offset + take);
    session_.token_offset = offset;
    RunPrefillChunk(chunk, offset, hidden);
    offset += take;
  }
  session_.token_offset = static_cast<int>(ids.size());
}

void TextEngine::ResetSession() {
  kv_.Reset();
  session_.Reset();
}

int TextEngine::ContextShift(int n_keep) {
  // Discard tokens from [n_keep, processed_tokens - 1]; the KV cache keeps
  // the leading n_keep resident rows and the caller replays the suffix.
  const TextContextShiftPlan plan = PlanContextShift(session_, n_keep);
  if (!plan.valid) {
    return 0;
  }
  kv_.CompactShift(plan.keep, plan.discard);
  session_.processed_tokens = plan.keep;
  session_.token_offset = plan.keep;
  return plan.discard;
}

bool TextEngine::AutoTruncate(int new_prompt_tokens, int max_new_tokens) {
  const TextAutoTruncatePlan plan =
      PlanAutoTruncate(session_, new_prompt_tokens, max_new_tokens);
  if (!plan.valid) {
    return false;
  }
  // The caller should now re-prefill the recent history using PrefillSuffix.
  ContextShift(plan.keep);
  return true;
}

void TextEngine::AddToHistory(const std::vector<int64_t>& tokens) {
  session_.AddHistory(tokens);
}

void TextEngine::ClearHistory() { session_.ClearHistory(); }

std::vector<int64_t> TextEngine::ContinueGenerate(
    const std::vector<int64_t>& full_ids, int max_new_tokens,
    const std::vector<float>* full_hidden) {
  return ContinueGenerateStream(full_ids, max_new_tokens, nullptr, full_hidden);
}

std::vector<int64_t> TextEngine::ContinueGenerateStream(
    const std::vector<int64_t>& full_ids, int max_new_tokens,
    TokenCallback on_token, const std::vector<float>* full_hidden) {
  // Stage 1 indexes full_hidden rows from the prompt start, so the buffer
  // must cover the whole sequence. Validate before any session mutation or
  // stage read; overflow-safe against the size multiplication.
  if (full_hidden != nullptr) {
    if (full_ids.size() >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(kHiddenSize)) {
      throw std::runtime_error("full_hidden size mismatch");
    }
    if (full_hidden->size() != full_ids.size() * static_cast<size_t>(kHiddenSize)) {
      throw std::runtime_error("full_hidden size mismatch");
    }
  }
  if (max_new_tokens <= 0) {
    return full_ids;
  }

  if (static_cast<int>(full_ids.size()) < session_.processed_tokens) {
    throw std::runtime_error("full_ids shorter than processed prefix");
  }

  const TextContinuationPlan alignment =
      PlanContinuationAlignment(session_, static_cast<int>(full_ids.size()));
  if (alignment.needs_alignment) {
    ContextShift(alignment.aligned_prefix);
    EmitDebug("KV reuse aligned to prefill boundary: keep=" +
              std::to_string(alignment.aligned_prefix) + " replay=" +
              std::to_string(alignment.replay_tokens));
  }

  if (static_cast<int>(full_ids.size()) > session_.processed_tokens) {
    PrefillSuffix(full_ids, session_.processed_tokens, full_hidden);
    session_.processed_tokens = static_cast<int>(full_ids.size());
  }

  session_.token_offset = session_.processed_tokens;
  const int last_idx = LastChunkRowIndex(session_.processed_tokens);

  // Stage 3 decode: greedy argmax over the last processed prefill row.
  std::vector<int64_t> out = full_ids;
  int64_t next = ArgmaxTextLogits(prefill_.outputs[0], last_idx, prefill_.seq_len,
                                  prefill_.OutputCapacity(0));
  out.push_back(next);

  if (on_token && !on_token(next)) {
    return out;
  }

  if (IsEos(next) || max_new_tokens <= 1) {
    return out;
  }

  int64_t last = next;
  for (int i = 1; i < max_new_tokens; ++i) {
    next = RunDecodeStep(last);
    session_.processed_tokens += 1;
    out.push_back(next);

    if (on_token && !on_token(next)) {
      break;
    }

    if (IsEos(next)) {
      break;
    }
    last = next;
  }
  return out;
}

std::vector<int64_t> TextEngine::GenerateStream(
    const std::vector<int64_t>& prompt_ids, int max_new_tokens,
    TokenCallback on_token) {
  ResetSession();
  return ContinueGenerateStream(prompt_ids, max_new_tokens, on_token);
}

std::vector<float> TextEngine::BuildPromptHidden(
    const std::vector<int64_t>& prompt_ids,
    const std::vector<float>& vision_features) const {
  return embeddings_.BuildPromptHidden(prompt_ids, vision_features);
}

PrefillChunkTensors TextEngine::ExportPrefillChunk(
    const std::vector<int64_t>& prompt_ids, int chunk_start,
    int chunk_valid) const {
  const int seq_len = prefill_.seq_len;
  PrefillChunkTensors out;
  out.input_ids.resize(static_cast<size_t>(seq_len));
  out.position_ids.resize(static_cast<size_t>(seq_len));
  out.inputs_embeds.resize(static_cast<size_t>(seq_len) * kHiddenSize);
  out.full_mask.resize(static_cast<size_t>(seq_len) * kCacheLen);
  out.sliding_mask.resize(static_cast<size_t>(seq_len) * kCacheLen);

  std::vector<int64_t> padded = prompt_ids;
  padded.resize(static_cast<size_t>(seq_len), 0);
  for (auto& id : padded) {
    if (id == kImageTokenId) {
      id = kPadTokenId;
    }
  }

  for (int i = 0; i < seq_len; ++i) {
    out.input_ids[static_cast<size_t>(i)] = padded[static_cast<size_t>(i)];
  }

  embeddings_.Lookup(padded, out.inputs_embeds.data());

  const int last_pos = chunk_start + std::max(chunk_valid - 1, 0);
  for (int i = 0; i < seq_len; ++i) {
    out.position_ids[static_cast<size_t>(i)] =
        (i < chunk_valid) ? (chunk_start + i) : last_pos;
  }

  BuildFullMask(out.full_mask.data(), chunk_start, chunk_valid, seq_len);
  BuildSlidingMask(out.sliding_mask.data(), chunk_start, chunk_valid, seq_len);
  return out;
}

int64_t TextEngine::RunDecodeStep(int64_t token_id) {
  const int pos = session_.token_offset;
  // Stage 1: one-row prepared context.
  const TextBatchInputs batch = PrepareDecodeInputs(embeddings_, token_id, pos);
  // Stage 2: strided write + one selective-flush inference.
  WriteBatchInputs(decode_, batch);
  RunSubgraphInference(decode_);
  // Stage 3: append this step's KV rows, decode the next token, advance.
  AppendKvStep(CollectKvOutputs(decode_, 1), pos);

  const int64_t next = ArgmaxTextLogits(decode_.outputs[0], 0, decode_.seq_len,
                                        decode_.OutputCapacity(0));
  session_.token_offset += 1;
  return next;
}

std::vector<int64_t> TextEngine::Generate(const std::vector<int64_t>& prompt_ids,
                                            int max_new_tokens) {
  ResetSession();
  return ContinueGenerate(prompt_ids, max_new_tokens, nullptr);
}

std::vector<int64_t> TextEngine::GenerateWithPromptEmbeddings(
    const std::vector<int64_t>& prompt_ids,
    const std::vector<float>& prompt_hidden, int max_new_tokens) {
  if (prompt_hidden.size() !=
      prompt_ids.size() * static_cast<size_t>(kHiddenSize)) {
    throw std::runtime_error("prompt_hidden size mismatch");
  }
  ResetSession();
  return ContinueGenerate(prompt_ids, max_new_tokens, &prompt_hidden);
}

BenchmarkResult TextEngine::Benchmark(const std::vector<int64_t>& prompt_ids,
                                      int max_new_tokens, int warmup_decode) {
  BenchmarkResult result;
  result.load_ms = load_ms_;

  ResetSession();

  auto pf0 = std::chrono::steady_clock::now();
  int offset = 0;
  int last_idx = 0;
  while (offset < static_cast<int>(prompt_ids.size())) {
    const int remain = static_cast<int>(prompt_ids.size()) - offset;
    const int take = std::min(kChunkSize, remain);
    std::vector<int64_t> chunk(prompt_ids.begin() + offset,
                               prompt_ids.begin() + offset + take);
    session_.token_offset = offset;
    RunPrefillChunk(chunk, offset);
    offset += take;
    last_idx = take - 1;
  }
  session_.token_offset = static_cast<int>(prompt_ids.size());
  auto pf1 = std::chrono::steady_clock::now();
  result.prefill_ms =
      std::chrono::duration<double, std::milli>(pf1 - pf0).count();

  int64_t last = ArgmaxTextLogits(prefill_.outputs[0], last_idx, prefill_.seq_len,
                                  prefill_.OutputCapacity(0));

  for (int i = 0; i < warmup_decode; ++i) {
    last = RunDecodeStep(last);
  }

  auto dc0 = std::chrono::steady_clock::now();
  for (int i = 0; i < max_new_tokens - 1; ++i) {
    last = RunDecodeStep(last);
    result.decode_steps += 1;
    if (IsEos(last)) {
      break;
    }
  }
  auto dc1 = std::chrono::steady_clock::now();
  result.decode_ms = std::chrono::duration<double, std::milli>(dc1 - dc0).count();

  if (result.decode_steps > 0 && result.decode_ms > 0) {
    result.tokens_per_sec =
        1000.0 * static_cast<double>(result.decode_steps) / result.decode_ms;
  }
  return result;
}

}  // namespace gemma4
