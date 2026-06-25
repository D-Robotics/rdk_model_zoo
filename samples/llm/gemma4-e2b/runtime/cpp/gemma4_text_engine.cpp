#include "gemma4_text_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "gemma4_config.h"
#include "hb_utils.h"

namespace gemma4 {

namespace {

ModelIo InitModelIo(hbDNNPackedHandle_t packed, const char* name) {
  ModelIo io;
  HBDNN_CHECK(hbDNNGetModelHandle(&io.handle, packed, name), name);

  int input_count = 0;
  HBDNN_CHECK(hbDNNGetInputCount(&input_count, io.handle), "input count");
  io.inputs.reserve(static_cast<size_t>(input_count));
  for (int i = 0; i < input_count; ++i) {
    io.inputs.push_back(MakeTensor(io.handle, true, i));
  }

  int output_count = 0;
  HBDNN_CHECK(hbDNNGetOutputCount(&output_count, io.handle), "output count");
  io.outputs.reserve(static_cast<size_t>(output_count));
  for (int i = 0; i < output_count; ++i) {
    io.outputs.push_back(MakeTensor(io.handle, false, i));
  }

  const auto& shape = io.inputs[0].properties.validShape;
  io.seq_len = shape.dimensionSize[0];
  return io;
}

}  // namespace

TextEngine::TextEngine(const std::string& text_hbm, const std::string& embed_path)
    : embeddings_(embed_path) {
  const char* path = text_hbm.c_str();
  const char* paths[] = {path};

  auto t0 = std::chrono::steady_clock::now();
  HBDNN_CHECK(hbDNNInitializeFromFiles(&packed_, paths, 1), "load hbm");
  auto t1 = std::chrono::steady_clock::now();
  load_ms_ =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  prefill_ = InitModelIo(packed_, "prefill");
  decode_ = InitModelIo(packed_, "decode");
  SetupZeroCopyKv();

  decode_hidden_.resize(kHiddenSize, 0.f);
  decode_mask_.resize(kCacheLen);
  decode_slide_mask_.resize(kCacheLen);
  decode_mask_q_.resize(kCacheLen);
  decode_slide_mask_q_.resize(kCacheLen);
}

TextEngine::~TextEngine() {
  // Null out virAddr for KV input tensors to prevent double-free
  // (the actual memory is owned by kv_)
  for (int i = 0; i < kNumKvLayers; ++i) {
    prefill_.inputs[5 + i].sysMem.virAddr = nullptr;
    prefill_.inputs[20 + i].sysMem.virAddr = nullptr;
    decode_.inputs[5 + i].sysMem.virAddr = nullptr;
    decode_.inputs[20 + i].sysMem.virAddr = nullptr;
  }
  
  FreeTensors(prefill_.inputs);
  FreeTensors(prefill_.outputs);
  FreeTensors(decode_.inputs);
  FreeTensors(decode_.outputs);
  if (packed_) {
    hbDNNRelease(packed_);
  }
}

// leap_llm mask algorithm: right-aligned cache layout.
// After prefill chunk with valid tokens [chunk_start, chunk_start+chunk_valid):
//   total_seen  = chunk_start + chunk_valid
//   cache_start_abs = total_seen - valid_total
//   cache_col_start = cache_len - chunk_valid - valid_total   (for first chunk)
// For each row r in [0, seq_len):
//   query_abs = chunk_start + r, clamped to total_seen-1 for pad rows
//   allowed [cache_start_abs .. query_abs]
//   col = cache_col_start + (abs_pos - cache_start_abs)
//
void TextEngine::BuildFullMask(const KvCache& /*kv*/, float* mask,
                               int cache_start, int chunk_start,
                               int chunk_valid, int seq_len) {
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
    float* row = mask + static_cast<size_t>(r) * kCacheLen;
    for (int c = start_col; c <= end_col; ++c) {
      row[c] = 0.f;
    }
  }
}

void TextEngine::BuildSlidingMask(const KvCache& /*kv*/, float* mask,
                                  int cache_start, int chunk_start,
                                  int chunk_valid, int seq_len) {
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
    float* row = mask + static_cast<size_t>(r) * kCacheLen;
    for (int c = start_col; c <= end_col; ++c) {
      row[c] = 0.f;
    }
  }
}

void TextEngine::QuantizeMask(const float* mask_f32, int16_t* mask_i16,
                              int rows, int cols) {
  for (int i = 0; i < rows * cols; ++i) {
    const float v = std::max(-32768.f, std::min(32767.f, std::round(mask_f32[i])));
    mask_i16[i] = static_cast<int16_t>(v);
  }
}

void TextEngine::SetupZeroCopyKv() {
  std::vector<int64_t> k_bytes(kNumKvLayers);
  std::vector<int64_t> v_bytes(kNumKvLayers);
  for (int i = 0; i < kNumKvLayers; ++i) {
    k_bytes[i] = decode_.inputs[5 + i].properties.alignedByteSize;
    v_bytes[i] = decode_.inputs[20 + i].properties.alignedByteSize;
  }
  kv_.Allocate(k_bytes, v_bytes);

  // Redirect KV input tensors to point at the shared cache memory.
  // After hbUCPFree, we null out virAddr so FreeTensors() will skip them.
  for (int i = 0; i < kNumKvLayers; ++i) {
    hbUCPFree(&prefill_.inputs[5 + i].sysMem);
    prefill_.inputs[5 + i].sysMem = kv_.KMem(i);
    hbUCPFree(&prefill_.inputs[20 + i].sysMem);
    prefill_.inputs[20 + i].sysMem = kv_.VMem(i);

    hbUCPFree(&decode_.inputs[5 + i].sysMem);
    decode_.inputs[5 + i].sysMem = kv_.KMem(i);
    hbUCPFree(&decode_.inputs[20 + i].sysMem);
    decode_.inputs[20 + i].sysMem = kv_.VMem(i);
  }
}

void TextEngine::FillCommonInputs(ModelIo& io,
                                  const std::vector<int64_t>& token_ids,
                                  int chunk_start, int chunk_valid, bool decode,
                                  const float* prebuilt_hidden) {
  const int seq_len = io.seq_len;
  std::vector<int64_t> padded = token_ids;
  if (!decode) {
    padded.resize(static_cast<size_t>(seq_len), 0);
  }

  // Image token ids → pad for PLE token-identity (input[1] and embed lookup base).
  std::vector<int64_t> ple_padded = padded;
  for (auto& id : ple_padded) {
    if (id == kImageTokenId) {
      id = kPadTokenId;
    }
  }

  // Build hidden buffer:
  // - For text-only (no prebuilt_hidden): PLE pad embedding lookup of padded ids.
  // - For VLM (with prebuilt_hidden): pad embedding lookup first, then
  //   overwrite valid positions with prebuilt_hidden (raw vision at image slots).
  std::vector<float> hidden(static_cast<size_t>(seq_len) * kHiddenSize, 0.f);
  if (prebuilt_hidden != nullptr) {
    embeddings_.Lookup(ple_padded, hidden.data());
    for (int i = 0; i < chunk_valid; ++i) {
      const float* src = prebuilt_hidden +
                         static_cast<size_t>(chunk_start + i) * kHiddenSize;
      float* dst = hidden.data() + static_cast<size_t>(i) * kHiddenSize;
      std::copy(src, src + kHiddenSize, dst);
    }
    std::cerr << "[DEBUG] FillCommonInputs: using prebuilt_hidden, chunk_start=" << chunk_start
              << " chunk_valid=" << chunk_valid << std::endl;
  } else {
    embeddings_.Lookup(ple_padded, hidden.data());
  }

  auto& embed_in = io.inputs[0];
  WriteInputTensor(embed_in, hidden.data());

  auto& token_in = io.inputs[1];
  int64_t token_buf[256];
  if (decode) {
    token_buf[0] = token_ids.back();
  } else {
    for (int i = 0; i < seq_len; ++i) {
      token_buf[i] = ple_padded[static_cast<size_t>(i)];
    }
  }
  WriteInputTensor(token_in, token_buf);

  auto& pos_in = io.inputs[2];
  int32_t pos_buf[256];
  const int last_pos = chunk_start + std::max(chunk_valid - 1, 0);
  for (int i = 0; i < seq_len; ++i) {
    if (i < chunk_valid) {
      pos_buf[i] = chunk_start + i;
    } else {
      pos_buf[i] = last_pos;
    }
  }
  WriteInputTensor(pos_in, pos_buf);

  std::vector<float> full_mask(static_cast<size_t>(seq_len) * kCacheLen);
  std::vector<float> slide_mask(static_cast<size_t>(seq_len) * kCacheLen);
  BuildFullMask(kv_, full_mask.data(), kv_.CacheStart(), chunk_start, chunk_valid,
                seq_len);
  BuildSlidingMask(kv_, slide_mask.data(), kv_.CacheStart(), chunk_start, chunk_valid,
                   seq_len);

  std::vector<int16_t> full_q(static_cast<size_t>(seq_len) * kCacheLen);
  std::vector<int16_t> slide_q(static_cast<size_t>(seq_len) * kCacheLen);
  QuantizeMask(full_mask.data(), full_q.data(), seq_len, kCacheLen);
  QuantizeMask(slide_mask.data(), slide_q.data(), seq_len, kCacheLen);
  WriteInputTensor(io.inputs[3], full_q.data());
  WriteInputTensor(io.inputs[4], slide_q.data());
}

void TextEngine::FillDecodeInputs(int64_t token_id, int pos) {
  embeddings_.Lookup(std::vector<int64_t>{token_id}, decode_hidden_.data());

  auto& embed_in = decode_.inputs[0];
  WriteInputTensor(embed_in, decode_hidden_.data());

  auto& token_in = decode_.inputs[1];
  int64_t token_buf[1] = {token_id};
  WriteInputTensor(token_in, token_buf);

  auto& pos_in = decode_.inputs[2];
  int32_t pos_buf[1] = {static_cast<int32_t>(pos)};
  WriteInputTensor(pos_in, pos_buf);

  BuildFullMask(kv_, decode_mask_.data(), kv_.CacheStart(), pos, 1, 1);
  BuildSlidingMask(kv_, decode_slide_mask_.data(), kv_.CacheStart(), pos, 1, 1);

  QuantizeMask(decode_mask_.data(), decode_mask_q_.data(), 1, kCacheLen);
  QuantizeMask(decode_slide_mask_.data(), decode_slide_mask_q_.data(), 1, kCacheLen);
  WriteInputTensor(decode_.inputs[3], decode_mask_q_.data());
  WriteInputTensor(decode_.inputs[4], decode_slide_mask_q_.data());
}

void TextEngine::RunPrefillChunk(const std::vector<int64_t>& chunk,
                                 int chunk_start,
                                 const float* prebuilt_hidden) {
  const int chunk_valid = static_cast<int>(chunk.size());
  FillCommonInputs(prefill_, chunk, chunk_start, chunk_valid, false,
                   prebuilt_hidden);
  // Selective flush: only flush non-KV input tensors (embedding/token/pos/masks).
  // KV cache inputs are BPU-owned, no CPU write happened.
  static const std::vector<int> flush_in = PrefillFlushIndices();
  // Flush ALL outputs — we need logits (0) + KV outputs (1..30) for CPU read.
  RunInferSelective(prefill_.handle, prefill_.inputs, prefill_.outputs, flush_in);

  const int8_t* k_outs[kNumKvLayers];
  const int8_t* v_outs[kNumKvLayers];
  int64_t row_strides[kNumKvLayers];
  for (int i = 0; i < kNumKvLayers; ++i) {
    k_outs[i] = static_cast<const int8_t*>(prefill_.outputs[1 + i].sysMem.virAddr);
    v_outs[i] = static_cast<const int8_t*>(prefill_.outputs[16 + i].sysMem.virAddr);
    row_strides[i] = OutputRowStrideBytes(prefill_.outputs[1 + i]);
  }
  kv_.AppendPrefillChunk(k_outs, v_outs, row_strides, chunk_start, chunk_valid);
}

int64_t TextEngine::OutputRowStrideBytes(const hbDNNTensor& tensor) {
  return tensor.properties.stride[0];
}

void TextEngine::PrefillSuffix(const std::vector<int64_t>& ids, int start,
                               const std::vector<float>* hidden) {
  int offset = start;
  while (offset < static_cast<int>(ids.size())) {
    const int remain = static_cast<int>(ids.size()) - offset;
    const int take = std::min(kChunkSize, remain);
    std::vector<int64_t> chunk(ids.begin() + offset,
                               ids.begin() + offset + take);
    token_offset_ = offset;
    const float* hptr = nullptr;
    if (hidden != nullptr && !hidden->empty()) {
      hptr = hidden->data();
    }
    RunPrefillChunk(chunk, offset, hptr);
    offset += take;
  }
  token_offset_ = static_cast<int>(ids.size());
}

void TextEngine::ResetSession() {
  kv_.Reset();
  token_offset_ = 0;
  processed_tokens_ = 0;
  chat_history_.clear();
}

int TextEngine::ContextShift(int n_keep) {
  if (n_keep < 0 || n_keep >= processed_tokens_) {
    return 0;
  }

  // Discard tokens from [n_keep, processed_tokens_ - 1]
  const int discard_len = processed_tokens_ - n_keep;
  if (discard_len <= 0) {
    return 0;
  }

  // Compact KV cache: truncate to only keep first n_keep tokens
  kv_.CompactShift(n_keep, discard_len);

  // Update token tracking
  processed_tokens_ = n_keep;
  token_offset_ = n_keep;

  return discard_len;
}

bool TextEngine::AutoTruncate(int new_prompt_tokens, int max_new_tokens) {
  const int available = kCacheLen - processed_tokens_;
  const int needed = new_prompt_tokens + max_new_tokens;

  if (available >= needed) {
    return false;  // No truncation needed
  }

  // Need to truncate. Keep system prompt (n_keep_ tokens) and discard old history.
  if (n_keep_ <= 0 || n_keep_ >= processed_tokens_) {
    return false;  // Can't truncate if no tokens to keep or nothing to discard
  }

  // Calculate how much to discard
  const int overflow = needed - available;
  const int discardable = processed_tokens_ - n_keep_;

  if (overflow > discardable) {
    // Even discarding everything won't help - this shouldn't happen in practice
    return false;
  }

  // Perform context shift to make room for new tokens
  ContextShift(n_keep_);
  
  // The caller should now re-prefill the recent history using PrefillSuffix
  return true;
}

void TextEngine::AddToHistory(const std::vector<int64_t>& tokens) {
  chat_history_.insert(chat_history_.end(), tokens.begin(), tokens.end());
}

void TextEngine::ClearHistory() {
  chat_history_.clear();
}

std::vector<int64_t> TextEngine::ContinueGenerate(
    const std::vector<int64_t>& full_ids, int max_new_tokens,
    const std::vector<float>* full_hidden) {
  return ContinueGenerateStream(full_ids, max_new_tokens, nullptr, full_hidden);
}

std::vector<int64_t> TextEngine::ContinueGenerateStream(
    const std::vector<int64_t>& full_ids, int max_new_tokens,
    TokenCallback on_token, const std::vector<float>* full_hidden) {
  if (static_cast<int>(full_ids.size()) < processed_tokens_) {
    throw std::runtime_error("full_ids shorter than processed prefix");
  }

  if (static_cast<int>(full_ids.size()) > processed_tokens_) {
    PrefillSuffix(full_ids, processed_tokens_, full_hidden);
    processed_tokens_ = static_cast<int>(full_ids.size());
  }

  token_offset_ = processed_tokens_;
  int last_idx = 0;
  if (processed_tokens_ > 0) {
    const int last_chunk_start =
        ((processed_tokens_ - 1) / kChunkSize) * kChunkSize;
    last_idx = processed_tokens_ - 1 - last_chunk_start;
    if (last_idx < 0 || last_idx >= kChunkSize) {
      last_idx = 0;
    }
  }

  std::vector<int64_t> out = full_ids;
  int64_t next = ArgmaxLogits(prefill_.outputs[0], last_idx);
  out.push_back(next);
  processed_tokens_ += 1;
  
  if (on_token && !on_token(next)) {
    return out;
  }

  if (IsEos(next) || max_new_tokens <= 1) {
    return out;
  }

  int64_t last = next;
  for (int i = 1; i < max_new_tokens; ++i) {
    next = RunDecodeStep(last);
    out.push_back(next);
    processed_tokens_ += 1;
    
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
  return ContinueGenerateStream(prompt_ids, max_new_tokens, on_token, nullptr);
}

int64_t TextEngine::ArgmaxLogits(const hbDNNTensor& logits_tensor, int seq_idx) {
  const auto& shape = logits_tensor.properties.validShape;
  const int vocab = shape.dimensionSize[shape.numDimensions - 1];
  const int16_t* logits = LogitsRowPtr(logits_tensor, seq_idx);
  int best = 0;
  float best_score = -1e30f;
  for (int i = 0; i < vocab; ++i) {
    const float score = static_cast<float>(logits[i]) * kLogitScale;
    if (score > best_score) {
      best_score = score;
      best = i;
    }
  }
  return best;
}

bool TextEngine::IsEos(int64_t token_id) {
  return token_id == kEosTokenId || token_id == kTurnEndTokenId;
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
  std::vector<int64_t> ple_padded = padded;
  for (auto& id : ple_padded) {
    if (id == kImageTokenId) {
      id = kPadTokenId;
    }
  }

  for (int i = 0; i < seq_len; ++i) {
    out.input_ids[static_cast<size_t>(i)] = ple_padded[static_cast<size_t>(i)];
  }

  embeddings_.Lookup(ple_padded, out.inputs_embeds.data());

  const int last_pos = chunk_start + std::max(chunk_valid - 1, 0);
  for (int i = 0; i < seq_len; ++i) {
    out.position_ids[static_cast<size_t>(i)] =
        (i < chunk_valid) ? (chunk_start + i) : last_pos;
  }

  BuildFullMask(kv_, out.full_mask.data(), kv_.CacheStart(), chunk_start,
                chunk_valid, seq_len);
  BuildSlidingMask(kv_, out.sliding_mask.data(), kv_.CacheStart(), chunk_start,
                   chunk_valid, seq_len);
  return out;
}

int64_t TextEngine::RunDecodeStep(int64_t token_id) {
  const int pos = token_offset_;
  FillDecodeInputs(token_id, pos);
  // Selective flush: only flush non-KV input tensors.
  // KV cache inputs are shared with BPU — no CPU write happened.
  static const std::vector<int> flush_in = DecodeFlushIndices();
  // Decode only needs logits output (0) — KV outputs write to shared memory.
  static const std::vector<int> flush_out = {static_cast<int>(kLogitsOutputIndex)};
  RunInferSelective(decode_.handle, decode_.inputs, decode_.outputs,
                    flush_in, flush_out);

  const int8_t* k_outs[kNumKvLayers];
  const int8_t* v_outs[kNumKvLayers];
  int64_t row_strides[kNumKvLayers];
  for (int i = 0; i < kNumKvLayers; ++i) {
    k_outs[i] = static_cast<const int8_t*>(decode_.outputs[1 + i].sysMem.virAddr);
    v_outs[i] = static_cast<const int8_t*>(decode_.outputs[16 + i].sysMem.virAddr);
    row_strides[i] = OutputRowStrideBytes(decode_.outputs[1 + i]);
  }
  kv_.AppendDecodeStep(k_outs, v_outs, row_strides, pos);

  const int64_t next = ArgmaxLogits(decode_.outputs[0], 0);
  token_offset_ += 1;
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
    token_offset_ = offset;
    RunPrefillChunk(chunk, offset);
    offset += take;
    last_idx = take - 1;
  }
  token_offset_ = static_cast<int>(prompt_ids.size());
  auto pf1 = std::chrono::steady_clock::now();
  result.prefill_ms =
      std::chrono::duration<double, std::milli>(pf1 - pf0).count();

  int64_t last = ArgmaxLogits(prefill_.outputs[0], last_idx);

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
