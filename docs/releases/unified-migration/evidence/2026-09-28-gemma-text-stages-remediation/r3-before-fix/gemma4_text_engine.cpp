/**
 * @file gemma4_text_engine.cpp
 * @brief Orchestrate Gemma4-E2B text prefill, decode, and KV-cache reuse.
 *
 * The engine sequences the pipeline stages — CPU input preparation
 * (gemma4_text_inputs), raw SDK transport (gemma4_text_transport), and
 * explicit output decoding plus KV/session update — for chunked prefill,
 * greedy decode, and reusable conversation prefixes. Session policy
 * decisions come from gemma4_text_session; this file executes them.
 *
 * @note TextEngine instances are not thread-safe.
 */

#include "gemma4_text_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "gemma4_config.hpp"
#include "gemma4_text_inputs.hpp"
#include "gemma4_text_transport.hpp"
#include "hb_utils.hpp"

namespace gemma4 {

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
// gemma4_text_inputs for the per-row window). Prepared per call by stage 1.

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
                                 const float* prebuilt_hidden) {
  const int chunk_valid = static_cast<int>(chunk.size());
  // Stage 1: prepared per-call context (embeddings, positions, masks).
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
    const float* hptr = nullptr;
    if (hidden != nullptr && !hidden->empty()) {
      hptr = hidden->data();
    }
    RunPrefillChunk(chunk, offset, hptr);
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
