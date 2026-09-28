// Stage-level behavior of the Text pipeline against the host SDK double:
// stage 1 prepared contexts, stage 2 transport (write/infer/collect), and
// failure ownership between stages. Not vendor ABI or model evidence.
#include "text_fixture.hpp"

#include <cassert>
#include <iostream>

namespace {

using text_fixture::kCacheLen;
using text_fixture::kChunkSize;
using text_fixture::kHeadDims;
using text_fixture::kHiddenSize;

template <class F> bool Throws(F action) {
  try {
    action();
  } catch (const std::exception &) {
    return true;
  }
  return false;
}

int16_t MaskValueAt(const std::vector<int16_t> &mask, int row, int col) {
  return mask[static_cast<size_t>(row) * kCacheLen + static_cast<size_t>(col)];
}

}  // namespace

int main() {
  using gemma4::BuildFullMask;
  using gemma4::BuildSlidingMask;
  using gemma4::CollectKvOutputs;
  using gemma4::InitTextSubgraph;
  using gemma4::QuantizeMask;
  using gemma4::ModelIo;
  using gemma4::PrepareBatchInputs;
  using gemma4::PrepareDecodeInputs;
  using gemma4::RunSubgraphInference;
  using gemma4::TokenEmbeddings;
  using gemma4::WriteBatchInputs;

  TokenEmbeddings embeddings("unused");

  // ---- stage 1: prepared context for a 5-token chunk ----
  {
    const auto batch =
        PrepareBatchInputs(embeddings, {11, 22, 33, 44, 55}, 0, 5, nullptr,
                           kChunkSize);
    assert(batch.token_ids.size() == kChunkSize);
    assert(batch.token_ids[0] == 11 && batch.token_ids[4] == 55);
    assert(batch.token_ids[5] == 0 && batch.token_ids.back() == 0);
    assert(batch.hidden.size() ==
           static_cast<size_t>(kChunkSize) * kHiddenSize);
    // The embedding double repeats each id: row boundaries carry the ids.
    assert(batch.hidden[0] == 11.f);
    assert(batch.hidden[4 * kHiddenSize] == 55.f);
    assert(batch.hidden[5 * kHiddenSize] == 0.f);
    assert(batch.positions[0] == 0 && batch.positions[4] == 4);
    for (int i = 5; i < kChunkSize; ++i)
      assert(batch.positions[static_cast<size_t>(i)] == 4);  // Last seen pos.
    // Masks: rows attend to the right-aligned window; everything else keeps
    // the source mask value.
    assert(MaskValueAt(batch.full_mask_q, 0, 3840) == 0);
    assert(MaskValueAt(batch.full_mask_q, 0, 3841) == static_cast<int16_t>(gemma4::kMaskValue));
    assert(MaskValueAt(batch.full_mask_q, 4, 3844) == 0);
    assert(MaskValueAt(batch.full_mask_q, 4, 3839) == static_cast<int16_t>(gemma4::kMaskValue));
    assert(MaskValueAt(batch.full_mask_q, 255, 3844) == 0);  // Pad rows clamp.
    assert(batch.full_mask_q == batch.slide_mask_q);  // Window > prompt here.
  }

  // ---- stage 1: image-token substitution and prebuilt-hidden override ----
  {
    std::vector<int64_t> ids = {11, gemma4::kImageTokenId, 33};
    // The prebuilt hidden must carry one row per valid chunk position.
    std::vector<float> vision(3 * kHiddenSize, 0.f);
    vision[0] = 9.f;
    vision[kHiddenSize] = 8.f;              // Row for the image slot.
    vision[2 * kHiddenSize] = 33.f;
    const auto batch = PrepareBatchInputs(embeddings, ids, 0, 3, &vision,
                                          kChunkSize);
    // PLE substitution rewrites the image token to the pad id for input[1].
    assert(batch.token_ids[1] == gemma4::kPadTokenId);
    assert(batch.token_ids[0] == 11 && batch.token_ids[2] == 33);
    // The prebuilt hidden replaces the first chunk_valid rows wholesale —
    // text and image rows alike (source behavior) — while later rows keep
    // the pad embedding.
    assert(batch.hidden[0] == 9.f);
    assert(batch.hidden[1 * kHiddenSize] == 8.f);
    assert(batch.hidden[2 * kHiddenSize] == 33.f);
    assert(batch.hidden[3 * kHiddenSize] == 0.f);
  }

  // ---- stage 1: decode step context ----
  {
    const auto batch = PrepareDecodeInputs(embeddings, 104, 5);
    assert(batch.token_ids == std::vector<int64_t>{104});
    assert(batch.hidden.size() == kHiddenSize);
    assert(batch.hidden[0] == 104.f);
    assert(batch.positions == std::vector<int32_t>{5});
    assert(MaskValueAt(batch.full_mask_q, 0, 4095) == 0);
    assert(MaskValueAt(batch.full_mask_q, 0, 4089) == static_cast<int16_t>(gemma4::kMaskValue));
    assert(batch.full_mask_q == batch.slide_mask_q);
  }

  // ---- stage 2: subgraph init, binding, write, infer, collect ----
  {
    text_fixture::Reset();
    ModelIo prefill = InitTextSubgraph(text_fixture::Packed(), "prefill", kChunkSize);
    ModelIo decode = InitTextSubgraph(text_fixture::Packed(), "decode", 1);
    gemma4::KvCache cache;
    BindKvCache(prefill, decode, cache);
    // The borrowed slots alias the cache exactly once.
    assert(prefill.inputs[5].sysMem.virAddr == decode.inputs[5].sysMem.virAddr);
    assert(prefill.inputs[20].sysMem.virAddr == cache.VMem(0).virAddr);

    const auto batch =
        PrepareBatchInputs(embeddings, {11, 22, 33, 44, 55}, 0, 5, nullptr,
                           kChunkSize);
    WriteBatchInputs(prefill, batch);
    // The write landed at the start of the embeds buffer, byte-exact.
    float carried = 0.f;
    std::memcpy(&carried, prefill.inputs[0].sysMem.virAddr, sizeof(carried));
    assert(carried == 11.f);
    const int infer_before = text_fixture::state().infer_calls;
    RunSubgraphInference(prefill);
    assert(text_fixture::state().infer_calls == infer_before + 1);
    assert(text_fixture::state().release_calls == infer_before + 1);

    const auto rows = CollectKvOutputs(prefill, 5);
    for (int layer = 0; layer < gemma4::kNumKvLayers; ++layer) {
      assert(rows.keys[layer].row_stride ==
             static_cast<int64_t>(kHeadDims[layer]));
      assert(rows.keys[layer].data[0] ==
             text_fixture::KvByte(false, layer, 0));
      assert(rows.values[layer].data[0] ==
             text_fixture::KvByte(true, layer, 0));
    }
    // Stage 3: append the collected rows; the cache rolls them right-aligned.
    {
      const int8_t *k_outs[gemma4::kNumKvLayers];
      const int8_t *v_outs[gemma4::kNumKvLayers];
      int64_t row_strides[gemma4::kNumKvLayers];
      for (int layer = 0; layer < gemma4::kNumKvLayers; ++layer) {
        k_outs[layer] = rows.keys[layer].data;
        v_outs[layer] = rows.values[layer].data;
        row_strides[layer] = rows.keys[layer].row_stride;
      }
      cache.AppendPrefillChunk(k_outs, v_outs, row_strides, 0, 5);
    }
    const auto *keys = static_cast<const int8_t *>(cache.KMem(0).virAddr);
    assert(keys[static_cast<int64_t>(kCacheLen - 5) * kHeadDims[0]] ==
           text_fixture::KvByte(false, 0, 0));

    // Row-count guards around collect.
    assert(Throws([&] { CollectKvOutputs(prefill, 0); }));
    assert(Throws([&] { CollectKvOutputs(prefill, kChunkSize + 1); }));

    // Failure ownership: an inference error propagates, buffers stay owned
    // by the ModelIo, no task leaks, and the stage is retryable.
    const size_t buffers_before = text_fixture::state().buffers.size();
    text_fixture::state().fail_infer = true;
    assert(Throws([&] { RunSubgraphInference(prefill); }));
    text_fixture::state().fail_infer = false;
    assert(text_fixture::state().buffers.size() == buffers_before);
    assert(text_fixture::state().tasks.empty());
    RunSubgraphInference(prefill);  // Retry succeeds.
    assert(text_fixture::state().tasks.empty());
  }

  // ---- stage 2: descriptor contract still enforced at init ----
  {
    for (const char *mutation :
         {"embed_dtype", "token_dtype", "prefill_seq", "hidden", "mask_cols",
          "kv_head", "kv_row_pad", "kv_rows", "logit_dtype", "vocab", "quanti",
          "overlap", "unknown_type", "zero_seq"}) {
      text_fixture::Reset();
      text_fixture::state().mutate = mutation;
      assert(Throws([&] {
        InitTextSubgraph(text_fixture::Packed(), "prefill", kChunkSize);
      }));
      assert(text_fixture::state().buffers.empty());  // Nothing allocated.
    }
    text_fixture::Reset();
  }

  // ---- fixed-context geometry contract (TEXT-R1) ----
  {
    std::vector<float> window(static_cast<size_t>(kChunkSize) * kCacheLen);
    // The reviewer case: a chunk starting beyond the 4096-token context
    // used to drive cache_col_start negative and write before the buffer.
    assert(Throws([&] {
      PrepareBatchInputs(embeddings, {11}, 4096, 1, nullptr, kChunkSize);
    }));
    assert(Throws([&] { BuildFullMask(window.data(), 4096, 1, kChunkSize); }));
    assert(Throws([&] { BuildSlidingMask(window.data(), 4096, 1, kChunkSize); }));
    // Negative and oversized geometry, in every argument position.
    assert(Throws([&] { BuildFullMask(window.data(), -1, 1, kChunkSize); }));
    assert(Throws([&] {
      BuildFullMask(window.data(), 0, kChunkSize + 1, kChunkSize);
    }));
    assert(Throws([&] { BuildFullMask(window.data(), 0, 5, 0); }));
    assert(Throws([&] {
      BuildFullMask(window.data(), 3841, kChunkSize, kChunkSize);  // 3841+256 > 4096
    }));
    assert(Throws([&] { BuildFullMask(nullptr, 0, 1, kChunkSize); }));
    assert(Throws([&] { BuildSlidingMask(nullptr, 0, 1, kChunkSize); }));
    assert(Throws([&] { QuantizeMask(nullptr, nullptr, -1, 4); }));

    // The last full window of the context stays valid and writes in bounds.
    BuildFullMask(window.data(), 3840, kChunkSize, kChunkSize);
    assert(window[0] == 0.f);  // Row 0 attends column 0 at this offset.
    assert(window[3840] == 0.f);  // Row 0 attends through the prompt start+3840.
    assert(window[3841] == static_cast<int16_t>(gemma4::kMaskValue));
    std::vector<int64_t> tail(kChunkSize, 70);
    const auto tail_batch = PrepareBatchInputs(embeddings, tail, 3840,
                                               kChunkSize, nullptr, kChunkSize);
    assert(tail_batch.positions.front() == 3840);
    assert(tail_batch.positions.back() == 4095);

    // Prepared token count must match the valid chunk length.
    assert(Throws([&] {
      PrepareBatchInputs(embeddings, {11, 22}, 0, 1, nullptr, kChunkSize);
    }));
    // The prebuilt hidden must cover the prompt rows it indexes.
    std::vector<float> short_hidden(2 * kHiddenSize, 0.f);
    assert(Throws([&] {
      PrepareBatchInputs(embeddings, {11, gemma4::kImageTokenId, 33}, 0, 3,
                         &short_hidden, kChunkSize);
    }));

    // Decode positions share the same fixed context.
    assert(Throws([&] { PrepareDecodeInputs(embeddings, 100, kCacheLen); }));
    assert(Throws([&] { PrepareDecodeInputs(embeddings, 100, -1); }));
    const auto edge = PrepareDecodeInputs(embeddings, 100, kCacheLen - 1);
    assert(edge.positions[0] == kCacheLen - 1);  // Last valid row works.
  }

  // Malformed prepared batches are rejected by the write stage.
  {
    text_fixture::Reset();
    ModelIo prefill = InitTextSubgraph(text_fixture::Packed(), "prefill", kChunkSize);
    auto batch = PrepareBatchInputs(embeddings, {11, 22, 33, 44, 55}, 0, 5,
                                    nullptr, kChunkSize);
    batch.token_ids.resize(kChunkSize - 1);  // Corrupt the prepared count.
    assert(Throws([&] { WriteBatchInputs(prefill, batch); }));
    assert(text_fixture::state().infer_calls == 0);
  }

  assert(text_fixture::state().buffers.empty());
  hbDNNRelease(text_fixture::Packed());
  assert(text_fixture::state().models.empty());
  std::cout << "text stage checks passed\n";
  return 0;
}
