// Gemma4-E2B Text: explicit pipeline stages and the high-level session.
//
// Host check build: the Horizon SDK entry points are host doubles from
// text_fixture.hpp (deterministic logits/KV patterns, no BPU). On a board
// the same code links the real libdnn/libhbucp and loads the prepared HBM;
// only the model acquisition line differs.
//
// Expected output (asserted by the host check):
//   stage first token: 104
//   session out: 11 22 33 44 55 104 100
//   session processed: 7
#include "text_fixture.hpp"

#include <iostream>
#include <vector>

int main() {
  // ---- Stage composition: prepare -> transport -> decode/KV update ----
  hbDNNPackedHandle_t packed = text_fixture::Packed();
  // Board: hbDNNInitializeFromFiles(&packed, paths, 1) instead.
  gemma4::TokenEmbeddings embeddings("tok_embeddings.bin");

  gemma4::ModelIo prefill =
      gemma4::InitTextSubgraph(packed, "prefill", gemma4::kChunkSize);
  gemma4::ModelIo decode =
      gemma4::InitTextSubgraph(packed, "decode", 1);
  gemma4::KvCache cache;
  // KV input slots borrow the cache's buffers; the cache owns the memory.
  gemma4::BindKvCache(prefill, decode, cache);

  const std::vector<int64_t> prompt = {11, 22, 33, 44, 55};
  // Stage 1: prepared per-call context (ids, embeddings, positions, masks).
  const gemma4::TextBatchInputs batch = gemma4::PrepareBatchInputs(
      embeddings, prompt, 0, static_cast<int>(prompt.size()), nullptr,
      gemma4::kChunkSize);
  // Stage 2: strided write, then one selective-flush inference.
  gemma4::WriteBatchInputs(prefill, batch);
  gemma4::RunSubgraphInference(prefill);
  // Stage 3: append the validated KV rows, then decode greedily.
  const gemma4::TextKvOutputSet rows =
      gemma4::CollectKvOutputs(prefill, static_cast<int>(prompt.size()));
  {
    const int8_t *k[gemma4::kNumKvLayers];
    const int8_t *v[gemma4::kNumKvLayers];
    int64_t strides[gemma4::kNumKvLayers];
    for (int layer = 0; layer < gemma4::kNumKvLayers; ++layer) {
      k[layer] = rows.keys[layer].data;
      v[layer] = rows.values[layer].data;
      strides[layer] = rows.keys[layer].row_stride;
    }
    cache.AppendPrefillChunk(k, v, strides, 0,
                             static_cast<int>(prompt.size()));
  }
  const int64_t first = gemma4::ArgmaxTextLogits(
      prefill.outputs[0], 4, prefill.seq_len, prefill.OutputCapacity(0));
  std::cout << "stage first token: " << first << std::endl;
  // Clear the subgraph owners before releasing the packed model.
  prefill.Clear();
  decode.Clear();

  // ---- High-level session: the same stages orchestrated multi-turn ----
  gemma4::TextEngine engine("text.hbm", "tok_embeddings.bin");
  // Diagnostics are opt-in; without a sink the engine prints nothing.
  engine.SetDebugSink([](const std::string &message) {
    std::cerr << "[gemma4] " << message << std::endl;
  });
  const auto out = engine.Generate(prompt, 2);
  std::cout << "session out:";
  for (int64_t token : out) std::cout << ' ' << token;
  std::cout << std::endl;
  const auto next = engine.ContinueGenerate(out, 1);
  (void)next;
  std::cout << "session processed: " << engine.ProcessedTokens() << std::endl;
  engine.ResetSession();
  // Self-check for the host run: these hold against the fixture doubles and
  // are the documented outputs of this example.
  bool ok = first == 104 && engine.ProcessedTokens() == 0 &&
            out.size() == prompt.size() + 2 && next.size() == out.size() + 1;
  return ok ? 0 : 1;
}
