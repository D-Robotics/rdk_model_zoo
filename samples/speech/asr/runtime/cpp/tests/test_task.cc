// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "asr.hpp"
#include <cassert>
#include <limits>
int main() {
  std::vector<std::string> vocabulary{"<pad>"};
  for (size_t i = 1; i < 3503; ++i)
    vocabulary.push_back("token" + std::to_string(i));
  vocabulary[5] = "A";
  std::vector<float> logits(4 * 3503, 0.f);
  logits[5] = 1;
  logits[3503 + 5] = 1;
  logits[3 * 3503 + 5] = 1;
  int calls = 0;
  asr::Runner runner = [&](const std::vector<float> &input) {
    assert(input.size() == 30000);
    ++calls;
    return logits;
  };
  // Explicit Ctc: the runtime default has been Legacy since 6c5cd7f2, and the
  // collapse semantics asserted below belong to the Ctc decoder.
  asr::ASR task(runner, 4, vocabulary, asr::DecodeMode::Ctc);
  const asr::AudioChunk audio{{.1f, .2f, .3f}, 16000, 1, 0, 0};
  auto prepared = task.preprocess(audio);
  assert(calls == 0 && prepared.valid_samples == 3);
  auto raw = task.infer(prepared);
  assert(calls == 1 && raw == logits);
  assert(task.postprocess(raw) == "AA" && calls == 1);
  logits.assign(logits.size(), 0.f);
  assert(task.postprocess(raw) == "AA");
  assert(task.predict(audio).text.empty() && calls == 2);
  asr::ASR legacy(runner, 4, vocabulary, asr::DecodeMode::Legacy);
  logits[5] = 1;
  logits[3503 + 5] = 1;
  logits[3 * 3503 + 5] = 1;
  const auto legacy_result = legacy.predict(audio);
  assert(legacy_result.text == "AAA" && legacy_result.valid_samples == 3);
  bool failed = false;
  try {
    task.infer(asr::PreparedChunk{{1}, 1});
  } catch (const std::invalid_argument &) {
    failed = true;
  }
  assert(failed && calls == 3);
  failed = false;
  prepared.values[0] = std::numeric_limits<float>::infinity();
  try {
    task.infer(prepared);
  } catch (const std::invalid_argument &) {
    failed = true;
  }
  assert(failed && calls == 3);
  failed = false;
  try {
    asr::ASR invalid(runner, 0, vocabulary);
  } catch (const std::invalid_argument &) {
    failed = true;
  }
  assert(failed);
  failed = false;
  try {
    asr::ASR invalid({}, 4, vocabulary);
  } catch (const std::invalid_argument &) {
    failed = true;
  }
  assert(failed);
}
