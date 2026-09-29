// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "asr.h"
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
  asr::ASR task(runner, 4, vocabulary);
  const asr::AudioChunk audio{{.1f, .2f, .3f}, 16000, 1, 0, 0};
  auto prepared = task.pre_process(audio);
  assert(calls == 0 && prepared.valid_samples == 3);
  auto raw = task.forward(prepared);
  assert(calls == 1 && raw == logits);
  assert(task.post_process(raw) == "AA" && calls == 1);
  logits.assign(logits.size(), 0.f);
  assert(task.post_process(raw) == "AA");
  assert(task.predict(audio).empty() && calls == 2);
  asr::ASR legacy(runner, 4, vocabulary, asr::DecodeMode::Legacy);
  logits[5] = 1;
  logits[3503 + 5] = 1;
  logits[3 * 3503 + 5] = 1;
  assert(legacy.predict(audio) == "AAA");
  bool failed = false;
  try {
    task.forward(asr::PreparedChunk{{1}, 1});
  } catch (const std::invalid_argument &) {
    failed = true;
  }
  assert(failed && calls == 3);
  failed = false;
  prepared.values[0] = std::numeric_limits<float>::infinity();
  try {
    task.forward(prepared);
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
