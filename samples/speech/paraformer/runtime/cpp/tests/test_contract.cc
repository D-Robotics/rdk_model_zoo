// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "pipeline.hpp"
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

template <class F> void rejects(F call) {
  bool rejected = false;
  try {
    call();
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
}

int main() {
  std::vector<float> weights(401, 0.f), hidden(401 * 512, 0.f);
  auto empty = paraformer::cif(weights, hidden, 400);
  assert(empty.token_count == 0 && empty.acoustic.size() == 100 * 512);
  for (float value : empty.acoustic)
    assert(value == 0.f);
  weights[0] = .75f;
  weights[1] = .75f;
  weights[2] = .5f;
  for (size_t h = 0; h < 512; ++h) {
    hidden[h] = 2;
    hidden[512 + h] = 6;
    hidden[1024 + h] = 10;
  }
  const auto before = weights;
  auto fractional = paraformer::cif(weights, hidden, 3);
  assert(fractional.token_count == 2 && weights == before);
  for (size_t h = 0; h < 512; ++h) {
    assert(fractional.acoustic[h] == 3.f);
    assert(fractional.acoustic[512 + h] == 8.f);
  }
  weights.assign(401, 0.f);
  weights[400] = 1.f;
  assert(paraformer::cif(weights, hidden, 400).token_count == 0);
  weights.assign(401, 1.f);
  for (size_t t = 0; t < 401; ++t)
    for (size_t h = 0; h < 512; ++h)
      hidden[t * 512 + h] = float(t);
  auto capped = paraformer::cif(weights, hidden, 400);
  assert(capped.token_count == 100);
  for (size_t t = 0; t < 100; ++t)
    assert(capped.acoustic[t * 512] == float(t));
  rejects([&] { paraformer::cif(weights, hidden, -1); });
  rejects([&] { paraformer::cif(weights, hidden, 401); });
  rejects([&] { paraformer::cif({}, hidden, 400); });
  weights[0] = -1.f;
  rejects([&] { paraformer::cif(weights, hidden, 400); });
  weights[0] = std::numeric_limits<float>::quiet_NaN();
  rejects([&] { paraformer::cif(weights, hidden, 400); });

  std::vector<std::string> vocabulary;
  for (int i = 0; i < 8404; ++i)
    vocabulary.push_back("token" + std::to_string(i));
  vocabulary[0] = "<blank>";
  vocabulary[1] = "中";
  vocabulary[2] = "文@@";
  vocabulary[3] = "</s>";
  vocabulary[4] = "a@@@@b";
  std::vector<float> logits(100 * 8404, 0.f);
  logits[1] = 1;
  logits[8404 + 1] = 1;
  logits[2 * 8404 + 2] = 1;
  logits[3 * 8404 + 3] = 1;
  logits[4 * 8404 + 4] = 1;
  auto decoded = paraformer::decode(logits, 5, vocabulary);
  assert(decoded.text == "中中文ab");
  assert((decoded.token_ids == std::vector<int>{1, 1, 2, 3, 4}));
  logits[2] = 1.f;
  assert((paraformer::decode(logits, 1, vocabulary).token_ids ==
          std::vector<int>{1})); // Equal scores retain the lowest token ID.
  assert(paraformer::decode(logits, 0, vocabulary).text.empty());
  rejects([&] { paraformer::decode(logits, 101, vocabulary); });
  rejects([&] { paraformer::decode({}, 1, vocabulary); });
  logits.back() = std::numeric_limits<float>::infinity();
  rejects([&] { paraformer::decode(logits, 1, vocabulary); });
  logits.back() = 0;
  vocabulary[0] = vocabulary[1];
  rejects([&] { paraformer::decode(logits, 1, vocabulary); });
}
