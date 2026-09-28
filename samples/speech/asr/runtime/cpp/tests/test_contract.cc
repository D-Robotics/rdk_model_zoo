#include "contract.h"
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
template <class F> void rejected(F f) {
  bool caught = false;
  try {
    f();
  } catch (const std::invalid_argument &) {
    caught = true;
  }
  assert(caught);
}
int main() {
  std::vector<std::string> vocab{"<pad>", "a", "b"};
  assert(asr::decode_ids({1, 1, 0, 1, 2, 2}, vocab) == "aab");
  assert(asr::decode_ids({1, 1, 0, 1, 2, 2}, vocab, asr::DecodeMode::Legacy) ==
         "aaabb");
  assert(asr::decode_ids({0, 0}, vocab).empty());
  assert(asr::decode_ids({}, vocab).empty());
  assert(asr::decode_logits({-3e35f, -2e35f, -3e35f, -3e35f, -2e35f, -2e35f}, 2,
                            vocab) == "a");
  rejected([&] { asr::decode_ids({3}, vocab); });
  rejected([&] { asr::decode_logits({0, 0}, 1, vocab); });
  rejected([&] {
    asr::decode_logits({0, std::numeric_limits<float>::quiet_NaN(), 0}, 1,
                       vocab);
  });
  rejected([&] { asr::decode_ids({1}, {"<pad>", "a", "a"}); });
  assert(asr::source_chunk_size(44100) == 82688);
  auto prepared = asr::normalize_and_pad({1, 2, 3});
  assert(prepared.values.size() == 30000 && prepared.valid_samples == 3);
  const float expected = -1 / std::sqrt(2.0f / 3 + 1e-5f);
  assert(std::abs(prepared.values[0] - expected) < 1e-6f);
  assert(prepared.values[3] == 0);
  auto constant = asr::normalize_and_pad({.5f, .5f});
  for (float value : constant.values)
    assert(value == 0);
  rejected([&] { asr::normalize_and_pad({}); });
  rejected([&] {
    asr::normalize_and_pad({std::numeric_limits<float>::infinity()});
  });
  rejected([&] { asr::source_chunk_size(0); });
}
