// Host callbacks are synthetic model boundaries, not SDK inference.
#include "pipeline.h"
#include <cassert>
#include <stdexcept>

int main() {
  std::vector<std::string> vocabulary;
  for (int i = 0; i < 8404; ++i)
    vocabulary.push_back("token" + std::to_string(i));
  vocabulary[3] = "中";
  vocabulary[4] = "文@@";
  vocabulary[5] = "</s>";
  std::vector<std::string> calls;
  bool empty = false;
  bool malformed_context = false;
  paraformer::Pipeline pipeline(
      [&](const std::vector<float> &features) {
        calls.push_back("encoder");
        assert(features[0] == 7.f);
        return std::vector<float>(malformed_context ? 1 : 400 * 512, 3.f);
      },
      [&](const std::vector<float> &context) {
        calls.push_back("predictor");
        assert(context[0] == 3.f);
        paraformer::PredictorOutput out{std::vector<float>(401, 0.f),
                                        std::vector<float>(401 * 512, 2.f)};
        if (!empty)
          for (int i = 0; i < 4; ++i)
            out.weights[i] = 1.f;
        out.weights[400] =
            1.f; // Must be masked regardless of other activations.
        return out;
      },
      [&](const paraformer::DecoderInput &input) {
        calls.push_back("decoder");
        assert(input.context[0] == 3.f);
        assert(input.token_count == 4);
        assert(input.acoustic[0] == 2.f && input.acoustic[4 * 512] == 0.f);
        for (float value : input.bias)
          assert(value == 0.f);
        std::vector<float> logits(100 * 8404, 0.f);
        const int ids[] = {3, 3, 4, 5};
        for (int i = 0; i < 4; ++i)
          logits[i * 8404 + ids[i]] = 1.f;
        return logits;
      },
      vocabulary);
  std::vector<float> features(400 * 560, 0.f);
  features[0] = 7.f;
  const auto prediction = pipeline.predict(features, 4);
  assert(
      (calls == std::vector<std::string>{"encoder", "predictor", "decoder"}));
  assert(prediction.text == "中中文" && prediction.token_count == 4);
  assert(prediction.decoder_executed &&
         prediction.timings.decoder_ms.has_value());
  assert(features[0] == 7.f);
  calls.clear();
  empty = true;
  const auto zero = pipeline.predict(features, 4);
  assert((calls == std::vector<std::string>{"encoder", "predictor"}));
  assert(zero.text.empty() && zero.token_ids.empty() && zero.token_count == 0);
  assert(!zero.decoder_executed && !zero.timings.decoder_ms.has_value());
  calls.clear();
  bool rejected = false;
  try {
    pipeline.predict(features, 0);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected && calls.empty());
  malformed_context = true;
  rejected = false;
  try {
    pipeline.predict(features, 4);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected && calls == std::vector<std::string>{"encoder"});
}
