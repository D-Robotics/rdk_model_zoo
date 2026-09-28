// R2 regression: non-finite or negative runtime measurements are rejected by
// name, zero throughput stays valid, and valid metrics pass through.
// Fixtures live in an atomically owned mkdtemp scratch directory; nothing
// outside that directory is read, written or removed.
#include <cmath>
#include <iostream>
#include <string>

#include "minicpm5.hpp"
#include "scratch_dir.hpp"

namespace {
template <typename Action>
bool throws(const Action& action, const std::string& needle,
            std::string* message = nullptr) {
  try {
    action();
  } catch (const std::exception& error) {
    if (message) *message = error.what();
    return std::string(error.what()).find(needle) != std::string::npos;
  }
  return false;
}
}  // namespace

int failures = 0;
#define CHECK(condition, step)              \
  do {                                      \
    if (!(condition)) {                     \
      std::cout << "FAIL " << step << "\n"; \
      ++failures;                           \
    }                                       \
  } while (0)

int main() {
  minicpm5_test::ScratchDir scratch("minicpm5-s600-metrics");
  const auto model = minicpm5_test::make_model_dir(scratch.path());

  // Pure validation: one message per metric, zero preserved, coercion banned.
  {
    std::string message;
    CHECK(throws([&] { minicpm5::validate_metrics(NAN, 53.25, 100.0); }, "TTFT",
                 &message),
          "non-finite TTFT rejected by name");
    CHECK(
        throws([&] { minicpm5::validate_metrics(-1.0, 53.25, 100.0); }, "TTFT"),
        "negative TTFT rejected");
    CHECK(throws([&] { minicpm5::validate_metrics(12.5, NAN, 100.0); },
                 "decode_tps"),
          "non-finite decode_tps rejected by name");
    CHECK(throws([&] { minicpm5::validate_metrics(12.5, -1.0, 100.0); },
                 "decode_tps"),
          "negative decode_tps rejected");
    CHECK(throws([&] { minicpm5::validate_metrics(12.5, 53.25, NAN); }, "e2e"),
          "non-finite e2e rejected by name");
    CHECK(throws([&] { minicpm5::validate_metrics(12.5, 53.25, -0.5); }, "e2e"),
          "negative e2e rejected");
    CHECK(throws([&] { minicpm5::validate_metrics(12.5, INFINITY, 100.0); },
                 "decode_tps"),
          "infinite decode_tps rejected");
    bool threw = false;
    try {
      minicpm5::validate_metrics(0.0, 0.0, 100.0);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(!threw, "zero measurements preserved");
  }

  // Full Generate path: invalid metrics fail the request instead of being
  // serialized as null JSON.
  {
    oellm::reset();
    minicpm5::MiniCPM5 valid({model.string(), 4});
    const auto result = valid.Generate("hello");
    CHECK(result.text == "fixture text" && result.tokens.size() == 3,
          "payload carried through");
    CHECK(result.status == 3, "normal-finish status carried");
    CHECK(std::abs(result.ttft_ms - 12.5) < 1e-9 &&
              std::abs(result.decode_tps - 53.25) < 1e-9 &&
              std::abs(result.e2e_ms - 100.0) < 1e-9,
          "valid metrics carried through");

    oellm::ttft = NAN;
    std::string message;
    CHECK(throws([&] { valid.Generate("hello"); }, "TTFT", &message),
          "Generate rejects non-finite TTFT");
    oellm::ttft = 12.5;
    oellm::decode_tps = -1.0;
    CHECK(throws([&] { valid.Generate("hello"); }, "decode_tps"),
          "Generate rejects negative decode_tps");
    oellm::decode_tps = 53.25;
    oellm::e2e = INFINITY;
    CHECK(throws([&] { valid.Generate("hello"); }, "e2e"),
          "Generate rejects non-finite e2e");

    // Documented one-token length-limited case: zero decode throughput and
    // length-limit status 6 stay a successful bounded request.
    oellm::reset();
    oellm::status = oellm::OellmStatus::kLengthFinished;
    oellm::decode_tps = 0.0;
    oellm::ttft = 0.0;
    const auto limited = valid.Generate("hello");
    CHECK(limited.status == 6 && limited.decode_tps == 0.0,
          "one-token length-limited request stays valid");
    oellm::reset();
  }
  if (failures) {
    std::cout << failures << " metric checks failed\n";
    return 1;
  }
  std::cout << "s600_metrics OK\n";
  return 0;
}
