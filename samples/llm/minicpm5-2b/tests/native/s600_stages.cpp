// S600 stage-boundary checks: constructor validation, pre-process request
// construction, response/status/metric failure mapping and the preserved
// two-turn conversation behavior.
// Fixtures live in an atomically owned mkdtemp scratch directory; nothing
// outside that directory is read, written or removed.
#include <iostream>
#include <string>

#include "minicpm5.hpp"
#include "scratch_dir.hpp"

namespace {
template <typename Action>
bool throws(const Action& action, const std::string& needle) {
  try {
    action();
  } catch (const std::exception& error) {
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
  minicpm5_test::ScratchDir scratch("minicpm5-s600-stages");
  const auto model = minicpm5_test::make_model_dir(scratch.path());

  // Constructor validation: output limits stay bounded.
  CHECK(throws([&] { minicpm5::MiniCPM5 failed({model.string(), 0}); },
               "max_new_tokens"),
        "zero output limit rejected");
  CHECK(throws([&] { minicpm5::MiniCPM5 failed({model.string(), 4097}); },
               "max_new_tokens"),
        "oversize output limit rejected");
  {
    oellm::reset();
    minicpm5::MiniCPM5 bounds({model.string(), 1});
    bounds.Generate("hello");
    CHECK(oellm::captured_requests.front().max_new_tokens == 1,
          "limit one accepted and passed");
    oellm::reset();
  }

  // Pre-process: empty prompts and out-of-range limits never reach the
  // runtime, so direct public callers cannot bypass the constructor bounds.
  CHECK(throws([&] { minicpm5::pre_process("", true, 128, 1); }, "empty"),
        "empty prompt rejected in pre_process");
  CHECK(throws([&] { minicpm5::pre_process("hi", true, 0, 1); },
               "max_new_tokens"),
        "zero limit rejected in pre_process");
  CHECK(throws([&] { minicpm5::pre_process("hi", true, 4097, 1); },
               "max_new_tokens"),
        "oversize limit rejected in pre_process");
  {
    const auto bounded_low = minicpm5::pre_process("hi", true, 1, 1);
    const auto bounded_high = minicpm5::pre_process("hi", true, 4096, 2);
    CHECK(bounded_low.oellm_requests.front().max_new_tokens == 1 &&
              bounded_high.oellm_requests.front().max_new_tokens == 4096,
          "boundary limits accepted in pre_process");
  }
  {
    const auto request = minicpm5::pre_process("hi", false, 32, 7);
    CHECK(request.oellm_requests.size() == 1, "single request built");
    const auto& item = request.oellm_requests.front();
    CHECK(item.request_id == 7 && item.conversation_id == 1 && !item.new_chat &&
              item.max_new_tokens == 32 && item.prompt.user_prompt.text == "hi",
          "request fields carried");
  }

  // Inference and post-processing failures map to named errors.
  {
    oellm::reset();
    minicpm5::MiniCPM5 instance({model.string(), 4});
    CHECK(throws([&] { instance.Generate(""); }, "empty"),
          "Generate rejects empty prompt");
    CHECK(oellm::captured_requests.empty(),
          "rejected prompt never reaches the runtime");

    oellm::infer_result = oellm::OellmErrorCode::kInferFailed;
    CHECK(throws([&] { instance.Generate("hello"); }, "inference failed"),
          "infer error return throws");
    oellm::reset();

    oellm::response_count = 2;
    CHECK(throws([&] { instance.Generate("hello"); }, "inference failed"),
          "malformed response shape throws");
    oellm::reset();

    oellm::null_first_response = true;
    CHECK(throws([&] { instance.Generate("hello"); }, "inference failed"),
          "null response throws");
    oellm::reset();

    oellm::status = oellm::OellmStatus::kAborted;
    CHECK(throws([&] { instance.Generate("hello"); },
                 "Unexpected generation status"),
          "unexpected status throws");
    oellm::reset();

    oellm::metric_result = oellm::OellmErrorCode::kMetricFailed;
    CHECK(throws([&] { instance.Generate("hello"); }, "metrics"),
          "metric failure throws");
    oellm::reset();

    oellm::metric_count = 2;
    CHECK(throws([&] { instance.Generate("hello"); }, "metrics"),
          "unexpected metric count throws");
    oellm::reset();
  }

  // Two-turn conversation: request ids advance and new_chat follows the flag.
  {
    oellm::reset();
    minicpm5::MiniCPM5 chat({model.string(), 128});
    const auto first = chat.Generate("What is 1+1?");
    const auto second = chat.Generate("Translate that into Chinese.", false);
    CHECK(oellm::captured_requests.size() == 2, "two requests captured");
    CHECK(oellm::captured_requests[0].request_id == 1 &&
              oellm::captured_requests[0].new_chat,
          "first request opens the conversation");
    CHECK(oellm::captured_requests[1].request_id == 2 &&
              !oellm::captured_requests[1].new_chat,
          "follow-up keeps the conversation");
    CHECK(oellm::captured_requests[0].conversation_id == 1 &&
              oellm::captured_requests[1].conversation_id == 1,
          "conversation id preserved");
    CHECK(first.status == 3 && second.status == 3, "both requests finished");
    oellm::reset();
  }
  if (failures) {
    std::cout << failures << " stage checks failed\n";
    return 1;
  }
  std::cout << "s600_stages OK\n";
  return 0;
}
