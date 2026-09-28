// Legacy request-stage checks: template loading limits, request construction
// and the preserved greedy sampling parameters sent to the SDK.
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "chat_template.hpp"
#include "minicpm5.hpp"

namespace {
std::string write_bytes(const std::string& contents) {
  auto path =
      (std::filesystem::temp_directory_path() / "minicpm5-j-XXXXXX").string();
  const int fd = mkstemp(path.data());
  if (fd < 0) throw std::runtime_error("mkstemp failed");
  ::close(fd);
  std::ofstream stream(path, std::ios::binary);
  stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  return path;
}

template <typename Action>
bool throws(const Action& action) {
  try {
    action();
  } catch (const std::exception&) {
    return true;
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
  // Template loader: missing, empty and oversize files are rejected.
  CHECK(throws([] { load_chat_template("/nonexistent/minicpm5-template"); }),
        "missing template rejected");
  const auto empty_file = write_bytes("");
  CHECK(throws([&] { load_chat_template(empty_file); }),
        "empty template rejected");
  std::filesystem::remove(empty_file);
  const auto oversize =
      write_bytes(std::string(kMaxChatTemplateBytes + 1, 'x'));
  CHECK(throws([&] { load_chat_template(oversize); }),
        "oversize template rejected");
  std::filesystem::remove(oversize);
  const auto max_size = write_bytes(std::string(kMaxChatTemplateBytes, 'j'));
  const auto loaded = load_chat_template(max_size);
  CHECK(loaded.size() == kMaxChatTemplateBytes, "max-size template accepted");
  std::filesystem::remove(max_size);

  // prepare_request: fixed single-request fields, owned strings wired in.
  {
    const auto prepared = prepare_request("介绍你自己", "{chat}");
    CHECK(prepared.request.request_id == 0, "request id zero");
    CHECK(prepared.request.type == XLM_INPUT_PROMPT, "request type prompt");
    CHECK(prepared.request.new_chat, "request opens a new chat");
    CHECK(prepared.request.infer_backend == XLM_INFER_BACKEND_BPU_ANY,
          "request backend");
    CHECK(prepared.request.prompt == prepared.prompt, "prompt wired");
    CHECK(prepared.request.chat_template == prepared.chat_template,
          "template wired");
    CHECK(std::string(prepared.request.prompt) == "介绍你自己",
          "prompt content");
    CHECK(prepared.input.request_num == 1, "one request per input");
    CHECK(prepared.input.requests == &prepared.request, "input views request");
  }

  // Greedy, non-thinking sampling parameters survive into xlm_init, and the
  // prepared request fields reach xlm_infer unchanged.
  {
    xlm_double::reset();
    const auto templ = write_bytes("template");
    MiniCPM5Config config;
    config.model_path = "/models/s100/model.hbm";
    config.tokenizer_path = "/models/s100/tokenizer";
    config.template_path = templ;
    MiniCPM5 model(config);
    model.init();
    const auto outcome = model.predict();
    CHECK(outcome.exit_code() == 0, "captured request completes");
    const auto& seen = xlm_double::capture;
    CHECK(seen.model_type == XLM_MODEL_TYPE_DEEPSEEK, "model type preserved");
    CHECK(seen.context_size == 4096, "context size preserved");
    CHECK(seen.temp == 0.0f && seen.top_k == 1 && seen.top_p == 1.0f &&
              seen.min_p == 0.0f && seen.penalty_repeat == 1.0f &&
              seen.penalty_freq == 0.0f && seen.penalty_present == 0.0f,
          "greedy sampling preserved");
    CHECK(seen.model_path == "/models/s100/model.hbm", "model path passed");
    CHECK(seen.token_config_path == "/models/s100/tokenizer",
          "tokenizer path passed");
    CHECK(seen.request_num == 1 && seen.request_id == 0 && seen.new_chat &&
              seen.request_type == XLM_INPUT_PROMPT &&
              seen.infer_backend == XLM_INFER_BACKEND_BPU_ANY,
          "request fields reached the SDK");
    CHECK(seen.prompt == "请用一句话介绍你自己。", "default prompt passed");
    CHECK(seen.chat_template == "template", "template content passed");
    CHECK(seen.infer_userdata == &model, "instance receives callback state");
    std::filesystem::remove(templ);
  }
  if (failures) {
    std::cout << failures << " request-stage checks failed\n";
    return 1;
  }
  std::cout << "legacy_request OK\n";
  return 0;
}
