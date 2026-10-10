// R1 regression: the legacy runtime enforces a single-use lifecycle.
// Construction performs SDK initialization; a first successful request can
// never be followed by a second request on the same instance, which would
// silently never receive END.
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "minicpm5.hpp"

namespace {
std::string make_template() {
  auto path =
      (std::filesystem::temp_directory_path() / "minicpm5-t-XXXXXX").string();
  const int fd = mkstemp(path.data());
  if (fd < 0) throw std::runtime_error("mkstemp failed");
  ::close(fd);
  std::ofstream(path).write("template", 8);
  return path;
}

// Sink target shared by the scenario configs; predict() streams into it.
std::ostringstream& streamed() {
  static std::ostringstream buffer;
  return buffer;
}

// Streams the tokens predict() delivers through the injected sink.
std::string captured_predict(MiniCPM5& model, RequestOutcome& outcome) {
  streamed().str({});
  streamed().clear();
  outcome = model.predict();
  return streamed().str();
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
  const auto templ = make_template();
  // 1. SDK initialization failure throws from the constructor with the raw
  // status, so no model instance exists in an uninitialized state. The
  // destructor does not run for a throwing constructor, so a nonnull partial
  // handle returned alongside the error must be destroyed exactly once
  // before the exception propagates; a null handle is left untouched.
  {
    xlm_double::reset();
    xlm_double::init_status = 3;  // Error plus a nonnull partial handle.
    MiniCPM5Config config;
    config.template_path = templ;
    config.text_sink = [](const char* chunk) { streamed() << chunk; };
    bool threw = false;
    std::string message;
    try {
      MiniCPM5 model(config);
    } catch (const std::exception& error) {
      threw = true;
      message = error.what();
    }
    CHECK(threw && message.find("xlm_init failed: 3") != std::string::npos,
          "constructor surfaces init failure");
    CHECK(xlm_double::destroy_calls == 1,
          "partial handle destroyed exactly once");
    xlm_double::reset();
    xlm_double::init_status = 3;
    xlm_double::init_handle = false;  // Error with a null handle.
    try {
      MiniCPM5Config null_config;
      null_config.template_path = templ;
      MiniCPM5 model(null_config);
    } catch (const std::exception&) {
    }
    CHECK(xlm_double::destroy_calls == 0, "null handle not destroyed");
    xlm_double::reset();
  }
  // 2. First request streams text, ends normally and cleans up.
  {
    xlm_double::reset();
    MiniCPM5Config config;
    config.template_path = templ;
    config.text_sink = [](const char* chunk) { streamed() << chunk; };
    MiniCPM5 model(config);
    RequestOutcome outcome;
    const auto streamed = captured_predict(model, outcome);
    CHECK(outcome.exit_code() == 0, "first request exit zero");
    CHECK(outcome.ended && !outcome.failed, "first request ended");
    CHECK(outcome.sdk_status == 0 && outcome.destroy_status == 0,
          "first request statuses");
    CHECK(streamed == "你好，很高兴认识你。", "streamed text preserved");
    CHECK(xlm_double::capture.infer_calls == 1, "exactly one infer call");
    CHECK(xlm_double::destroy_calls == 1, "request teardown destroys handle");
  }
  // 3. A second predict on the completed instance is rejected; no second
  // request runs.
  {
    xlm_double::reset();
    MiniCPM5Config config;
    config.template_path = templ;
    config.text_sink = [](const char* chunk) { streamed() << chunk; };
    MiniCPM5 model(config);
    RequestOutcome first;
    captured_predict(model, first);
    CHECK(first.exit_code() == 0, "baseline request before reuse");
    bool predict_threw = false;
    try {
      model.predict();
    } catch (const std::exception&) {
      predict_threw = true;
    }
    CHECK(predict_threw, "predict after finalize rejected");
    CHECK(xlm_double::capture.infer_calls == 1,
          "no second request ever started");
  }
  // 4. A request that never receives END reports ended=0 and exit one.
  {
    xlm_double::reset();
    xlm_double::deliver_end = false;
    MiniCPM5Config config;
    config.template_path = templ;
    config.text_sink = [](const char* chunk) { streamed() << chunk; };
    MiniCPM5 model(config);
    RequestOutcome outcome;
    captured_predict(model, outcome);
    CHECK(outcome.exit_code() == 1 && !outcome.ended && !outcome.failed,
          "missing END reported");
  }
  // 5. An SDK ERROR state marks the request failed with exit one.
  {
    xlm_double::reset();
    xlm_double::deliver_error = true;
    MiniCPM5Config config;
    config.template_path = templ;
    config.text_sink = [](const char* chunk) { streamed() << chunk; };
    MiniCPM5 model(config);
    RequestOutcome outcome;
    captured_predict(model, outcome);
    CHECK(outcome.exit_code() == 1 && outcome.failed && !outcome.ended,
          "ERROR state reported");
  }
  // 6. Destroy failure keeps exit one and the raw status.
  {
    xlm_double::reset();
    xlm_double::destroy_status = 5;
    MiniCPM5Config config;
    config.template_path = templ;
    config.text_sink = [](const char* chunk) { streamed() << chunk; };
    MiniCPM5 model(config);
    RequestOutcome outcome;
    captured_predict(model, outcome);
    CHECK(outcome.exit_code() == 1 && outcome.destroy_status == 5,
          "destroy failure reported");
  }
  // 7. A nonzero infer status propagates with exit one.
  {
    xlm_double::reset();
    xlm_double::infer_status = 7;
    MiniCPM5Config config;
    config.template_path = templ;
    config.text_sink = [](const char* chunk) { streamed() << chunk; };
    MiniCPM5 model(config);
    RequestOutcome outcome;
    captured_predict(model, outcome);
    CHECK(outcome.exit_code() == 1 && outcome.sdk_status == 7,
          "infer status propagated");
  }
  std::filesystem::remove(templ);
  if (failures) {
    std::cout << failures << " lifecycle checks failed\n";
    return 1;
  }
  std::cout << "legacy_lifecycle OK\n";
  return 0;
}
