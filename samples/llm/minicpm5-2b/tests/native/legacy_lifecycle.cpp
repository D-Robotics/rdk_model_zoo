// R1 regression: the legacy runtime enforces a single-use lifecycle.
// Scenario order matches the reviewer baseline reproduction: a first
// successful request must be followed by rejected reinitialization instead of
// a second request that silently never receives END.
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
  // 1. predict() before init() throws.
  {
    MiniCPM5Config config;
    config.template_path = templ;
    config.text_sink = [](const char* chunk) { streamed() << chunk; };
    MiniCPM5 model(config);
    bool threw = false;
    try {
      model.predict();
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw, "predict before init throws");
  }
  // 2. First request streams text, ends normally and cleans up.
  {
    xlm_double::reset();
    MiniCPM5Config config;
    config.template_path = templ;
    config.text_sink = [](const char* chunk) { streamed() << chunk; };
    MiniCPM5 model(config);
    model.init();
    RequestOutcome outcome;
    const auto streamed = captured_predict(model, outcome);
    CHECK(outcome.exit_code() == 0, "first request exit zero");
    CHECK(outcome.ended && !outcome.failed, "first request ended");
    CHECK(outcome.sdk_status == 0 && outcome.destroy_status == 0,
          "first request statuses");
    CHECK(streamed == "你好，很高兴认识你。", "streamed text preserved");
    CHECK(xlm_double::capture.infer_calls == 1, "exactly one infer call");
  }
  // 3. Reinitialization after predict is rejected; no second request runs.
  {
    xlm_double::reset();
    MiniCPM5Config config;
    config.template_path = templ;
    config.text_sink = [](const char* chunk) { streamed() << chunk; };
    MiniCPM5 model(config);
    model.init();
    RequestOutcome first;
    captured_predict(model, first);
    CHECK(first.exit_code() == 0, "baseline request before reinit");
    bool init_threw = false;
    try {
      model.init();
    } catch (const std::exception&) {
      init_threw = true;
    }
    CHECK(init_threw, "init after predict rejected");
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
    model.init();
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
    model.init();
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
    model.init();
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
    model.init();
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
