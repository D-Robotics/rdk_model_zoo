// CORE-R3 regression: streaming output goes through an injected sink (the
// library performs no console IO), with END/ERROR text suppression kept and
// consumer exceptions contained inside the vendor callback boundary.
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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

  // 1. A custom sink receives the streamed chunks; no library console IO.
  {
    xlm_double::reset();
    std::vector<std::string> chunks;
    MiniCPM5Config config;
    config.template_path = templ;
    config.text_sink = [&](const char* chunk) { chunks.emplace_back(chunk); };
    MiniCPM5 model(config);
    const auto outcome = model.predict();
    CHECK(outcome.exit_code() == 0, "sink request succeeds");
    CHECK(chunks.size() == 1 && chunks.front() == "你好，很高兴认识你。",
          "custom sink receives the chunk");
    CHECK(xlm_double::capture.infer_calls == 1, "one infer call");
  }
  // 2. Empty sink: the library consumes silently (no console required).
  {
    xlm_double::reset();
    MiniCPM5Config config;
    config.template_path = templ;
    MiniCPM5 model(config);
    const auto outcome = model.predict();
    CHECK(outcome.exit_code() == 0 && outcome.ended && !outcome.stream_error,
          "silent request succeeds");
  }
  // 3. END-state text stays suppressed, exactly as in the source.
  {
    xlm_double::reset();
    xlm_double::end_text = "END-CARRIED";
    std::vector<std::string> chunks;
    MiniCPM5Config config;
    config.template_path = templ;
    config.text_sink = [&](const char* chunk) { chunks.emplace_back(chunk); };
    MiniCPM5 model(config);
    const auto outcome = model.predict();
    CHECK(outcome.exit_code() == 0 && outcome.ended, "END with text finishes");
    CHECK(chunks.size() == 1 && chunks.front() == "你好，很高兴认识你。",
          "END-carried text suppressed");
  }
  // 4. ERROR-state text stays suppressed and the request still fails.
  {
    xlm_double::reset();
    xlm_double::deliver_error = true;
    xlm_double::error_text = "ERROR-CARRIED";
    std::vector<std::string> chunks;
    MiniCPM5Config config;
    config.template_path = templ;
    config.text_sink = [&](const char* chunk) { chunks.emplace_back(chunk); };
    MiniCPM5 model(config);
    const auto outcome = model.predict();
    CHECK(outcome.exit_code() == 1 && outcome.failed && !outcome.ended,
          "ERROR state reported");
    CHECK(chunks.empty(), "ERROR-carried text suppressed");
  }
  // 5. A throwing consumer is contained: no exception crosses the vendor
  // callback, the failure is recorded once, streaming stops, and the
  // lifecycle and status mapping are unchanged.
  {
    xlm_double::reset();
    xlm_double::running_text = "你好，很高兴认识你。";
    int sink_calls = 0;
    MiniCPM5Config config;
    config.template_path = templ;
    config.text_sink = [&](const char*) {
      ++sink_calls;
      throw std::runtime_error("consumer exploded");
    };
    MiniCPM5 model(config);
    bool threw = false;
    RequestOutcome outcome;
    try {
      outcome = model.predict();
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(!threw, "consumer exception contained");
    CHECK(sink_calls == 1, "streaming stops after the consumer failure");
    CHECK(outcome.ended && !outcome.failed && outcome.sdk_status == 0 &&
              outcome.destroy_status == 0,
          "status mapping unchanged through consumer failure");
    CHECK(outcome.exit_code() == 0, "exit code follows source mapping");
    CHECK(outcome.stream_error, "stream failure recorded in the outcome");
  }
  // 6. Multiple chunks stream incrementally before END within one call.
  {
    xlm_double::reset();
    xlm_double::running_text = "A";
    xlm_double::running_chunks = 2;
    std::vector<std::string> chunks;
    MiniCPM5Config config;
    config.template_path = templ;
    config.text_sink = [&](const char* chunk) { chunks.emplace_back(chunk); };
    MiniCPM5 model(config);
    const auto outcome = model.predict();
    CHECK(outcome.exit_code() == 0, "multi-chunk request finishes");
    CHECK(chunks.size() == 2 && chunks.front() == "A",
          "all chunks delivered incrementally");
  }
  std::filesystem::remove(templ);
  if (failures) {
    std::cout << failures << " stream-sink checks failed\n";
    return 1;
  }
  std::cout << "legacy_stream_sink OK\n";
  return 0;
}
