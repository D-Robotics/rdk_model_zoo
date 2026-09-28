// R3 regression: the temporary OELLM configuration file is removed on the
// success, SDK-error-return and injected-exception paths alike.
// Each scenario gets an atomically owned mkdtemp scratch directory; TMPDIR is
// restored afterwards, including an initially unset state, and nothing outside
// the owned directories is read, written or removed.
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "minicpm5.hpp"
#include "runtime_config.hpp"
#include "scratch_dir.hpp"

namespace {
template <typename Action>
bool throws(const Action& action) {
  try {
    action();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

// Counts temporary minicpm5-* files in the directory used by the runtime.
std::size_t temp_files(const std::filesystem::path& scratch) {
  std::size_t count = 0;
  for (const auto& item : std::filesystem::directory_iterator(scratch)) {
    if (item.path().filename().string().rfind("minicpm5-", 0) == 0) ++count;
  }
  return count;
}

// Snapshot of the process TMPDIR state, including "unset".
std::string tmpdir_state() {
  const char* value = std::getenv("TMPDIR");
  return value ? std::string("set:") + value : std::string("unset");
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
  const std::string initial_tmpdir = tmpdir_state();

  // 1. Successful construction leaves no temporary configuration file behind.
  {
    minicpm5_test::ScratchDir scratch("minicpm5-s600-cleanup");
    const auto model = minicpm5_test::make_model_dir(scratch.path());
    minicpm5::MiniCPM5 model_handle({model.string(), 4});
    CHECK(temp_files(scratch.path()) == 0, "success path leaves no temp file");
    CHECK(!oellm::init_config_path.empty() &&
              !std::filesystem::exists(oellm::init_config_path),
          "runtime received the removed config path");
  }
  // 2. An SDK error return from Init leaves no temporary file.
  {
    minicpm5_test::ScratchDir scratch("minicpm5-s600-cleanup");
    const auto model = minicpm5_test::make_model_dir(scratch.path());
    oellm::init_result = oellm::OellmErrorCode::kInitFailed;
    CHECK(throws([&] { minicpm5::MiniCPM5 failed({model.string(), 4}); }),
          "Init error return throws");
    CHECK(temp_files(scratch.path()) == 0, "error path leaves no temp file");
    oellm::reset();
  }
  // 3. An injected C++ exception from Init leaves no temporary file.
  {
    minicpm5_test::ScratchDir scratch("minicpm5-s600-cleanup");
    const auto model = minicpm5_test::make_model_dir(scratch.path());
    oellm::throw_init = true;
    CHECK(throws([&] { minicpm5::MiniCPM5 failed({model.string(), 4}); }),
          "Init exception propagates");
    CHECK(temp_files(scratch.path()) == 0,
          "exception path leaves no temp file");
    oellm::reset();
  }
  // 4. Model directory validation still guards every required file.
  {
    minicpm5_test::ScratchDir scratch("minicpm5-s600-cleanup");
    const auto model = minicpm5_test::make_model_dir(scratch.path());
    std::filesystem::remove(model / "tokenizer.json");
    CHECK(throws([&] { minicpm5::MiniCPM5 failed({model.string(), 4}); }),
          "missing model file throws");
    CHECK(temp_files(scratch.path()) == 0,
          "validation failure leaves no temp file");
  }
  // 5. TMPDIR ends in exactly the state main() started in.
  CHECK(tmpdir_state() == initial_tmpdir, "TMPDIR state restored");
  if (failures) {
    std::cout << failures << " config-cleanup checks failed\n";
    return 1;
  }
  std::cout << "s600_config_cleanup OK\n";
  return 0;
}
