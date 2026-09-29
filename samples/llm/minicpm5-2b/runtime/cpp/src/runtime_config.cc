/** @file runtime_config.cc
 * @brief OELLM runtime configuration; file IO stays outside inference stages.
 */
#include "runtime_config.hpp"

#include <unistd.h>

#include <cstdio>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace minicpm5 {
namespace {
constexpr const char* kModel =
    "MiniCPM5-2B_language_chunk_256_cache_4096_w8_nash-p_corenum_4_4.hbm";
constexpr const char* kEmbedding = "MiniCPM5-2B_embed_tokens.bin";

/** @brief Owns one mkstemp descriptor and its path until scope exit.
 *
 * The destructor closes the descriptor and removes the file, covering the
 * success, SDK error return and exception paths without manual cleanup.
 */
class TemporaryFile {
 public:
  TemporaryFile() {
    auto pattern =
        (std::filesystem::temp_directory_path() / "minicpm5-XXXXXX").string();
    const int descriptor = mkstemp(pattern.data());
    if (descriptor < 0)
      throw std::runtime_error("Cannot create temporary runtime configuration");
    descriptor_ = descriptor;
    path_ = std::move(pattern);
  }
  ~TemporaryFile() {
    if (descriptor_ >= 0) ::close(descriptor_);
    if (!path_.empty()) std::remove(path_.c_str());
  }
  TemporaryFile(const TemporaryFile&) = delete;
  TemporaryFile& operator=(const TemporaryFile&) = delete;
  /** @brief Write all bytes, continuing after short writes. */
  void write(const std::string& contents) {
    size_t offset = 0;
    while (offset < contents.size()) {
      const ssize_t written = ::write(descriptor_, contents.data() + offset,
                                      contents.size() - offset);
      if (written < 0)
        throw std::runtime_error("Cannot write runtime configuration");
      offset += static_cast<size_t>(written);
    }
  }
  /** @brief Path of the temporary file, valid for the lifetime of the owner. */
  const std::string& path() const noexcept { return path_; }

 private:
  int descriptor_ = -1;
  std::string path_;
};
}  // namespace

std::filesystem::path validate_model_directory(
    const std::filesystem::path& model_directory) {
  const auto directory = std::filesystem::canonical(model_directory);
  for (const auto* name :
       {kModel, kEmbedding, "tokenizer.json", "tokenizer_config.json"}) {
    if (!std::filesystem::is_regular_file(directory / name)) {
      throw std::runtime_error(std::string("Missing model file: ") + name);
    }
  }
  return directory;
}

void configure_runtime(const std::filesystem::path& model_directory,
                       oellm::OellmRuntime& runtime) {
  const auto directory = validate_model_directory(model_directory);
  // OELLM backend enums differ from hbm_infer's zero-based core IDs.
  const nlohmann::json settings = {
      {"work_dir", directory.string()},
      {"lm_model_file", kModel},
      {"embed_weight_name", kEmbedding},
      {"runtime_type", "LLM"},
      {"max_batch_size", 1},
      {"max_conv_cache_num", 0},
      {"backends", {{"prefill", {1, 2, 3, 4}}, {"decode", {1, 2, 3, 4}}}}};
  TemporaryFile file;  // The destructor removes the file on every exit path.
  file.write(settings.dump());
  const auto code = runtime.Init(file.path());
  if (code != oellm::OellmErrorCode::kOk) {
    throw std::runtime_error("OELLM initialization failed: " +
                             std::to_string(static_cast<int>(code)));
  }
}
}  // namespace minicpm5
