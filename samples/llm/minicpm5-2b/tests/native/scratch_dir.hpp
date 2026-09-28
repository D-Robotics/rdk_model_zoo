// Test-only scratch isolation helpers for the MiniCPM native drivers.
// ScratchDir owns one atomically allocated unique directory (mkdtemp), points
// TMPDIR at it for the scope, and restores the previous TMPDIR value —
// including an initially unset state — while removing only the directory it
// created. Nothing outside the owned directory is ever written or removed.
#pragma once
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace minicpm5_test {

class ScratchDir {
 public:
  explicit ScratchDir(const std::string& label) {
    const char* previous = std::getenv("TMPDIR");
    original_tmpdir_set_ = previous != nullptr;
    if (previous) original_tmpdir_ = previous;
    auto pattern =
        (std::filesystem::temp_directory_path() / (label + "-XXXXXX")).string();
    // mkdtemp creates the directory atomically under a unique name, so
    // concurrent driver instances can never share or collide on a path.
    if (!mkdtemp(pattern.data()))
      throw std::runtime_error("Cannot create scratch directory");
    path_ = pattern;
    setenv("TMPDIR", path_.c_str(), 1);
  }
  ~ScratchDir() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);  // Owned path only.
    if (original_tmpdir_set_)
      setenv("TMPDIR", original_tmpdir_.c_str(), 1);
    else
      unsetenv("TMPDIR");
  }
  ScratchDir(const ScratchDir&) = delete;
  ScratchDir& operator=(const ScratchDir&) = delete;
  /** Directory created atomically for this instance; removed at scope exit. */
  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
  bool original_tmpdir_set_ = false;
  std::string original_tmpdir_;
};

/** Model fixture payload written inside an owned scratch directory; the
 * content is never interpreted as weights. Removal happens with the scratch
 * directory that owns it. */
inline std::filesystem::path make_model_dir(
    const std::filesystem::path& scratch) {
  std::filesystem::path dir = scratch / "model";
  std::filesystem::create_directories(dir);
  for (const auto* name :
       {"MiniCPM5-2B_language_chunk_256_cache_4096_w8_nash-p_corenum_4_4.hbm",
        "MiniCPM5-2B_embed_tokens.bin", "tokenizer.json",
        "tokenizer_config.json"}) {
    std::ofstream(dir / name).write("fixture", 7);
  }
  return dir;
}
}  // namespace minicpm5_test
