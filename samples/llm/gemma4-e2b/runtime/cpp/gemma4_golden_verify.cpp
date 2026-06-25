#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gemma4_config.h"
#include "gemma4_text_engine.h"

namespace {

std::vector<uint8_t> ReadFile(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("cannot open: " + path);
  }
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(ifs)),
                              std::istreambuf_iterator<char>());
}

template <typename T>
std::vector<T> LoadBin(const std::string& path) {
  const auto bytes = ReadFile(path);
  if (bytes.size() % sizeof(T) != 0) {
    throw std::runtime_error("bad bin size: " + path);
  }
  std::vector<T> out(bytes.size() / sizeof(T));
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

bool CompareF32(const std::string& name, const std::vector<float>& got,
                const std::vector<float>& golden, float tol) {
  if (got.size() != golden.size()) {
    std::cerr << name << ": size mismatch got=" << got.size()
              << " golden=" << golden.size() << std::endl;
    return false;
  }
  float max_diff = 0.f;
  size_t max_idx = 0;
  double cos_dot = 0, cos_a2 = 0, cos_b2 = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const float d = std::fabs(got[i] - golden[i]);
    if (d > max_diff) {
      max_diff = d;
      max_idx = i;
    }
    cos_dot += static_cast<double>(got[i]) * static_cast<double>(golden[i]);
    cos_a2 += static_cast<double>(got[i]) * static_cast<double>(got[i]);
    cos_b2 += static_cast<double>(golden[i]) * static_cast<double>(golden[i]);
  }
  const double cosine = (cos_a2 > 0 && cos_b2 > 0)
      ? cos_dot / (std::sqrt(cos_a2) * std::sqrt(cos_b2))
      : 0;
  const bool ok = max_diff <= tol;
  std::cout << name << ": " << (ok ? "OK" : "FAIL")
            << " max_diff=" << max_diff << " at " << max_idx
            << " cosine=" << cosine << std::endl;
  return ok;
}

bool CompareI64(const std::string& name, const std::vector<int64_t>& got,
                const std::vector<int64_t>& golden) {
  if (got.size() != golden.size()) {
    std::cerr << name << ": size mismatch got=" << got.size()
              << " golden=" << golden.size() << std::endl;
    return false;
  }
  int mismatches = 0;
  size_t first_mismatch = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != golden[i]) {
      if (mismatches == 0) first_mismatch = i;
      ++mismatches;
    }
  }
  if (mismatches > 0) {
    std::cerr << name << ": FAIL " << mismatches << "/" << got.size()
              << " mismatches, first at " << first_mismatch
              << " got=" << got[first_mismatch]
              << " golden=" << golden[first_mismatch] << std::endl;
    // Show surrounding mismatched values
    size_t show_end = std::min(first_mismatch + 20, got.size());
    std::cerr << "  got:    ";
    for (size_t i = first_mismatch; i < show_end; ++i)
      std::cerr << got[i] << ",";
    std::cerr << "\n  golden: ";
    for (size_t i = first_mismatch; i < show_end; ++i)
      std::cerr << golden[i] << ",";
    std::cerr << std::endl;
    return false;
  }
  std::cout << name << ": OK" << std::endl;
  return true;
}

bool CompareI32(const std::string& name, const std::vector<int32_t>& got,
                const std::vector<int32_t>& golden) {
  if (got.size() != golden.size()) {
    std::cerr << name << ": size mismatch" << std::endl;
    return false;
  }
  int mismatches = 0;
  size_t first_mismatch = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != golden[i]) {
      if (mismatches == 0) first_mismatch = i;
      ++mismatches;
    }
  }
  if (mismatches > 0) {
    std::cerr << name << ": FAIL " << mismatches << "/" << got.size()
              << " mismatches, first at " << first_mismatch << std::endl;
    return false;
  }
  std::cout << name << ": OK" << std::endl;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const char* env_home = std::getenv("GEMMA4_HOME");
  const std::string home = (env_home && *env_home) ? env_home : ".";
  std::string golden_root = home + "/golden_mask_kv";
  std::string prompt_id = "prompt_0";
  std::string hbm = home + "/model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm";
  std::string embed = home + "/model/tok_embeddings.bin";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if ((arg == "--golden-root" || arg == "--prompt" || arg == "--hbm" ||
         arg == "--embed") &&
        i + 1 < argc) {
      const std::string val = argv[++i];
      if (arg == "--golden-root") {
        golden_root = val;
      } else if (arg == "--prompt") {
        prompt_id = val;
      } else if (arg == "--hbm") {
        hbm = val;
      } else if (arg == "--embed") {
        embed = val;
      }
    }
  }

  try {
    const std::string chunk_dir =
        golden_root + "/" + prompt_id + "/prefill_chunk_0";
    std::cout << "Golden dir: " << chunk_dir << std::endl;

    const auto golden_ids = LoadBin<int64_t>(chunk_dir + "/input_ids.int64.bin");
    const auto golden_pos =
        LoadBin<int32_t>(chunk_dir + "/position_ids.int32.bin");
    const auto golden_embeds =
        LoadBin<float>(chunk_dir + "/inputs_embeds.f32.bin");
    const auto golden_full =
        LoadBin<float>(chunk_dir + "/full_mask.f32.bin");
    const auto golden_slide =
        LoadBin<float>(chunk_dir + "/sliding_mask.f32.bin");

    // Extract valid tokens from golden input_ids (non-zero, non-pad prefix)
    int valid_len = 0;
    for (size_t i = 0; i < golden_ids.size(); ++i) {
      if (golden_ids[i] == 0 && i >= 8) break;  // pad starts after valid
      ++valid_len;
    }
    // For prompt_0, valid_len=8 (from meta.json)
    // But golden_ids already has the correct valid tokens at positions 0..7
    std::vector<int64_t> prompt_ids(golden_ids.begin(),
                                    golden_ids.begin() + valid_len);

    std::cout << "Golden prompt_ids (" << valid_len << "): ";
    for (int i = 0; i < valid_len; ++i) {
      if (i) std::cout << ',';
      std::cout << prompt_ids[static_cast<size_t>(i)];
    }
    std::cout << std::endl;

    gemma4::TextEngine engine(hbm, embed);
    engine.ResetSession();

    const auto got = engine.ExportPrefillChunk(prompt_ids, 0, valid_len);

    bool ok = true;
    ok &= CompareI64("input_ids", got.input_ids, golden_ids);
    ok &= CompareI32("position_ids", got.position_ids, golden_pos);
    ok &= CompareF32("inputs_embeds", got.inputs_embeds, golden_embeds, 1e-3f);
    ok &= CompareF32("full_mask", got.full_mask, golden_full, 0.f);
    ok &= CompareF32("sliding_mask", got.sliding_mask, golden_slide, 0.f);

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << (ok ? "ALL PASSED" : "SOME FAILED") << std::endl;
    return ok ? 0 : 1;
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << std::endl;
    return 1;
  }
}
