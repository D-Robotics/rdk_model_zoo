#include "gemma4_embeddings.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace gemma4 {

TokenEmbeddings::TokenEmbeddings(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    throw std::runtime_error("failed to open embeddings: " + path);
  }
  const auto size = static_cast<size_t>(in.tellg());
  in.seekg(0);

  const size_t expected_f32 =
      static_cast<size_t>(kVocabSize) * kHiddenSize * sizeof(float);
  const size_t expected_f16 =
      static_cast<size_t>(kVocabSize) * kHiddenSize * sizeof(uint16_t);

  if (size == expected_f32) {
    table_.resize(static_cast<size_t>(kVocabSize) * kHiddenSize);
    in.read(reinterpret_cast<char*>(table_.data()),
            static_cast<std::streamsize>(size));
  } else if (size == expected_f16) {
    std::vector<uint16_t> f16(static_cast<size_t>(kVocabSize) * kHiddenSize);
    in.read(reinterpret_cast<char*>(f16.data()),
            static_cast<std::streamsize>(size));
    table_.resize(f16.size());
    for (size_t i = 0; i < f16.size(); ++i) {
      // Minimal f16->f32 conversion for lookup; board file is f32 in practice.
      uint16_t h = f16[i];
      uint32_t sign = (h >> 15) & 1;
      uint32_t exp = (h >> 10) & 0x1f;
      uint32_t mant = h & 0x3ff;
      float val = 0.f;
      if (exp == 0) {
        val = (mant ? static_cast<float>(mant) / 1024.f * std::ldexp(1.f, -14)
                    : 0.f);
      } else if (exp == 31) {
        val = mant ? NAN : (sign ? -INFINITY : INFINITY);
      } else {
        val = std::ldexp(1.f + static_cast<float>(mant) / 1024.f,
                         static_cast<int>(exp) - 15);
      }
      if (sign) {
        val = -val;
      }
      table_[i] = val;
    }
  } else {
    throw std::runtime_error("unexpected tok_embeddings size " +
                             std::to_string(size));
  }
}

void TokenEmbeddings::Lookup(const std::vector<int64_t>& ids, float* out) const {
  for (size_t i = 0; i < ids.size(); ++i) {
    const int64_t id = ids[i];
    if (id < 0 || id >= kVocabSize) {
      throw std::runtime_error("token id out of range: " + std::to_string(id));
    }
    const float* row = table_.data() + static_cast<size_t>(id) * kHiddenSize;
    std::copy(row, row + kHiddenSize, out + i * kHiddenSize);
  }
}

const float* TokenEmbeddings::GetRow(int64_t id) const {
  if (id < 0 || id >= kVocabSize) {
    throw std::runtime_error("token id out of range: " + std::to_string(id));
  }
  return table_.data() + static_cast<size_t>(id) * kHiddenSize;
}

std::vector<float> TokenEmbeddings::BuildPromptHidden(
    const std::vector<int64_t>& ids,
    const std::vector<float>& vision_features) const {
  const size_t seq_len = ids.size();
  std::vector<float> hidden(seq_len * static_cast<size_t>(kHiddenSize));

  // PLE path: embed pad_token_id at image positions (not image token id).
  std::vector<int64_t> ple_ids = ids;
  for (auto& id : ple_ids) {
    if (id == static_cast<int64_t>(kImageTokenId)) {
      id = kPadTokenId;
    }
  }
  Lookup(ple_ids, hidden.data());

  if (vision_features.empty()) {
    return hidden;
  }

  const size_t expected =
      static_cast<size_t>(kVisionSoftTokens) * static_cast<size_t>(kHiddenSize);
  if (vision_features.size() != expected) {
    throw std::runtime_error("vision feature size mismatch: got " +
                             std::to_string(vision_features.size()) +
                             " expected " + std::to_string(expected));
  }

  // Debug: count image tokens in ids
  int img_count = 0;
  for (size_t i = 0; i < seq_len; ++i) {
    if (ids[i] == static_cast<int64_t>(kImageTokenId)) ++img_count;
  }
  fprintf(stderr, "[DEBUG] BuildPromptHidden: seq_len=%zu, image_tokens=%d, vision_features.size=%zu\n",
          seq_len, img_count, vision_features.size());

  // Debug: vision feature statistics
  {
    double sum = 0, sq = 0;
    float vmin = vision_features[0], vmax = vision_features[0];
    for (size_t i = 0; i < vision_features.size(); ++i) {
      sum += vision_features[i];
      sq += vision_features[i] * vision_features[i];
      if (vision_features[i] < vmin) vmin = vision_features[i];
      if (vision_features[i] > vmax) vmax = vision_features[i];
    }
    double mean = sum / vision_features.size();
    double var = sq / vision_features.size() - mean * mean;
    fprintf(stderr, "[DEBUG] vision_features: min=%.6f max=%.6f mean=%.6f std=%.6f\n",
            vmin, vmax, mean, sqrt(var));
  }

  int run_start = -1;
  for (size_t i = 0; i <= seq_len; ++i) {
    const bool is_image =
        i < seq_len && ids[i] == static_cast<int64_t>(kImageTokenId);
    if (is_image && run_start < 0) {
      run_start = static_cast<int>(i);
    } else if (!is_image && run_start >= 0) {
      const int run_len = static_cast<int>(i) - run_start;
      if (run_len != kVisionSoftTokens) {
        throw std::runtime_error("image placeholder run length " +
                                 std::to_string(run_len) + " != " +
                                 std::to_string(kVisionSoftTokens));
      }
      // PLE token-identity path uses pad embedding at image positions
      // (Gemma4Model.forward replaces image token ids with pad_token_id).
      // Vision HBM output [280,1536] is injected raw — no L2-norm / sqrt scaling.
      fprintf(stderr, "[VLM-FIX] Injecting raw vision features at positions [%d, %d)\n",
              run_start, run_start + kVisionSoftTokens);

      for (int t = 0; t < kVisionSoftTokens; ++t) {
        const float* src = vision_features.data() +
                           static_cast<size_t>(t) * kHiddenSize;
        float* dst = hidden.data() +
                     static_cast<size_t>(run_start + t) * kHiddenSize;
        std::copy(src, src + kHiddenSize, dst);
      }

      // Self-check: injected vision features std and L2/row (per fix doc §4 P0)
      {
        double inj_sq = 0, inj_l2_sum = 0;
        for (int t = 0; t < kVisionSoftTokens; ++t) {
          const float* row = hidden.data() +
                             static_cast<size_t>(run_start + t) * kHiddenSize;
          double row_sq = 0;
          for (int d = 0; d < kHiddenSize; ++d) {
            inj_sq += static_cast<double>(row[d]) * row[d];
            row_sq += static_cast<double>(row[d]) * row[d];
          }
          inj_l2_sum += std::sqrt(row_sq);
        }
        const double inj_std = std::sqrt(inj_sq / (kVisionSoftTokens * kHiddenSize));
        const double inj_l2_avg = inj_l2_sum / kVisionSoftTokens;
        fprintf(stderr, "[VLM-FIX] Injected stats: std=%.4f L2/row=%.2f "
                "(expect std≈0.78 L2/row≈30.5)\n",
                inj_std, inj_l2_avg);
        if (inj_std > 1.5) {
          fprintf(stderr, "[VLM-FIX] WARNING: injected std=%.4f >> 0.78, "
                  "possible L2-norm scaling still applied!\n", inj_std);
        }
      }
      run_start = -1;
    }
  }

  // Debug: hidden buffer statistics after injection
  {
    double sum = 0, sq = 0;
    float hmin = hidden[0], hmax = hidden[0];
    for (size_t i = 0; i < hidden.size(); ++i) {
      sum += hidden[i];
      sq += hidden[i] * hidden[i];
      if (hidden[i] < hmin) hmin = hidden[i];
      if (hidden[i] > hmax) hmax = hidden[i];
    }
    double mean = sum / hidden.size();
    double var = sq / hidden.size() - mean * mean;
    fprintf(stderr, "[DEBUG] hidden buffer: size=%zu, min=%.6f max=%.6f mean=%.6f std=%.6f\n",
            hidden.size(), hmin, hmax, mean, sqrt(var));
  }

  return hidden;
}

}  // namespace gemma4
