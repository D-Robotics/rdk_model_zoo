/** @file minicpm5.cc
 * @brief Portable configuration and synchronous OELLM text inference.
 */
#include "minicpm5.hpp"

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <unistd.h>
#include <nlohmann/json.hpp>

namespace minicpm5 {
namespace {
constexpr const char* kModel = "MiniCPM5-2B_language_chunk_256_cache_4096_w8_nash-p_corenum_4_4.hbm";
constexpr const char* kEmbedding = "MiniCPM5-2B_embed_tokens.bin";
}

MiniCPM5::MiniCPM5(const Config& config) : config_(config) {
  if (config.max_new_tokens < 1 || config.max_new_tokens > 4096) {
    throw std::runtime_error("max_new_tokens must be between 1 and 4096");
  }
  const auto directory = std::filesystem::canonical(config.model_path);
  for (const auto* name : {kModel, kEmbedding, "tokenizer.json", "tokenizer_config.json"}) {
    if (!std::filesystem::is_regular_file(directory / name)) {
      throw std::runtime_error(std::string("Missing model file: ") + name);
    }
  }
  // OELLM backend enums differ from hbm_infer's zero-based core IDs.
  const nlohmann::json settings = {
      {"work_dir", directory.string()}, {"lm_model_file", kModel},
      {"embed_weight_name", kEmbedding}, {"runtime_type", "LLM"},
      {"max_batch_size", 1}, {"max_conv_cache_num", 0},
      {"backends", {{"prefill", {1, 2, 3, 4}}, {"decode", {1, 2, 3, 4}}}}};
  std::string filename = (std::filesystem::temp_directory_path() / "minicpm5-XXXXXX").string();
  const int fd = mkstemp(filename.data());
  if (fd < 0) throw std::runtime_error("Cannot create temporary runtime configuration");
  const std::string serialized = settings.dump();
  const auto written = write(fd, serialized.data(), serialized.size());
  close(fd);
  if (written != static_cast<ssize_t>(serialized.size())) {
    std::remove(filename.c_str());
    throw std::runtime_error("Cannot write runtime configuration");
  }
  const auto code = runtime_.Init(filename);
  std::remove(filename.c_str());
  if (code != oellm::OellmErrorCode::kOk) {
    throw std::runtime_error("OELLM initialization failed: " + std::to_string(static_cast<int>(code)));
  }
}

Result MiniCPM5::Generate(const std::string& prompt, bool new_chat) {
  if (prompt.empty()) throw std::runtime_error("Prompt must not be empty");
  oellm::OellmRequest request;
  oellm::OellmRequest::Request item;
  item.request_id = ++request_id_;
  item.conversation_id = 1;
  item.new_chat = new_chat;
  item.max_new_tokens = config_.max_new_tokens;
  oellm::OellmPrompt::TextPrompt text_prompt;
  text_prompt.text = prompt;
  item.prompt.user_prompt = text_prompt;
  request.oellm_requests.push_back(item);
  oellm::OellmResponse response;
  const auto code = runtime_.Infer(request, response);
  if (code != oellm::OellmErrorCode::kOk || response.response_datas.size() != 1 ||
      !response.response_datas.front()) {
    throw std::runtime_error("OELLM inference failed: " + std::to_string(static_cast<int>(code)));
  }
  const auto& data = *response.response_datas.front();
  Result result;
  result.text = data.text_result;
  result.tokens.assign(data.token_ids.begin(), data.token_ids.end());
  result.status = static_cast<int>(data.status);
  if (data.status != oellm::OellmStatus::kNormalFinished &&
      data.status != oellm::OellmStatus::kLengthFinished &&
      data.status != oellm::OellmStatus::kMaxContextFinished) {
    throw std::runtime_error("Unexpected generation status: " + std::to_string(result.status));
  }
  oellm::OellmMetric metrics;
  if (runtime_.GetOellmInferMetric(metrics) != oellm::OellmErrorCode::kOk ||
      metrics.metric_datas.size() != 1 || !metrics.metric_datas.front()) {
    throw std::runtime_error("Cannot obtain request metrics");
  }
  const auto& metric = *metrics.metric_datas.front();
  result.ttft_ms = metric.ttft;
  result.decode_tps = metric.decode_tps;
  result.e2e_ms = metric.e2e;
  return result;
}
}  // namespace minicpm5
