// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "gemma4.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include "oellm_runtime_basic/oellm_runtime_version.h"

static_assert(OELLM_VERSION_MAJOR == 2 && OELLM_VERSION_MINOR == 0,
              "Build this sample with the supplied OELLM 2.0 headers");

namespace gemma4 {

Gemma4::Gemma4(const Config& config) : config_(config) {
  std::ifstream identity("/sys/class/boardinfo/soc_name");
  std::string soc;
  identity >> soc;
  std::transform(soc.begin(), soc.end(), soc.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (soc != "S600") throw std::runtime_error("This model requires an identified S600 board");
  if (config.max_new_tokens < 1 || config.max_new_tokens > 4096)
    throw std::runtime_error("max_new_tokens must be between 1 and 4096");
  std::ifstream stream(config.config_path);
  if (!stream) throw std::runtime_error("Cannot open the prepared runtime config");
  const auto settings = nlohmann::json::parse(stream);
  if (settings.value("runtime_type", "") != "GEMMA4" || settings.value("soc", "") != "s600")
    throw std::runtime_error("Use the prepared SDK 2.0 GEMMA4 S600 configuration");
  const std::filesystem::path directory = settings.at("work_dir").get<std::string>();
  if (!directory.is_absolute()) throw std::runtime_error("work_dir must be absolute");
  for (const auto* key : {"lm_model_file", "embed_weight_name", "ple_embed_weight_name", "vit_model_file"}) {
    const auto filename = settings.at(key).get<std::string>();
    if (filename.empty() || !std::filesystem::is_regular_file(directory / filename))
      throw std::runtime_error(std::string("Missing prepared file: ") + key);
  }
  const auto result = runtime_.Init(config.config_path);
  if (result != oellm::OellmErrorCode::kOk)
    throw std::runtime_error("OELLM Init failed: " + std::to_string(static_cast<int>(result)));
}

nlohmann::json Gemma4::Generate(const std::string& prompt) {
  if (prompt.empty()) throw std::runtime_error("The text prompt must not be empty");
  oellm::OellmRequest request;
  auto& item = request.oellm_requests.emplace_back();
  item.request_id = ++request_id_;
  item.conversation_id = 1;
  item.max_new_tokens = config_.max_new_tokens;
  item.enable_thinking = false;
  oellm::OellmSampleParams sampling;
  sampling.do_sample = false;
  item.sample_params = sampling;
  oellm::OellmPrompt::TextPrompt text;
  text.text = prompt;
  item.prompt.user_prompt = text;
  // SDK 2.0 requires full history on every turn; conversation_id is only a tag.
  item.prompt.history = history_;
  oellm::OellmResponse response;
  const auto code = runtime_.Infer(request, response);
  if (code != oellm::OellmErrorCode::kOk || response.response_datas.size() != 1 || !response.response_datas.front())
    throw std::runtime_error("OELLM Infer failed: " + std::to_string(static_cast<int>(code)));
  const auto& data = *response.response_datas.front();
  if (data.status != oellm::OellmStatus::kNormalFinished &&
      data.status != oellm::OellmStatus::kLengthFinished &&
      data.status != oellm::OellmStatus::kMaxContextFinished)
    throw std::runtime_error("Generation failed, status: " + std::to_string(static_cast<int>(data.status)));
  oellm::OellmPrompt::HistoryMessage user, assistant;
  user.role = "user";
  user.content = prompt;
  assistant.role = "assistant";
  assistant.content = data.text_result;
  history_.push_back(std::move(user));
  history_.push_back(std::move(assistant));
  nlohmann::json output = {{"text", data.text_result}, {"token_ids", data.token_ids},
      {"status", static_cast<int>(data.status)}, {"runtime_version", oellm::GetVersion()},
      {"ttft_ms", nullptr}, {"decode_tokens_per_second", nullptr}, {"e2e_ms", nullptr}};
  oellm::OellmMetric metrics;
  if (runtime_.GetOellmInferMetric(metrics) == oellm::OellmErrorCode::kOk &&
      metrics.metric_datas.size() == 1 && metrics.metric_datas.front()) {
    const auto& metric = *metrics.metric_datas.front();
    output["ttft_ms"] = metric.ttft;
    output["decode_tokens_per_second"] = metric.decode_tps;
    output["e2e_ms"] = metric.e2e;
  }
  return output;
}

void Gemma4::Reset() {
  history_.clear();
  runtime_.ReleaseConversation(1);
}

}  // namespace gemma4
