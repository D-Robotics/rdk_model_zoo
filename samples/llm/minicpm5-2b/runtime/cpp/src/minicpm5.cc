/** @file minicpm5.cc
 * @brief OELLM generation stages: pre-process, inference and post-processing.
 *
 * Configuration and file IO live in runtime_config.cc; this file holds only
 * the generation stages and their validation.
 */
#include "minicpm5.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "runtime_config.hpp"

namespace minicpm5 {

MiniCPM5::MiniCPM5(const Config& config) : config_(config) {
  if (config_.max_new_tokens < 1 || config_.max_new_tokens > 4096) {
    throw std::runtime_error("max_new_tokens must be between 1 and 4096");
  }
  configure_runtime(config_.model_path, runtime_);
}

void validate_metrics(double ttft_ms, double decode_tps, double e2e_ms) {
  if (!std::isfinite(ttft_ms))
    throw std::runtime_error("Non-finite TTFT metric");
  if (ttft_ms < 0) throw std::runtime_error("Negative TTFT metric");
  if (!std::isfinite(decode_tps))
    throw std::runtime_error("Non-finite decode_tps metric");
  if (decode_tps < 0) throw std::runtime_error("Negative decode_tps metric");
  if (!std::isfinite(e2e_ms)) throw std::runtime_error("Non-finite e2e metric");
  if (e2e_ms < 0) throw std::runtime_error("Negative e2e metric");
}

oellm::OellmRequest pre_process(const std::string& prompt, bool new_chat,
                                int32_t max_new_tokens, int32_t request_id) {
  if (prompt.empty()) throw std::runtime_error("Prompt must not be empty");
  // Enforced here because this is a public entry: direct callers cannot
  // bypass the bounds that the constructor also checks.
  if (max_new_tokens < 1 || max_new_tokens > 4096)
    throw std::runtime_error("max_new_tokens must be between 1 and 4096");
  oellm::OellmRequest request;
  oellm::OellmRequest::Request item;
  item.request_id = request_id;
  item.conversation_id = 1;
  item.new_chat = new_chat;
  item.max_new_tokens = max_new_tokens;
  oellm::OellmPrompt::TextPrompt text_prompt;
  text_prompt.text = prompt;
  item.prompt.user_prompt = text_prompt;
  request.oellm_requests.push_back(item);
  return request;
}

void infer(oellm::OellmRuntime& runtime, const oellm::OellmRequest& request,
           oellm::OellmResponse& response) {
  const auto code = runtime.Infer(request, response);
  if (code != oellm::OellmErrorCode::kOk ||
      response.response_datas.size() != 1 || !response.response_datas.front()) {
    throw std::runtime_error("OELLM inference failed: " +
                             std::to_string(static_cast<int>(code)));
  }
}

Result post_process(const oellm::OellmResponse& response,
                    oellm::OellmRuntime& runtime) {
  const auto& data = *response.response_datas.front();
  Result result;
  result.text = data.text_result;
  result.tokens.assign(data.token_ids.begin(), data.token_ids.end());
  result.status = static_cast<int>(data.status);
  if (data.status != oellm::OellmStatus::kNormalFinished &&
      data.status != oellm::OellmStatus::kLengthFinished &&
      data.status != oellm::OellmStatus::kMaxContextFinished) {
    throw std::runtime_error("Unexpected generation status: " +
                             std::to_string(result.status));
  }
  oellm::OellmMetric metrics;
  if (runtime.GetOellmInferMetric(metrics) != oellm::OellmErrorCode::kOk ||
      metrics.metric_datas.size() != 1 || !metrics.metric_datas.front()) {
    throw std::runtime_error("Cannot obtain request metrics");
  }
  const auto& metric = *metrics.metric_datas.front();
  validate_metrics(metric.ttft, metric.decode_tps, metric.e2e);
  result.ttft_ms = metric.ttft;
  result.decode_tps = metric.decode_tps;
  result.e2e_ms = metric.e2e;
  return result;
}

Result MiniCPM5::Generate(const std::string& prompt, bool new_chat) {
  const int32_t request_id = ++request_id_;
  const auto request =
      pre_process(prompt, new_chat, config_.max_new_tokens, request_id);
  oellm::OellmResponse response;
  infer(runtime_, request, response);
  return post_process(response, runtime_);
}
}  // namespace minicpm5
