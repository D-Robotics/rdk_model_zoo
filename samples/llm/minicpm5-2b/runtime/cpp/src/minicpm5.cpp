/** @file minicpm5.cpp
 * @brief OELLM generation stages and the MiniCPM5 model that owns the runtime.
 *
 * Model validation, temporary configuration-file handling and the generation
 * stages all live in this file, so no CLI code holds SDK setup.
 */
#include "minicpm5.hpp"

#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

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

oellm::OellmRequest preprocess(const std::string& prompt, bool new_chat,
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

Result postprocess(const oellm::OellmResponse& response,
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

Result MiniCPM5::predict(const std::string& prompt, bool new_chat) {
  const int32_t request_id = ++request_id_;
  const auto request =
      preprocess(prompt, new_chat, config_.max_new_tokens, request_id);
  oellm::OellmResponse response;
  infer(runtime_, request, response);
  return postprocess(response, runtime_);
}
}  // namespace minicpm5
