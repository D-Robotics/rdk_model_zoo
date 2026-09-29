// Host test double for the S600 OELLM 2.0 "oellm_runtime_basic" API used by
// runtime/cpp. This is NOT the vendor SDK header: it performs no tokenization,
// BPU work or generation. It gives deterministic responses, metrics and
// failure injection so the production stage logic can run on a host machine.
#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace oellm {
enum class OellmErrorCode {
  kOk = 0,
  kInitFailed = 1,
  kInferFailed = 2,
  kMetricFailed = 3
};
enum class OellmStatus {
  kNormalFinished = 3,
  kMaxContextFinished = 4,
  kAborted = 5,
  kLengthFinished = 6
};
struct OellmPrompt {
  struct TextPrompt {
    std::string text;
  };
  TextPrompt user_prompt;
};
struct OellmRequest {
  struct Request {
    int request_id = 0, conversation_id = 0, max_new_tokens = 0;
    bool new_chat = false;
    OellmPrompt prompt;
  };
  std::vector<Request> oellm_requests;
};
struct ResponseData {
  std::string text_result = "fixture text";
  std::vector<int> token_ids{1, 2, 3};
  OellmStatus status = OellmStatus::kNormalFinished;
};
struct OellmResponse {
  std::vector<std::shared_ptr<ResponseData>> response_datas;
};
struct MetricData {
  double ttft = 12.5, decode_tps = 53.25, e2e = 100.0;
};
struct OellmMetric {
  std::vector<std::shared_ptr<MetricData>> metric_datas;
};

struct CapturedRequest {
  int request_id, conversation_id, max_new_tokens;
  bool new_chat;
  std::string prompt_text;
};

// Test controls and captured observations; call reset() between scenarios.
inline std::vector<CapturedRequest> captured_requests;
inline std::string init_config_path;  // Init() argument
inline bool throw_init = false;       // Init throws instead of returning
inline OellmErrorCode init_result = OellmErrorCode::kOk;
inline OellmErrorCode infer_result = OellmErrorCode::kOk;
inline OellmErrorCode metric_result = OellmErrorCode::kOk;
inline int response_count = 1;  // response_datas size produced
inline bool null_first_response = false;
inline int metric_count = 1;  // metric_datas size produced
inline bool null_first_metric = false;
inline double ttft = 12.5, decode_tps = 53.25, e2e = 100.0;
inline OellmStatus status = OellmStatus::kNormalFinished;

inline void reset() {
  captured_requests.clear();
  init_config_path.clear();
  throw_init = false;
  init_result = OellmErrorCode::kOk;
  infer_result = OellmErrorCode::kOk;
  metric_result = OellmErrorCode::kOk;
  response_count = 1;
  null_first_response = false;
  metric_count = 1;
  null_first_metric = false;
  ttft = 12.5;
  decode_tps = 53.25;
  e2e = 100.0;
  status = OellmStatus::kNormalFinished;
}

class OellmRuntime {
 public:
  OellmErrorCode Init(const std::string& config_path) {
    if (throw_init) throw std::runtime_error("injected Init exception");
    init_config_path = config_path;
    return init_result;
  }
  OellmErrorCode Infer(const OellmRequest& request, OellmResponse& response) {
    if (infer_result != OellmErrorCode::kOk) return infer_result;
    for (const auto& item : request.oellm_requests) {
      captured_requests.push_back({item.request_id, item.conversation_id,
                                   item.max_new_tokens, item.new_chat,
                                   item.prompt.user_prompt.text});
    }
    for (int i = 0; i < response_count; ++i)
      response.response_datas.push_back(std::make_shared<ResponseData>());
    if (null_first_response)
      response.response_datas.front() = nullptr;
    else
      response.response_datas.front()->status = status;
    return OellmErrorCode::kOk;
  }
  OellmErrorCode GetOellmInferMetric(OellmMetric& metrics) {
    if (metric_result != OellmErrorCode::kOk) return metric_result;
    for (int i = 0; i < metric_count; ++i)
      metrics.metric_datas.push_back(std::make_shared<MetricData>());
    if (null_first_metric)
      metrics.metric_datas.front() = nullptr;
    else {
      metrics.metric_datas.front()->ttft = ttft;
      metrics.metric_datas.front()->decode_tps = decode_tps;
      metrics.metric_datas.front()->e2e = e2e;
    }
    return OellmErrorCode::kOk;
  }
};
}  // namespace oellm
