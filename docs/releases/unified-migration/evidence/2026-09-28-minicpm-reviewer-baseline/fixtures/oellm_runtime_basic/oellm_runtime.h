#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
namespace oellm {
enum class OellmErrorCode{kOk};
enum class OellmStatus{kNormalFinished=3,kLengthFinished=6,kMaxContextFinished=4};
struct OellmPrompt{struct TextPrompt{std::string text;};TextPrompt user_prompt;};
struct OellmRequest{struct Request{int request_id,conversation_id,max_new_tokens;bool new_chat;OellmPrompt prompt;};std::vector<Request> oellm_requests;};
struct ResponseData{std::string text_result="fixture";std::vector<int> token_ids{1};OellmStatus status=OellmStatus::kNormalFinished;};
struct OellmResponse{std::vector<std::shared_ptr<ResponseData>> response_datas;};
struct MetricData{double ttft=std::numeric_limits<double>::quiet_NaN(),decode_tps=-1,e2e=0;};
struct OellmMetric{std::vector<std::shared_ptr<MetricData>> metric_datas;};
inline bool throw_init=false;
class OellmRuntime{public:
OellmErrorCode Init(const std::string&){if(throw_init)throw std::runtime_error("injected Init exception");return OellmErrorCode::kOk;}
OellmErrorCode Infer(const OellmRequest&,OellmResponse&r){r.response_datas.push_back(std::make_shared<ResponseData>());return OellmErrorCode::kOk;}
OellmErrorCode GetOellmInferMetric(OellmMetric&m){m.metric_datas.push_back(std::make_shared<MetricData>());return OellmErrorCode::kOk;}
};
}
