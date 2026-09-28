/** @file minicpm5.cc OELLM legacy request stages and single-use lifecycle.
 *
 * Template file IO lives in chat_template.cc and all console rendering lives
 * in main.cc; this file holds SDK initialization, streaming through the
 * injected sink, and the request stages. It performs no console IO itself.
 */
#include "minicpm5.hpp"

#include <stdexcept>
#include <utility>

#include "chat_template.hpp"

MiniCPM5::MiniCPM5(MiniCPM5Config config) : config_(std::move(config)) {}
MiniCPM5::~MiniCPM5() {
  if (handle_) xlm_destroy(&handle_);
}

void MiniCPM5::callback(xlm_result_t* result, xlm_state_t state,
                        void* userdata) {
  // The SDK invokes this callback asynchronously; streamed text is delivered
  // to the caller-injected sink. End-state text is suppressed as in the
  // source, and a throwing consumer is contained here — no exception may
  // cross the C SDK callback boundary.
  auto* self = static_cast<MiniCPM5*>(userdata);
  if (state == XLM_STATE_ERROR) {
    self->failed_ = true;
    return;
  }
  if (result && result->text && state != XLM_STATE_END &&
      self->config_.text_sink && !self->stream_error_) {
    try {
      self->config_.text_sink(result->text);
    } catch (...) {
      self->stream_error_ = true;  // Stop streaming; keep the request going.
    }
  }
  if (state == XLM_STATE_END) self->ended_ = true;
}

void MiniCPM5::init() {
  if (finalized_)
    throw std::runtime_error(
        "Request already completed; create a new MiniCPM5 instance");
  if (handle_) throw std::runtime_error("Already initialized");
  auto params = xlm_create_default_param();
  params.model_path = config_.model_path.c_str();
  params.token_config_path = config_.tokenizer_path.c_str();
  params.model_type = XLM_MODEL_TYPE_DEEPSEEK;
  params.context_size = 4096;
  params.sampling.temp = 0.0f;
  params.sampling.top_k = 1;
  params.sampling.top_p = 1.0f;
  params.sampling.min_p = 0.0f;
  params.sampling.penalty_repeat = 1.0f;
  params.sampling.penalty_freq = 0.0f;
  params.sampling.penalty_present = 0.0f;
  const int status = xlm_init(&params, callback, &handle_);
  if (status)
    throw std::runtime_error("xlm_init failed: " + std::to_string(status));
}

PreparedRequest::PreparedRequest(const std::string& prompt,
                                 const std::string& chat_template)
    : prompt(prompt), chat_template(chat_template) {
  request.request_id = 0;
  request.type = XLM_INPUT_PROMPT;
  request.new_chat = true;
  request.infer_backend = XLM_INFER_BACKEND_BPU_ANY;
  bind();
}

void PreparedRequest::bind() {
  request.prompt = prompt.c_str();
  request.chat_template = chat_template.c_str();
  input.request_num = 1;
  input.requests = &request;
}

PreparedRequest::PreparedRequest(const PreparedRequest& other)
    : prompt(other.prompt),
      chat_template(other.chat_template),
      request(other.request),
      input(other.input) {
  bind();
}

PreparedRequest::PreparedRequest(PreparedRequest&& other) noexcept
    : prompt(std::move(other.prompt)),
      chat_template(std::move(other.chat_template)),
      request(other.request),
      input(other.input) {
  bind();
  other.bind();  // Keep the moved-from carrier pointing at its own storage.
}

PreparedRequest& PreparedRequest::operator=(const PreparedRequest& other) {
  if (this != &other) {
    prompt = other.prompt;
    chat_template = other.chat_template;
    request = other.request;
    input = other.input;
    bind();
  }
  return *this;
}

PreparedRequest& PreparedRequest::operator=(PreparedRequest&& other) noexcept {
  if (this != &other) {
    prompt = std::move(other.prompt);
    chat_template = std::move(other.chat_template);
    request = other.request;
    input = other.input;
    bind();
    other.bind();
  }
  return *this;
}

PreparedRequest prepare_request(const std::string& prompt,
                                const std::string& chat_template) {
  return PreparedRequest(prompt, chat_template);
}

RequestOutcome MiniCPM5::predict() {
  if (finalized_)
    throw std::runtime_error(
        "Request already completed; create a new MiniCPM5 instance");
  if (!handle_) throw std::runtime_error("Call init first");
  RequestOutcome outcome;
  // Pre-process: template loading lives in chat_template.cc; the carrier
  // owns copies of the prompt and template and rebinds its own view, so the
  // SDK pointers stay valid for the whole synchronous call below.
  const std::string templ = load_chat_template(config_.template_path);
  PreparedRequest prepared = prepare_request(config_.prompt, templ);
  // Inference: one synchronous SDK call streaming through the sink.
  outcome.sdk_status = xlm_infer(handle_, &prepared.input, this);
  // Single-use teardown; request state is never reused across requests.
  outcome.destroy_status = xlm_destroy(&handle_);
  handle_ = nullptr;
  finalized_ = true;
  outcome.ended = ended_;
  outcome.failed = failed_;
  outcome.stream_error = stream_error_;
  return outcome;
}
