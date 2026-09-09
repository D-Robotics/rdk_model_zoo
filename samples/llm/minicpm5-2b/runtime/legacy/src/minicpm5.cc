/** @file minicpm5.cc OELLM legacy text generation and lifecycle. */
#include "minicpm5.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <utility>
MiniCPM5::MiniCPM5(MiniCPM5Config config) : config_(std::move(config)) {}
MiniCPM5::~MiniCPM5() { if (handle_) xlm_destroy(&handle_); }
void MiniCPM5::callback(xlm_result_t* result, xlm_state_t state, void* userdata) {
  auto* self = static_cast<MiniCPM5*>(userdata);
  if (state == XLM_STATE_ERROR) { self->failed_ = true; return; }
  if (result && result->text && state != XLM_STATE_END)
    std::cout << result->text << std::flush;
  if (state == XLM_STATE_END) self->ended_ = true;
}
void MiniCPM5::init() {
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
  if (status) throw std::runtime_error("xlm_init failed: " + std::to_string(status));
}
int MiniCPM5::predict() {
  if (!handle_) throw std::runtime_error("Call init first");
  std::ifstream input_template(config_.template_path, std::ios::binary);
  if (!input_template) throw std::runtime_error("Cannot open chat template");
  const std::string templ{std::istreambuf_iterator<char>(input_template), {}};
  if (templ.empty() || templ.size() > 65535) throw std::runtime_error("Invalid chat template size");
  xlm_lm_request_t request{};
  // SDK 1.0.0 single-request indexing starts at zero.
  request.request_id = 0;
  request.type = XLM_INPUT_PROMPT;
  request.new_chat = true;
  request.prompt = config_.prompt.c_str();
  request.chat_template = templ.c_str();
  request.infer_backend = XLM_INFER_BACKEND_BPU_ANY;
  xlm_input_t input{};
  input.request_num = 1;
  input.requests = &request;
  const int status = xlm_infer(handle_, &input, this);
  const int destroyed = xlm_destroy(&handle_);
  handle_ = nullptr;
  std::cout << "\nRESULT status=" << status << " ended=" << ended_
            << " failed=" << failed_ << " destroy=" << destroyed << '\n';
  return status || !ended_ || failed_ || destroyed ? 1 : 0;
}
