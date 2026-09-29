// Host test double for the OELLM 1.0.0 "xlm" API used by runtime/legacy.
// This is NOT the vendor SDK header: it performs no tokenization, BPU work or
// generation. It exists so the production lifecycle and status logic can be
// exercised on a host machine with deterministic callback sequences.
#pragma once
#include <string>

using xlm_handle_t = void*;
enum xlm_state_t { XLM_STATE_ERROR, XLM_STATE_END, XLM_STATE_RUNNING };
struct xlm_result_t {
  const char* text = nullptr;
};
struct sampling_t {
  float temp, top_p, min_p, penalty_repeat, penalty_freq, penalty_present;
  int top_k;
};
struct param_t {
  const char* model_path;
  const char* token_config_path;
  int model_type, context_size;
  sampling_t sampling;
};
constexpr int XLM_MODEL_TYPE_DEEPSEEK = 1, XLM_INPUT_PROMPT = 1,
              XLM_INFER_BACKEND_BPU_ANY = 1;
struct xlm_lm_request_t {
  int request_id, type;
  bool new_chat;
  const char* prompt;
  const char* chat_template;
  int infer_backend;
};
struct xlm_input_t {
  int request_num;
  xlm_lm_request_t* requests;
};
using Callback = void (*)(xlm_result_t*, xlm_state_t, void*);

// Test controls and captured observations; call reset() between scenarios.
namespace xlm_double {
struct Capture {
  int model_type = -1, context_size = -1;
  float temp = -1, top_p = -1, min_p = -1;
  float penalty_repeat = -1, penalty_freq = -1, penalty_present = -1;
  int top_k = -1;
  std::string model_path, token_config_path;
  std::string prompt, chat_template;
  int request_id = -1, request_type = -1, request_num = -1, infer_backend = -1;
  bool new_chat = false;
  void* infer_userdata = nullptr;
  int infer_calls = 0;
};
inline Capture capture;
inline Callback registered_callback = nullptr;
inline int init_status = 0;         // xlm_init return value
inline int infer_status = 0;        // xlm_infer return value
inline int destroy_status = 0;      // xlm_destroy return value
inline bool deliver_end = true;     // END delivered on the end_call_index call
inline int end_call_index = 1;      // 1-based infer call that delivers END
inline bool deliver_error = false;  // ERROR state delivered before returning
inline const char* running_text = "你好，很高兴认识你。";
inline const char* end_text = nullptr;    // Text carried by the END state
inline const char* error_text = nullptr;  // Text carried by the ERROR state
inline int running_chunks = 1;  // RUNNING text callbacks within one call
inline void reset() {
  capture = Capture{};
  registered_callback = nullptr;
  init_status = infer_status = destroy_status = 0;
  deliver_end = true;
  end_call_index = 1;
  deliver_error = false;
  end_text = nullptr;
  error_text = nullptr;
  running_chunks = 1;
}
}  // namespace xlm_double

inline param_t xlm_create_default_param() { return {}; }

inline int xlm_init(param_t* params, Callback callback, xlm_handle_t* handle) {
  auto& seen = xlm_double::capture;
  seen.model_type = params->model_type;
  seen.context_size = params->context_size;
  seen.temp = params->sampling.temp;
  seen.top_p = params->sampling.top_p;
  seen.min_p = params->sampling.min_p;
  seen.penalty_repeat = params->sampling.penalty_repeat;
  seen.penalty_freq = params->sampling.penalty_freq;
  seen.penalty_present = params->sampling.penalty_present;
  seen.top_k = params->sampling.top_k;
  seen.model_path = params->model_path ? params->model_path : "";
  seen.token_config_path =
      params->token_config_path ? params->token_config_path : "";
  xlm_double::registered_callback = callback;
  *handle = new int(1);
  return xlm_double::init_status;
}

inline int xlm_infer(xlm_handle_t handle, xlm_input_t* input, void* userdata) {
  (void)handle;
  auto& seen = xlm_double::capture;
  ++seen.infer_calls;
  seen.request_num = input->request_num;
  seen.request_id = input->requests[0].request_id;
  seen.request_type = input->requests[0].type;
  seen.new_chat = input->requests[0].new_chat;
  seen.prompt = input->requests[0].prompt ? input->requests[0].prompt : "";
  seen.chat_template =
      input->requests[0].chat_template ? input->requests[0].chat_template : "";
  seen.infer_backend = input->requests[0].infer_backend;
  seen.infer_userdata = userdata;
  if (xlm_double::deliver_error) {
    xlm_result_t error_chunk;
    error_chunk.text = xlm_double::error_text;
    xlm_double::registered_callback(&error_chunk, XLM_STATE_ERROR, userdata);
    return xlm_double::infer_status;
  }
  if (xlm_double::deliver_end &&
      seen.infer_calls == xlm_double::end_call_index) {
    for (int i = 0; i < xlm_double::running_chunks; ++i) {
      xlm_result_t chunk;
      chunk.text = xlm_double::running_text;
      xlm_double::registered_callback(&chunk, XLM_STATE_RUNNING, userdata);
    }
    xlm_result_t end_chunk;
    end_chunk.text = xlm_double::end_text;
    xlm_double::registered_callback(&end_chunk, XLM_STATE_END, userdata);
  }
  return xlm_double::infer_status;
}

inline int xlm_destroy(xlm_handle_t* handle) {
  delete static_cast<int*>(*handle);
  *handle = nullptr;
  return xlm_double::destroy_status;
}
