#pragma once
using xlm_handle_t=void*;
enum xlm_state_t { XLM_STATE_ERROR, XLM_STATE_END, XLM_STATE_RUNNING };
struct xlm_result_t {const char* text=nullptr;};
struct sampling_t {float temp,top_p,min_p,penalty_repeat,penalty_freq,penalty_present;int top_k;};
struct param_t {const char* model_path;const char* token_config_path;int model_type,context_size;sampling_t sampling;};
constexpr int XLM_MODEL_TYPE_DEEPSEEK=1,XLM_INPUT_PROMPT=1,XLM_INFER_BACKEND_BPU_ANY=1;
struct xlm_lm_request_t {int request_id,type;bool new_chat;const char* prompt;const char* chat_template;int infer_backend;};
struct xlm_input_t {int request_num;xlm_lm_request_t* requests;};
using Callback=void(*)(xlm_result_t*,xlm_state_t,void*);
param_t xlm_create_default_param();
int xlm_init(param_t*,Callback,xlm_handle_t*);
int xlm_infer(xlm_handle_t,xlm_input_t*,void*);
int xlm_destroy(xlm_handle_t*);
