/** @file minicpm5.hpp Single-request MiniCPM5 inference with OELLM 1.0.0. */
#pragma once
#include <functional>
#include <string>

#include "xlm.h"
/** Paths and prompt for a fixed W8, chunk-256, context-4096 model. */
struct MiniCPM5Config {
  /** Path to the board-specific W8, chunk-256, context-4096 HBM file. */
  std::string model_path;
  /** Directory containing the prepared OELLM 1.0.0 tokenizer metadata. */
  std::string tokenizer_path;
  /** Path to the prepared non-thinking text Jinja template file. */
  std::string template_path;
  /** UTF-8 user text for one new conversation; defaults to self-introduction.
   */
  std::string prompt = "请用一句话介绍你自己。";
  /** Optional streaming sink invoked once per streamed text chunk; empty
   * disables streaming output so the library stays console-free. main.cc
   * injects stdout here. Exceptions thrown by the sink are contained by the
   * callback (recorded as RequestOutcome::stream_error), never propagated
   * across the vendor C callback. */
  std::function<void(const char*)> text_sink;
};
/** Outcome of one predict() request; main.cc renders the RESULT line. */
struct RequestOutcome {
  int sdk_status = 0;      ///< Raw xlm_infer return value; zero is success.
  bool ended = false;      ///< XLM_STATE_END arrived for this request.
  bool failed = false;     ///< XLM_STATE_ERROR arrived for this request.
  int destroy_status = 0;  ///< Raw xlm_destroy return value; zero is success.
  /** A consumer sink threw mid-stream; the request still completed. */
  bool stream_error = false;
  /** Zero only for normal EOS with successful cleanup, as in the source. */
  int exit_code() const {
    return sdk_status || !ended || failed || destroy_status ? 1 : 0;
  }
};
/** Pre-process carrier owning the strings referenced by its embedded SDK
 * view. Copy and move constructors and assignments rebind the view (and keep
 * a moved-from instance self-consistent), so every instance owns its own
 * pointers. Copy and move are allowed; pass &input to xlm_infer only while
 * the instance is alive and unmodified. */
struct PreparedRequest {
  std::string prompt;
  std::string chat_template;
  xlm_lm_request_t request{};
  xlm_input_t input{};
  /** Empty carrier; no SDK fields are set until strings are assigned. */
  PreparedRequest() = default;
  /** Store copies of the texts and bind the fixed single-request fields;
   * SDK 1.0.0 single-request indexing starts at zero. */
  PreparedRequest(const std::string& prompt, const std::string& chat_template);
  PreparedRequest(const PreparedRequest& other);
  PreparedRequest(PreparedRequest&& other) noexcept;
  PreparedRequest& operator=(const PreparedRequest& other);
  PreparedRequest& operator=(PreparedRequest&& other) noexcept;
  /** Point the embedded view at this instance's own storage. */
  void bind();
};
/** Build the single fixed request for one new conversation. */
PreparedRequest prepare_request(const std::string& prompt,
                                const std::string& chat_template);
/** Owns the SDK handle; each instance serves exactly one synchronous request.
 * init() after predict() is rejected — create a fresh instance per request. */
class MiniCPM5 {
 public:
  /** Store configuration without allocating runtime resources.
   * @param config Model/tokenizer/template paths, the single-request prompt
   * and the optional streaming sink.
   */
  explicit MiniCPM5(MiniCPM5Config config);
  /** Release the runtime handle if no request consumed it. */
  ~MiniCPM5();
  MiniCPM5(const MiniCPM5&) = delete;
  MiniCPM5& operator=(const MiniCPM5&) = delete;
  /** Load the model and tokenizer.
   * @throws std::runtime_error If already initialized, already finalized by a
   * previous predict(), or SDK initialization fails.
   */
  void init();
  /** Stream one greedy non-thinking response through the injected sink and
   * release the SDK handle. Outcome status is returned unprinted; main.cc
   * renders the RESULT line. Text is suppressed for END and ERROR states as
   * in the source; a throwing sink is contained and recorded.
   * @return RequestOutcome; exit_code() maps it to the source exit statuses.
   * @throws std::runtime_error If uninitialized, already finalized, or the
   * chat template cannot be loaded (see load_chat_template).
   */
  RequestOutcome predict();

 private:
  static void callback(xlm_result_t*, xlm_state_t, void*);
  MiniCPM5Config config_;
  xlm_handle_t handle_ = nullptr;
  bool ended_ = false;
  bool failed_ = false;
  bool finalized_ = false;
  bool stream_error_ = false;
};
