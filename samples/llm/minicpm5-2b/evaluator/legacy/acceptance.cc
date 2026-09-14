/** @file acceptance.cc Greedy text checks for the OELLM 1.0.0 board runtime. */
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <string>

#include "xlm.h"
using nlohmann::json;
/** Text and terminal states for one synchronous request. */
struct State {
  std::string text;
  bool ended = false, failed = false;
};
/** Collect streamed text and error/end states for the active request.
 * @param r SDK-owned streamed result, possibly null.
 * @param s SDK event state.
 * @param u Pointer to the current request State.
 */
void callback(xlm_result_t* r, xlm_state_t s, void* u) {
  auto& v = *static_cast<State*>(u);
  if (s == XLM_STATE_ERROR) v.failed = true;
  if (s == XLM_STATE_END)
    v.ended = true;
  else if (r && r->text)
    v.text += r->text;
}
/** Run reference, conversation, long-context and repeat checks.
 * @param argc Number of command-line arguments, including executable.
 * @param argv Executable, HBM, tokenizer directory, references and long
 * prompts.
 * @return Zero when all comparisons and destruction pass; nonzero otherwise.
 */
int main(int argc, char** argv) {
  if (argc != 5) return 2;
  std::string mp = argv[1], tp = argv[2];
  std::ifstream f(tp + "/simple-chat.jinja");
  std::string templ{std::istreambuf_iterator<char>(f), {}};
  if (templ.empty()) return 3;
  auto p = xlm_create_default_param();
  p.model_path = mp.c_str();
  p.token_config_path = tp.c_str();
  p.model_type = XLM_MODEL_TYPE_DEEPSEEK;
  p.context_size = 4096;
  p.sampling.temp = 0;
  p.sampling.top_k = 1;
  p.sampling.top_p = 1;
  p.sampling.min_p = 0;
  p.sampling.penalty_repeat = 1;
  p.sampling.penalty_freq = 0;
  p.sampling.penalty_present = 0;
  void* h = nullptr;
  if (xlm_init(&p, callback, &h)) return 4;
  json refs, longs;
  std::ifstream(argv[3]) >> refs;
  std::ifstream(argv[4]) >> longs;
  int failures = 0;
  auto check = [&](std::string name, std::string prompt, bool fresh,
                   std::string expected) {
    State state;
    xlm_lm_request_t req{};
    req.request_id = 0;
    req.type = XLM_INPUT_PROMPT;
    req.new_chat = fresh;
    req.prompt = prompt.c_str();
    req.chat_template = templ.c_str();
    req.infer_backend = XLM_INFER_BACKEND_BPU_ANY;
    xlm_input_t in{};
    in.request_num = 1;
    in.requests = &req;
    auto t = std::chrono::steady_clock::now();
    int status = xlm_infer(h, &in, &state);
    bool pass =
        status == 0 && state.ended && !state.failed && state.text == expected;
    failures += !pass;
    std::cout << "CHECK "
              << json{{"name", name},
                      {"passed", pass},
                      {"text", state.text},
                      {"expected", expected},
                      {"status", status},
                      {"ended", state.ended},
                      {"failed", state.failed},
                      {"e2e_ms", std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - t)
                                     .count()}}
                     .dump()
              << std::endl;
  };
  int i = 0;
  for (auto& r : refs)
    check("reference-" + std::to_string(++i), r.at("prompt"), true,
          r.at("baseline"));
  check("conversation-en",
        "What is the capital of France? Answer with the city name only.", true,
        "Paris");
  check("conversation-zh",
        "Translate the previous answer into Chinese. Answer with the city name "
        "only.",
        false, u8"\u5df4\u9ece");
  for (auto& r : longs) check(r.at("name"), r.at("prompt"), true, "VIOLET-853");
  for (int k = 0; k < 50; k++)
    check("repeat-" + std::to_string(k + 1),
          "What is the capital of France? Answer with the city name only.",
          true, "Paris");
  int d = xlm_destroy(&h);
  std::cout << "SUMMARY " << json{{"failures", failures}, {"destroy", d}}.dump()
            << std::endl;
  return failures || d ? 1 : 0;
}
