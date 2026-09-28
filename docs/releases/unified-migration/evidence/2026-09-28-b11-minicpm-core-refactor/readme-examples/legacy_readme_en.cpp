#include <iostream>

#include "minicpm5.hpp"

int main() {
  MiniCPM5Config config;
  config.model_path = "model/s100/minicpm5-2b_ctx4096_s100.hbm";
  config.tokenizer_path = "model/s100/tokenizer";
  config.template_path = "model/s100/tokenizer/simple-chat.jinja";
  config.prompt = "请用一句话介绍你自己。";
  config.text_sink = [](const char* chunk) {  // optional; omit for silent use
    std::cout << chunk << std::flush;
  };
  MiniCPM5 model(config);                    // stores settings only
  model.init();                              // loads model and tokenizer; one request per instance
  RequestOutcome outcome = model.predict();  // streams via the sink
  // consume: outcome.ended, outcome.failed, outcome.sdk_status, outcome.destroy_status,
  // outcome.stream_error; outcome.exit_code() is zero only for normal EOS with cleanup.
  return outcome.exit_code();
}
