#include "minicpm5.hpp"

int main() {
  minicpm5::Config config;              // model_path defaults to ../../model/s600
  config.max_new_tokens = 128;          // 1-4096
  minicpm5::MiniCPM5 model(config);     // validates settings, prepares the runtime; throws on failure
  minicpm5::Result first = model.Generate("What is 1+1?");  // opens the conversation
  if (first.status == 3) {              // 3 EOS; 6 output limit; 4 context limit
    // consume first.text, first.tokens, first.ttft_ms, first.decode_tps, first.e2e_ms
  }
  minicpm5::Result follow = model.Generate("Translate that.", false);  // same conversation
  return first.status == 3 && follow.status == 3 ? 0 : 1;
}
