// Shared state between the chat-app host doubles and the scenario driver.
//
// The doubles in chat_app_doubles.cpp replace the model engines/tokenizer
// for the host check of the production application session source. They are
// clearly marked doubles: no HBM is loaded, no BPU runs, no tokenization
// happens, and nothing here is board evidence.
#pragma once

#include <cstdint>
#include <vector>

namespace chat_test {

struct DoubleState {
  int vision_ctors = 0;
  int vision_dtors = 0;
  int vision_infers = 0;
  int text_ctors = 0;
  int text_dtors = 0;
  int continues = 0;
  int resets = 0;
  int prompt_hiddens = 0;
  int tokenizer_ctors = 0;
  int tokenizer_decodes = 0;
  int predict_visions = 0;
  int load_images = 0;
  std::vector<int> last_max_tokens;
  std::vector<char> hidden_flags;
  std::vector<int64_t> last_full_ids;
  std::vector<size_t> full_id_sizes;
};

// Defined in chat_app_doubles.cpp; reset by the driver between scenarios.
DoubleState& Doubles();

}  // namespace chat_test
