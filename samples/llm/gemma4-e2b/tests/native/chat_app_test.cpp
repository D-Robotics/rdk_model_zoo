// Scenario driver for the production interactive-chat application source.
//
// Links the real src/gemma4_chat_app.cpp against the engine doubles in
// chat_app_doubles.cpp and drives the REPL through redirected stdin. Each
// scenario asserts the observable session behavior (engine call counts,
// hidden-injection flags, reset counts) and prints "OK <scenario>". The
// doubles never load an HBM or touch a BPU; board behavior is not-run.

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "chat_app_doubles.hpp"
#include "gemma4_chat_app.hpp"
#include "gemma4_config.hpp"

namespace {

int fail(const std::string& message) {
  std::cerr << "FAIL: " << message << std::endl;
  return 1;
}

gemma4::chat::InteractiveChatApp make_app(bool rebuild_each_turn = false) {
  gemma4::chat::ChatPaths paths;
  paths.text_hbm = "double-text.hbm";
  paths.vision_hbm = "double-vision.hbm";
  paths.tok_embeddings = "double-embed.bin";
  paths.tokenizer_json = "double-tokenizer.json";
  gemma4::chat::ChatSettings settings;
  settings.rebuild_context_each_turn = rebuild_each_turn;
  return gemma4::chat::InteractiveChatApp(paths, settings);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: chat_app_test <scenario> [image-path]" << std::endl;
    return 2;
  }
  const std::string scenario = argv[1];
  const auto& state = chat_test::Doubles();

  if (scenario == "text_turn") {
    make_app().Run();
    if (state.continues != 1) return fail("continues != 1");
    if (state.hidden_flags.size() != 1 || state.hidden_flags[0] != 0)
      return fail("text turn must not inject vision hidden states");
    if (state.resets != 0) return fail("clean first turn must not reset");
    if (state.vision_infers != 0) return fail("text turn must not run vision");
  } else if (scenario == "image_turn") {
    if (argc < 3) return fail("image_turn needs an image path");
    make_app().Run();
    if (state.load_images != 1) return fail("load_images != 1");
    if (state.predict_visions != 1) return fail("predict_visions != 1");
    if (state.vision_infers != 1) return fail("vision runner not called once");
    if (state.continues != 1) return fail("continues != 1");
    if (state.hidden_flags.empty() || state.hidden_flags[0] != 1)
      return fail("image turn must inject prompt hidden states");
    if (state.prompt_hiddens != 1) return fail("prompt_hiddens != 1");
  } else if (scenario == "reset_context") {
    make_app().Run();
    if (state.resets != 1) return fail("resets != 1");
    if (state.continues != 0) return fail("no generation expected");
  } else if (scenario == "oversize_prompt") {
    make_app().Run();
    if (state.continues != 0) return fail("oversize prompt must not generate");
  } else if (scenario == "history_trim") {
    make_app().Run();
    if (state.continues != 2) return fail("continues != 2");
    if (state.resets < 1) return fail("trim must reset the session");
    if (state.full_id_sizes.size() != 2 ||
        state.full_id_sizes[1] >= static_cast<size_t>(gemma4::kCacheLen))
      return fail("trimmed prompt must fit the KV cache");
  } else if (scenario == "rebuild_each_turn") {
    make_app(/*rebuild_each_turn=*/true).Run();
    if (state.continues != 2) return fail("continues != 2");
    if (state.resets < 2) return fail("rebuild mode must reset every turn");
  } else if (scenario == "session_growth") {
    make_app().Run();
    if (state.continues != 2) return fail("continues != 2");
    if (state.full_id_sizes.size() != 2 ||
        state.full_id_sizes[1] <= state.full_id_sizes[0])
      return fail("continuation context must grow across turns");
  } else {
    return fail("unknown scenario: " + scenario);
  }

  std::cout << "OK " << scenario << std::endl;
  return 0;
}
