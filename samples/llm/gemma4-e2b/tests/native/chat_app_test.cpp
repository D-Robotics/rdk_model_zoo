// Scenario driver for the production Gemma4 chat model source.
//
// Links the real src/gemma4.cpp and src/cli.cpp against the engine
// doubles in chat_app_doubles.cpp and drives the Gemma4 model API directly:
// one predict call per chat turn, LoadImage for image turns, Reset and
// ContextUsage for the session commands. Each scenario asserts the
// observable session behavior (engine call counts, hidden-injection flags,
// reset counts) and prints "OK <scenario>" after exercising the real CLI
// presentation helpers, so the report lines stay the production ones. The
// doubles never load an HBM or touch a BPU; board behavior is not-run.

#include <iostream>
#include <string>
#include <vector>

#include "chat_app_doubles.hpp"
#include "cli.hpp"
#include "gemma4.hpp"
#include "gemma4_config.hpp"

namespace {

int fail(const std::string& message) {
  std::cerr << "FAIL: " << message << std::endl;
  return 1;
}

gemma4::Gemma4 MakeModel(bool rebuild_each_turn = false) {
  gemma4::ChatPaths paths;
  paths.text_hbm = "double-text.hbm";
  paths.vision_hbm = "double-vision.hbm";
  paths.tok_embeddings = "double-embed.bin";
  paths.tokenizer_json = "double-tokenizer.json";
  gemma4::ChatSettings settings;
  settings.rebuild_context_each_turn = rebuild_each_turn;
  return gemma4::Gemma4(paths, settings, gemma4::cli::StatusLineSink(),
                        gemma4::cli::DebugLineSink());
}

gemma4::ChatTurnResult RunTurn(gemma4::Gemma4& model, const std::string& text) {
  const gemma4::ChatTurnResult turn =
      model.predict(text, gemma4::cli::StreamSink());
  gemma4::cli::ReportTurn(turn);
  return turn;
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
    gemma4::Gemma4 model = MakeModel();
    const gemma4::ChatTurnResult turn = RunTurn(model, "hello");
    gemma4::cli::ReportContext(model.ContextUsage());
    if (turn.prompt_too_long) return fail("first turn must generate");
    if (turn.reply != "WW") return fail("streamed reply text mismatch");
    if (turn.streamed_tokens != 2) return fail("two tokens must stream");
    if (state.continues != 1) return fail("continues != 1");
    if (state.hidden_flags.size() != 1 || state.hidden_flags[0] != 0)
      return fail("text turn must not inject vision hidden states");
    if (state.resets != 0) return fail("clean first turn must not reset");
    if (state.vision_infers != 0) return fail("text turn must not run vision");
  } else if (scenario == "image_turn") {
    if (argc < 3) return fail("image_turn needs an image path");
    gemma4::Gemma4 model = MakeModel();
    gemma4::cli::ReportImageProcessing(argv[2]);
    gemma4::cli::ReportImageLoaded(model.LoadImage(argv[2]));
    const gemma4::ChatTurnResult turn = RunTurn(model, "what is this");
    if (turn.prompt_too_long) return fail("image turn must generate");
    if (state.load_images != 1) return fail("load_images != 1");
    if (state.predict_visions != 1) return fail("predict_visions != 1");
    if (state.vision_infers != 1) return fail("vision runner not called once");
    if (state.continues != 1) return fail("continues != 1");
    if (state.hidden_flags.empty() || state.hidden_flags[0] != 1)
      return fail("image turn must inject prompt hidden states");
    if (state.prompt_hiddens != 1) return fail("prompt_hiddens != 1");
  } else if (scenario == "reset_context") {
    gemma4::Gemma4 model = MakeModel();
    model.Reset();
    gemma4::cli::ReportSessionReset();
    gemma4::cli::ReportContext(model.ContextUsage());
    if (state.resets != 1) return fail("resets != 1");
    if (state.continues != 0) return fail("no generation expected");
  } else if (scenario == "oversize_prompt") {
    gemma4::Gemma4 model = MakeModel();
    const gemma4::ChatTurnResult turn =
        RunTurn(model, std::string(5000, 'a'));
    if (!turn.prompt_too_long) return fail("oversize prompt must be rejected");
    if (turn.prompt_tokens < 5000) return fail("prompt token count mismatch");
    if (state.continues != 0) return fail("oversize prompt must not generate");
  } else if (scenario == "history_trim") {
    gemma4::Gemma4 model = MakeModel();
    RunTurn(model, std::string(3000, 'x'));
    RunTurn(model, std::string(3000, 'y'));
    if (state.continues != 2) return fail("continues != 2");
    if (state.resets < 1) return fail("trim must reset the session");
    if (state.full_id_sizes.size() != 2 ||
        state.full_id_sizes[1] >= static_cast<size_t>(gemma4::kCacheLen))
      return fail("trimmed prompt must fit the KV cache");
  } else if (scenario == "rebuild_each_turn") {
    gemma4::Gemma4 model = MakeModel(/*rebuild_each_turn=*/true);
    RunTurn(model, "one");
    RunTurn(model, "two");
    if (state.continues != 2) return fail("continues != 2");
    if (state.resets < 2) return fail("rebuild mode must reset every turn");
  } else if (scenario == "session_growth") {
    gemma4::Gemma4 model = MakeModel();
    RunTurn(model, "first");
    RunTurn(model, "second");
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
