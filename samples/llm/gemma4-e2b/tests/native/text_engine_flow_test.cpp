// Behavior suite for the refactored TextEngine orchestration against the
// host SDK double: public call compositions, continuation and reset, EOS and
// callback stops, image embeddings, malformed inputs, failure ownership and
// session reuse, benchmark scope, and the explicit debug sink. Host doubles
// are not vendor ABI or model evidence.
#include "text_fixture.hpp"

#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

using gemma4::TextEngine;
using text_fixture::kHiddenSize;

template <class F> bool Throws(F action) {
  try {
    action();
  } catch (const std::exception &) {
    return true;
  }
  return false;
}

// Captures stderr through a temp file and asserts the library printed
// nothing while no sink is installed.
class QuietStderr {
 public:
  QuietStderr() {
    ::fflush(stderr);
    file_ = std::tmpfile();
    assert(file_ != nullptr);
    original_ = dup(fileno(stderr));
    assert(dup2(fileno(file_), fileno(stderr)) >= 0);
  }
  ~QuietStderr() {
    std::fflush(stderr);
    dup2(original_, fileno(stderr));
    ::close(original_);
    std::fclose(file_);
  }
  bool Empty() {
    std::fflush(stderr);
    return std::ftell(file_) == 0;
  }

 private:
  std::FILE *file_ = nullptr;
  int original_ = -1;
};

}  // namespace

int main() {
  // ---- one-shot generation through the composed pipeline ----
  {
    text_fixture::Reset();
    TextEngine engine("fixture.hbm", "unused");
    text_fixture::state().expect_cache_rows = 5;  // Verified by first decode.
    const auto out = engine.Generate({11, 22, 33, 44, 55}, 3);
    // Prefill argmax row 4 -> 104; each decode step argmaxes row 0 -> 100.
    assert((out == std::vector<int64_t>{11, 22, 33, 44, 55, 104, 100, 100}));
    assert(engine.ProcessedTokens() == 7);  // Five prompt + two decode steps.
    assert(text_fixture::state().infer_calls ==
           1 + 2);  // One prefill chunk + two decode steps.
  }
  assert(text_fixture::state().buffers.empty());
  assert(text_fixture::state().tasks.empty());

  // ---- streaming with an early callback stop ----
  {
    text_fixture::Reset();
    TextEngine engine("fixture.hbm", "unused");
    std::vector<int64_t> seen;
    const auto out = engine.GenerateStream(
        {11, 22, 33, 44, 55}, 5, [&](int64_t token) {
          seen.push_back(token);
          return seen.size() < 2;
        });
    assert(seen.size() == 2);           // Callback stopped after two tokens.
    assert(out.size() == 7);
    assert(out.back() == seen.back());  // Prefix already generated is kept.
    // Streaming resets first: this run prefilled exactly one chunk.
    assert(text_fixture::state().infer_calls == 1 + 1);
  }
  assert(text_fixture::state().buffers.empty());

  // ---- EOS stop (fixture row 6 -> token 106 = kTurnEndTokenId) ----
  {
    text_fixture::Reset();
    TextEngine engine("fixture.hbm", "unused");
    // A 255-token prompt prefills one chunk; argmax row 254 -> 354.
    std::vector<int64_t> prompt(255, 40);
    prompt[0] = 11;
    const auto out = engine.Generate(prompt, 4);
    assert(out.size() == 255 + 4);
    assert(engine.ProcessedTokens() == 258);  // Prompt + three decode steps.
    // Continue from the generated prefix: 259 % 256 != 0 -> align to 256,
    // replay 3 tokens through a context shift, then prefill the suffix.
    const int infer_before = text_fixture::state().infer_calls;
    const auto next = engine.ContinueGenerate(out, 2);
    // The returned vector always extends the caller's full id list.
    assert(next.size() == out.size() + 2);
    assert(std::equal(out.begin(), out.end(), next.begin()));
    assert(engine.ProcessedTokens() == 260);
    assert(text_fixture::state().infer_calls > infer_before);
    // ContextShift keeps the prefix: shift to 256 discards 260 - 256 = 4.
    assert(engine.ContextShift(256) == 4);
    assert(engine.ProcessedTokens() == 256);
  }
  assert(text_fixture::state().buffers.empty());

  // ---- unaligned continuation replays the prefix (source alignment) ----
  {
    text_fixture::Reset();
    TextEngine engine("fixture.hbm", "unused");
    std::vector<int64_t> prompt(250, 40);
    const auto first = engine.Generate(prompt, 2);
    assert(engine.ProcessedTokens() == 251);  // Prompt + one decode step.
    // 251 % 256 != 0: alignment aligns DOWN to the chunk boundary (0 here),
    // so the whole prefix is replayed through a context shift before the
    // suffix prefill — exactly the source's continuation behavior.
    const int infer_before = text_fixture::state().infer_calls;
    const auto second = engine.ContinueGenerate(first, 3);
    assert(second.size() == first.size() + 3);
    assert(std::equal(first.begin(), first.end(), second.begin()));
    assert(engine.ProcessedTokens() == 254);
    assert(text_fixture::state().infer_calls > infer_before);  // Replay prefill.
  }
  assert(text_fixture::state().buffers.empty());

  // ---- aligned continuation: complete ids mean no prefill ----
  {
    text_fixture::Reset();
    TextEngine engine("fixture.hbm", "unused");
    std::vector<int64_t> prompt(255, 40);
    prompt[0] = 11;
    const auto out = engine.Generate(prompt, 2);  // processed = 256 (aligned).
    assert(engine.ProcessedTokens() == 256);
    // out carries one undecoded-id ahead of the session (the last decoded
    // token is only prefilled by the next continuation); feeding exactly the
    // processed prefix means no new ids and therefore no prefill inference.
    const std::vector<int64_t> prefix(out.begin(), out.begin() + 256);
    const int infer_before = text_fixture::state().infer_calls;
    const auto next = engine.ContinueGenerate(prefix, 1);
    assert(next.size() == prefix.size() + 1);
    assert(engine.ProcessedTokens() == 256);  // max_new=1 runs no decode.
    assert(text_fixture::state().infer_calls == infer_before);
  }
  assert(text_fixture::state().buffers.empty());

  // ---- image embeddings path (prebuilt hidden carried into inputs) ----
  {
    text_fixture::Reset();
    TextEngine engine("fixture.hbm", "unused");
    std::vector<int64_t> prompt = {11, 22, gemma4::kImageTokenId, 44};
    std::vector<float> hidden = engine.BuildPromptHidden(
        prompt, std::vector<float>(kHiddenSize, 7.5f));
    assert(hidden.size() == prompt.size() * kHiddenSize);
    // One generated token keeps the prefill as the last inference, so the
    // recorded first embed float is the prefill row: the text id 11 (the
    // image slot row 7.5 sits further into the same buffer).
    const auto out = engine.GenerateWithPromptEmbeddings(prompt, hidden, 1);
    assert(out.size() == prompt.size() + 1);
    assert(text_fixture::state().last_first_embed == 11.f);
    assert(engine.ProcessedTokens() == 4);  // Prefill only, no decode step.
  }
  assert(text_fixture::state().buffers.empty());

  // ---- malformed inputs ----
  {
    text_fixture::Reset();
    TextEngine engine("fixture.hbm", "unused");
    const auto out = engine.Generate({11, 22, 33, 44, 55}, 2);
    assert(engine.ProcessedTokens() == 6);
    // Shorter than the processed prefix.
    assert(Throws([&] { engine.ContinueGenerate({1, 2}, 3); }));
    assert(engine.ProcessedTokens() == 6);  // Session unchanged.
    // Hidden-size mismatch is rejected before any session change.
    assert(Throws([&] {
      engine.GenerateWithPromptEmbeddings({11, 22}, {1.f, 2.f}, 2);
    }));
    assert(engine.ProcessedTokens() == 6);
    // The engine remains usable afterwards.
    const auto again = engine.Generate({11, 22, 33, 44, 55}, 1);
    assert(again.size() == 6 && again.back() == 104);
  }
  assert(text_fixture::state().buffers.empty());

  // ---- failure ownership and session reuse ----
  {
    text_fixture::Reset();
    TextEngine engine("fixture.hbm", "unused");
    const auto out = engine.Generate({11, 22, 33, 44, 55}, 1);
    assert(engine.ProcessedTokens() == 5);  // max_new=1 runs no decode.
    text_fixture::state().fail_infer = true;
    assert(Throws([&] { engine.ContinueGenerate(out, 3); }));
    // Source semantics: the continuation aligns the unaligned prefix first
    // (shift to zero), so the failed prefill leaves the session at the
    // aligned state, not at the pre-call state.
    assert(engine.ProcessedTokens() == 0);
    text_fixture::state().fail_infer = false;
    // Failure released no engine buffers: a retry through a fresh session
    // produces the documented outputs.
    const auto retried = engine.Generate({11, 22, 33, 44, 55}, 2);
    assert((retried ==
            std::vector<int64_t>{11, 22, 33, 44, 55, 104, 100}));
    assert(engine.ProcessedTokens() == 6);
  }
  assert(text_fixture::state().buffers.empty());

  // ---- auto truncate and context shift execution ----
  {
    text_fixture::Reset();
    TextEngine engine("fixture.hbm", "unused");
    const auto out = engine.Generate({11, 22, 33, 44, 55}, 3);
    assert(engine.ProcessedTokens() == 7);
    engine.SetKeepTokens(2);
    // Fits: no truncation.
    assert(!engine.AutoTruncate(64, 16));
    assert(engine.ProcessedTokens() == 7);
    // Marginal overflow (needed 4090 > available 4089, overflow 1 <= the 5
    // discardable tokens): keep 2, discard 5.
    assert(engine.AutoTruncate(4000, 90));
    assert(engine.ProcessedTokens() == 2);
    // Continuation replays the requested ids from the retained prefix
    // (alignment shifts the unaligned count of 2 to zero and re-prefills).
    const auto next = engine.ContinueGenerate(out, 1);
    assert(next.size() == out.size() + 1);
    assert(engine.ProcessedTokens() == 8);
  }
  assert(text_fixture::state().buffers.empty());

  // ---- benchmark scope (warmup outside the timed window, EOS break) ----
  {
    text_fixture::Reset();
    TextEngine engine("fixture.hbm", "unused");
    const auto bench = engine.Benchmark({11, 22, 33, 44, 55}, 2, 1);
    assert(bench.decode_steps == 1);  // Timed loop runs max_new_tokens - 1.
    assert(bench.load_ms >= 0.0 && bench.prefill_ms >= 0.0 &&
           bench.decode_ms >= 0.0);
    assert(bench.tokens_per_sec >= 0.0);
  }
  assert(text_fixture::state().buffers.empty());

  // ---- explicit debug sink; no implicit library output ----
  {
    text_fixture::Reset();
    TextEngine engine("fixture.hbm", "unused");
    std::vector<std::string> messages;
    engine.SetDebugSink([&](const std::string &message) {
      messages.push_back(message);
    });
    std::vector<int64_t> prompt = {11, 22, gemma4::kImageTokenId, 44};
    std::vector<float> hidden =
        engine.BuildPromptHidden(prompt, std::vector<float>(kHiddenSize, 7.5f));
    engine.GenerateWithPromptEmbeddings(prompt, hidden, 1);
    assert(messages.size() == 1);
    assert(messages[0].find("using prebuilt_hidden") != std::string::npos);
    assert(messages[0].find("chunk_start=0") != std::string::npos);
  }
  {
    text_fixture::Reset();
    TextEngine engine("fixture.hbm", "unused");
    std::vector<std::string> messages;
    engine.SetDebugSink([&](const std::string &message) {
      messages.push_back(message);
    });
    // Continuation alignment also reports through the sink.
    std::vector<int64_t> prompt(300, 40);
    const auto out = engine.Generate(prompt, 2);
    engine.ContinueGenerate(out, 1);
    bool alignment_reported = false;
    for (const auto &message : messages)
      if (message.find("KV reuse aligned") != std::string::npos)
        alignment_reported = true;
    assert(alignment_reported);
  }
  {
    // Without a sink the library writes nothing to stderr.
    text_fixture::Reset();
    TextEngine engine("fixture.hbm", "unused");
    QuietStderr quiet;
    std::vector<int64_t> prompt = {11, 22, gemma4::kImageTokenId, 44};
    std::vector<float> hidden =
        engine.BuildPromptHidden(prompt, std::vector<float>(kHiddenSize, 7.5f));
    engine.GenerateWithPromptEmbeddings(prompt, hidden, 2);
    engine.ContinueGenerate(engine.Generate({11, 22, 33, 44, 55}, 2), 1);
    assert(quiet.Empty());
  }
  // ---- public continuation validates the hidden extent (TEXT-R2) ----
  {
    text_fixture::Reset();
    TextEngine engine("fixture.hbm", "unused");
    std::vector<float> short_hidden(1, 0.f);
    // Both public entry points reject a hidden that cannot cover the prompt
    // rows stage 1 indexes.
    assert(Throws([&] {
      engine.ContinueGenerate({11, 22}, 1, &short_hidden);
    }));
    assert(Throws([&] {
      engine.ContinueGenerateStream({11, 22}, 1,
                                    [](int64_t) { return true; }, &short_hidden);
    }));
    assert(engine.ProcessedTokens() == 0);  // No session mutation happened.
    // Rejection leaves a clean, reusable session.
    const auto out = engine.Generate({11, 22, 33, 44, 55}, 1);
    assert(out.size() == 6 && out.back() == 104);
    assert(engine.ProcessedTokens() == 5);

    // Nonzero prefix offset: a mismatched hidden is still rejected before
    // any state change.
    std::vector<float> wrong(3 * kHiddenSize, 0.f);  // Needs six rows here.
    assert(Throws([&] { engine.ContinueGenerate(out, 1, &wrong); }));
    assert(engine.ProcessedTokens() == 5);
    assert(Throws([&] {
      engine.ContinueGenerateStream(out, 1, [](int64_t) { return true; },
                                    &wrong);
    }));
    assert(engine.ProcessedTokens() == 5);

    // Full-size hidden covering the whole prompt is accepted. The prefix
    // (5 tokens) is unaligned, so the source alignment shifts to zero and
    // re-prefills the whole sequence from prompt row 0 with the provided
    // hidden — demonstrating whole-prompt indexing, not suffix storage.
    std::vector<float> full(out.size() * kHiddenSize, 0.f);
    for (size_t i = 0; i < out.size(); ++i)
      full[i * kHiddenSize] = static_cast<float>(out[i]);
    const auto next = engine.ContinueGenerateStream(
        out, 1, [](int64_t) { return true; }, &full);
    assert(next.size() == out.size() + 1);
    assert(text_fixture::state().last_first_embed == 11.f);  // Prompt row 0.
    assert(engine.ProcessedTokens() == 6);
  }
  assert(text_fixture::state().buffers.empty());

  // ---- valid 4096-window generation stays supported (TEXT-R1 boundary) ----
  {
    text_fixture::Reset();
    TextEngine engine("fixture.hbm", "unused");
    std::vector<int64_t> prompt(4096, 40);  // Chunks 0..3840: the full window.
    prompt[0] = 11;
    // One generated token: the prefill is the only inference, so the last
    // chunk (chunk_start 3840) must prepare and run without any write
    // outside its buffers.
    const auto out = engine.Generate(prompt, 1);
    assert(out.size() == 4097);
    assert(engine.ProcessedTokens() == 4096);
    // Continuing past the fixed context is rejected by the geometry
    // contract at preparation time instead of silently clamping attention
    // (documented limit; AutoTruncate is the application tool to stay
    // inside it). The session is unchanged by the rejection.
    assert(Throws([&] { engine.ContinueGenerate(out, 1); }));
    assert(engine.ProcessedTokens() == 4096);
  }
  assert(text_fixture::state().buffers.empty());

  std::cout << "text engine flow checks passed\n";
  return 0;
}
