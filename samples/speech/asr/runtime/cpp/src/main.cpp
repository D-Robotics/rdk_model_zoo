// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "asr.hpp"
#include "cli.hpp"
#include <iostream>
#include <optional>
int main(int argc, char **argv) {
  std::optional<asr::RunWorkspace> workspace;
  try {
    const auto options = asr::parse_cli(argc, argv);
    if (options.help) {
      std::cout << asr::cli_help();
      return 0;
    }
    // Admit board/model/vocabulary before any output is created.
    auto gate = asr::make_preflight(options.model_sha256, options.vocabulary);
    gate(options.model);
    const auto vocabulary = asr::load_vocabulary(options.vocabulary);
    const auto audio_sha = asr::file_digest(options.audio);
#ifdef ASR_HOST_FIXTURE
    const char *backend = "host-fixture";
#else
    const char *backend = "native-sdk";
#endif
    workspace.emplace(options, audio_sha, backend);
    // Named model construction: the ASR owns its native runner and metadata.
    asr::ASR model(options.model, gate, vocabulary,
                   options.decode_mode == "ctc" ? asr::DecodeMode::Ctc
                                                : asr::DecodeMode::Legacy);
    workspace->note_metadata(model.metadata());
    asr::AudioReader reader(options.audio);
    asr::AudioChunk chunk;
    while (reader.next(chunk))
      workspace->add_chunk(chunk, model.predict(chunk));
    if (!workspace->chunk_count())
      throw std::invalid_argument("No audio chunks processed");
    workspace->complete();
    std::cout << workspace->text() << "\nReport: "
              << (workspace->output() / "result.json").string() << '\n';
    return 0;
  } catch (const std::exception &e) {
    if (workspace)
      workspace->fail(e);
    std::cerr << "error: " << e.what() << '\n';
    return 2;
  }
}
