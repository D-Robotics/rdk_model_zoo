// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cli.hpp"
#include "pipeline.hpp"
#include <array>
#include <exception>
#include <iostream>
#include <optional>
#include <string>

int main(int argc, char **argv) {
  std::optional<paraformer::RunWorkspace> workspace;
  std::string current;
  try {
    const auto options = paraformer::parse_cli(argc, argv);
#ifdef PARAFORMER_HOST_FIXTURE
    const std::string backend = "host-fixture";
#else
    const std::string backend = "native-sdk";
#endif
    if (options.help) {
      std::cout << paraformer::cli_help() << "Backend: " << backend << '\n';
      return 0;
    }
    auto gate = paraformer::make_preflight(options.models, options.vocabulary);
    const auto vocabulary = paraformer::load_vocabulary(options.vocabulary);
    const auto manifest_sha = paraformer::file_digest(options.manifest);
    const auto items =
        paraformer::load_prepared_manifest(options.manifest, options.max_utts);
    paraformer::require_manifest_intact(options.manifest, manifest_sha);
    workspace.emplace(options, manifest_sha, backend);
    // The named three-stage model owns all SDK runners; main only constructs
    // it, reads prepared features and calls predict once per utterance.
    paraformer::Pipeline pipeline(options.models, gate, vocabulary);
    const std::array<paraformer::Stage, 3> stages{
        paraformer::Stage::Encoder, paraformer::Stage::Predictor,
        paraformer::Stage::Decoder};
    for (size_t i = 0; i < stages.size(); ++i)
      workspace->note_metadata(pipeline.metadata(stages[i]), i);
    for (const auto &item : items) {
      current = item.utt_id;
      const auto features = paraformer::load_features(item);
      workspace->mark_attempted();
      const auto result = pipeline.predict(features, item.valid_frames);
      workspace->add_record(item, result);
      std::cout << item.utt_id << ": " << result.text << '\n';
    }
    current.clear();
    gate(options.models[0].model);
    workspace->complete(items);
    std::cout << "Report: " << workspace->output() << '\n';
    return 0;
  } catch (const std::exception &error) {
    if (workspace)
      workspace->fail(error, current);
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
