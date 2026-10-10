// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cli.hpp"
#include "policy.hpp"
#include "sha256.h"
#include <iostream>
#include <utility>

int main(int argc, char **argv) {
  try {
    auto options = himloco::parse_cli(argc, argv);
    if (options.help) {
      std::cout << himloco::cli_help();
      return 0;
    }
    // Admit the exact published model before any output is created.
    himloco::verify_native_model(options.model.model_path);
    auto model_digest = rdk::sha256_file(options.model.model_path);
    auto inputs = himloco::discover_inputs(options.input);
    himloco::RunWorkspace workspace(options, std::move(inputs), model_digest);
    try {
      himloco::HimLoco model(options.model);
      workspace.note_runtime(model.model_name(), model.runtime_version(),
                             model.input_metadata(),
                             model.output_metadata());
      workspace.begin_warmup();
      auto first = himloco::load_input(workspace.inputs().records.front());
      for (int i = 0; i < options.warmup; ++i) {
        model.predict(first.first);
        workspace.warmup_completed(i + 1);
      }
      workspace.flush();
      for (const auto &input : workspace.inputs().records) {
        workspace.begin_record(input);
        auto loaded = himloco::load_input(input);
        auto result = model.predict(loaded.first);
        workspace.add_record(input, loaded.second, result);
      }
      workspace.complete();
    } catch (const std::exception &e) {
      workspace.fail(e);
      throw;
    }
    std::cout << "completed report=" << options.report << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "HIMLoco: " << e.what() << '\n';
    return 2;
  }
}
