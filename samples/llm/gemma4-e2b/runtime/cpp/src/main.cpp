/**
 * @file main.cpp
 * @brief Entry point for the interactive Gemma4-E2B VLM chat.
 *
 * This file stays deliberately thin: the CLI layer (cli.hpp/.cpp) parses the
 * command line into ChatOptions, then main visibly constructs the named
 * Gemma4 model and runs the read-dispatch loop — one model.predict call per
 * chat turn over the parsed TerminalCommand. Invalid terminal input
 * re-prompts and continues; only end-of-input or /quit exits. Console
 * presentation (banner, prompt, result lines) lives in the CLI; all model
 * execution and conversation bookkeeping live in gemma4.hpp/.cpp and the
 * engines.
 *
 * @note Primary executable of this Model Zoo sample; built as `main`.
 */
#include <iostream>
#include <string>

#include "cli.hpp"
#include "gemma4.hpp"

int main(int argc, char** argv) {
  gemma4::cli::ChatOptions options;
  if (!gemma4::cli::ParseOptions(argc, argv, &options)) {
    return 2;
  }

  try {
    gemma4::cli::PrintBanner();
    gemma4::Gemma4 model(options.paths, options.settings,
                         gemma4::cli::StatusLineSink(),
                         gemma4::cli::DebugLineSink());

    gemma4::cli::PrintHelp();
    gemma4::cli::PrintPrompt();
    std::string line;
    for (;;) {
      const gemma4::cli::ReadStatus status = gemma4::cli::ReadLine(&line);
      if (status == gemma4::cli::ReadStatus::kEof) {
        break;
      }
      if (status == gemma4::cli::ReadStatus::kInvalid) {
        // ReadLine printed the terminal-configuration error; re-prompt and
        // keep the session alive on undecodable input.
        gemma4::cli::PrintPrompt();
        continue;
      }

      const gemma4::cli::TerminalCommand command =
          gemma4::cli::ParseCommand(line);
      switch (command.command) {
        case gemma4::cli::Command::kQuit:
          return 0;
        case gemma4::cli::Command::kHelp:
          gemma4::cli::PrintHelp();
          break;
        case gemma4::cli::Command::kReset:
          model.Reset();
          gemma4::cli::ReportSessionReset();
          break;
        case gemma4::cli::Command::kContext:
          gemma4::cli::ReportContext(model.ContextUsage());
          break;
        case gemma4::cli::Command::kImage: {
          std::string img_path;
          if (!gemma4::cli::ParseImageArgument(command.text, &img_path)) {
            break;
          }
          gemma4::cli::ReportImageProcessing(img_path);
          gemma4::cli::ReportImageLoaded(model.LoadImage(img_path));
          break;
        }
        case gemma4::cli::Command::kEmpty:
          break;
        case gemma4::cli::Command::kChat:
          gemma4::cli::ReportTurn(
              model.predict(command.text, gemma4::cli::StreamSink()));
          break;
      }
      gemma4::cli::PrintPrompt();
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << std::endl;
    return 1;
  }
}
