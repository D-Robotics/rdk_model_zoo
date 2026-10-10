/**
 * @file cli.hpp
 * @brief Command-line and console layer for the interactive Gemma4-E2B chat.
 *
 * The CLI owns everything the user sees and everything argument-shaped: the
 * gflags command line with $GEMMA4_HOME default-path resolution and
 * validation (ParseOptions), the terminal-input REPL parsing (ReadLine +
 * ParseCommand), the banner/help/prompt, UTF-8/GB18030 terminal-input
 * normalization and the presentation of model results (image loading,
 * context usage, streamed-token echo wiring, per-turn timing). It performs
 * no inference and no session bookkeeping — the Gemma4 model in gemma4.hpp
 * owns those, and main.cpp stays a thin entry that constructs the named
 * model and calls predict per turn.
 */

#pragma once

#include <string>

#include "gemma4.hpp"

namespace gemma4::cli {

/** @brief Parsed command-line configuration for the chat executable. */
struct ChatOptions {
  ChatPaths paths;     ///< Resolved model/tokenizer locations.
  ChatSettings settings;  ///< Generation budget for every turn.
};

/**
 * @brief Parse the gflags command line into options.
 *
 * Owns the flag definitions, the usage message, the $GEMMA4_HOME-based
 * default-path resolution and the generation-setting validation. On a
 * validation failure the error is printed to stderr and false is returned
 * (the caller exits with code 2).
 *
 * @param argc    Argument count exactly as received by main.
 * @param argv    Argument vector exactly as received by main.
 * @param options Output for the resolved paths and settings.
 */
bool ParseOptions(int argc, char** argv, ChatOptions* options);

/** @brief One parsed REPL input line. */
enum class Command {
  kChat,   ///< Ordinary chat text; #text carries the message.
  kHelp,   ///< "/help".
  kReset,  ///< "/reset".
  kContext,  ///< "/context".
  kImage,  ///< "/image <path>"; #text carries the raw argument.
  kQuit,   ///< "/quit" or "/exit".
  kEmpty,  ///< An empty line.
};

/** @brief Result of parsing one terminal line. */
struct TerminalCommand {
  Command command = Command::kEmpty;
  std::string text;  ///< Chat message (kChat) or raw image argument (kImage).
};

/**
 * @brief Classify one normalized terminal line into a REPL command.
 *
 * "/quit"/"/exit", "/help", "/reset" and "/context" match exactly; "/image "
 * (with trailing space) introduces an image argument, which #text carries
 * unvalidated — ParseImageArgument checks it. Everything else, including
 * unknown slash words, is chat text, and an empty line is kEmpty.
 */
TerminalCommand ParseCommand(const std::string& line);

/** @brief Sink printing model status lines to stdout (used as model status). */
ChatEventSink StatusLineSink();

/** @brief Sink printing model debug lines to stderr (used as model debug). */
ChatEventSink DebugLineSink();

/** @brief Sink echoing streamed generation fragments to stdout. */
ChatEventSink StreamSink();

/** @brief Print the ANSI banner shown once at startup. */
void PrintBanner();

/** @brief Print the command reference shown at startup and on /help. */
void PrintHelp();

/** @brief Print the REPL prompt ("gemma4> ") without a newline. */
void PrintPrompt();

/** @brief Outcome of reading one terminal line. */
enum class ReadStatus {
  kEof,      ///< stdin is exhausted.
  kInvalid,  ///< Neither valid UTF-8 nor GB18030; an error was printed.
  kOk,       ///< A normalized UTF-8 line is available.
};

/**
 * @brief Read one line from stdin and normalize it to UTF-8.
 *
 * Strips a trailing CR, validates UTF-8 and falls back to GB18030
 * conversion (printing the conversion notice on stderr). On invalid input
 * the terminal-configuration error is printed and kInvalid is returned;
 * the caller re-prompts and continues rather than exiting.
 */
ReadStatus ReadLine(std::string* line);

/**
 * @brief Validate the argument of an "/image <path>" command line.
 *
 * Trims whitespace and checks that the file is readable, printing the
 * original error message and returning false otherwise.
 */
bool ParseImageArgument(const std::string& argument, std::string* path);

/** @brief Report that vision processing of @p path has started. */
void ReportImageProcessing(const std::string& path);

/** @brief Report the loaded image feature count and the follow-up hint. */
void ReportImageLoaded(size_t feature_count);

/** @brief Report "Session reset." after Gemma4::Reset(). */
void ReportSessionReset();

/** @brief Print the KV-cache usage line for /context. */
void ReportContext(const ChatContextUsage& usage);

/** @brief Print the oversized-prompt error or the per-turn timing line. */
void ReportTurn(const ChatTurnResult& turn);

}  // namespace gemma4::cli
