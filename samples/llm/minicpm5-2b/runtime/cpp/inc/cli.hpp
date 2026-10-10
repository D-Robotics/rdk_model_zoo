/** @file cli.hpp
 * @brief gflags command line and RESULT reporting for the S600 CLI.
 */
#pragma once

#include <cstdint>
#include <exception>
#include <string>

#include "minicpm5.hpp"

namespace minicpm5 {

/** @brief CLI flags controlling the model location and both turns. */
struct CliOptions {
  std::string model_path = "../../model/s600";  ///< Verified model directory.
  std::string prompt =
      "What is 1+1? Give a short answer.";      ///< First UTF-8 user prompt.
  std::string follow_up;                        ///< Optional second prompt.
  int32_t max_new_tokens = 128;  ///< Per-request limit, from 1 to 4096.
};

/** @brief Parse gflags into options; usage errors exit inside gflags.
 * @param[in,out] argc Argument count from main.
 * @param[in,out] argv Argument values from main.
 * @param[out] options Receives the parsed flag values.
 * @return True when parsing succeeded; false on an unexpected positional
 * argument, for which the caller exits with status 2 after this function
 * printed the diagnostic.
 */
bool parse_cli(int argc, char* argv[], CliOptions& options);

/** @brief Emit one machine-readable RESULT line for a completed turn.
 * @param turn One-based turn index.
 * @param result Completed generation returned by MiniCPM5::predict.
 */
void emit_result(int turn, const Result& result);

/** @brief Report a fatal error on stderr.
 * @param error Exception that aborted the run.
 * @return Process exit code 1.
 */
int fail(const std::exception& error);
}  // namespace minicpm5
