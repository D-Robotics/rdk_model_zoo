/** @file cli.hpp Command-line parsing and RESULT rendering for S100/S100P. */
#pragma once
#include <exception>
#include <functional>
#include <string>

#include "minicpm5.hpp"
/** Explicit paths and prompt normally supplied by run.sh. */
struct CliOptions {
  std::string model_path;      ///< Board-specific HBM model file.
  std::string tokenizer_path;  ///< Prepared tokenizer metadata directory.
  std::string template_path;   ///< Prepared non-thinking Jinja template.
  /** UTF-8 user text; defaults to the self-introduction prompt. */
  std::string prompt = "请用一句话介绍你自己。";
};
/** Parse explicit options.
 * @param[in,out] argc Argument count from main.
 * @param[in,out] argv Argument values from main.
 * @param[out] options Receives the parsed values.
 * @return True when a request should run; false after --help printed usage
 * (the caller exits zero).
 * @throws std::runtime_error With the source messages for a missing value
 * ("Missing value for <option>"), an unknown option ("Unknown option:
 * <option>") or an empty required field ("Model, tokenizer, template and
 * prompt must be nonempty; use run.sh for defaults").
 */
bool parse_cli(int argc, char* argv[], CliOptions& options);
/** Streaming sink writing each chunk to stdout, as in the source main. */
std::function<void(const char*)> stdout_sink();
/** Render the source RESULT line for one completed request, preceded by a
 * WARNING line when the streaming consumer failed mid-request.
 * @param outcome Unprinted predict() return value.
 */
void emit_result(const RequestOutcome& outcome);
/** Report a fatal error on stderr.
 * @param error Exception that aborted the run.
 * @return Process exit code 1.
 */
int fail(const std::exception& error);
