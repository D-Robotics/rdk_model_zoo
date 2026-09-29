/** @file runtime_config.hpp
 * @brief Model validation and OELLM runtime configuration file handling.
 *
 * Configuration and file IO live here, outside the inference-stage file.
 */
#pragma once

#include <filesystem>

#include "oellm_runtime_basic/oellm_runtime.h"

namespace minicpm5 {

/** @brief Verify the required model files inside an extracted directory.
 * @param model_directory Candidate extracted S600 model directory.
 * @return The canonical directory path used for runtime configuration.
 * @throws std::runtime_error if the path is missing any required file.
 */
std::filesystem::path validate_model_directory(
    const std::filesystem::path& model_directory);

/** @brief Write the runtime JSON configuration and initialize the runtime.
 *
 * The temporary configuration file is owned by an RAII guard, so it is
 * removed on success, SDK error return and exception paths alike.
 * @param model_directory Extracted S600 model directory.
 * @param runtime Uninitialized OELLM runtime instance.
 * @throws std::runtime_error on temporary-file or SDK initialization failure.
 */
void configure_runtime(const std::filesystem::path& model_directory,
                       oellm::OellmRuntime& runtime);
}  // namespace minicpm5
