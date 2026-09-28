/** @file chat_template.hpp Non-thinking chat template loading for OELLM 1.0.0.
 *
 * Template file IO lives here, outside the inference-stage file.
 */
#pragma once
#include <cstddef>
#include <string>

/** Largest accepted chat template payload in bytes. */
constexpr std::size_t kMaxChatTemplateBytes = 65535;

/** Read a prepared non-thinking text Jinja template file.
 * @param template_path Path to the prepared template file.
 * @return Raw template bytes; prepare_request copies them before inference.
 * @throws std::runtime_error If the file cannot be opened, is empty, or
 * exceeds kMaxChatTemplateBytes.
 */
std::string load_chat_template(const std::string& template_path);
