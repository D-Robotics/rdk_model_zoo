/** @file chat_template.cc Chat template file loading for the legacy runtime. */
#include "chat_template.hpp"

#include <fstream>
#include <iterator>
#include <stdexcept>

std::string load_chat_template(const std::string& template_path) {
  std::ifstream input_template(template_path, std::ios::binary);
  if (!input_template) throw std::runtime_error("Cannot open chat template");
  const std::string templ{std::istreambuf_iterator<char>(input_template), {}};
  if (templ.empty() || templ.size() > kMaxChatTemplateBytes)
    throw std::runtime_error("Invalid chat template size");
  return templ;
}
