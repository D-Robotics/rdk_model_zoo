#include "gemma4_tokenizer.h"

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace gemma4 {

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

// Gemma chat template constants (the strings the tokenizer recognizes as
// single special tokens via its added_tokens table).
constexpr const char* kBos = "<bos>";
constexpr const char* kTurnOpen = "<|turn>";
constexpr const char* kTurnClose = "<turn|>";
constexpr const char* kImageMark = "\xF0\x9F\x96\xBC";  // 🖼 (U+1F5BC, no VS16)
constexpr const char* kBoi = "<|image>";
constexpr const char* kEoi = "<image|>";
constexpr int kSoftImageTokens = 280;

// Resolve the tokenizer directory from GEMMA4_HOME (preferred) or a few
// well-known relative locations. The directory must contain tokenizer.json
// and tokenizer_config.json.
std::string ResolveTokenizerDir(const std::string& hint) {
  if (!hint.empty() && fs::exists(fs::path(hint) / "tokenizer.json")) {
    return hint;
  }
  const char* env = std::getenv("GEMMA4_HOME");
  if (env && *env) {
    const std::string p = std::string(env) + "/tokenizer";
    if (fs::exists(fs::path(p) / "tokenizer.json")) return p;
  }
  for (const char* rel : {"tokenizer", "../tokenizer", "../../tokenizer"}) {
    if (fs::exists(fs::path(rel) / "tokenizer.json")) return rel;
  }
  throw std::runtime_error(
      "TokenizerBridge: cannot locate tokenizer.json. Set GEMMA4_HOME or pass "
      "the tokenizer directory explicitly.");
}

// JSON string unescape (inverse of JsonEscape in chat.cpp).
std::string JsonUnescape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\\' && i + 1 < s.size()) {
      switch (s[++i]) {
        case '\\': out += '\\'; break;
        case '"':  out += '"'; break;
        case 'n':  out += '\n'; break;
        case 'r':  out += '\r'; break;
        case 't':  out += '\t'; break;
        case '/':  out += '/'; break;
        default:   out += s[i]; break;
      }
    } else {
      out += s[i];
    }
  }
  return out;
}

// Render messages_json to the Gemma prompt text. Supports:
//  - content as string: "[{\"role\":\"user\",\"content\":\"hi\"}]"
//  - content as array:  content:[{type:image},{type:text,text:...}]
// This mirrors the non-tool branch of chat_template.jinja.
std::string RenderChat(const std::string& messages_json) {
  json msgs = json::parse(messages_json);
  std::string out = kBos;
  for (const auto& m : msgs) {
    const std::string role_raw = m.value("role", "user");
    const std::string role = (role_raw == "assistant") ? "model" : role_raw;
    out += std::string(kTurnOpen) + role + "\n";

    const auto& content = m["content"];
    if (content.is_string()) {
      out += JsonUnescape(content.get<std::string>());
    } else if (content.is_array()) {
      for (const auto& part : content) {
        const std::string type = part.value("type", "");
        if (type == "text") {
          out += JsonUnescape(part.value("text", std::string{}));
        } else if (type == "image") {
          out += kImageMark;
        }
      }
    }
    out += std::string(kTurnClose) + "\n";
  }
  // add_generation_prompt: open the model turn.
  out += std::string(kTurnOpen) + "model\n";
  return out;
}

// Expand each image mark 🖼️ into <|image> + (🖼️)*280 + <image|>, matching the
// python bridge's expand_image_tokens().
std::string ExpandImageTokens(const std::string& text) {
  std::string soft_block;
  soft_block.reserve(std::strlen(kImageMark) * kSoftImageTokens);
  for (int i = 0; i < kSoftImageTokens; ++i) soft_block += kImageMark;
  const std::string replacement = std::string(kBoi) + soft_block + kEoi;

  std::string out;
  out.reserve(text.size());
  const std::string mark = kImageMark;
  size_t pos = 0;
  while (pos < text.size()) {
    const size_t hit = text.find(mark, pos);
    if (hit == std::string::npos) {
      out.append(text, pos, std::string::npos);
      break;
    }
    out.append(text, pos, hit - pos);
    out += replacement;
    pos = hit + mark.size();
  }
  return out;
}

}  // namespace

TokenizerBridge::TokenizerBridge(const std::string& tokenizer_dir) {
  const std::string dir = ResolveTokenizerDir(tokenizer_dir);
  tokenizer_ = Tokenizer::Create(dir);
  if (!tokenizer_) {
    throw std::runtime_error("TokenizerBridge: failed to create tokenizer from " + dir);
  }
}

std::vector<int64_t> TokenizerBridge::EncodeMessagesJson(
    const std::string& messages_json, bool expand_images) const {
  std::string rendered = RenderChat(messages_json);
  if (expand_images) {
    rendered = ExpandImageTokens(rendered);
  }
  // The tokenizer's added_tokens table makes special tokens (<bos>, <|turn>,
  // <|image>, <image|>) match as single tokens even with add_special_tokens off.
  const auto ids = tokenizer_->Encode(rendered);
  return std::vector<int64_t>(ids.begin(), ids.end());
}

std::string TokenizerBridge::DecodeIds(const std::vector<int64_t>& ids) const {
  std::vector<int32_t> i32(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    i32[i] = static_cast<int32_t>(ids[i]);
  }
  return tokenizer_->Decode(i32);
}

}  // namespace gemma4
