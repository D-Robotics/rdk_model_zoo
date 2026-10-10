/**
 * @file gemma4_server.cpp
 * @brief OpenAI-compatible HTTP text chat server for Gemma4-E2B.
 *
 * Keeps the Text HBM loaded, reuses the KV cache when consecutive requests
 * share a token prefix, and exposes /v1/chat/completions for ChatBox and other
 * OpenAI-compatible clients. Image chat remains in the interactive main
 * executable because this server currently implements the text-only request
 * schema and runtime path.
 */
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "gflags/gflags.h"
#include <nlohmann/json.hpp>

#include "gemma4_config.hpp"
#include "gemma4_text_engine.hpp"
#include "gemma4_tokenizer.hpp"

// Empty path defaults are resolved from $GEMMA4_HOME.
DEFINE_string(text_hbm, "",
              "Path to text LLM *.hbm. Default: $GEMMA4_HOME/model/"
              "gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm");
DEFINE_string(tok_embeddings, "",
              "Path to tok_embeddings.bin. Default: "
              "$GEMMA4_HOME/model/tok_embeddings.bin");
DEFINE_string(tokenizer_path, "",
              "Path to tokenizer.json. Default: "
              "$GEMMA4_HOME/tokenizer/tokenizer.json");
DEFINE_string(host, "0.0.0.0", "HTTP listen address.");
DEFINE_int32(port, 8000, "HTTP listen port.");
DEFINE_string(model, "gemma4-e2b", "Model id returned by the OpenAI API.");
DEFINE_int32(max_tokens, 0,
             "Default maximum new tokens per request. 0 uses all KV capacity "
             "remaining after the prompt.");
DEFINE_int32(min_response_tokens, 256,
             "Minimum response capacity preserved when old turns are trimmed.");
DEFINE_int32(request_limit_mb, 4, "Maximum HTTP request body size in MiB.");

namespace {

using json = nlohmann::json;

constexpr size_t kMaxHeaderBytes = 64 * 1024;

class HttpError : public std::runtime_error {
 public:
  HttpError(int status, const std::string& message)
      : std::runtime_error(message), status_(status) {}

  int status() const { return status_; }

 private:
  int status_;
};

/**
 * @brief Represent a parsed HTTP request.
 */
struct HttpRequest {
  std::string method;
  std::string path;
  std::map<std::string, std::string> headers;
  std::string body;
};

/**
 * @brief Represent a single chat message (role + content).
 */
struct ChatMessage {
  std::string role;
  std::string content;
};

/**
 * @brief Represent a parsed OpenAI-style chat completion request.
 */
struct ChatRequest {
  std::vector<ChatMessage> messages;
  int max_tokens = 0;
  bool stream = false;
  bool include_usage = false;
};

/**
 * @brief Represent a prepared prompt ready for one inference call.
 */
struct PreparedChat {
  std::vector<int64_t> prompt_ids;
  int max_new_tokens = 0;
  size_t trimmed_messages = 0;
};

/**
 * @brief Represent the decoded result of one chat completion.
 */
struct ChatResult {
  std::string text;
  std::string finish_reason;
  int prompt_tokens = 0;
  int completion_tokens = 0;
  int output_budget = 0;
  size_t trimmed_messages = 0;
  bool cache_reused = false;
};

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  return value;
}

std::string Trim(const std::string& value) {
  const size_t first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const size_t last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

const char* StatusText(int status) {
  switch (status) {
    case 200:
      return "OK";
    case 204:
      return "No Content";
    case 400:
      return "Bad Request";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 413:
      return "Payload Too Large";
    case 431:
      return "Request Header Fields Too Large";
    case 500:
      return "Internal Server Error";
    case 501:
      return "Not Implemented";
    default:
      return "Error";
  }
}

bool SendAll(int file_descriptor, const std::string& data) {
  size_t offset = 0;
  while (offset < data.size()) {
    const ssize_t written =
        send(file_descriptor, data.data() + offset, data.size() - offset,
             MSG_NOSIGNAL);
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written <= 0) {
      return false;
    }
    offset += static_cast<size_t>(written);
  }
  return true;
}

bool SendResponse(int file_descriptor, int status, const std::string& body,
                  const std::string& content_type =
                      "application/json; charset=utf-8") {
  std::ostringstream headers;
  headers << "HTTP/1.1 " << status << ' ' << StatusText(status) << "\r\n"
          << "Content-Type: " << content_type << "\r\n"
          << "Content-Length: " << body.size() << "\r\n"
          << "Connection: close\r\n"
          << "Access-Control-Allow-Origin: *\r\n"
          << "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
          << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
          << "\r\n";
  return SendAll(file_descriptor, headers.str()) &&
         (body.empty() || SendAll(file_descriptor, body));
}

bool SendSseHeaders(int file_descriptor) {
  const std::string headers =
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: text/event-stream; charset=utf-8\r\n"
      "Cache-Control: no-cache\r\n"
      "Connection: close\r\n"
      "X-Accel-Buffering: no\r\n"
      "Access-Control-Allow-Origin: *\r\n"
      "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
      "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
      "\r\n";
  return SendAll(file_descriptor, headers);
}

bool SendSseData(int file_descriptor, const json& payload) {
  return SendAll(file_descriptor, "data: " + payload.dump() + "\n\n");
}

json ErrorBody(const std::string& message, const std::string& type,
               const std::string& code) {
  return json{{"error",
               {{"message", message},
                {"type", type},
                {"param", nullptr},
                {"code", code}}}};
}

void SendError(int file_descriptor, int status, const std::string& message) {
  const bool client_error = status >= 400 && status < 500;
  const json body = ErrorBody(
      message, client_error ? "invalid_request_error" : "server_error",
      client_error ? "invalid_request" : "internal_error");
  SendResponse(file_descriptor, status, body.dump());
}

/**
 * @brief Read one HTTP request from a client socket.
 *
 * @param file_descriptor Open client socket file descriptor.
 * @param max_body_bytes Maximum accepted request body size in bytes.
 *
 * @return Parsed HTTP request (method, path, headers, body).
 *
 * @throws HttpError On malformed headers, unsupported HTTP version, or a
 *         request body larger than @p max_body_bytes.
 */
HttpRequest ReadHttpRequest(int file_descriptor, size_t max_body_bytes) {
  std::string received;
  received.reserve(8192);
  char buffer[8192];
  size_t header_end = std::string::npos;

  while (header_end == std::string::npos) {
    const ssize_t count = recv(file_descriptor, buffer, sizeof(buffer), 0);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      throw HttpError(400, "connection closed before HTTP headers completed");
    }
    received.append(buffer, static_cast<size_t>(count));
    if (received.size() > kMaxHeaderBytes) {
      throw HttpError(431, "HTTP headers exceed 64 KiB");
    }
    header_end = received.find("\r\n\r\n");
  }

  HttpRequest request;
  std::istringstream header_stream(received.substr(0, header_end));
  std::string request_line;
  if (!std::getline(header_stream, request_line)) {
    throw HttpError(400, "missing HTTP request line");
  }
  if (!request_line.empty() && request_line.back() == '\r') {
    request_line.pop_back();
  }

  std::string target;
  std::string version;
  std::istringstream request_line_stream(request_line);
  if (!(request_line_stream >> request.method >> target >> version)) {
    throw HttpError(400, "invalid HTTP request line");
  }
  if (version != "HTTP/1.1" && version != "HTTP/1.0") {
    throw HttpError(400, "unsupported HTTP version");
  }
  const size_t query = target.find('?');
  request.path = target.substr(0, query);

  std::string line;
  while (std::getline(header_stream, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    const size_t colon = line.find(':');
    if (colon == std::string::npos) {
      throw HttpError(400, "invalid HTTP header");
    }
    request.headers[ToLower(Trim(line.substr(0, colon)))] =
        Trim(line.substr(colon + 1));
  }

  const auto transfer_encoding = request.headers.find("transfer-encoding");
  if (transfer_encoding != request.headers.end() &&
      ToLower(transfer_encoding->second) != "identity") {
    throw HttpError(501, "chunked request bodies are not supported");
  }

  size_t content_length = 0;
  const auto length_header = request.headers.find("content-length");
  if (length_header != request.headers.end()) {
    try {
      size_t parsed = 0;
      const unsigned long long length =
          std::stoull(length_header->second, &parsed, 10);
      if (parsed != length_header->second.size()) {
        throw std::invalid_argument("trailing data");
      }
      if (length > max_body_bytes) {
        throw HttpError(413, "HTTP request body exceeds configured limit");
      }
      content_length = static_cast<size_t>(length);
    } catch (const HttpError&) {
      throw;
    } catch (const std::exception&) {
      throw HttpError(400, "invalid Content-Length header");
    }
  }

  const size_t body_start = header_end + 4;
  if (received.size() > body_start) {
    request.body.assign(received.data() + body_start,
                        received.size() - body_start);
  }

  const auto expect_header = request.headers.find("expect");
  if (expect_header != request.headers.end() &&
      ToLower(expect_header->second) == "100-continue" &&
      request.body.size() < content_length) {
    if (!SendAll(file_descriptor, "HTTP/1.1 100 Continue\r\n\r\n")) {
      throw HttpError(400, "client disconnected before request body");
    }
  }

  while (request.body.size() < content_length) {
    const size_t remaining = content_length - request.body.size();
    const ssize_t count =
        recv(file_descriptor, buffer, std::min(remaining, sizeof(buffer)), 0);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      throw HttpError(400, "connection closed before request body completed");
    }
    request.body.append(buffer, static_cast<size_t>(count));
  }
  if (request.body.size() > content_length) {
    request.body.resize(content_length);
  }
  return request;
}

std::string ReadMessageContent(const json& content) {
  if (content.is_string()) {
    return content.get<std::string>();
  }
  if (!content.is_array()) {
    throw HttpError(400, "each message content must be a string or array");
  }

  std::string text;
  for (const auto& part : content) {
    if (part.is_string()) {
      text += part.get<std::string>();
      continue;
    }
    if (!part.is_object()) {
      throw HttpError(400, "invalid message content part");
    }
    const std::string type = part.value("type", "");
    if (type == "text" || type == "input_text" ||
        (type.empty() && part.contains("text"))) {
      if (!part.contains("text") || !part["text"].is_string()) {
        throw HttpError(400, "text content part requires a string text field");
      }
      text += part["text"].get<std::string>();
    } else if (type == "image" || type == "image_url" ||
               type == "input_image") {
      throw HttpError(
          400,
          "gemma4_server is text-only; use the interactive main executable "
          "for image chat");
    } else {
      throw HttpError(400, "unsupported message content part type: " + type);
    }
  }
  return text;
}

int ReadRequestMaxTokens(const json& request, int default_max_tokens) {
  const json* value = nullptr;
  if (request.contains("max_tokens") && !request["max_tokens"].is_null()) {
    value = &request["max_tokens"];
  } else if (request.contains("max_completion_tokens") &&
             !request["max_completion_tokens"].is_null()) {
    value = &request["max_completion_tokens"];
  }
  if (value == nullptr) {
    return default_max_tokens;
  }
  if (!value->is_number_integer()) {
    throw HttpError(400, "max_tokens must be a non-negative integer");
  }
  const int64_t requested = value->get<int64_t>();
  if (requested < 0) {
    throw HttpError(400, "max_tokens must be a non-negative integer");
  }
  return static_cast<int>(std::min<int64_t>(
      requested, std::numeric_limits<int>::max()));
}

/**
 * @brief Parse a JSON chat-completion request body.
 *
 * @param body Raw JSON request body.
 * @param default_max_tokens Fallback output limit when the request omits
 *        `max_tokens`.
 *
 * @return Parsed chat request.
 *
 * @throws HttpError On invalid JSON, a missing `messages` array, or a
 *         negative `max_tokens`.
 */
ChatRequest ParseChatRequest(const std::string& body,
                             int default_max_tokens) {
  json request;
  try {
    request = json::parse(body);
  } catch (const std::exception& error) {
    throw HttpError(400, std::string("invalid JSON body: ") + error.what());
  }
  if (!request.is_object()) {
    throw HttpError(400, "request body must be a JSON object");
  }
  if (!request.contains("messages") || !request["messages"].is_array() ||
      request["messages"].empty()) {
    throw HttpError(400, "messages must be a non-empty array");
  }
  if (request.contains("n") && request["n"].is_number_integer() &&
      request["n"].get<int>() != 1) {
    throw HttpError(400, "only n=1 is supported");
  }

  ChatRequest parsed;
  parsed.max_tokens = ReadRequestMaxTokens(request, default_max_tokens);
  if (request.contains("stream")) {
    if (!request["stream"].is_boolean()) {
      throw HttpError(400, "stream must be a boolean");
    }
    parsed.stream = request["stream"].get<bool>();
  }
  if (request.contains("stream_options") &&
      request["stream_options"].is_object()) {
    const auto& options = request["stream_options"];
    if (options.contains("include_usage") &&
        options["include_usage"].is_boolean()) {
      parsed.include_usage = options["include_usage"].get<bool>();
    }
  }

  for (const auto& item : request["messages"]) {
    if (!item.is_object() || !item.contains("role") ||
        !item["role"].is_string() || !item.contains("content")) {
      throw HttpError(400, "each message requires string role and content");
    }
    std::string role = item["role"].get<std::string>();
    if (role == "developer") {
      role = "system";
    } else if (role == "model") {
      role = "assistant";
    }
    if (role != "system" && role != "user" && role != "assistant") {
      throw HttpError(400, "unsupported message role: " + role);
    }
    parsed.messages.push_back({role, ReadMessageContent(item["content"])});
  }
  if (parsed.messages.back().role != "user") {
    throw HttpError(400, "the last message must have role=user");
  }
  return parsed;
}

std::string BuildMessagesJson(const std::vector<ChatMessage>& messages) {
  json output = json::array();
  for (const auto& message : messages) {
    output.push_back({{"role", message.role}, {"content", message.content}});
  }
  return output.dump();
}

size_t DropOldestTurn(std::vector<ChatMessage>* messages) {
  if (messages == nullptr || messages->size() <= 1) {
    return 0;
  }

  size_t first = 0;
  while (first + 1 < messages->size() &&
         (*messages)[first].role == "system") {
    ++first;
  }
  if (first >= messages->size() - 1) {
    return 0;
  }

  size_t erase_count = 1;
  if ((*messages)[first].role == "user" &&
      first + 1 < messages->size() - 1 &&
      (*messages)[first + 1].role == "assistant") {
    erase_count = 2;
  }
  messages->erase(messages->begin() + static_cast<std::ptrdiff_t>(first),
                  messages->begin() +
                      static_cast<std::ptrdiff_t>(first + erase_count));
  return erase_count;
}

bool PrefixMatches(const std::vector<int64_t>& prompt,
                   const std::vector<int64_t>& cached, int length) {
  if (length <= 0) {
    return true;
  }
  if (static_cast<int>(prompt.size()) < length ||
      static_cast<int>(cached.size()) < length) {
    return false;
  }
  for (int index = 0; index < length; ++index) {
    if (prompt[static_cast<size_t>(index)] !=
        cached[static_cast<size_t>(index)]) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Own the text engine and tokenizer, and serve chat completions.
 *
 * Loads the Text HBM once at construction and keeps it resident. Consecutive
 * requests that share a token prefix reuse the KV cache.
 */
class ChatService {
 public:
  ChatService(const std::string& text_hbm, const std::string& embeddings,
              const std::string& tokenizer_path) {
    std::cout << "Loading Text HBM (one-time)..." << std::endl;
    engine_ = std::make_unique<gemma4::TextEngine>(text_hbm, embeddings);
    tokenizer_ =
        std::make_unique<gemma4::TokenizerBridge>(tokenizer_path);
    std::cout << "Text ready (" << engine_->LoadMs() << " ms)" << std::endl;
  }

  /**
   * @brief Encode messages into a prompt that fits the KV cache.
   *
   * Trims the oldest complete turns until the prompt plus the requested
   * output reserve fit into the fixed 4096-token KV cache.
   *
   * @param request Parsed chat request.
   * @param min_response_tokens Minimum output capacity to preserve.
   *
   * @return Prepared prompt with the final output budget.
   *
   * @throws HttpError If the prompt itself still exceeds the KV cache.
   */
  PreparedChat Prepare(const ChatRequest& request,
                       int min_response_tokens) const {
    std::vector<ChatMessage> messages = request.messages;
    const int desired_reserve = request.max_tokens == 0
        ? min_response_tokens
        : std::min(request.max_tokens, gemma4::kCacheLen - 1);

    PreparedChat prepared;
    while (true) {
      prepared.prompt_ids =
          tokenizer_->EncodeMessagesJson(BuildMessagesJson(messages), true);
      const bool prompt_fits =
          prepared.prompt_ids.size() < static_cast<size_t>(gemma4::kCacheLen);
      const bool reserve_fits =
          prepared.prompt_ids.size() + static_cast<size_t>(desired_reserve) <=
          static_cast<size_t>(gemma4::kCacheLen);
      if (prompt_fits && reserve_fits) {
        break;
      }
      const size_t removed = DropOldestTurn(&messages);
      if (removed == 0) {
        break;
      }
      prepared.trimmed_messages += removed;
    }

    if (prepared.prompt_ids.size() >=
        static_cast<size_t>(gemma4::kCacheLen)) {
      std::ostringstream error;
      error << "prompt uses " << prepared.prompt_ids.size()
            << " tokens, but the HBM KV cache is limited to "
            << gemma4::kCacheLen;
      throw HttpError(400, error.str());
    }

    const int available =
        gemma4::kCacheLen - static_cast<int>(prepared.prompt_ids.size());
    prepared.max_new_tokens = request.max_tokens == 0
        ? available
        : std::min(request.max_tokens, available);
    return prepared;
  }

  /**
   * @brief Run inference for a prepared chat and return the decoded result.
   *
   * Reuses the KV cache when the new prompt shares a token prefix with the
   * previously processed context; otherwise resets the cache first.
   *
   * @param prepared Prepared prompt and output budget.
   * @param on_text Optional streaming callback invoked per generated chunk.
   *
   * @return Decoded completion text, finish reason, and token accounting.
   */
  ChatResult Generate(
      const PreparedChat& prepared,
      const std::function<bool(const std::string&)>& on_text = nullptr) {
    const int previous_processed = engine_->ProcessedTokens();
    const bool prefix_matches = PrefixMatches(
        prepared.prompt_ids, cached_ids_, previous_processed);
    const bool cache_reused =
        prepared.trimmed_messages == 0 && previous_processed > 0 &&
        prefix_matches;
    if (previous_processed > 0 &&
        (prepared.trimmed_messages > 0 || !prefix_matches)) {
      engine_->ResetSession();
      cached_ids_.clear();
    }

    std::cout << "[request] prompt=" << prepared.prompt_ids.size()
              << ", output_budget=" << prepared.max_new_tokens
              << ", capacity=" << gemma4::kCacheLen
              << ", cache_reused=" << (cache_reused ? "yes" : "no")
              << ", trimmed_messages=" << prepared.trimmed_messages
              << std::endl;

    gemma4::TokenCallback token_callback;
    if (on_text) {
      token_callback = [&](int64_t token_id) {
        if (token_id == gemma4::kEosTokenId ||
            token_id == gemma4::kTurnEndTokenId) {
          return true;
        }
        return on_text(tokenizer_->DecodeIds({token_id}));
      };
    }

    std::vector<int64_t> output;
    try {
      output = engine_->ContinueGenerateStream(
          prepared.prompt_ids, prepared.max_new_tokens, token_callback);
    } catch (...) {
      engine_->ResetSession();
      cached_ids_.clear();
      throw;
    }

    size_t generation_end = output.size();
    bool stopped = false;
    while (generation_end > prepared.prompt_ids.size() &&
           (output[generation_end - 1] == gemma4::kEosTokenId ||
            output[generation_end - 1] == gemma4::kTurnEndTokenId)) {
      stopped = true;
      --generation_end;
    }
    const std::vector<int64_t> generated(
        output.begin() + static_cast<std::ptrdiff_t>(prepared.prompt_ids.size()),
        output.begin() + static_cast<std::ptrdiff_t>(generation_end));

    cached_ids_ = output;

    ChatResult result;
    result.text = tokenizer_->DecodeIds(generated);
    result.finish_reason = stopped ? "stop" : "length";
    result.prompt_tokens = static_cast<int>(prepared.prompt_ids.size());
    result.completion_tokens = static_cast<int>(generated.size());
    result.output_budget = prepared.max_new_tokens;
    result.trimmed_messages = prepared.trimmed_messages;
    result.cache_reused = cache_reused;
    return result;
  }

  double LoadMs() const { return engine_->LoadMs(); }
  int CachedTokens() const { return engine_->ProcessedTokens(); }

 private:
  std::unique_ptr<gemma4::TextEngine> engine_;
  std::unique_ptr<gemma4::TokenizerBridge> tokenizer_;
  std::vector<int64_t> cached_ids_;
};

int64_t UnixTime() {
  return static_cast<int64_t>(std::time(nullptr));
}

std::string MakeCompletionId() {
  static uint64_t sequence = 0;
  const uint64_t milliseconds = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  std::ostringstream id;
  id << "chatcmpl-gemma4-" << milliseconds << '-' << ++sequence;
  return id.str();
}

json CompletionChunk(const std::string& id, int64_t created,
                     const std::string& model, const json& delta,
                     const json& finish_reason) {
  json chunk;
  chunk["id"] = id;
  chunk["object"] = "chat.completion.chunk";
  chunk["created"] = created;
  chunk["model"] = model;
  chunk["choices"] = json::array(
      {{{"index", 0}, {"delta", delta}, {"finish_reason", finish_reason}}});
  return chunk;
}

/**
 * @brief Handle one POST /v1/chat/completions request.
 *
 * Parses the request, prepares the prompt, runs inference, and writes either
 * a single JSON response or an SSE stream depending on the request's
 * `stream` flag.
 *
 * @param file_descriptor Client socket to write the response to.
 * @param request Parsed HTTP request.
 * @param service Shared chat service holding the resident text engine.
 * @param model Model ID to report in the response.
 * @param default_max_tokens Fallback output token limit.
 * @param min_response_tokens Minimum output capacity preserved while
 *        trimming history.
 */
void HandleChatCompletion(int file_descriptor, const HttpRequest& request,
                          ChatService* service, const std::string& model,
                          int default_max_tokens,
                          int min_response_tokens) {
  const ChatRequest chat_request =
      ParseChatRequest(request.body, default_max_tokens);
  const PreparedChat prepared =
      service->Prepare(chat_request, min_response_tokens);
  const std::string completion_id = MakeCompletionId();
  const int64_t created = UnixTime();

  if (!chat_request.stream) {
    const ChatResult result = service->Generate(prepared);
    json response;
    response["id"] = completion_id;
    response["object"] = "chat.completion";
    response["created"] = created;
    response["model"] = model;
    response["choices"] = json::array(
        {{{"index", 0},
          {"message", {{"role", "assistant"}, {"content", result.text}}},
          {"finish_reason", result.finish_reason}}});
    response["usage"] = {
        {"prompt_tokens", result.prompt_tokens},
        {"completion_tokens", result.completion_tokens},
        {"total_tokens", result.prompt_tokens + result.completion_tokens}};
    SendResponse(file_descriptor, 200, response.dump());
    return;
  }

  if (!SendSseHeaders(file_descriptor)) {
    return;
  }
  bool connected = SendSseData(
      file_descriptor,
      CompletionChunk(completion_id, created, model,
                      json{{"role", "assistant"}}, nullptr));
  if (!connected) {
    return;
  }

  try {
    const ChatResult result = service->Generate(
        prepared, [&](const std::string& text) {
          if (text.empty()) {
            return true;
          }
          connected = SendSseData(
              file_descriptor,
              CompletionChunk(completion_id, created, model,
                              json{{"content", text}}, nullptr));
          return connected;
        });
    if (!connected) {
      return;
    }

    connected = SendSseData(
        file_descriptor,
        CompletionChunk(completion_id, created, model, json::object(),
                        result.finish_reason));
    if (connected && chat_request.include_usage) {
      json usage_chunk;
      usage_chunk["id"] = completion_id;
      usage_chunk["object"] = "chat.completion.chunk";
      usage_chunk["created"] = created;
      usage_chunk["model"] = model;
      usage_chunk["choices"] = json::array();
      usage_chunk["usage"] = {
          {"prompt_tokens", result.prompt_tokens},
          {"completion_tokens", result.completion_tokens},
          {"total_tokens", result.prompt_tokens + result.completion_tokens}};
      connected = SendSseData(file_descriptor, usage_chunk);
    }
    if (connected) {
      SendAll(file_descriptor, "data: [DONE]\n\n");
    }
  } catch (const std::exception& error) {
    if (connected) {
      SendSseData(file_descriptor,
                  ErrorBody(error.what(), "server_error", "internal_error"));
      SendAll(file_descriptor, "data: [DONE]\n\n");
    }
  }
}

/**
 * @brief Serve one client connection until the request is answered.
 *
 * Reads a single HTTP request, dispatches it to the matching endpoint
 * (`/health`, `/v1/models`, `/v1/chat/completions`), and writes the response.
 * Any HttpError is converted into the corresponding HTTP error body.
 *
 * @param file_descriptor Client socket.
 * @param max_body_bytes Maximum accepted request body size.
 * @param service Shared chat service.
 * @param model Model ID to report.
 * @param default_max_tokens Fallback output token limit.
 * @param min_response_tokens Minimum output capacity preserved while
 *        trimming history.
 */
void HandleClient(int file_descriptor, size_t max_body_bytes,
                  ChatService* service, const std::string& model,
                  int default_max_tokens, int min_response_tokens) {
  try {
    const HttpRequest request =
        ReadHttpRequest(file_descriptor, max_body_bytes);

    if (request.method == "OPTIONS") {
      SendResponse(file_descriptor, 204, "");
      return;
    }
    if (request.method == "GET" &&
        (request.path == "/health" || request.path == "/v1/health")) {
      const json body = {{"status", "ok"},
                         {"model", model},
                         {"context_length", gemma4::kCacheLen},
                         {"cached_tokens", service->CachedTokens()},
                         {"load_ms", service->LoadMs()}};
      SendResponse(file_descriptor, 200, body.dump());
      return;
    }
    if (request.method == "GET" && request.path == "/v1/models") {
      const json body = {
          {"object", "list"},
          {"data",
           json::array({{{"id", model},
                         {"object", "model"},
                         {"created", 0},
                         {"owned_by", "d-robotics"}}})}};
      SendResponse(file_descriptor, 200, body.dump());
      return;
    }
    if (request.method == "GET" && request.path == "/") {
      const json body = {
          {"name", "Gemma4-E2B OpenAI-compatible server"},
          {"model", model},
          {"context_length", gemma4::kCacheLen},
          {"endpoints",
           json::array({"GET /health", "GET /v1/models",
                        "POST /v1/chat/completions"})}};
      SendResponse(file_descriptor, 200, body.dump());
      return;
    }
    if (request.method == "POST" &&
        request.path == "/v1/chat/completions") {
      HandleChatCompletion(file_descriptor, request, service, model,
                           default_max_tokens, min_response_tokens);
      return;
    }
    if (request.path == "/v1/chat/completions") {
      throw HttpError(405, "use POST for /v1/chat/completions");
    }
    throw HttpError(404, "endpoint not found");
  } catch (const HttpError& error) {
    SendError(file_descriptor, error.status(), error.what());
  } catch (const std::exception& error) {
    std::cerr << "Request error: " << error.what() << std::endl;
    SendError(file_descriptor, 500, error.what());
  }
}

/**
 * @brief Create and bind a listening TCP socket.
 *
 * @param host Bind address (e.g. "0.0.0.0").
 * @param port Listen port.
 *
 * @return The listening socket file descriptor.
 *
 * @throws std::runtime_error On socket creation, bind, listen, or
 *         SO_REUSEADDR failure.
 */
int CreateListenSocket(const std::string& host, int port) {
  struct addrinfo hints {};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;

  struct addrinfo* addresses = nullptr;
  const std::string port_text = std::to_string(port);
  const char* node = (host.empty() || host == "*") ? nullptr : host.c_str();
  const int resolve_result =
      getaddrinfo(node, port_text.c_str(), &hints, &addresses);
  if (resolve_result != 0) {
    throw std::runtime_error(std::string("getaddrinfo failed: ") +
                             gai_strerror(resolve_result));
  }

  int listen_socket = -1;
  for (struct addrinfo* address = addresses; address != nullptr;
       address = address->ai_next) {
    listen_socket =
        socket(address->ai_family, address->ai_socktype, address->ai_protocol);
    if (listen_socket < 0) {
      continue;
    }
    const int reuse = 1;
    setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    if (::bind(listen_socket, address->ai_addr, address->ai_addrlen) == 0 &&
        listen(listen_socket, 16) == 0) {
      break;
    }
    close(listen_socket);
    listen_socket = -1;
  }
  freeaddrinfo(addresses);

  if (listen_socket < 0) {
    throw std::runtime_error(std::string("failed to bind HTTP socket: ") +
                             std::strerror(errno));
  }
  return listen_socket;
}

}  // namespace

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "OpenAI-compatible text chat server for Gemma4-E2B.\n"
      "Usage: ./gemma4_server [--host 0.0.0.0] [--port 8000] "
      "[--max_tokens 0]");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_port <= 0 || FLAGS_port > 65535) {
    std::cerr << "--port must be in [1, 65535]" << std::endl;
    return 2;
  }
  if (FLAGS_max_tokens < 0) {
    std::cerr << "--max_tokens must be zero or positive" << std::endl;
    return 2;
  }
  if (FLAGS_min_response_tokens <= 0) {
    std::cerr << "--min_response_tokens must be positive" << std::endl;
    return 2;
  }
  if (FLAGS_request_limit_mb <= 0 || FLAGS_request_limit_mb > 64) {
    std::cerr << "--request_limit_mb must be in [1, 64]" << std::endl;
    return 2;
  }

  const char* env_home = std::getenv("GEMMA4_HOME");
  const std::string home = (env_home && *env_home) ? env_home : ".";
  const std::string text_hbm = FLAGS_text_hbm.empty()
      ? home + "/model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm"
      : FLAGS_text_hbm;
  const std::string embeddings = FLAGS_tok_embeddings.empty()
      ? home + "/model/tok_embeddings.bin"
      : FLAGS_tok_embeddings;
  const std::string tokenizer_path = FLAGS_tokenizer_path.empty()
      ? home + "/tokenizer/tokenizer.json"
      : FLAGS_tokenizer_path;
  const int min_response_tokens =
      std::min(FLAGS_min_response_tokens, gemma4::kCacheLen - 1);
  const size_t max_body_bytes =
      static_cast<size_t>(FLAGS_request_limit_mb) * 1024 * 1024;

  std::signal(SIGPIPE, SIG_IGN);

  try {
    ChatService service(text_hbm, embeddings, tokenizer_path);
    const int listen_socket = CreateListenSocket(FLAGS_host, FLAGS_port);
    std::cout << "Listening on http://" << FLAGS_host << ':' << FLAGS_port
              << std::endl;
    std::cout << "OpenAI base URL: http://<board-ip>:" << FLAGS_port << "/v1"
              << std::endl;
    std::cout << "KV cache: " << gemma4::kCacheLen
              << " tokens; default max output: "
              << (FLAGS_max_tokens == 0
                      ? std::string("all capacity remaining after the prompt")
                      : std::to_string(FLAGS_max_tokens))
              << std::endl;

    while (true) {
      const int client_socket = accept(listen_socket, nullptr, nullptr);
      if (client_socket < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw std::runtime_error(std::string("accept failed: ") +
                                 std::strerror(errno));
      }
      HandleClient(client_socket, max_body_bytes, &service, FLAGS_model,
                   FLAGS_max_tokens, min_response_tokens);
      close(client_socket);
    }
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << std::endl;
    return 1;
  }
}
