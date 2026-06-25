/*
 * Copyright (C) 2024 Shanghai Gua Technology Co., Ltd.
 * All rights reserved
 */

#include "gemma4_native_tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#define TAG "tokenizer"
// Local logging shims (no hlog dependency).
#define LOGE(tag, fmt, ...) fprintf(stderr, "[E][" tag "] " fmt "\n", ##__VA_ARGS__)
#ifndef NDEBUG
#define LOGD(tag, fmt, ...) fprintf(stderr, "[D][" tag "] " fmt "\n", ##__VA_ARGS__)
#else
#define LOGD(tag, fmt, ...) ((void)0)
#endif

namespace {
/* @brief The method to post-process the tokens to their original strings.
 * Possible values (each refers to a kind of tokenizer):
 * - "byte_fallback": The same as the byte-fallback BPE tokenizer, including
 * LLaMA-2, Mixtral-7b, etc. E.g. "▁of" -> " of", "<0x1B>" -> "\x1B". This
 * method: 1) Transform tokens like <0x1B> to hex char byte 1B. (so-called
 * byte-fallback) 2) Replace \\u2581 "▁" with space.
 * - "byte_level": The same as the byte-level BPE tokenizer, including LLaMA-3,
 * GPT-2, Phi-2, etc. E.g. "Ġin" -> " in", "ě" -> "\x1B" This method inverses
 * the bytes-to-unicode transformation in the encoding process in
 *   https://github.com/huggingface/transformers/blob/87be06ca77166e6a6215eee5a990ab9f07238a18/src/transformers/models/gpt2/tokenization_gpt2.py#L38-L59
 */
static const char kByteFallback[] = "byte_fallback";
static const char kByteLevel[] = "byte_level";
}  // namespace

namespace gemma4 {

std::string LoadBytesFromFile(std::string const &path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  fs.close();
  return data;
}

std::string SpaceReplacerDecoder(std::string const &token) {
  // \u2581 is the unicode for "lower one eighth block"
  // UTF8 encoding for \u2581 is 0xE2 0x96 0x81
  std::string result;
  for (size_t i = 0; i < token.size(); ++i) {
    if (i + 2 < token.size() && token[i] == static_cast<char>(0xE2) &&
        token[i + 1] == static_cast<char>(0x96) &&
        token[i + 2] == static_cast<char>(0x81)) {
      result += ' ';
      i += 2;
    } else {
      result += token[i];
    }
  }
  return result;
}

bool GetPostprocType(std::string const &tokenizer_json,
                     std::string &token_postproc_method) {
  std::ifstream file(tokenizer_json, std::ifstream::in);
  if (!file.good()) {
    LOGE(TAG, "Read file failed: %s !!!", tokenizer_json.c_str());
    return false;
  }
  nlohmann::json json = nlohmann::json::parse(file);
  file.close();
  if (json.is_discarded() || json.empty()) {
    LOGE(TAG, "Parse config file failed: '%s' !!!", tokenizer_json.c_str());
    return false;
  }
  if (json.contains("model") && json["model"].is_object()) {
    nlohmann::json &models_json = json["model"];
    if (models_json.contains("byte_fallback") &&
        models_json["byte_fallback"].is_boolean() &&
        models_json["byte_fallback"]) {
      token_postproc_method = kByteFallback;
    }
  } else {
    token_postproc_method = kByteLevel;
  }
  return true;
}

bool GetEosToken(std::string const &config_file, std::string &eos_token) {
  std::ifstream file(config_file, std::ifstream::in);
  if (!file.good()) {
    LOGE(TAG, "Read file failed: %s !!!", config_file.c_str());
    return false;
  }
  nlohmann::json json = nlohmann::json::parse(file);
  file.close();
  if (json.is_discarded() || json.empty()) {
    LOGE(TAG, "Parse config file failed: '%s' !!!", config_file.c_str());
    return false;
  }
  if (json.contains("eos_token")) {
    if (json["eos_token"].is_string()) {
      eos_token = json["eos_token"];
    } else if (json["eos_token"].is_object()) {
      nlohmann::json &eos_token_json = json["eos_token"];
      if (eos_token_json.contains("content") &&
          eos_token_json["content"].is_string()) {
        eos_token = eos_token_json["content"];
      } else {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

std::mutex Tokenizer::global_tokenizer_mutex_{};
std::unordered_map<std::string, TokenizerInfo> Tokenizer::global_tokenizer_{};

Tokenizer::Tokenizer(TokenizerInfo const &tokenizer_info) {
  tokenizer_path_ = tokenizer_info.tokenizer_path;
  tokenizer_ = tokenizer_info.tokenizer;
  stop_token_id_ = TokenToId(tokenizer_info.eos_token);
  token_postproc_method_ = tokenizer_info.token_postproc_method;
}

std::unique_ptr<Tokenizer> Tokenizer::Create(
    std::string const &tokenizer_path) {
  std::lock_guard<std::mutex> lock(global_tokenizer_mutex_);
  auto it = global_tokenizer_.find(tokenizer_path);
  if (it != global_tokenizer_.end()) {
    LOGD(TAG, "Tokenizer loaded from global cache: %s", tokenizer_path.c_str());
    auto &tokenizer_info = it->second;
    tokenizer_info.ref_count++;
    return std::make_unique<Tokenizer>(it->second);
  }
  TokenizerInfo tokenizer_info;
  tokenizer_info.tokenizer_path = tokenizer_path;
  std::filesystem::path path(tokenizer_path);
  std::filesystem::path sentencepiece;
  std::filesystem::path huggingface;
  std::filesystem::path config_file;
  if (!std::filesystem::exists(path)) {
    std::cerr << "Cannot find tokenizer via path: " << tokenizer_path;
  }
  if (std::filesystem::is_directory(path)) {
    sentencepiece = path / "tokenizer.model";
    huggingface = path / "tokenizer.json";
    config_file = path / "tokenizer_config.json";
  } else {
    sentencepiece = path.parent_path() / "tokenizer.model";
    huggingface = path.parent_path() / "tokenizer.json";
    config_file = path.parent_path() / "tokenizer_config.json";
  }

  // 读取eos_token
  if (!GetEosToken(config_file.string(), tokenizer_info.eos_token)) {
    LOGE(TAG, "Get EosToken Failed, please check tokenizer_config.json");
    return nullptr;
  }

  if (std::filesystem::exists(huggingface)) {
    // 读取 postproc_type
    if (!GetPostprocType(huggingface, tokenizer_info.token_postproc_method)) {
      LOGE(TAG, "Get PostprocType Failed, please check tokenizer.json");
      return nullptr;
    }
    LOGD(TAG, "token_postproc_method %s", tokenizer_info.token_postproc_method.c_str());
    tokenizer_info.tokenizer = std::move(tokenizers::Tokenizer::FromBlobJSON(
        LoadBytesFromFile(huggingface.string())));
  } else if (std::filesystem::exists(sentencepiece)) {
    tokenizer_info.token_postproc_method = kByteFallback;
    tokenizer_info.tokenizer =
        std::move(tokenizers::Tokenizer::FromBlobSentencePiece(
            LoadBytesFromFile(sentencepiece.string())));
  } else {
    // check byte-level BPE
    std::filesystem::path merges_path = path / "merges.txt";
    std::filesystem::path vocab_path = path / "vocab.json";
    std::filesystem::path added_tokens_path = path / "added_tokens.json";
    if (std::filesystem::exists(merges_path) &&
        std::filesystem::exists(vocab_path) &&
        std::filesystem::exists(added_tokens_path)) {
      std::string vocab = LoadBytesFromFile(vocab_path.string());
      std::string merges = LoadBytesFromFile(merges_path.string());
      std::string added_tokens = LoadBytesFromFile(added_tokens_path.string());
      tokenizer_info.token_postproc_method = kByteLevel;
      tokenizer_info.tokenizer =
          std::move(tokenizers::Tokenizer::FromBlobByteLevelBPE(vocab, merges,
                                                                added_tokens));
    }
  }

  if (tokenizer_info.tokenizer != nullptr) {
    LOGD(TAG, "Load tokenizer success: %s", tokenizer_path.c_str());
    tokenizer_info.ref_count = 1U;
    global_tokenizer_[tokenizer_path] = tokenizer_info;
    return std::make_unique<Tokenizer>(tokenizer_info);
  }

  LOGE(TAG, "Load tokenizer failed!!!");
  return nullptr;
}

std::vector<int32_t> Tokenizer::Encode(std::string const &text) const {
  return tokenizer_->Encode(text);
}

std::vector<std::vector<int32_t>> Tokenizer::EncodeBatch(
    std::vector<std::string> const &texts) const {
  return tokenizer_->EncodeBatch(texts);
}

std::string Tokenizer::Decode(std::vector<int32_t> const &token_ids) const {
  return tokenizer_->Decode(token_ids);
}

std::string Tokenizer::Decode(int32_t token_id) {
  if (token_postproc_method_ == kByteFallback) {
    return SpaceReplacerDecoder(IdToToken(token_id));
  } else {
    return tokenizer_->Decode({token_id});
  }
}

std::string Tokenizer::DecodeIncremental(
    const std::vector<int32_t> &delta_tokens) {
  if (delta_tokens.empty()) return "";

  std::string ret;

  for (int32_t token : delta_tokens) {
    pending_tokens_.push_back(token);

    // 拼接前缀 + pending tokens
    std::vector<int32_t> all_tokens;
    all_tokens.reserve(prefix_tokens_.size() + pending_tokens_.size());
    all_tokens.insert(all_tokens.end(), prefix_tokens_.begin(),
                      prefix_tokens_.end());
    all_tokens.insert(all_tokens.end(), pending_tokens_.begin(),
                      pending_tokens_.end());

    std::string prefix_str =
        prefix_tokens_.empty() ? "" : Decode(prefix_tokens_);
    std::string full_str = Decode(all_tokens);

    std::string validated_str;
    std::vector<int32_t> new_pending_tokens;

    if (full_str.compare(0, prefix_str.length(), prefix_str) == 0) {
      // prefix ok
      validated_str = full_str.substr(prefix_str.length());

      // 回退半字符 token
      while (!pending_tokens_.empty() && validated_str.size() >= 3 &&
             validated_str.compare(validated_str.size() - 3, 3,
                                   "\xef\xbf\xbd") == 0) {
        new_pending_tokens.push_back(pending_tokens_.back());
        pending_tokens_.pop_back();
        all_tokens.pop_back();
        validated_str = Decode(all_tokens).substr(prefix_str.length());
      }
    } else {
      // prefix 不匹配，弹出最多 3 个 token
      if (pending_tokens_.size() >= 3) {
        bool ok = false;
        while (!pending_tokens_.empty() && new_pending_tokens.size() < 3) {
          new_pending_tokens.push_back(pending_tokens_.back());
          pending_tokens_.pop_back();
          all_tokens.pop_back();
          full_str = Decode(all_tokens);
          if (full_str.compare(0, prefix_str.length(), prefix_str) == 0) {
            ok = true;
            break;
          }
        }
        if (ok) {
          validated_str = full_str.substr(prefix_str.length());
        } else {
          validated_str = Decode(pending_tokens_);
        }
      }
    }

    if (!pending_tokens_.empty()) {
      prefix_tokens_ = pending_tokens_;
    }

    std::reverse(new_pending_tokens.begin(), new_pending_tokens.end());
    pending_tokens_ = new_pending_tokens;

    ret += validated_str;
  }

  return ret;
}

std::string Tokenizer::Finish() {
  std::vector<int32_t> all_tokens;
  all_tokens.reserve(prefix_tokens_.size() + pending_tokens_.size());
  all_tokens.insert(all_tokens.end(), prefix_tokens_.begin(),
                    prefix_tokens_.end());
  all_tokens.insert(all_tokens.end(), pending_tokens_.begin(),
                    pending_tokens_.end());

  std::string prefix_str = prefix_tokens_.empty() ? "" : Decode(prefix_tokens_);
  std::string full_str = all_tokens.empty() ? "" : Decode(all_tokens);

  if (full_str.compare(0, prefix_str.length(), prefix_str) == 0) {
    return full_str.substr(prefix_str.length());
  } else {
    return Decode(pending_tokens_);
  }
}

std::string Tokenizer::IdToToken(int32_t token_id) const {
  return tokenizer_->IdToToken(token_id);
}

int32_t Tokenizer::TokenToId(std::string const &token) const {
  if (token.empty()) {
    return -1;
  }
  return tokenizer_->TokenToId(token);
}

bool Tokenizer::IsEos(int32_t token_id) {
  if (token_id == stop_token_id_) {
    return true;
  }

  return false;
}

Tokenizer::~Tokenizer() {
  if (!tokenizer_path_.empty()) {
    std::lock_guard<std::mutex> lock(global_tokenizer_mutex_);
    auto it = global_tokenizer_.find(tokenizer_path_);
    if (it != global_tokenizer_.end()) {
      it->second.ref_count--;
      if (it->second.ref_count == 0) {
        global_tokenizer_.erase(it);
        LOGD(TAG, "Tokenizer removed from global cache: %s", tokenizer_path_.c_str());
      }
    }
  }
}

}  // namespace gemma4
