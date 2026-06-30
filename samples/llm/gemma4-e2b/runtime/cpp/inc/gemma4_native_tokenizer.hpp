/*
 * Copyright (C) 2024 Shanghai Gua Technology Co., Ltd.
 * All rights reserved
 */

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "tokenizers_cpp.h"

namespace gemma4 {

struct TokenizerInfo {
  std::string tokenizer_path;
  std::shared_ptr<tokenizers::Tokenizer> tokenizer;
  std::string eos_token;
  std::string token_postproc_method;
  size_t ref_count;
};

class Tokenizer {
 public:
  Tokenizer() = default;
  explicit Tokenizer(TokenizerInfo const &tokenizer_info);
  ~Tokenizer();

  /**
   * @brief 创建tokenizer实例，支持两种格式：
   *        - huggingface_tokenizer格式
   *        - sentencepiece 格式
   *
   * @param filename tokenizer模型文件路径
   * @return 返回tokenizer实例
   */
  static std::unique_ptr<Tokenizer> Create(std::string const &filename);

  /**
   * @brief 判断token是否为结束符
   *
   * @param token_id token id
   * @return 返回是否为结束符
   */
  bool IsEos(int32_t token_id);

  /**
   * @brief 将字符串编码为token id列表
   *
   * @param str 输入字符串
   * @return 返回token id列表
   */
  std::vector<int32_t> Encode(std::string const &str) const;

  /**
   * @brief 将字符串列表编码为token id列表, 支持多段文本
   *
   * @param texts 输入字符串列表
   * @return 返回token id列表
   */
  std::vector<std::vector<int32_t>> EncodeBatch(
      std::vector<std::string> const &texts) const;

  /**
   * @brief 将token id列表解码为字符串
   *
   * @param token_ids 输入token id列表
   * @return 返回解码后的字符串
   */
  std::string Decode(std::vector<int32_t> const &token_ids) const;

  /**
   * @brief 将token id解码为字符串
   *
   * @param token_id 输入token id
   * @return 返回解码后的字符串
   */
  std::string Decode(int32_t token_id);

  /**
   * @brief 获取词表大小
   *
   * @return 返回词表大小
   */
  size_t GetVocabSize() const { return tokenizer_->GetVocabSize(); }

  // TODO(cdliang): 这两个接口不对输出的token做后处理
  //                第三方库(tokenizers)也分为Decode/ IdToToken 两种
  std::string IdToToken(int32_t token_id) const;
  int32_t TokenToId(std::string const &token) const;

  int32_t inline eos_token_id() const { return stop_token_id_; }
  std::string DecodeIncremental(const std::vector<int32_t> &delta_tokens);
  std::string Finish();

 private:
  int32_t stop_token_id_;
  std::string token_postproc_method_;
  std::shared_ptr<tokenizers::Tokenizer> tokenizer_;
  std::string tokenizer_path_;

  static std::mutex global_tokenizer_mutex_;
  static std::unordered_map<std::string, TokenizerInfo> global_tokenizer_;

  std::vector<int32_t> pending_tokens_;  // 增量 token 缓存
  std::vector<int32_t> prefix_tokens_;   // 已确认 decode 的 token 前缀
};

}  // namespace gemma4
