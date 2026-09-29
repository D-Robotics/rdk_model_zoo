#include "minicpm5.hpp"

int main() {
  minicpm5::Config config;              // model_path 默认 ../../model/s600
  config.max_new_tokens = 128;          // 1-4096
  minicpm5::MiniCPM5 model(config);     // 校验配置并准备 runtime；失败抛异常
  minicpm5::Result first = model.Generate("What is 1+1?");  // 开启会话
  if (first.status == 3) {              // 3 EOS；6 生成上限；4 上下文上限
    // 消费 first.text、first.tokens、first.ttft_ms、first.decode_tps、first.e2e_ms
  }
  minicpm5::Result follow = model.Generate("Translate that.", false);  // 同一会话
  return first.status == 3 && follow.status == 3 ? 0 : 1;
}
