#include <iostream>

#include "minicpm5.hpp"

int main() {
  MiniCPM5Config config;
  config.model_path = "model/s100/minicpm5-2b_ctx4096_s100.hbm";
  config.tokenizer_path = "model/s100/tokenizer";
  config.template_path = "model/s100/tokenizer/simple-chat.jinja";
  config.prompt = "请用一句话介绍你自己。";
  config.text_sink = [](const char* chunk) {  // 可选；不设置则静默
    std::cout << chunk << std::flush;
  };
  MiniCPM5 model(config);                    // 仅保存配置
  model.init();                              // 加载模型与分词器；每个实例一次请求
  RequestOutcome outcome = model.predict();  // 经 sink 流式输出
  // 消费：outcome.ended、outcome.failed、outcome.sdk_status、outcome.destroy_status、
  // outcome.stream_error；仅正常 EOS 且成功释放时 outcome.exit_code() 为零。
  return outcome.exit_code();
}
