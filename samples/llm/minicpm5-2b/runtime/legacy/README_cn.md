> 下文的板测结果、精度与 SDK 发布说明为 S 源发布的记录；板端运行按本指南命令执行。

[English](README.md) | [简体中文](README_cn.md)

# S100 / S100P：OELLM 1.0.0 C++ Runtime

本入口支持单轮、贪心、关闭思考的纯文本流式生成。`runtime/cpp` 是 S600 的 OELLM 2.0 入口；两个接口和模型包不能混用。

## 目录结构

```text
legacy/
├── inc/  # C++ 公开接口
├── src/  # C++ 推理与命令行入口
├── CMakeLists.txt  # 原生构建配置
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── run.sh  # 运行示例
```

## 环境与内存

板端需要 `build-essential cmake curl`，以及单独下载的 [S100 SDK 1.0.0](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s100/1.0.0/D-Robotics_LLM_S100_1.0.0_SDK.tar.gz) 中 `oellm_runtime/include/xlm.h` 和 `oellm_runtime/lib/`。已测 UCP/DNN 3.7.3、HBRT 4.2.11。不要覆盖系统库，脚本通过 `LD_LIBRARY_PATH` 选择 SDK。板端不需要安装 PyTorch 或编译器 Python 包。

实测模型约 2.9 GB，需要足够大的连续 BPU 内存。两板验证时 `/boot/config.txt` 使用以下 ION 分配，重启后确认实际生效：

```ini
ion=ion_cma_size=0x40000000
ion=ion_reserved_size=0xf0000000
ion=ion_carveout_size=0xf0000000
```

即 CMA 1 GiB、reserved/carveout 各 3.75 GiB。修改前备份原配置，结合板卡实际容量确认；**脚本不会自动修改启动配置或重启**。该配置下 S100 的 Linux 内存约 2.8 GiB，S100P 约 14 GiB，具体以硬件版本实际容量为准。S100 请关闭非必要应用，不能只根据 `free` 判断 BPU 连续内存是否足够。下载与解压需约 6 GB 可用存储。

## 一键运行

```bash
export OELLM_SDK_ROOT=/path/to/D-Robotics_LLM_S100_1.0.0_SDK
cd samples/llm/minicpm5-2b/runtime/legacy
BOARD=s100 bash ../../model/download_model.sh
BOARD=s100 bash run.sh --build
BOARD=s100 bash run.sh -- --prompt 'What is the capital of France?'
# On S100P:
BOARD=s100p bash ../../model/download_model.sh
BOARD=s100p bash run.sh --build
BOARD=s100p bash run.sh -- --prompt '请用一句话介绍你自己。'
```

准备命令负责校验/下载模型，`--build` 只构建，普通启动只运行已有程序。本包装脚本必须显式设置 `BOARD`；共享启动器在构建或运行前检查实际目标。见[启动器参数](../README_cn.md)。

| 配置 | 说明 |
|---|---|
| `OELLM_SDK_ROOT` | 解压后的 S100 1.0.0 SDK 根目录 |
| `OELLM_RUNTIME_ROOT` | 可直接指定含 include/lib 的 runtime 目录，优先于 SDK_ROOT |
| `MODEL_DIR` | 默认 `model/$BOARD`；已有文件仍须通过固定 SHA256 校验 |
| `INFERENCE_TIMEOUT` | 推理超时秒数，默认 120；不包括下载/编译 |
| `--prompt TEXT` | 默认中文自我介绍；单次调用一个请求 |

成功时输出回答和 `RESULT status=0 ended=1 failed=0 destroy=0`。非正常结束返回非零；超时返回 124。SDK 日志的 Performance 行可作短请求参考，回调中的零值性能字段不作测量。

手动构建：

```bash
cmake -S . -B build -DMINICPM_TARGET=s100 -DOELLM_RUNTIME_ROOT="$OELLM_SDK_ROOT/oellm_runtime"
cmake --build build --parallel 2
export LD_LIBRARY_PATH="$OELLM_SDK_ROOT/oellm_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
timeout 120 ./build/main --model-path ../../model/s100/minicpm5-2b_ctx4096_s100.hbm   --tokenizer-path ../../model/s100/tokenizer   --template-path ../../model/s100/tokenizer/simple-chat.jinja --prompt 'What is the capital of France?'
```

`inc/minicpm5.hpp` 保存配置、`prepare_request`（前处理）与 `RequestOutcome` 状态记录；`src/chat_template.cc` 在推理文件之外加载并检查对话模板大小；`src/minicpm5.cc` 管理 SDK 初始化、经注入 sink 的流式回调与单次请求的释放；`src/main.cc` 解析参数、注入 stdout sink、输出 RESULT 行并映射 `RequestOutcome::exit_code`。推理文件自身不做任何控制台输出。SDK 承担分词、模板渲染、BPU 推理和采样。每个实例只服务一次请求：`predict` 之后再次 `init` 会被拒绝，新请求请创建新实例或新进程。流式 token 在 SDK 运行期间送达 sink；随后的 RESULT 行携带与原先一致的状态值。

完整的原生库使用示例（自包含程序）——可直接复制、对照 SDK 头文件编译并运行：

```cpp
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
```

抛异常的 sink 会被回调内部兜住：异常不会越过厂商 C 回调边界，失败记录在 `stream_error`，流式输出停止，请求仍按原状态映射完成。END 与 ERROR 状态携带的文本按源逻辑被抑制。`PreparedRequest` 拥有自己的 prompt/template 字符串，并在复制和移动时重绑内嵌 SDK 视图；仅在实例存活且未被修改期间把 `&request.input` 传给 SDK。

## 已知边界

编译 chunk=256、cache=4096；输入和输出共享上下文。此旧接口未提供本示例可用的输出 token 上限，因此提供进程超时。独立的 [全量评估入口](../../evaluator/legacy/README_cn.md)覆盖 PPL、双轮、长输入与 50 次连续请求；当前 PPL 与参考文本匹配未达到 ≤3% 精度目标，详见 [结果](../../evaluator/README_cn.md)。本 CLI 为单请求入口；工具调用、多模态输入与长时间稳定性不在其范围内。

旧 tokenizer 需要字符串形式 BPE merges 和简化的非思考模板；部署主 EOS 为已有 `<|im_end|>`（130073）。转换脚本不改原始 checkpoint。单请求使用 SDK 示例约定的 request_id=0。
