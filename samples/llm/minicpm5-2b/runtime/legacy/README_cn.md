[English](README.md) | [简体中文](README_cn.md)

# S100 / S100P：OELLM 1.0.0 C++ Runtime

本入口支持单轮、贪心、关闭思考的纯文本流式生成。`runtime/cpp` 是 S600 的 OELLM 2.0 入口；两个接口和模型包不能混用。

## 环境与内存

板端需要 `build-essential cmake curl`，以及单独下载的 [S100 SDK 1.0.0](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s100/1.0.0/D-Robotics_LLM_S100_1.0.0_SDK.tar.gz) 中 `oellm_runtime/include/xlm.h` 和 `oellm_runtime/lib/`。已测 UCP/DNN 3.7.3、HBRT 4.2.11。不要覆盖系统库，脚本通过 `LD_LIBRARY_PATH` 选择 SDK。板端不需要安装 PyTorch 或编译器 Python 包。

实测模型约 2.9 GB，需要足够大的连续 BPU 内存。两板验证时 `/boot/config.txt` 使用以下 ION 分配，重启后确认实际生效：

```ini
ion=ion_cma_size=0x40000000
ion=ion_reserved_size=0xf0000000
ion=ion_carveout_size=0xf0000000
```

即 CMA 1 GiB、reserved/carveout 各 3.75 GiB。修改前备份原配置，结合板卡实际容量确认；**脚本不会自动修改启动配置或重启**。该配置下本次 S100 的 Linux 内存约 2.8 GiB，S100P 约 14 GiB，不代表所有硬件版本容量。S100 请关闭非必要应用，不能只根据 `free` 判断 BPU 连续内存是否足够。下载与解压需约 6 GB 可用存储。

## 一键运行

```bash
export OELLM_SDK_ROOT=/path/to/D-Robotics_LLM_S100_1.0.0_SDK
cd samples/llm/minicpm5-2b/runtime/legacy
BOARD=s100 bash run.sh --prompt 'What is the capital of France?'
# On S100P:
BOARD=s100p bash run.sh --prompt '请用一句话介绍你自己。'
```

首次执行会校验下载、CMake 编译并运行。`BOARD` 默认 `s100`，在 S100P 上必须显式设置 `s100p`。不支持自动识别或跨板加载。

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
cmake -S . -B build -DOELLM_RUNTIME_ROOT="$OELLM_SDK_ROOT/oellm_runtime"
cmake --build build --parallel 2
export LD_LIBRARY_PATH="$OELLM_SDK_ROOT/oellm_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
timeout 120 ./build/main --model-path ../../model/s100/minicpm5-2b_ctx4096_s100.hbm   --tokenizer-path ../../model/s100/tokenizer   --template-path ../../model/s100/tokenizer/simple-chat.jinja --prompt 'What is the capital of France?'
```

`inc/minicpm5.hpp` 保存配置与模型类，`src/minicpm5.cc` 管理 SDK 生命周期和回调，`src/main.cc` 解析参数。SDK 承担分词、模板渲染、BPU 推理和采样，无需复制通用视觉处理工具。

## 已知边界

编译 chunk=256、cache=4096；输入和输出共享上下文。此旧接口未提供本示例可用的输出 token 上限，因此提供进程超时。独立的 [全量评估入口](../../evaluator/legacy/README_cn.md)覆盖 PPL、双轮、长输入与 50 次连续请求；当前 PPL 与参考文本匹配未达到验收目标，详见 [结果](../../evaluator/README_cn.md)。本 CLI 仍为单请求入口。未验证工具调用、多模态或长时间稳定性。

旧 tokenizer 需要字符串形式 BPE merges 和简化的非思考模板；部署主 EOS 为已有 `<|im_end|>`（130073）。转换脚本不改原始 checkpoint。单请求使用 SDK 示例约定的 request_id=0。
