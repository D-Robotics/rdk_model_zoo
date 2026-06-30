# 第三方依赖

**简体中文** | [English](./README.md)

本目录用于存放 Gemma4-E2B 示例所依赖的第三方源码。

## tokenizers-cpp

HuggingFace tokenizers 的 C++ 绑定 + sentencepiece，用于推理时的原生 C++
分词（板端无需 Python）。

**不随 git 提交。** 首次编译时 `install_tokenizers_cpp.sh` 会从
[mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) 拉取
固定 commit 的源码。

`runtime/cpp/run.sh` 在执行 `cmake` 之前会自动调用此脚本。如需手动安装：

```bash
bash third_party/install_tokenizers_cpp.sh
```

依赖 `curl` 及外网访问；编译过程还需要 `cargo`（Rust 工具链）来构建
tokenizers 的 Rust binding。
