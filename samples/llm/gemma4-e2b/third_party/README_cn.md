# 第三方依赖

**简体中文** | [English](./README.md)

本目录用于存放 Gemma4-E2B 示例所依赖的第三方源码。

## tokenizers-cpp

HuggingFace tokenizers 的 C++ 绑定 + sentencepiece，用于推理时的原生 C++
分词（原生推理无需 Python；启动器使用 Python 3）。

**不随 git 提交。** 显式执行 `install_tokenizers_cpp.sh` 时会从
[mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) 拉取
固定 commit 的源码。

启动器和 CMake 不会自动运行安装脚本。从 sample 根目录显式准备：

```bash
bash third_party/install_tokenizers_cpp.sh
```

依赖 `curl` 及外网访问；编译过程还需要 `cargo`（Rust 工具链）来构建
tokenizers 的 Rust binding。 Rust 版本需不低于 1.80；若系统版本过旧，安装脚本会在 `$HOME/.cargo` 下引导安装当前稳定版 rustup 工具链。
