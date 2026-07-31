# third_party

[简体中文](./README_cn.md) | **English**

This directory holds third-party dependencies used by the Gemma4-E2B sample.

## tokenizers-cpp

HuggingFace tokenizers C++ binding + sentencepiece, used for native C++
tokenization at inference time (no Python required on board).

**Not vendored in git.** The source is downloaded on first build by
`install_tokenizers_cpp.sh`, which fetches a pinned commit from
[mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp).

`runtime/cpp/run.sh` calls the installer automatically before `cmake`. To
set it up manually:

```bash
bash third_party/install_tokenizers_cpp.sh
```

This requires `curl` and network access. The build itself additionally
needs `cargo` (Rust toolchain) to compile the tokenizers Rust binding. Rust 1.79 or newer is required;
when the system version is older, the installer bootstraps the current stable
rustup toolchain under `$HOME/.cargo`.
