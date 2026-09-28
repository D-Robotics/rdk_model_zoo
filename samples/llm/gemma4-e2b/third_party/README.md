# third_party

[简体中文](./README_cn.md) | **English**

This directory holds third-party dependencies used by the Gemma4-E2B sample.

## tokenizers-cpp

HuggingFace tokenizers C++ binding + sentencepiece, used for native C++
tokenization at inference time (native inference needs no Python; the launcher uses Python 3).

**Not vendored in git.** The source is downloaded by explicitly running
`install_tokenizers_cpp.sh`, which fetches a pinned commit from
[mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp).

Neither `runtime/cpp/run.sh` nor CMake runs the installer. Prepare explicitly from the sample root:

```bash
bash third_party/install_tokenizers_cpp.sh
```

This requires `curl` and network access. The build itself additionally
needs `cargo` (Rust toolchain) to compile the tokenizers Rust binding. Rust 1.80 or newer is required;
when the system version is older, the installer bootstraps the current stable
rustup toolchain under `$HOME/.cargo`.

## Pinned source and local changes

- Upstream: `mlc-ai/tokenizers-cpp`, pinned source commit `c586c52f93f7b060753bd2388eb96a105cb7374d`.
- Rust binding: `tokenizers 0.21.2` + `onig`; sentencepiece/msgpack submodules are also initialized.
- Destination: `tokenizers-cpp/` below this directory, excluded from the Model Zoo Git repository.
- The installer normalizes Rust `Cargo.lock` format 4 to 3, a local compatibility change rather than a new upstream commit.

The installer reuses directories containing its expected CMake files without rechecking their Git commit.
For an unknown existing tree, preserve and inspect local changes; directory presence does not prove version identity.
The source script recreates a directory missing those CMake files, so preserve your own contents before running it.
Configure proxies through `HTTP_PROXY`/`HTTPS_PROXY`. Success prints `tokenizers-cpp ready` or an existing-directory message;
this establishes dependency preparation only, not native build or inference success.

Next, from the sample root, run `bash runtime/cpp/run.sh --target s600 --build`;
see [build and run](../runtime/cpp/README.md#build). Dependency and submodule licenses remain in their downloaded source trees.
