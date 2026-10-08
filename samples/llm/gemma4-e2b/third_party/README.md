English | [简体中文](README_cn.md)

# third_party


This directory holds third-party dependencies used by the Gemma4-E2B sample.

## Directory structure

```text
third_party/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── install_tokenizers_cpp.sh  # Shell command
```

## tokenizers-cpp

HuggingFace tokenizers C++ binding + sentencepiece, used for native C++
tokenization at inference time (native inference needs no Python; the launcher uses Python 3).

**Not vendored in git.** The source is downloaded by explicitly running
`install_tokenizers_cpp.sh`, which fetches the recorded upstream revision from
[mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp).

Neither `runtime/cpp/run.sh` nor CMake runs the installer. Prepare explicitly from the sample root:

```bash
bash third_party/install_tokenizers_cpp.sh --dry-run
bash third_party/install_tokenizers_cpp.sh
```

Preparation requires Git, network access, and an explicitly installed stable Rust
1.80+ toolchain (`rustc` and `cargo` on PATH, or under `$HOME/.cargo/bin`). The script
checks these prerequisites and never installs or upgrades Rust. `--dry-run` prints
the pinned source and destination without network access or file creation; it does
not require the build tools. `--help` prints usage.

If Rust is missing or too old, install a suitable toolchain before rerunning this
step; when rustup is already available, `rustup toolchain install stable --profile minimal`
and `rustup default stable` select a stable toolchain explicitly. Confirm with
`rustc --version` and `cargo --version`. This changes your active Rust toolchain,
so choose a project override instead when other projects need another version.

## Pinned source and local changes

- Upstream: `mlc-ai/tokenizers-cpp`, pinned source commit `c586c52f93f7b060753bd2388eb96a105cb7374d`.
- Rust binding: `tokenizers 0.21.2` + `onig`; sentencepiece/msgpack submodules are also initialized.
- Destination: `tokenizers-cpp/` below this directory, excluded from the Model Zoo Git repository.
- The installer normalizes Rust `Cargo.lock` format 4 to 3, a local compatibility change rather than a new upstream commit.

New downloads are prepared in a temporary sibling directory and moved into place
only after checking the root commit and recursively initialized submodule pins.
Existing checkouts must match the pin, contain the expected CMake files and have
no local changes except the documented exact Cargo.lock format patch. Modified,
incomplete, mismatched or symlink destinations are rejected and preserved; inspect
and relocate your own work before retrying. A failed clone removes only its own
temporary directory. The installer does not overwrite or reset an existing tree.

Configure proxies through `HTTP_PROXY`/`HTTPS_PROXY`. Success prints
`tokenizers-cpp ready` or `already prepared` with the full commit and confirms
source preparation; the native build follows in the next step. Building the Rust
binding can still contact package registries unless dependencies are cached.

Next, from the sample root, run `bash runtime/cpp/run.sh --target s600 --build`;
see [build and run](../runtime/cpp/README.md#build). Dependency and submodule licenses remain in their downloaded source trees.
