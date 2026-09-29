# MiniCPM 原生启动编排

**简体中文** | [English](README.md)

启动器只依赖 Python 3 标准库，用于选择原生 SDK 实现；分词、模型执行与生成仍在 C++。
S100/S100P 使用 [OELLM 1.0.0 legacy](legacy/README_cn.md)，S600 使用
[OELLM 2.0 实现](cpp/README_cn.md)。上述版本属于固定源记录；文件名相同不代表 SDK 或模型可互换。

## 显式准备、构建、运行

在对应板卡上，从仓库根目录开始：

```bash
cd samples/llm/minicpm5-2b
# 单独获取匹配 SDK，设置其解压后的 runtime 目录。
export OELLM_RUNTIME_ROOT=/path/to/matching-sdk/oellm_runtime
BOARD=s600 bash model/download_model.sh
python3 runtime/launcher.py --target s600 --build
python3 runtime/launcher.py --target s600 -- --prompt 'What is the capital of France?'
```

S100/S100P 请同时更改两个目标选择，并使用其 OELLM 1.0.0 SDK。启动器不安装软件包、不下载模型、不执行量化。
`--build` 只构建，不运行推理；普通运行不会自动编译缺失的程序。各目标使用独立模型/构建目录，原生参数放在 `--` 后。

## 无板主机预览

```bash
python3 samples/llm/minicpm5-2b/runtime/launcher.py --target s100p --dry-run
python3 samples/llm/minicpm5-2b/runtime/launcher.py --target s600 --build --dry-run
```

输出 JSON 包含 SDK 分支、路径、命令与超时，不执行子进程、不读模型、不创建文件，也不要求安装 SDK。
未设置 runtime 路径时显示 `null` / `<set-runtime-root>`。真正构建/运行先检查板型，拒绝不匹配或无法识别的主机。
预览不证明运行兼容；本轮迁移未运行板测。

## 参数与环境

| 启动器参数 | 默认值 / 行为 |
| --- | --- |
| `--target` | 默认 `auto`；支持 `s100`、`s100p`、`s600`；无板预览需显式指定 |
| `--runtime-root` | 优先 `OELLM_RUNTIME_ROOT`，其次 `$OELLM_SDK_ROOT/oellm_runtime`；不自动下载 SDK |
| `--model-dir` | 优先 `MODEL_DIR`，否则 sample 的 `model/<target>` |
| `--build-dir` | `runtime/cpp/build-s600` 或 `runtime/legacy/build-<target>` |
| `--build` | 只执行 CMake 配置/构建，不接受推理参数 |
| `--dry-run` | 只打印所选操作 |
| `--timeout` | S100/S100P 超时，必须为正有限秒数；取 `INFERENCE_TIMEOUT` 或 120；S600 不额外限时 |
| `-- 原生参数` | 保留参数边界，传给对应 C++ CLI |

S600 原生参数为 `--model_path`、`--prompt`、`--follow_up`、`--max_new_tokens`；
legacy 为 `--model-path`、`--tokenizer-path`、`--template-path`、`--prompt`，没有等价的输出 token 上限。
模型选择优先使用启动器 `--model-dir`。各运行指南保留原生直接调用方式。
启动器使用按目标划分的构建目录；手动 CMake 示例则使用显式指定的 `build` 目录。

启动器把 SDK `lib` 加到 `LD_LIBRARY_PATH` 首部；S600 设置 L2M 为 `6:6:6:6`。
原生退出码直接传回，编排错误返回 2，legacy 超时返回 124。
源模型准备命令校验固定归档/清单哈希，启动时不重复校验，详见[模型说明](../model/README_cn.md)。

## 迁移边界

原生推理核心与完整 README 契约仍在重构。源量化/评估文件与历史精度未达标结论保留，不重新运行量化。
启动器主机测试通过不代表这些迁移项完成，也不把历史板测升级成本轮证据。
