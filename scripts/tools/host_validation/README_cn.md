[English](README.md) | 简体中文

# 主机验证执行器（维护者工具）

`run.py` 是面向维护者的单一命令：以一次确定性、相互隔离的运行执行统一源码的
完整主机门禁。它**不是**面向用户的推理 CLI，也不新增任何模型能力：用户与
Agent 继续使用各 Sample 的原生入口（ADR-0003）。本工具只做测试编排——
不下载、不加载板端 SDK、不初始化 VLA 子模块、不在板上执行。

```bash
python scripts/tools/host_validation/run.py --repo PATH --report PATH [--python PYTHON]
```

`--repo` 是被校验的 checkout（执行器本身可以位于另一个 checkout）。
`--report` 是机器可读 JSON 报告路径，**必须位于源码树之外**（写入 checkout
内部的报告会被拒绝：那会污染工作区并使源码身份门禁失效）。构建产物与各套件
日志写入报告旁边的 `host-validation-artifacts/`。`--python` 选择执行测试套件
的解释器（默认为运行 `run.py` 的解释器）。退出码 0 表示所有部分全部通过。

## 一次运行执行的内容

| 部分 | 内容 |
| --- | --- |
| Python 套件 | 所有第一方 `unittest` 目录：`samples/` 下全部 Sample 的 `tests/`（覆盖全部 51 个原生 Sample，含嵌套的 `samples/vision/yoloe/conversion/tests` 与 `evaluator/tests`）、`utils/py_utils/tests`（含 `test_vla_integration.py`——父仓库 gitlink/固定提交完整性检查，无需初始化子模块即可运行，绝不执行上游 ACT/Pi0 代码）、`scripts/tools/board_validation/tests`、`scripts/tools/sample_contract/tests`、`skills/tests` 以及本目录自身的测试 |
| Sample 覆盖核对 | 发现结果与已接受的 51 行 Sample 清单（`docs/releases/unified-migration/2026-10-05-all-sample-coverage.json`）交叉核对——清单是**必需凭据**：清单所列 Sample 或其 `tests/` 目录从源码树中被删除、清单之外多出 Sample、或清单缺失/不可解析/rows 为空/存在畸形或重复行，均以显式结构化原因判失败（清单行被校验并上报，绝不静默过滤掉；强制范围跟随已评审的清单，执行器中不硬编码任何 Sample 数量） |
| 原生前置检查 | C++17 编译器、cmake/ctest、git、`native_dependencies.py` 的可移植解析（nlohmann-json、gflags、iconv）——均为声明必需；缺失即失败，不允许隐藏原生覆盖。libsndfile/libsamplerate 一并探测并记录（它们是默认开启的 ASR CTest 开关的门禁） |
| 静态契约 | `scripts/tools/sample_contract/check.py --scope migration --parser-mode import`，检查报告一并保留 |
| 原生 CTest | 六个主机安全工程，**默认开启**完整门禁开关：`gemma4-e2b/tests/native`（C++17 + OpenCV + 固定 platforms 提交）、`yoloe/runtime/cpp/tests`（`YOLOE_TEST_OPENCV=ON`：真实 OpenCV 图像/掩码/流水线测试 + SDK 替身与 CLI fixture）、`asr/runtime/cpp/tests`（`ASR_AUDIO_TESTS=ON` + `ASR_CLI_TESTS=ON`：真实 libsndfile/libsamplerate 前端与主机 CLI 测试）、`ultralytics_yolo/runtime/cpp/test`（面向窄 API 替身的共享辅助与描述符适配器）、`paraformer/runtime/cpp`（`PARAFORMER_BUILD_TESTS=ON` + `PARAFORMER_BUILD_IO=ON` + `PARAFORMER_SANITIZERS=ON`：带消毒剂的 contract/pipeline 测试、面向显式伪头文件的 SDK 替身测试、基于 Git 跟踪证据文件（相对路径、不下载/不导出）的 preflight 与 prepared-feature 检查、`PARAFORMER_HOST_FIXTURE` CLI 帮助测试）、`himloco/runtime/cpp`（`HIMLOCO_BUILD_TESTS=ON`：基于仓库内置 `obs_history` fixture 的无 SDK 数值策略测试）——仅 SDK 替身与主机库；无厂商 SDK、无模型文件、无下载、不编译生产 SDK 工程，也无板端 SDK 与真实推理。两个 `runtime/cpp` 工程的厂商 SDK 适配器与生产 CLI 开关（`PARAFORMER_BUILD_SDK`/`PARAFORMER_BUILD_CLI`、`HIMLOCO_BUILD_SDK`/`HIMLOCO_BUILD_CLI`）被强制保持 **OFF**：`runtime/cpp` 下的源码目录被配置本身并不意味着厂商执行——这些 OFF 默认值是强制性安全要求，不是主机测试范围的缩减，试图开启的 `--cmake-define` 会在任何构建之前被拒绝（`VAR:BOOL`/`VAR:STRING`/`VAR:PATH` 这类带类型的 CMake 缓存拼写会在选项解析阶段被直接拒绝：CMake 允许靠后的类型化定义覆盖先前的裸值，因此类型化键名正是绕过强制 OFF 开关、或在不记录范围缩减的情况下悄悄关闭默认 ON 开关的途径；带空白填充的假值同样被拒绝——CMake 保留 `-D` 值的前导空白，`" OFF "` 会使开关保持开启，只有精确无填充的 CMake 假常量重申才是可接受的无操作：OFF/FALSE/0/NO/N/IGNORE 与空值大小写不敏感，精确的 `NOTFOUND` 与任何以 `-NOTFOUND` 结尾的值则区分大小写） |
| Catalog | `scripts/tools/catalog-publisher` 内执行 `npm run check`（来源校验、Vitest 套件、构建、`catalog:check`）——仅在包自身 `engines` 声明的 Node 范围内执行（范围缺失、不可解析或不满足即带原因失败）；**最先运行**（先于 Python 套件），使其构建阶段生成的 `dist/catalog.json` 在 ultralytics_yolo 资产/清单快照套件读取之前就已存在 |

每个套件在独立子进程中运行，因此各 Sample 中同名测试模块相互隔离。每个套件
子进程都以被请求的仓库为工作目录启动：无论维护者从哪个目录启动执行器，相对
示例路径与子解释器对仓库根模块的导入都能确定性解析。结果来自机器可读的
`unittest` 结果：精确计数、逐条 skip 的身份与原因、逐条失败信息。

部分顺序是有意义的：**catalog 检查先于 Python 套件运行**，因为其构建阶段会
生成 `scripts/tools/catalog-publisher/dist/catalog.json`，而 ultralytics_yolo 的
资产/清单快照套件（`test_platform_assets`、`test_yolo26`）读取的正是这个
文件。`dist/` 是被忽略的构建产物，干净 checkout 上并不存在，若在这些套件
之后才构建 catalog，它们必然失败（只有脏开发 checkout 上遗留的被忽略
`dist/` 掩盖了这一依赖）。catalog 部分仍在同一源码稳定性区间内运行——位于
起始快照之后、结束快照之前——顺序调整绝不弱化内容门禁；`scripts/tools/catalog-publisher`
内的 `npm ci` 仍是维护者前置条件：执行器从不安装任何东西。

运行还会校验源码树自身声明的历史 Git 对象（`utils/py_utils/legacy_platforms.py`、
Gemma 原生 CMake 固定提交、catalog 提交来源）：缺失固定提交即为显式失败，
并给出对应的 `git fetch` 命令——绝不静默跳过读取固定来源的套件。因此完整的
clone 历史是前置条件。

## 源码身份（基于内容，而非仅 HEAD）

运行前后各做一次快照：HEAD、分支、`git status --porcelain` 脏文件清单，以及
对**文件内容**的确定性 sha256 摘要——覆盖所有被跟踪文件（排除 mode-160000
gitlink：树中不存在其上游代码）加上所有未被 Git 忽略规则排除的未跟踪文件
（因此 `__pycache__`、`node_modules` 等被忽略的产物绝不算漂移）。运行期间
HEAD 变化、脏文件清单变化或内容摘要变化都会使门禁失败；报告记录两份摘要，
以及"脏但稳定"checkout 的脏状态出处。不是 Git 仓库（无法解析 HEAD）的
checkout 绝不可能通过：源码身份是前置条件，不是可选注记。

## Skip 策略（默认严格模式）

| 类别 | 含义 | 严格模式 |
| --- | --- | --- |
| `optional_export` | 恰好是已声明的可选导出范围：`*/conversion/tests` 目录下的 Torch/FunASR/Ultralytics 导入失败（记录为 `optional_missing`），以及 Paraformer 导出阶段套件的条件性框架 skip（`samples/speech/paraformer/tests/test_export_stages`） | 允许；始终记录身份与原因，绝不计入已执行测试数 |
| `native_prerequisite` | 缺少 C++ 编译器、nlohmann-json、gflags、iconv 或 CMake | **拒绝**（运行失败）；`--allow-native-skips` 可改为仅记录 |
| `conditional` | 合法的条件性 skip（例如安装了板端 SDK 导致导入失败路径不可达） | 允许并记录 |
| 其他 | 未声明的 skip——包括无关运行时/模型/原生套件中以 torch/funasr/ultralytics 为名的 skip：可选范围绝不掩盖依赖回归 | **拒绝** |

发现 0 个测试、声明的目录缺失、无法解释的测试目录、worker 崩溃、加载器崩溃、
套件超时、CTest 分阶段超时/可执行文件失败、CTest 发现输出损坏、声明数与实际
运行数不一致、已接受 Sample 清单缺失/畸形/不匹配与源码漂移都绝不可能报告
成功。CTest 用例数单独记录在独立部分（按实际运行工程的真实用例数，无硬编码
总数），刻意不与 Python unittest 总数相加。

## 选项

| 选项 | 作用 |
| --- | --- |
| `--python PYTHON` | 执行测试套件所用的解释器 |
| `--timeout N` | 单套件超时秒数（默认 1800） |
| `--suite SUBSTRING` | 只运行目录匹配的套件（可重复；报告标记 `ci_equivalent: false`） |
| `--list` | 打印发现结果（套件、仅原生目录、Sample 覆盖、固定提交、CTest 注册表与默认开关）并退出 |
| `--skip-ctest` / `--skip-contract` / `--skip-catalog` | 跳过一个部分，显式记录为 `skipped-explicit`；不等价于 CI；`--skip-catalog` 不会构建 `dist/catalog.json`，ultralytics_yolo 资产/清单快照套件因此需要事先生成的 catalog（缺失即失败） |
| `--allow-native-skips` | 记录而非拒绝原生前置 skip；不等价于 CI |
| `--cmake` / `--ctest` | 可执行文件覆盖（也会在解释器旁查找，例如 pip 安装的 cmake） |
| `--cmake-define PROJECT:VAR=VALUE` | 覆盖某个 CTest 工程的定义；完整门禁默认值（`YOLOE_TEST_OPENCV`、`ASR_AUDIO_TESTS`、`ASR_CLI_TESTS`、`PARAFORMER_BUILD_TESTS`/`BUILD_IO`/`SANITIZERS`、`HIMLOCO_BUILD_TESTS` = `ON`）自动合并——以任意 CMake 假常量关闭默认 ON 的开关，会记录为以原始值命名的范围缩减并将 `ci_equivalent` 置 `false`。分类依据真实 CMake 对 `-D` 值的实际读取方式（已对照 CMake 4.4.4 验证）：命名常量 OFF/FALSE/0/NO/N/IGNORE 与空值大小写不敏感；精确的 `NOTFOUND` 与任何以 `-NOTFOUND` 结尾的值区分大小写（`notfound`/`X-notfound` 为真值，不是假常量）；尾随空白由 CMake 的 `-D` 缓存写入自行剥离（`NO ` 实际为 OFF），前导空白则保留（` NO` 与 `" NO "` 使开关保持开启，不记录缩减）。厂商 SDK / 生产 CLI 开关（`PARAFORMER_BUILD_SDK`/`BUILD_CLI`、`HIMLOCO_BUILD_SDK`/`BUILD_CLI`）被强制 OFF：试图开启的覆盖在任何构建之前即被拒绝，原因输出到 stderr（只有精确、无填充的假常量重申是无操作——CMake 保留 `-D` 值的前导空白，守卫绝不猜测 CMake 会剥离哪种填充，因此带空白填充的假值一律以失败关闭方式拒绝）。仅接受无类型的 `PROJECT:VAR=VALUE` 拼写——带类型的 CMake 缓存键（`VAR:BOOL`、`VAR:STRING`、`VAR:PATH` 等）属于不支持的语法，会在选项解析阶段以可操作的报错直接拒绝，先于任何 configure/build |

任何缩小范围的选项（跳过部分、套件过滤、允许原生 skip、关闭默认开关）都会
写入报告并将 `ci_equivalent` 置为 `false`，因此本地的局部运行不可能被误认为
完整 CI 门禁。

## OpenCV 解析

需要 `find_package(OpenCV)` 的工程在配置前先做探测解析：先尝试普通
`find_package`（apt 的 `libopencv-dev` 在此解析成功），再尝试文档化的
`OPENCV_DIR`/`OpenCV_DIR` 环境变量覆盖，最后是标准包管理器前缀
（`/opt/homebrew/opt/opencv*/lib/cmake/opencv{4,5}`、`/usr/local/opt/opencv*/…`、
`/usr/lib/<arch>-linux-gnu/cmake/opencv4`）——因为 Homebrew 将配置安装为
`opencv5`，CMake 无法经 `opencv4` 布局找到。解析模式与目录记录在 CTest 部分；
失败时给出安装命令。不硬编码任何个人路径。

## 前置依赖

- Python 3.10 或 3.12，并执行 `pip install -r scripts/tools/host_validation/requirements.txt`
  （numpy、opencv-python-headless、PyYAML、scipy、onnx、onnxruntime、
  pycocotools、pillow、ftfy、regex——以及 Sample 已声明的核心主机依赖
  `lap==0.5.12` 与 `cython-bbox==0.1.5`（ByteTrack 跟踪器真实的 CPU 依赖）
  和 `jsonschema`（skills 证据校验工具所需且拒绝自动安装）；三者均为必需，
  不存在围绕它们的自动跳过）。SciPy 使用环境标记：Darwin 且
  Python ≥ 3.12 要求 ≥ 1.17.1（1.15.3 macOS arm64 wheel 不可用——
  `scipy.sparse.linalg` 导入失败，见 scipy/scipy#25635）；Python 3.10 保持
  一般的 ≥ 1.10 下限（SciPy 1.17 不支持 3.10；该故障仅限 Darwin）。
  opencv/pycocotools 下限即 Sample 自身声明的约束（≥ 4.8 / ≥ 2.0.7）：
  曾在 opencv 4.12.0.88 与 pycocotools 2.0.10 上观察到的门禁失败是
  Zoo 自身 fixture/评估器的缺陷（固定 legacy 加载周围的整表
  `sys.modules` 回滚，以及对评估器公开契约从未要求的无 `info` 文档调用
  `loadRes`），两者均已在 Sample 内修复并带有新进程回归——升级任一库从来
  不是修复手段。Torch/FunASR/Ultralytics 按设计保持可选；其导出套件报告
  `optional_export`。
- C++17 编译器、CMake ≥ 3.18 与 CTest、OpenCV C++ 开发头文件（Gemma 与
  YOLOE OpenCV 门禁测试需要）、gflags 与 nlohmann-json（标准系统路径或
  pkg-config；环境变量覆盖见 `native_dependencies.py`）、默认开启的 ASR
  audio/CLI 测试所需的 **libsndfile 与 libsamplerate**（`brew install
  libsndfile libsamplerate` / `apt install libsndfile1-dev
  libsamplerate0-dev`）、pkg-config、含完整历史与声明固定提交的 git clone。
- Node.js ≥ 22.12 且 < 23，并在 `scripts/tools/catalog-publisher` 执行过 `npm ci`
  （catalog 部分需要；执行器自身从不安装任何东西；Node 不在声明的
  `engines` 范围内时该部分直接失败，而不是声称一次不受支持的通过）。

## CI

`.github/workflows/host-validation.yml` 在 Ubuntu（Python 3.10 与 3.12）和
macOS 上、于 `develop`/`main` 的 push 与 pull request 时运行本命令，使用完整
clone 历史（不含子模块）、安装声明的原生前置依赖，报告写入 `$RUNNER_TEMP`
（绝不写入工作区，内容身份门禁不会被自身报告触发）且即使失败也上传。本地
绿灯不是 CI 结论：CI 结果只以 GitHub 任务实际输出为准。

## 文件

- `run.py` —— 编排器与逐套件 worker（即上述 `--repo/--report/--python` 接口）。
- `test_run.py` —— 针对执行器的 fixture 测试（发现、隔离、套件工作目录、
  skip 策略、超时、固定提交、源码漂移、CTest 分阶段失败与注册工程的安全
  默认值/禁止覆盖守卫、catalog engines、catalog 先于 Python 的顺序、必需的
  Sample 清单、报告结构），基于合成仓库。
- `requirements.txt` —— 核心主机依赖集合（含环境标记与实测版本记录）。
- `native_dependencies.py` / `test_native_dependencies.py` —— 与 LLM 主机测试
  共用的可移植原生依赖发现（解析顺序与环境变量覆盖见模块文档字符串）。
