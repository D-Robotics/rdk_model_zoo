<!-- Template: runtime/cpp README (English). Contract: readme-contract.md §4.4.
     Keep anchors; replace ⟪…⟫; delete guidance when done. This README exists
     only where a C++ implementation exists — its absence must never be papered
     over by claiming dual-language support elsewhere. -->

# C++ Runtime — ⟪model name⟫

<a id="supported-boards"></a>
## Supported Boards

> **Must answer:** exactly which boards this build runs on and which are
> excluded, with reasons. “All platforms” requires per-board evidence — do not
> write it otherwise.

| Board | Status | Note |
| --- | --- | --- |
| ⟪board⟫ | ⟪supported-verified / supported-not-run / not-supported⟫ | ⟪reason/link⟫ |

<a id="dependencies"></a>
## Dependencies

> **Must answer:** cross-compile/host toolchain, on-board libraries (libdnn,
> libhbucp, …), header search paths, CMake version. Concrete versions.

- Toolchain: ⟪e.g. OE cross toolchain ⟪version⟫⟫
- On-board libs: ⟪libdnn / libhbucp + version⟫
- CMake ≥ ⟪version⟫

<a id="build"></a>
## Build

> **Must answer:** the full command sequence with cwd; note the SoC macro
> detection (reads `/sys/class/boardinfo/soc_name` at configure time) and any
> per-target CMake switches.

```bash
# cwd: samples/⟪domain⟫/⟪name⟫/runtime/cpp
mkdir -p build && cd build
cmake .. && make -j
# expect: ⟪binary path/name⟫
```

<a id="run"></a>
## Run

> **Must answer:** cwd, prerequisites (artifact prepared), default and
> customized commands, expected observable result.

```bash
# cwd: samples/⟪domain⟫/⟪name⟫/runtime/cpp/build
./⟪binary⟫ --model_path=⟪artifact⟫ ⟪other args with actual defaults⟫
# expect: ⟪observable result / output file⟫
```

<a id="parameters"></a>
## Parameters

> **Must answer:** every gflags parameter (snake_case) with the default that the
> code actually defines — statically machine-checked.

| Parameter | Default | Description |
| --- | --- | --- |
| `--model_path` | ⟪default⟫ | ⟪…⟫ |

<a id="interface-lifecycle"></a>
## Interface & Resource Lifecycle

> **Must answer:** the public surface (config struct + model class), when
> resources are allocated/released (constructor vs `init()`/`deinit()`), the
> free-function data flow (pre_process/infer/post_process by reference), and
> threading boundaries. Reference real symbols from the headers.

- Config: `⟪XxxConfig⟫` — ⟪fields with defaults⟫
- Model: `⟪XxxModel⟫` — ⟪init() loads model & allocates tensors; resources freed in ⟪deinit()/dtor⟫⟫
- Data flow: `pre_process(…) → infer(…) → post_process(…)` passing tensors by reference

<a id="results-interpretation"></a>
## Interpreting Results

> **Must answer:** what stdout/stdored files mean — formats, coordinate
> conventions, exit codes.

⟪e.g. prints one line per detection: [x1,y1,x2,y2] score class_id (pixel
coords); exit 0 on success; result image at ⟪path⟫⟫
