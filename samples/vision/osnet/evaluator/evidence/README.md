# Imported experiment evidence / 原实验记录

Source chat: **转换 OSNet 并优化 RDK X5 推理**,
`01a1839a-979c-7741-bad7-9244602fd153`, experiment date 2026-10-29.
Original workspace: `d-robotics-dev:/data/zhengyi/osnet-x5-20261029`.
The delivery ZIP is pinned in `../../model/artifacts.json`.

`2026-10-29-report.zh-CN.md` and `normalized-retrieval.json` preserve the
original report and complete normalized retrieval scores. They were recovered
from the referenced chat's recorded outputs. They describe the original
artifact, source implementation and environment, not a new run of this sample.

原始转换环境：X5 OE v1.2.8-py310，hb_mapper 1.24.3，HBDK 3.49.15。
原始板端环境：RDK X5 V1.0，libdnn 1.24.5，HBRT 3.15.55.0，BPU 996 MHz，
CPU `schedutil`。校准集为 128 张 Market1501 训练图，seed=20261029。

The original upstream source URL is
[deep-person-reid OSNet](https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/osnet.py),
with [MIT license](https://github.com/KaiyangZhou/deep-person-reid/blob/master/LICENSE).
The experiment checkpoint is the
[official x1.0 Market1501 checkpoint](https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view),
SHA-256 `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154`.
The delivery contains the actual `source/osnet.py`, its MIT license,
`results/export.json`, conversion configs and source-file digest. A moving
`master` URL is not a pinned Git revision; a missing upstream commit must remain
unknown and must not be inferred from the artifact date.

The source has also been fetched from upstream and verified against the original
export digest: `c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae`.
The unmodified bytes and MIT license are kept in `../../third_party/`.
`export.json` preserves the original source/weight/ONNX pairing; its ONNX hash
identifies the initial `[1,512]` graph, not the spatial-head export.

## Migration status / 接入状态

On 2026-10-29, the original X5 files were recovered read-only through dev-s600.
`2026-10-29-readback.json` records the model/fixture identities and checks.
The three BIN hashes match the original delivery records; each original board
capture records 19281 images, and its model hash matches the recovered BIN.
The original C++ timing JSONs match `candidate-summary.json` exactly.
Raw small board receipts are preserved in `x5-readback/`.

The new Python preprocessing matches all 32 real fixture inputs byte-for-byte.
The numerical C++ source matches the original board file; only introductory
build comments and the main function name differ. These checks did not load
the SDK or execute inference on either a host or a board.

| Check | Status |
| --- | --- |
| Original X5 full Market1501 evaluation | Historical executed result, 19281 images |
| Original C++ fixed-input verification | Historical executed result, 32 inputs per candidate, max feature error 0 |
| Original varied-size verification | Historical executed result, 12 sizes |
| New Python/C++ entrypoints on X5 | `not-run` |
| Original BIN availability | Three files retrieved locally and copied to dev-server; SHA-256 verified |
| Host preprocessing of original fixed inputs | 32/32 byte-exact |
| New conversion workflow in OE | `not-run` |
| S100/S100P/S600 artifact, accuracy, performance | Unavailable / `not-run` |
| Public Web release | Not published |

Host unit tests are independent of these board statuses. The imported C++
implementation preserves the experiment's numerical code; the main entry name
is changed only to link it with the standard `main.cpp` launcher. Python keeps
the verified Pillow/planar-byte/L2 operations and adds explicit validation and
artifact identity checks.

公开 Web 发布仍缺少稳定公开制品/OE 报告 URL，以及当前提交指南要求的双路 Runtime
和 C++ E2E 证据；不能用此次历史单路结果填补这些字段。
