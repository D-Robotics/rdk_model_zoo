English | [简体中文](README_cn.md)

# VLA and manipulation-policy integrations

[中文](README_cn.md)

These entries preserve complete upstream repositories as Git submodules. Choose
the policy **and its target/version** before preparing the environment. They are
separate from the offline [HIMLoco locomotion sample](../robotics/himloco/README.md).
The release manifest records ACT/Pi0 as manual integrations, without published
Model Zoo assets or an automatic model download.

| Capability | Checkout / entry | Source environment | Guide |
| --- | --- | --- | --- |
| ACT on S100 | `act/` | Older LeRobot, v2.1 datasets, SO101 control | [ACT](guides/act.md) |
| ACT on S600 | `pi0/models/act/` | LeRobot v0.5.2, v3.0 datasets, SO100 | [ACT](guides/act.md) |
| Pi0 on S600 | `pi0/models/pi0/` | LeRobot v0.5.2, S600 SDK 1.0.2, SO100 dual-camera | [Pi0](guides/pi0.md) |

`pi0/` names the second integration, not a Pi0-only repository. It also contains
the S600 ACT backend. Initializing only `act/` does not provide S600 ACT.

## Initialize the exact sources

From Model Zoo repository root:

```bash
git submodule sync -- samples/vla/act samples/vla/pi0
git submodule update --init --checkout samples/vla/act samples/vla/pi0
git submodule status -- samples/vla/act samples/vla/pi0
```

Expected pins:

- ACT: `326ea043be204de25223d95c7d918efe8672dc66`.
- Pi0/S600 tools: `a32de276bc1681a2b1531012de111eaa1c16acb6`.

Both originate from `D-Robotics/rdk_LeRobot_tools`. The parent Git tree and
[integration registry](integrations.json) pin the versions; do not use
`git submodule update --remote` or switch to a moving branch to reproduce them.
Initialization fetches source only; it does not install dependencies, download
models, build a runtime or connect to a robot. A normal clone without submodule
initialization leaves the source directories empty.

## Read and run

The guides below distinguish source preparation, model conversion, offline board
inference and live robot control. Each upstream guide's “repository root” refers
to its submodule root, not Model Zoo root. Preserve training checkpoints,
normalization statistics, camera names and robot calibration as one compatible
set; these are supplied by the operator and are not included here.

- [ACT guide](guides/act.md): S100/S600 differences, export inputs, runtime entry,
  source measurements and full workflow links.
- [Pi0 guide](guides/pi0.md): three-model chain, fixed deployment config, offline
  inputs/output, source results and live-control boundary.
- [Upstream ACT README](act/README.md) and [S600 tools README](pi0/README.md)
  become available locally after initialization, together with all source code,
  demos and workflow documents.

Offline board inference workflows are documented in the guides. Live robot
control follows the upstream guides and uses operator-supplied hardware and
calibration.

## License and changes

Both pinned repositories include Apache-2.0 [licenses](act/LICENSE); checkpoints,
datasets, vendor SDKs and robot hardware retain their respective terms. Changes
to a submodule require an explicit new upstream commit and an intentional parent
pin update. Initialize and use the unified paths above.
