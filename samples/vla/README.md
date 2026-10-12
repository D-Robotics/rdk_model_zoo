English | [简体中文](README_cn.md)

# VLA and manipulation-policy integrations

Use ACT or Pi0 to run manipulation policies on the supported boards. Choose the policy and board below, then get the source and prepare its models.

| Capability | Checkout / entry | Source environment | Guide |
| --- | --- | --- | --- |
| ACT on S100 | `act/` | Older LeRobot, v2.1 datasets, SO101 control | [ACT](guides/act.md) |
| ACT on S600 | `pi0/models/act/` | LeRobot v0.5.2, v3.0 datasets, SO100 | [ACT](guides/act.md) |
| Pi0 on S600 | `pi0/models/pi0/` | LeRobot v0.5.2, S600 SDK 1.0.2, SO100 dual-camera | [Pi0](guides/pi0.md) |

For S100 ACT, use `act/`. For S600 ACT and Pi0, use `pi0/`.

## Get the source

From Model Zoo repository root:

```bash
git submodule sync -- samples/vla/act samples/vla/pi0
git submodule update --init --checkout samples/vla/act samples/vla/pi0
git submodule status -- samples/vla/act samples/vla/pi0
```

Source versions:

- ACT: `326ea043be204de25223d95c7d918efe8672dc66`.
- Pi0/S600 tools: `a32de276bc1681a2b1531012de111eaa1c16acb6`.

Both submodules come from `D-Robotics/rdk_LeRobot_tools`. The commands above check out the versions in the [integration registry](integrations.json). Then follow the guides below to install dependencies and prepare models.

## Read and run

Run upstream commands from the corresponding submodule directory (`act/` or `pi0/`). Prepare matching training checkpoints, normalization statistics, camera names and robot calibration data.

- [ACT guide](guides/act.md): S100/S600 setup, model export and inference.
- [Pi0 guide](guides/pi0.md): model preparation, offline inference and robot control.
- [ACT README](act/README.md) and [S600 tools README](pi0/README.md): source documentation available after submodule initialization.

For live robot control, prepare the hardware and calibration described in the upstream guide.

## License

The submodules use Apache-2.0; see their [license](act/LICENSE). Checkpoints, datasets and vendor SDKs retain their respective licenses.
