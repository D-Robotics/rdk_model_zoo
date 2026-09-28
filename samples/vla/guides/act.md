# ACT: choose the S100 or S600 source

[中文](act_cn.md) · [Integration overview](../README.md)

## Versions and prerequisites

| Target | Exact checkout | LeRobot/data | Runtime |
| --- | --- | --- | --- |
| S100 | `samples/vla/act`, `326ea043be204de25223d95c7d918efe8672dc66` | D-Robotics older fork; v2.1 | Root `bpu_control_robot.py`, SO101 default |
| S600 | `samples/vla/pi0`, `a32de276bc1681a2b1531012de111eaa1c16acb6` | Hugging Face LeRobot v0.5.2; v3.0 | `models/act/bpu_control_robot.py`, SO100Follower |

The S600 source explicitly excludes the older D-Robotics fork. A source config
listing other marches is not verified board support. Use the board-image BPU
runtime and the environment matching the selected guide. Source-supported target
claims are inherited; no board execution was performed in this migration.

## Initialization and reading

From Model Zoo root:

```bash
git submodule update --init --checkout samples/vla/act samples/vla/pi0
git -C samples/vla/act rev-parse HEAD
git -C samples/vla/pi0 rev-parse HEAD
```

Then use the complete [S100 guide](../act/README.md),
[S100 workflow](../act/doc/WORKFLOW_GUIDE_EN.md) or
[S600 ACT guide](../pi0/models/act/README.md). Demo images and training examples
remain in those submodules. All commands in the remainder of this page run from
the selected submodule root.

## Trusted export/compilation recipe

Provide the trained ACT directory (`config.json`, `model.safetensors`), matching
dataset and calibration statistics. Set their paths in the selected YAML.
For S100 use `bpu_export_config.yaml`; for S600 use
`bpu_export_config_s600_calfix.yaml` (`nash-p`). The generic S600-tree template
defaults to `nash-e`, so it is not the S600 recipe.

```bash
# S100, cwd samples/vla/act
python export_bpu_actpolicy.py --config bpu_export_config.yaml
# S600, cwd samples/vla/pi0
python models/act/export_bpu_actpolicy.py --config bpu_export_config_s600_calfix.yaml
```

Run the generated `build_all.sh` inside the appropriate OE environment as explained
by the source guide. Keep the two final VisionEncoder/TransformerLayers HBM files
and all generated normalization `.npy` files together in `bpu_output/`.
Camera names in the statistics must match runtime camera names. The S600 source
specifies `uint8 → /255 → (image-mean)/std`; do not mix older calibration assumptions.
These instructions are preserved source recipes and were not executed here.

## Runtime and outputs

The runtime is a **robot-control application**, not an offline image classifier.
On the prepared board, with trained model files and the correct robot/camera
configuration, the source entry is:

```bash
# S100, cwd samples/vla/act; source uses SO101
python bpu_control_robot.py --bpu-act-path /data/bpu_output
# S600, cwd samples/vla/pi0; source uses SO100Follower
python models/act/bpu_control_robot.py --bpu-act-path /data/bpu_output \
  --robot-port /dev/ttyACM0 --camera-index 0 --camera-name front \
  --fps 30 --inference-time 60
```

`/data/bpu_output` denotes the operator-prepared model directory. These commands
open robot/camera devices and send actions; they are not host smoke checks.
Hardware ports, calibration and robot type must match the selected source. S600
ACT returns 100-step chunks; the source warns against forcing one action step as
a workaround. The full source guides retain environment installation and all
runtime options; no conversion or actuator command was run by this migration.

## Historical measurements and boundaries

The S600 source records 20 warmups and 200 measured BPU calls: VisionEncoder
3.92 ms, TransformerLayers 2.29 ms, combined 6.20 ms (161.2 inferences/s).
That is BPU inference throughput, not camera/control-loop FPS or task success rate.
The S100 pin reports deployment verification without a published numeric benchmark.
Do not extend either claim to S100P/X5, different checkpoints or a different robot.

Code licensing remains in each submodule's LICENSE. Model/data rights and SDK
terms remain separate. See [integration checks](../README.md) for pin preservation.
