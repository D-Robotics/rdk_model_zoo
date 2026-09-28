# Pi0 on S600: source, deployment and offline entry

[中文](pi0_cn.md) · [Integration overview](../README.md)

## Exact source and scope

`samples/vla/pi0` pins `D-Robotics/rdk_LeRobot_tools` at
`a32de276bc1681a2b1531012de111eaa1c16acb6`. The model implementation starts at
`models/pi0/`; the same checkout also contains [S600 ACT](act.md).
From Model Zoo root:

```bash
git submodule update --init --checkout samples/vla/pi0
git -C samples/vla/pi0 rev-parse HEAD
```

Read the full [upstream guide](../pi0/models/pi0/README.md) after initialization.
The pinned source uses LeRobot v0.5.2, SO100Follower with six joint dimensions in
degrees, two real cameras and one masked empty image slot. Its standalone runtime
uses D-Robotics LLM S600 SDK 1.0.2 (`nash-p`) without a runtime `libxlm.so` dependency.
There is no corresponding Model Zoo downloadable bundle or validated X5/S100/S100P
entry. Supply the trained checkpoint, three matching HBMs, normalization stats,
robot calibration and images as described upstream.

## Model chain and source recipe

Front/side images and six-dimensional joint state flow through SigLIP (three
slots), PaliGemma (36 KV tensors) and the Action Expert (ten flow-matching steps).
Output is `[50,6]` absolute joint targets. PaliGemma KV belongs to a single image
pair/request and is reused only within its ten Expert denoise steps.

The source recipe is chain-aware: SigLIP HBM outputs calibrate PaliGemma, whose
real HBM KV outputs calibrate Expert. The complete training/conversion tools and
constraints remain in the upstream guide; they were not rerun in this migration.
Do not mix an individually rebuilt HBM into the fixed deployment bundle.

The default deployment file, relative to `models/pi0/`, is
`configs/deployments/pi0_full_v5_2cam_positionfp16_siglip_fixed16_paligemma_hbmkv_expert_20260801.json`.
Its companion manifest records file identities and sizes. Inspect and adapt local
resource locations as a separate deployment configuration; the source retains
original `/root` and `/home/sunrise` environment assumptions.

## Build and offline board inference

The following is the preserved S600 workflow, **not a host-only test**. Install the
SDK/dependencies documented upstream separately. From Model Zoo root:

```bash
cd samples/vla/pi0
export D_ROBOTICS_LLM_SDK_ROOT=/root/D-Robotics_LLM_S600_1.0.2_SDK/oellm_runtime
bash models/pi0/native/build_standalone_pi0.sh
cd models/pi0
/home/sunrise/lerobot/.venv/bin/python validate_pi0_config.py \
  configs/deployments/pi0_full_v5_2cam_positionfp16_siglip_fixed16_paligemma_hbmkv_expert_20260801.json
/home/sunrise/lerobot/.venv/bin/python pi0_standalone_offline.py \
  --front /data/front.jpg --side /data/side.jpg --state 0 0 0 0 0 0 \
  --output-dir /data/pi0_offline_result
```

Supply real image paths and a new output directory. This entry does not open robot
serial ports, but it launches the local native BPU engine and a TCP exchange.
It writes `actions.npy` (`[50,6]`), `result.json` (input paths/state/task, first
action, output shape and request latency) and `engine.log`. Request latency includes
the synchronous message exchange; it is not a pure BPU benchmark. Exit 0 means the
script completed; inspect the report and engine log for the run's actual result.

| Offline option | Default | Meaning |
| --- | --- | --- |
| `--front`, `--side` | Required | Real front/side image files |
| `--state` | Required, six floats | Joint positions in the source coordinate convention |
| `--output-dir` | Required, new directory | Action array, result and engine log |
| `--task` | `Place the RDK camera box on top of the black MCU box.` | Fixed natural-language task |
| `--config` | Deployment JSON named above | File/engine configuration |
| `--engine-runner` | `run_pi0_standalone_config.sh` beside the script | Engine launcher |
| `--fixed-noise-file` | `configs/fixed_noise_cv_12345678_fp16.bin` | Fixed inference noise |
| `--connect-timeout-s` | `120.0` | Engine connection timeout |

Build output is `models/pi0/native/install/bin/pi0_standalone_sdk102`.
`D_ROBOTICS_LLM_SDK_ROOT` selects SDK headers/libraries;
`PI0_STANDALONE_BIN` can select the engine binary. The config launcher validates
artifacts before running the engine. An incorrect SDK path, missing bundle or
mismatched hash must be fixed in preparation, not hidden by renaming artifacts.

## Live control and historical results

`models/pi0/run_live_sync.sh` is the source live-control entry. It opens the SO100
and cameras and calls `send_action()` at 30 Hz. It uses synchronous 50-step chunks
with `prefetch_steps=0` and `--force-model-actions`; it does not add relative target
limiting or chunk blending. The offline command above must not be confused with
this hardware-control workflow. Follow the full upstream setup/calibration and
operator-supervision instructions before choosing live operation.

The source reports 64 synchronous hardware chunks and one fixed real-input BF16/HBM
comparison: MAE 0.4405°, RMSE 0.5903°, maximum error 1.6888°, relative L2 0.852%,
cosine similarity 0.999981858. This is chain-integrity evidence for one input,
not task-success rate. No board inference, live control, training or quantization
was executed in this migration. The upstream Apache-2.0 LICENSE and separate
model/data/SDK terms remain applicable.
