English | [简体中文](README_cn.md)

# MobileSAM Migration Evaluator

<a id="dataset"></a>
## Dataset

This evaluator compares one fixed `test_data/dogs.jpg` image through the source legacy entrypoint and the unified runtime on the same target. The default box is `(185,120,380,445)` in resized `512x512` coordinates. It measures fixed-image migration consistency, not dataset accuracy or performance.

<a id="environment"></a>
## Environment

Run from the repository root with Python 3.10+ on the target board and its matching runtime installed. The evaluator does not download models. The selected pair must already exist at the manifest-derived paths, or both custom stage paths and exact manifest asset IDs must be supplied. Board SDK/system versions are unknown; board execution is not-run in this migration.

<a id="command"></a>
## Command

The output directory must be new; an existing directory is rejected. The command gates the requested target before executing legacy and unified stages with the same image, pair, box, priority and scheduling settings:

```bash
# cwd: repository root; prerequisite: target runtime and prepared pair
python3 samples/vision/mobile_sam/evaluator/compare.py \
  --target x5 \
  --output-dir /absolute/new-evidence-dir \
  --box 185,120,380,445
```

Use `--target s100|s100p|s600` for S targets. Optional `--test-img`, stage model paths and matching asset IDs, `--priority` (default `0`) and `--bpu-cores` are passed to both sides. Omit `--bpu-cores` to use S core `[0]`; X5 has no explicit core selection. Exit `0` means every comparison check passed, `1` means the runs completed but a check failed, and `2` means target gating, argument, model, image or execution failed.

Scheduling control: the evaluator first calls the fixed source helper's own `set_scheduling_params` unchanged and records every native scheduling call with its raw outcome. Because the fixed X5 source helper passes a scalar priority that the native API rejects (the source swallows that TypeError and applies nothing), the evaluator additionally applies an explicit verified per-model control mapping to both sides' native runtimes so the comparison executes under identical, actually-applied scheduling. `comparison.json` records each stage's model name, the exact native arguments, and both rejected and applied calls. This explicit control is an evaluator provision, not the fixed source CLI's own scheduling behavior, and it does not change either side's pre/forward/post processing.

| Argument | Default | Meaning |
|---|---|---|
| `--target` | required | `x5`, `s100`, `s100p` or `s600`; gates the actual execution target |
| `--output-dir` | required | New absolute or relative evidence directory; existing directories are rejected |
| `--test-img` | `samples/vision/mobile_sam/test_data/dogs.jpg` | Fixed BGR fixture |
| `--box` | `185,120,380,445` | Ordered box in resized `512x512` coordinates |
| `--encoder-model-path`, `--decoder-model-path` | manifest-derived paths | Custom paths, each requiring its matching exact asset ID |
| `--encoder-asset-id`, `--decoder-asset-id` | `null` | Qualified manifest identities for custom paths |
| `--priority` | `0` | Integer scheduling priority in `0..255` |
| `--bpu-cores` | `null` | S-series core indexes; omitted means `[0]`, and X5 rejects explicit cores |

<a id="metrics"></a>
## Metrics

The comparison requires exact input shapes, dtypes and values; exact output shapes and dtypes with raw numeric absolute tolerance `1e-5` and relative tolerance `0`; exact boolean mask and selected mask index; IoU absolute tolerance `1e-6`; and low-resolution float32 mask tolerance `1e-5`. There is no tie or changed-mask exemption.

<a id="outputs"></a>
## Outputs

The new directory contains `comparison.json` and NumPy arrays for both sides: encoder/decoder inputs and raw outputs, selected boolean mask and low-resolution masks. The JSON records target, complete asset identities and observed hashes, image hash/shape/dtype, box, metadata, scheduling, source reference, command context, code hashes, per-check decisions and tolerances. Failed execution or comparison retains the evidence written so far.

<a id="reference-results"></a>
## Reference Results

The following values are historical measurements copied from the fixed source evaluator READMEs. They are context only, are not results from this unified tree, and do not establish current board support:

| Source target | Stage | Threads | Historical latency (ms) | Historical FPS |
|---|---|---:|---:|---:|
| X5 | encoder | 1 | 1402.542 | 0.712979 |
| X5 | encoder | 8 | 2091.229 | 3.772934 |
| X5 | decoder | 1 | 96.198 | 10.393262 |
| X5 | decoder | 8 | 171.752 | 45.783301 |
| S100 | encoder | 1 / 2 | 10.97 / 21.29 | 91.03 / 93.61 |
| S100 | decoder | 1 / 2 | 3.30 / 4.42 | 297.47 / 443.42 |
| S100P | encoder | 1 / 2 | 8.52 / 16.50 | 117.26 / 120.79 |
| S100P | decoder | 1 / 2 | 2.84 / 3.88 | 345.15 / 505.87 |
| S600 | encoder | 1 / 12 | 5.52 / 15.87 | 180.70 / 738.48 |
| S600 | decoder | 1 / 12 | 2.58 / 6.42 | 381.09 / 1772.04 |

A successful local or board invocation is the reference evidence for its selected target, pair and box; preserve the complete output directory for review.

### Preserved per-model performance procedure

`compare.py` checks migration consistency. The source samples separately expose encoder/decoder `hrt_model_exec perf` measurements; the procedure below preserves that capability and was not executed in this migration. The tool comes from the matching board development kit, not pip. Prepare the pair using the model README and set `TARGET` to the actual board. This shell does not gate hardware identity itself; verify identity as described in the runtime README before running it.

```bash
# Bash; cwd: repository root; run only on the matching prepared board
cd samples/vision/mobile_sam/evaluator
TARGET=s100
CORE_ARGS=()
case "$TARGET" in
  x5)
    ENCODER=../model/mobile_sam_image_encoder_norm_512x512_allint16.bin
    DECODER=../model/mobile_sam_decoder_512_box_default.bin
    THREADS=8 ;;
  s100|s100p|s600)
    case "$TARGET" in
      s100) MARCH=nash-e; SUFFIX=nashe; THREADS=2 ;;
      s100p) MARCH=nash-m; SUFFIX=nashm; THREADS=2 ;;
      s600) MARCH=nash-p; SUFFIX=nashp; THREADS=12; CORE_ARGS=(--core_id 1,2,3,4) ;;
    esac
    ENCODER=../model/$MARCH/mobile_sam_image_encoder_norm_512x512_$SUFFIX.hbm
    DECODER=../model/$MARCH/mobile_sam_decoder_512_$SUFFIX.hbm ;;
  *) exit 2 ;;
esac
for STAGE_MODEL in "$ENCODER" "$DECODER"; do
  hrt_model_exec perf --model_file "$STAGE_MODEL" --thread_num 1
  hrt_model_exec perf --model_file "$STAGE_MODEL" --thread_num "$THREADS" "${CORE_ARGS[@]}"
done
```

The X5 source used the tool's default 200 frames. The S source did not record a frame count; current tool defaults and SDK versions remain unverified. S100/S100P use 2 threads for throughput, S600 uses 12 with explicit `--core_id 1,2,3,4`. Do not copy this tool's core IDs into the Python runtime's `--bpu-cores` option. Preserve the complete commands, tool/system versions, board identity and output when retesting, not only the FPS summary.

### Historical measurement definitions

The S source tables also record the following model facts. Parameter counts and FLOPs describe the original FP32 models, not a new calculation or the compiled artifact size. FLOPs use `2×MACs`. Classes are `-` (class agnostic); CPU pre/post-processing latency was not reported.

| Stage | Input size | Params (M) | FLOPs (G) |
| --- | --- | ---: | ---: |
| encoder | RGB 512×512 | 6.07 | 20.78 |
| decoder | 256×32×32 embedding + 1×4 box | 4.06 | 0.94 |

The S source defines BPU task latency from submission to completion, including cache warmup; streaming measurements reuse preallocated buffers and exclude allocation/deallocation. Inputs are float32 tensors, not NV12. The stages execute sequentially, and end-to-end latency also includes CPU preprocessing and mask resizing. Per-stage FPS is not whole-sample throughput. These are historical conditions, not measured performance guarantees for the unified implementation.

<a id="boundaries"></a>
## Boundaries

`compare.py` does not download assets, test a full dataset, measure latency, certify accuracy, or prove a target that was not explicitly gated and executed. It compares the fixed source legacy implementation with the unified implementation using the same runtime call path.
