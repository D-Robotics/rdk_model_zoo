# Python runtime (bad_cli_drift fixture)

<a id="environment"></a>
## Environment

Board image with `hbm_runtime`; host tests need Python only.

<a id="usage"></a>
## Usage

Run from the repository root (cwd):

```bash
python3 samples/tools/fixture/bad_cli_drift/runtime/python/main.py --target x5
```

Success is exit code 0.

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | choice | auto | execution target |
| `--top-k` / `--topk` | int | 3 | number of results (drifted: parser says 5) |
| `--priority` | int | 0 | scheduling priority |
| `--bpu-cores` | int list | [0] | BPU core indexes |
| `--extra-opt` | int | 1 | documented but absent from the parser |

`--threshold` is intentionally missing from this table.

<a id="results"></a>
## Results

Prints the top-5 list.

<a id="integration-example"></a>
## Integration example

```python
image = load_bgr("test_data/input.jpg")
task = FixtureTask(selection)
result = task.predict(image)
```

All input variables are defined in the example scope.

<a id="stage-io"></a>
## Stage I/O

`pre_process` → tensors; `forward` → raw outputs; `post_process` → top-5.

<a id="troubleshooting"></a>
## Troubleshooting

- Missing model file: run `model/download.sh` first.
