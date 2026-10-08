# Python runtime (fixture)

<a id="environment"></a>
## Environment

Board image with `hbm_runtime`, NumPy, and OpenCV-Python; `hbm_runtime`
exists only on board images, and `main.py` imports it lazily.

<a id="usage"></a>
## Usage

Run from the repository root (cwd):

```bash
python3 samples/tools/fixture/good_sample/runtime/python/main.py --target x5
```

Success is exit code 0 and a printed top-5 list.

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | choice | auto | execution target |
| `--top-k` / `--topk` | int | 5 | number of results |
| `--priority` | int | 0 | scheduling priority |
| `--bpu-cores` | int list | [0] | BPU core indexes |
| `--threshold` | float | null | optional score threshold |

<a id="results"></a>
## Results

`main.py` prints the top-5 `(label, score)` list; it writes files only when
`--img-save-path` style options are given (not in this fixture).

<a id="integration-example"></a>
## Integration example

```python
image = load_bgr("test_data/input.jpg")  # defined below
task = FixtureTask(selection)
result = task.predict(image)

def load_bgr(path):
    with open(path, "rb") as handle:
        return handle.read()
```

Every input variable is defined inside the example.

<a id="stage-io"></a>
## Stage I/O

`pre_process` takes one BGR `uint8` image and returns resized tensors;
`forward` feeds the bound tensor contract and returns raw outputs;
`post_process` returns the top-5 `(label, score)` list.

<a id="troubleshooting"></a>
## Troubleshooting

- Missing model file: run `model/download.sh --target <target>` first.
- Unknown board: the runtime reports the unresolved SoC instead of falling
  back to another target.
