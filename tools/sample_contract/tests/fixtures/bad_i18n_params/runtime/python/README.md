# Python runtime (bad_i18n_params fixture, en)

No `main.py` ships with this fixture on purpose: the CLI-defaults check must
report itself as skipped, not as passed, while the bilingual comparison runs.

<a id="environment"></a>
## Environment

Board image with `hbm_runtime`.

<a id="usage"></a>
## Usage

Run from the repository root (cwd).

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | choice | auto | execution target |
| `--top-k` | int | 5 | number of results |

<a id="results"></a>
## Results

Prints the top-5 list.

<a id="integration-example"></a>
## Integration example

```python
image = load_bgr("test_data/input.jpg")
result = task.predict(image)
```

<a id="stage-io"></a>
## Stage I/O

`pre_process` → tensors; `forward` → raw outputs; `post_process` → top-5.

<a id="troubleshooting"></a>
## Troubleshooting

- Missing model file: run `model/download.sh` first.
