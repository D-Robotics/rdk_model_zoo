"""Reproduce archived source defects and execute both customer API examples."""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import re
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h, numpy_helper as nh, TensorProto as T

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "platforms/s/samples/speech/paraformer/conversion"


def load(name):
    path = SOURCE / name
    spec = importlib.util.spec_from_file_location("archived_graph_step", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(path.read_bytes()).hexdigest()


def model(nodes, inputs, outputs, initializers=()):
    return h.make_model(h.make_graph(nodes, "source-regression", inputs, outputs,
                                   initializer=list(initializers)),
                        opset_imports=[h.make_opsetid("", 15)], ir_version=10)


results = {}
with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    source, digest = load("03_convert_gather_int64_to_int32.py")
    graph = model(
        [h.make_node("Gather", ["x", "i"], ["g"]),
         h.make_node("Add", ["i", "offset"], ["other"])],
        [h.make_tensor_value_info("x", T.FLOAT, [2])],
        [h.make_tensor_value_info("g", T.FLOAT, [1]),
         h.make_tensor_value_info("other", T.INT64, [1])],
        [nh.from_array(np.array([1], np.int64), "i"),
         nh.from_array(np.array([2], np.int64), "offset")])
    src, dst = directory / "input.onnx", directory / "output.onnx"
    onnx.save(graph, src)
    source.main(str(src), str(dst))
    try:
        ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    except Exception as exc:
        assert "Type" in str(exc) and "Add" in str(exc), str(exc)
        results["source_shared_gather"] = {"source_sha256": digest, "reproduced": True,
                                           "error": str(exc)}
    else:
        raise AssertionError("Expected source shared-constant type defect")
    source, digest = load("04_topsort.py")
    graph = model([h.make_node("Identity", ["b"], ["a"]),
                   h.make_node("Identity", ["a"], ["b"])], [],
                  [h.make_tensor_value_info("a", T.FLOAT, [1])])
    onnx.save(graph, src)
    dst.unlink()
    try:
        source.main(str(src), str(dst))
    except onnx.checker.ValidationError as exc:
        assert dst.is_file() and len(onnx.load(dst).graph.node) == 0
        results["source_cycle_partial_save"] = {"source_sha256": digest,
            "reproduced": True, "saved_node_count": 0, "error": str(exc)}
    else:
        raise AssertionError("Expected source partial graph checker failure")

for language in ("README.md", "README_cn.md"):
    path = ROOT / "samples/speech/paraformer/conversion" / language
    text = path.read_text()
    for target in re.findall(r"\]\(([^)]+)\)", text):
        assert (path.parent / target).exists(), (path, target)
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert len(blocks) == 1
    exec(compile(blocks[0], str(path), "exec"), {})
    results[language] = {"api_blocks_executed": 1, "local_links_valid": True}
results["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                           "onnx": onnx.__version__, "onnxruntime": ort.__version__}
results["scope"] = "Synthetic host graph checks only; real weights/OE/SDK/board not-run"
(OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
