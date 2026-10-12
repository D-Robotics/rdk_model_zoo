"""
Fold audio2mouth (model_1.onnx) constant inputs into the graph.

audio2mouth_cpu.py feeds three inputs as CONSTANTS every call:
  - input_ph  [1,30,2]   = zeros
  - input_sp  [1]        = int64(2)
  - w         [] (scalar)= float32(1.0)
Only input_au [1,30,50,512] varies.

This script replaces those three graph inputs with Constant initializers,
then constant-folds + simplifies, producing a single-input ONNX
`model_1_folded.onnx` suitable for clean PTQ.

Usage:
  python scripts/fold_audio2mouth.py [--src weights/model_1.onnx] \
      [--dst model_1_folded.onnx]
"""
import os, sys, argparse
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto, shape_inference

try:
    import onnxsim
    HAVE_ONNXSIM = True
except Exception:
    HAVE_ONNXSIM = False


def fold(src, dst):
    m = onnx.load(src)
    g = m.graph

    input_b = {i.name for i in g.input}
    const_specs = {
        # name -> (np_array, TensorProto dtype)
        "input_ph": (np.zeros((1, 30, 2), dtype=np.float32), TensorProto.FLOAT),
        "input_sp": (np.array([2], dtype=np.int64), TensorProto.INT64),
        "w":        (np.array([1.0], dtype=np.float32), TensorProto.FLOAT),
    }

    # 1) replace graph inputs with Constant initializer nodes
    new_init_names = set(i.name for i in g.initializer)
    added_nodes = []
    for name, (arr, dt) in const_specs.items():
        if name not in input_b:
            print(f"  skip {name}: not a graph input")
            continue
        # remove from graph.input
        g.input.remove(next(i for i in g.input if i.name == name))
        # add as initializer (constant tensor) via standard helper.from_array
        if name not in new_init_names:
            g.initializer.append(numpy_helper.from_array(arr, name=name))
            new_init_names.add(name)
        print(f"  folded {name} -> constant {arr.shape} {arr.dtype}")

    # rebuild graph with added constant nodes
    g.node.extend(added_nodes)

    # 2) shape inference + constant folding via onnxsim if available, else onnx optimizer
    onnx.checker.check_model(m)
    if HAVE_ONNXSIM:
        print("onnxsim simplify (constant folding) ...")
        m2, ok = onnxsim.simplify(m, input_shapes={"input_au": [1, 30, 50, 512]},
                                  overwrite_input_shapes=True)
        if not ok:
            print("  onnxsim reported not-ok, keeping best-effort")
        m = m2
    else:
        print("onnxsim missing — running shape_inference only (no constant fold)")
        m = shape_inference.infer_shapes(m)

    # prune spurious onnxsim outputs — keep only the two audio2mouth actually
    # uses (audio2mouth_cpu.py:88 runs ['output','feat']). Stripping the
    # intermediate '1815'/'3080'-style leaves keeps hb_compile + cosine clean.
    # NOTE: operate on m.graph (onnxsim may have returned a fresh model, so the
    # `g` captured before simplify can be stale).
    keep = {"output", "feat"}
    pruned = [o for o in m.graph.output if o.name in keep]
    del m.graph.output[:]
    m.graph.output.extend(pruned)

    onnx.save(m, dst, save_as_external_data=False)
    onnx.checker.check_model(m)

    # 3) report
    print("\n=== folded ONNX ===")
    print("inputs:", [i.name for i in m.graph.input])
    print("outputs:", [o.name for o in m.graph.output])
    print("nodes:", len(m.graph.node))
    print(f"saved: {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="weights/model_1.onnx")
    p.add_argument("--dst", default="model_1_folded.onnx")
    a = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    fold(a.src, a.dst)
