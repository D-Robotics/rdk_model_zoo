"""
Step 06: Aggressive shape freeze via onnxsim, then normalize axes + fold Gather.

Combines:
  a) onnxsim.simplify with fixed input shapes → propagates constants,
     eliminates dynamic Shape/Gather/Concat/Reshape chains
  b) Normalize negative axis to positive (Split axis=-1 crashes hbdk4 hbir slice)
  c) Fold `/predictor/Gather_output_0` graph input to Constant(1) —
     this is the batch dim scalar, always 1; folding it removes a dynamic
     Tile(repeats) that hbdk4 rejects
  d) Re-run onnxsim to propagate the newly-constant paths

Usage:
    python 06_shape_freeze.py <input.onnx> <output.onnx>
"""
import sys, argparse
sys.setrecursionlimit(200000)
import onnx, numpy as np, onnx.numpy_helper as nh
import onnxsim
from onnx import helper, TensorProto, shape_inference
from collections import deque
import onnxruntime as ort


INPUT_SHAPES = {
    "/encoder/after_norm/Add_1_output_0": [1, 400, 512],
    "bias_embed":                         [1, 1, 512],
    "onnx::Shape_8609":                   [1, 100, 512],
    "token_num":                          [1],
    "/predictor/Gather_output_0":         [],
}

OPS_WITH_AXIS_ATTR = {
    "Split", "Softmax", "LogSoftmax", "Concat", "Gather",
    "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin",
    "Squeeze", "Flatten",
}


def get_rank(vi_map, tensor_name):
    vi = vi_map.get(tensor_name)
    if vi is None:
        return None
    return len(vi.type.tensor_type.shape.dim)


def normalize_axes(m):
    mi = shape_inference.infer_shapes(m, check_type=False, strict_mode=False)
    vi_map = {vi.name: vi for vi in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output)}
    init_map = {i.name: i for i in m.graph.initializer}

    cnt = 0
    # Attribute-based axis
    for n in m.graph.node:
        if n.op_type not in OPS_WITH_AXIS_ATTR or not n.input:
            continue
        r = get_rank(vi_map, n.input[0])
        if r is None:
            continue
        for a in n.attribute:
            if a.name == "axis" and a.i < 0:
                a.i += r; cnt += 1
            elif a.name == "axes" and any(x < 0 for x in a.ints):
                new_axes = [x + r if x < 0 else x for x in a.ints]
                del a.ints[:]; a.ints.extend(new_axes); cnt += 1

    # Opset-13+ axes as tensor input (Squeeze / ReduceX only)
    for n in m.graph.node:
        if n.op_type not in ("Squeeze", "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin"):
            continue
        if len(n.input) < 2 or not n.input[1]:
            continue
        r = get_rank(vi_map, n.input[0])
        if r is None:
            continue
        tensor = init_map.get(n.input[1])
        if tensor:
            arr = nh.to_array(tensor)
            if arr.dtype == np.int64 and any(int(x) < 0 for x in arr):
                new_arr = np.array([int(x) + r if int(x) < 0 else int(x) for x in arr], dtype=np.int64)
                tensor.CopyFrom(nh.from_array(new_arr, name=tensor.name))
                cnt += 1
    print(f"[axis_norm] normalized {cnt} axis references")
    return m


def fold_gather_output(m):
    NAME = "/predictor/Gather_output_0"
    new_inputs = [i for i in m.graph.input if i.name != NAME]
    if len(new_inputs) == len(m.graph.input):
        return m  # not present
    del m.graph.input[:]
    m.graph.input.extend(new_inputs)

    const_tensor = nh.from_array(np.array(1, dtype=np.int64), name=NAME + "_val")
    const_node = helper.make_node(
        "Constant", inputs=[], outputs=[NAME],
        name="const_predictor_Gather_output_0", value=const_tensor,
    )
    m.graph.node.insert(0, const_node)

    # Topo-sort
    producer = {}
    for i, n in enumerate(m.graph.node):
        for o in n.output:
            producer[o] = i
    init_names = {i.name for i in m.graph.initializer}
    graph_inputs = {gi.name for gi in m.graph.input}
    nodes = list(m.graph.node)
    n_count = len(nodes)
    deps = [set() for _ in range(n_count)]
    for i, node in enumerate(nodes):
        for inp in node.input:
            if inp in init_names or inp in graph_inputs or inp == "":
                continue
            j = producer.get(inp)
            if j is not None and j != i:
                deps[i].add(j)
    indeg = [len(d) for d in deps]
    reverse = [[] for _ in range(n_count)]
    for i, d in enumerate(deps):
        for j in d: reverse[j].append(i)
    q = deque(i for i in range(n_count) if indeg[i] == 0)
    order = []
    while q:
        i = q.popleft(); order.append(i)
        for k in reverse[i]:
            indeg[k] -= 1
            if indeg[k] == 0: q.append(k)
    sorted_nodes = [nodes[i] for i in order]

    new_graph = helper.make_graph(sorted_nodes, m.graph.name, list(m.graph.input),
                                  list(m.graph.output), initializer=list(m.graph.initializer))
    m_new = helper.make_model(new_graph, opset_imports=list(m.opset_import), ir_version=m.ir_version)
    print(f"[fold_gather] folded {NAME} → Constant(1)")
    return m_new


def main(src, dst):
    m = onnx.load(src)
    print(f"input: {src}  nodes={len(m.graph.node)}")

    # a) onnxsim pass 1
    valid_shapes = {k: v for k, v in INPUT_SHAPES.items() if k in {i.name for i in m.graph.input}}
    m, ok = onnxsim.simplify(m, overwrite_input_shapes=valid_shapes, perform_optimization=True)
    print(f"[onnxsim#1] ok={ok}  nodes={len(m.graph.node)}")

    # b) axis normalize (skip Unsqueeze — output-rank semantics)
    m = normalize_axes(m)

    # c) fold /predictor/Gather_output_0 to Constant
    m = fold_gather_output(m)

    # d) onnxsim pass 2 (propagate constants from folded Gather)
    valid_shapes = {k: v for k, v in INPUT_SHAPES.items() if k in {i.name for i in m.graph.input}}
    m, ok = onnxsim.simplify(m, overwrite_input_shapes=valid_shapes, perform_optimization=True)
    print(f"[onnxsim#2] ok={ok}  nodes={len(m.graph.node)}")

    onnx.save(m, dst, save_as_external_data=False)
    onnx.checker.check_model(m)
    print(f"\nSaved: {dst}  nodes={len(m.graph.node)}")

    # Runtime sanity check
    sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    print(f"final inputs: {[gi.name for gi in sess.get_inputs()]}")
    np.random.seed(0)
    feed = {}
    for gi in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in gi.shape]
        if gi.type == "tensor(int32)":
            feed[gi.name] = np.full(shape, 100, dtype=np.int32)
        elif gi.type == "tensor(int64)":
            feed[gi.name] = np.array(1, dtype=np.int64) if not shape else np.full(shape, 1, dtype=np.int64)
        else:
            feed[gi.name] = np.random.randn(*shape).astype(np.float32)
    outs = sess.run(None, feed)
    for name, arr in zip([o.name for o in sess.get_outputs()], outs):
        print(f"  verify {name}: shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src", nargs="?", default="./out/decoder_only_norange.onnx")
    p.add_argument("dst", nargs="?", default="./out/decoder_only_final.onnx")
    args = p.parse_args()
    main(args.src, args.dst)
