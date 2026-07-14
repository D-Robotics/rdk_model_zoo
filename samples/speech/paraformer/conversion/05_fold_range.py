"""
Step 05: Fold all Range ops to Constants.

hbdk4 rejects the ONNX Range operator ("Operator Range should be optimized").
We use ORT to probe each Range's runtime output on a dummy input and replace
the Range node with a Constant node.

Usage:
    python 05_fold_range.py <input.onnx> <output.onnx>
"""
import sys, argparse
from collections import deque
import onnx, numpy as np, onnx.numpy_helper as nh
import onnxruntime as ort
from onnx import helper, TensorProto


def main(src, dst):
    m = onnx.load(src)
    range_nodes = [n for n in m.graph.node if n.op_type == "Range"]
    print(f"Range nodes: {len(range_nodes)}")
    if not range_nodes:
        onnx.save(m, dst, save_as_external_data=False)
        return

    mi = onnx.shape_inference.infer_shapes(m, check_type=False, strict_mode=False)
    vi_map = {vi.name: vi for vi in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output)}

    # Build probe: add Range outputs as graph outputs with inferred dtype
    m_probe = onnx.load(src)
    existing = {o.name for o in m_probe.graph.output}
    for n in range_nodes:
        out = n.output[0]
        if out in existing:
            continue
        vi = vi_map.get(out)
        if vi and vi.type.tensor_type.elem_type != TensorProto.UNDEFINED:
            dtype = vi.type.tensor_type.elem_type
        else:
            start_vi = vi_map.get(n.input[0])
            dtype = start_vi.type.tensor_type.elem_type if start_vi else TensorProto.INT64
        m_probe.graph.output.append(helper.make_tensor_value_info(out, dtype, None))

    probe_path = "/tmp/dp_range_probe.onnx"
    onnx.save(m_probe, probe_path, save_as_external_data=False)
    sess = ort.InferenceSession(probe_path, providers=["CPUExecutionProvider"])

    # Build dummy feed based on graph inputs
    np.random.seed(0)
    feed = {}
    for gi in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in gi.shape]
        if gi.type == "tensor(float)":
            feed[gi.name] = np.random.randn(*shape).astype(np.float32)
        elif gi.type == "tensor(int32)":
            feed[gi.name] = np.full(shape, 100, dtype=np.int32)
        elif gi.type == "tensor(int64)":
            feed[gi.name] = np.array(1, dtype=np.int64) if not shape else np.full(shape, 1, dtype=np.int64)
        else:
            feed[gi.name] = np.zeros(shape, dtype=np.float32)
    out_names = [o.name for o in sess.get_outputs()]
    val = dict(zip(out_names, sess.run(out_names, feed)))

    folded_nodes = []
    for n in range_nodes:
        v = val[n.output[0]]
        print(f"  fold {n.name}: dtype={v.dtype} shape={v.shape}")
        tensor = nh.from_array(v.copy(), name=n.output[0] + "_val")
        folded_nodes.append(helper.make_node(
            "Constant", inputs=[], outputs=[n.output[0]],
            name="fold_" + n.name, value=tensor,
        ))

    new_nodes = [n for n in m.graph.node if n.op_type != "Range"]
    new_nodes = folded_nodes + new_nodes

    # Topo-sort
    init_names = {i.name for i in m.graph.initializer}
    input_names = {i.name for i in m.graph.input}
    producer = {o: i for i, n in enumerate(new_nodes) for o in n.output}
    deps = [set() for _ in new_nodes]
    for i, node in enumerate(new_nodes):
        for inp in node.input:
            if inp in init_names or inp in input_names or inp == "":
                continue
            j = producer.get(inp)
            if j is not None and j != i:
                deps[i].add(j)
    indeg = [len(d) for d in deps]
    reverse = [[] for _ in new_nodes]
    for i, d in enumerate(deps):
        for j in d: reverse[j].append(i)
    q = deque(i for i in range(len(new_nodes)) if indeg[i] == 0)
    order = []
    while q:
        i = q.popleft(); order.append(i)
        for k in reverse[i]:
            indeg[k] -= 1
            if indeg[k] == 0: q.append(k)
    sorted_nodes = [new_nodes[i] for i in order]

    new_graph = helper.make_graph(sorted_nodes, m.graph.name, list(m.graph.input),
                                  list(m.graph.output), initializer=list(m.graph.initializer))
    m_out = helper.make_model(new_graph, opset_imports=list(m.opset_import), ir_version=m.ir_version)
    onnx.save(m_out, dst, save_as_external_data=False)
    onnx.checker.check_model(m_out)
    print(f"Saved: {dst}  nodes={len(sorted_nodes)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src", nargs="?", default="./out/decoder_only_topo.onnx")
    p.add_argument("dst", nargs="?", default="./out/decoder_only_norange.onnx")
    args = p.parse_args()
    main(args.src, args.dst)
