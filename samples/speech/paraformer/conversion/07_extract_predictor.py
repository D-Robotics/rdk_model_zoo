"""
Step 07: Extract predictor-only subgraph from model.onnx.

The predictor is small (~12 nodes after onnxsim) with a single input
(encoder output) and 2 outputs: alphas + Concat_5_output_0.

Usage:
    python 07_extract_predictor.py <model.onnx> <predictor_only.onnx>
"""
import sys, os, argparse
sys.setrecursionlimit(200000)
import numpy as np, onnx, onnx.numpy_helper as nh, onnxruntime as ort
from onnx import helper, TensorProto, shape_inference
from onnx.tools import update_model_dims
import onnxsim


KEEP_INPUTS = {"/encoder/after_norm/Add_1_output_0"}
KEEP_OUTPUTS = [
    "/predictor/Add_output_0",       # alphas [1, 401]
    "/predictor/Concat_5_output_0",  # [1, 401, 512]
]


def main(src, dst):
    m = onnx.load(src)
    m = update_model_dims.update_inputs_outputs_dims(
        m,
        input_dims={"speech": [1, 400, 560], "speech_lengths": [1], "bias_embed": [1, 1, 512]},
        output_dims={"logits": [1, 100, 8404], "token_num": [1]},
    )

    producer = {o: n for n in m.graph.node for o in n.output}
    init_names = {i.name for i in m.graph.initializer}

    keep_names, boundary, seen = set(), set(), set()
    queue = list(KEEP_OUTPUTS)
    while queue:
        t = queue.pop()
        if t in seen or t in init_names or t == "":
            continue
        seen.add(t)
        if t in KEEP_INPUTS:
            boundary.add(t); continue
        p = producer.get(t)
        if p is None:
            boundary.add(t); continue
        if p.name.startswith("/encoder") or p.name.startswith("/make_pad_mask") or p.name.startswith("/decoder"):
            boundary.add(t); continue
        keep_names.add(p.name)
        for i in p.input:
            queue.append(i)

    print(f"kept nodes: {len(keep_names)}, boundary: {len(boundary)}")
    inputs_to_keep = boundary & KEEP_INPUTS
    tensors_to_fold = boundary - inputs_to_keep

    # Probe FP32 to capture folded values
    m_probe = onnx.load(src)
    m_probe = update_model_dims.update_inputs_outputs_dims(
        m_probe,
        input_dims={"speech": [1, 400, 560], "speech_lengths": [1], "bias_embed": [1, 1, 512]},
        output_dims={"logits": [1, 100, 8404], "token_num": [1]},
    )
    existing = {o.name for o in m_probe.graph.output}
    mi = shape_inference.infer_shapes(m_probe, check_type=False, strict_mode=False)
    vi_map = {vi.name: vi for vi in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output)}
    for t in list(tensors_to_fold) + KEEP_OUTPUTS:
        if t in existing: continue
        vi = vi_map.get(t)
        dt = vi.type.tensor_type.elem_type if (vi and vi.type.tensor_type.elem_type) else TensorProto.FLOAT
        m_probe.graph.output.append(helper.make_tensor_value_info(t, dt, None))
    onnx.save(m_probe, "/tmp/pred_probe.onnx", save_as_external_data=False)
    sess = ort.InferenceSession("/tmp/pred_probe.onnx", providers=["CPUExecutionProvider"])
    speech = np.zeros((1, 400, 560), dtype=np.float32)
    speech_lengths = np.array([400], dtype=np.int32)
    bias_embed = np.zeros((1, 1, 512), dtype=np.float32)
    out_names = [o.name for o in sess.get_outputs()]
    val = dict(zip(out_names, sess.run(out_names, {"speech": speech, "speech_lengths": speech_lengths, "bias_embed": bias_embed})))

    np_to_tp = {np.dtype("float32"): TensorProto.FLOAT, np.dtype("int64"): TensorProto.INT64,
                np.dtype("int32"): TensorProto.INT32, np.dtype("bool"): TensorProto.BOOL}
    const_nodes = []
    for t in tensors_to_fold:
        arr = val[t]
        const_nodes.append(helper.make_node(
            "Constant", [], [t],
            name=f"const_folded_{t.replace('/','_').replace(':','_')}",
            value=nh.from_array(arr, name=t + "_val"),
        ))
        print(f"  fold {t}: shape={arr.shape} dtype={arr.dtype}")

    kept_nodes = [n for n in m.graph.node if n.name in keep_names]
    final_nodes = const_nodes + kept_nodes

    new_inputs = []
    for name in inputs_to_keep:
        orig = next((gi for gi in m.graph.input if gi.name == name), None)
        if orig:
            new_inputs.append(orig)
        else:
            new_inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 400, 512]))

    new_outputs = []
    for name in KEEP_OUTPUTS:
        arr = val[name]
        new_outputs.append(helper.make_tensor_value_info(name, np_to_tp[arr.dtype], list(arr.shape)))

    used = set()
    for n in final_nodes:
        for i in n.input: used.add(i)
    kept_inits = [i for i in m.graph.initializer if i.name in used]

    new_graph = helper.make_graph(final_nodes, "predictor_only", new_inputs, new_outputs, initializer=kept_inits)
    m_out = helper.make_model(new_graph, opset_imports=list(m.opset_import), ir_version=m.ir_version)

    # onnxsim to collapse to minimal graph
    m_out, ok = onnxsim.simplify(
        m_out, overwrite_input_shapes={"/encoder/after_norm/Add_1_output_0": [1, 400, 512]},
        perform_optimization=True,
    )
    print(f"[onnxsim] ok={ok}  final nodes={len(m_out.graph.node)}")

    onnx.save(m_out, dst, save_as_external_data=False)
    onnx.checker.check_model(m_out)
    print(f"Saved: {dst}  nodes={len(m_out.graph.node)}")

    # Verify
    s = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    np.random.seed(0)
    enc = np.random.randn(1, 400, 512).astype(np.float32)
    outs = s.run(None, {"/encoder/after_norm/Add_1_output_0": enc})
    for name, arr in zip([o.name for o in s.get_outputs()], outs):
        print(f"  verify {name}: shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src", nargs="?", default="./models/paraformer/model.onnx")
    p.add_argument("dst", nargs="?", default="./out/predictor_only.onnx")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    main(args.src, args.dst)
