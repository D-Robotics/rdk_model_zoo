"""
Step 02: Extract the decoder-only subgraph from the FunASR-exported model.onnx.

The result is a 5-input ONNX where the CIF is fully excluded (its outputs become
graph inputs — `pre_acoustic_embeds` and `token_num`). This is because the CIF's
GatherND with dynamic index cannot be compiled by hbdk4.

Boundary tensors that are constant (all-1 masks, etc.) get folded to Constant
nodes at extraction time.

Usage:
    python 02_extract_decoder.py <input_model.onnx> <output_decoder_only.onnx>

Downstream:
    03_convert_gather_int64_to_int32.py → 04_topsort.py → 05_fold_range.py → 06_shape_freeze.py
"""
import sys, os, argparse
sys.setrecursionlimit(200000)
import numpy as np
import onnx, onnx.numpy_helper as nh, onnxruntime as ort
from onnx import helper, TensorProto
from onnx.tools import update_model_dims

# Runtime-varying inputs kept as graph inputs.
# `onnx::Shape_8609` is the CIF's output frame_fires (=pre_acoustic_embeds) tensor
# name in this specific export. It may differ across FunASR versions — verify with
# `netron model.onnx` if the extraction fails.
KEEP_INPUTS = {
    "/encoder/after_norm/Add_1_output_0",   # encoder output (K/V for cross-attn)
    "bias_embed",                           # Contextual hotword embedding
    "onnx::Shape_8609",                     # pre_acoustic_embeds (CIF output)
    "token_num",                            # predicted token count
    "/predictor/Gather_output_0",           # batch dim (=1); folded later
}
KEEP_OUTPUTS = ["logits", "token_num"]


def main(src, dst):
    stage1 = "/tmp/dp_stage1.onnx"
    stage2 = "/tmp/dp_stage2_probe.onnx"

    print("=== Stage 1: load + fix shapes + inline If ===")
    m = onnx.load(src)
    m = update_model_dims.update_inputs_outputs_dims(
        m,
        input_dims={"speech": [1, 400, 560], "speech_lengths": [1], "bias_embed": [1, 1, 512]},
        output_dims={"logits": [1, 100, 8404], "token_num": [1]},
    )
    # Inline any top-level If (the patched CIF removes them, so usually skipped)
    if_nodes = [n for n in m.graph.node if n.op_type == "If"]
    if if_nodes:
        if_n = if_nodes[0]
        else_branch = next(a.g for a in if_n.attribute if a.name == "else_branch")
        else_out = else_branch.output[0].name
        for init in else_branch.initializer:
            m.graph.initializer.append(init)
        new_nodes = [n for n in m.graph.node if n.name != if_n.name]
        new_nodes.extend(list(else_branch.node))
        for n in new_nodes:
            for i, inp in enumerate(n.input):
                if inp == if_n.output[0]:
                    n.input[i] = else_out
        del m.graph.node[:]
        m.graph.node.extend(new_nodes)
        print(f"  inlined If, {len(m.graph.node)} nodes")
    else:
        print("  no If nodes (patched CIF removed the data-dependent branch)")
    onnx.save(m, stage1, save_as_external_data=False)

    print("\n=== Stage 2: reachability from decoder outputs ===")
    producer = {o: n for n in m.graph.node for o in n.output}
    init_names = {i.name for i in m.graph.initializer}

    # Cut at KEEP_INPUTS and at any non-/decoder producer
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
        is_decoder = p.name.startswith("/decoder")
        is_tail = p.op_type == "LogSoftmax" or p.name in ("/LogSoftmax", "/Floor")
        if not (is_decoder or is_tail):
            boundary.add(t); continue
        keep_names.add(p.name)
        for i in p.input:
            queue.append(i)
    print(f"  kept {len(keep_names)} nodes, {len(boundary)} boundary tensors")

    inputs_to_keep = boundary & KEEP_INPUTS
    tensors_to_fold = boundary - inputs_to_keep
    print(f"  keep as input: {sorted(inputs_to_keep)}")
    print(f"  fold to const: {sorted(tensors_to_fold)}")

    print("\n=== Stage 3: probe FP32 model to capture boundary values ===")
    m_probe = onnx.load(stage1)
    try:
        mi = onnx.shape_inference.infer_shapes(m_probe, check_type=False, strict_mode=False)
        vi_map = {vi.name: vi for vi in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output)}
    except Exception:
        vi_map = {}
    existing = {o.name for o in m_probe.graph.output}
    for t in tensors_to_fold:
        if t in existing:
            continue
        vi = vi_map.get(t)
        dt = vi.type.tensor_type.elem_type if (vi and vi.type.tensor_type.elem_type) else TensorProto.FLOAT
        m_probe.graph.output.append(helper.make_tensor_value_info(t, dt, None))
    onnx.save(m_probe, stage2, save_as_external_data=False)
    sess = ort.InferenceSession(stage2, providers=["CPUExecutionProvider"])
    speech = np.zeros((1, 400, 560), dtype=np.float32)
    speech_lengths = np.array([400], dtype=np.int32)
    bias_embed = np.zeros((1, 1, 512), dtype=np.float32)
    feed = {"speech": speech, "speech_lengths": speech_lengths, "bias_embed": bias_embed}
    out_names = [o.name for o in sess.get_outputs()]
    val = dict(zip(out_names, sess.run(out_names, feed)))

    print("\n=== Stage 4: build reduced graph ===")
    np_to_tp = {
        np.dtype("float32"): TensorProto.FLOAT,
        np.dtype("float16"): TensorProto.FLOAT16,
        np.dtype("int64"): TensorProto.INT64,
        np.dtype("int32"): TensorProto.INT32,
        np.dtype("int8"): TensorProto.INT8,
        np.dtype("bool"): TensorProto.BOOL,
    }
    const_nodes = []
    for t in tensors_to_fold:
        arr = val[t]
        tensor = nh.from_array(arr, name=t + "_val")
        const_nodes.append(helper.make_node(
            "Constant", [], [t],
            name=f"const_folded_{t.replace('/','_').replace(':','_')}",
            value=tensor,
        ))
        print(f"  fold {t}: shape={arr.shape} dtype={arr.dtype}")

    kept_nodes = [n for n in m.graph.node if n.name in keep_names]
    final_nodes = const_nodes + kept_nodes

    INPUT_SHAPES = {
        "/encoder/after_norm/Add_1_output_0": ([1, 400, 512], TensorProto.FLOAT),
        "bias_embed":                         ([1, 1, 512],   TensorProto.FLOAT),
        "onnx::Shape_8609":                   ([1, 100, 512], TensorProto.FLOAT),
        "token_num":                          ([1],           TensorProto.INT32),
        "/predictor/Gather_output_0":         ([],            TensorProto.INT64),
    }
    new_inputs = []
    for name in inputs_to_keep:
        orig = next((gi for gi in m.graph.input if gi.name == name), None)
        if orig:
            new_inputs.append(orig)
        elif name in INPUT_SHAPES:
            shp, tp = INPUT_SHAPES[name]
            new_inputs.append(helper.make_tensor_value_info(name, tp, shp))

    used = set()
    for n in final_nodes:
        for i in n.input: used.add(i)
    kept_inits = [i for i in m.graph.initializer if i.name in used]

    new_graph = helper.make_graph(final_nodes, "decoder_only", new_inputs, list(m.graph.output), initializer=kept_inits)
    m_out = helper.make_model(new_graph, opset_imports=list(m.opset_import), ir_version=m.ir_version)
    onnx.save(m_out, dst, save_as_external_data=False)
    print(f"\nSaved: {dst}")

    # Verify
    sess2 = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    np.random.seed(0)
    feed2 = {}
    for gi in sess2.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in gi.shape]
        if gi.type == "tensor(int32)":
            feed2[gi.name] = np.full(shape, 100, dtype=np.int32)
        elif gi.type == "tensor(int64)":
            feed2[gi.name] = np.full(shape, 1, dtype=np.int64) if shape else np.array(1, dtype=np.int64)
        else:
            feed2[gi.name] = np.random.randn(*shape).astype(np.float32)
    outs = sess2.run(None, feed2)
    for name, arr in zip([o.name for o in sess2.get_outputs()], outs):
        print(f"  verify output {name}: shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src", nargs="?", default="./models/paraformer/model.onnx")
    p.add_argument("dst", nargs="?", default="./out/decoder_only.onnx")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    main(args.src, args.dst)
