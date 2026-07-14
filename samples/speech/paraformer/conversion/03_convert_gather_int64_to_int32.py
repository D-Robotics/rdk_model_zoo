"""
Step 03: Convert all Gather nodes' INT64 index inputs to INT32.

HMCT's `adjust_multi_output_use_quant_info_pass` asserts if any Gather has an
INT64 index. Converting these constants/initializers to INT32 fixes it.

Usage:
    python 03_convert_gather_int64_to_int32.py <input.onnx> <output.onnx>
"""
import sys, argparse
import onnx
import onnx.numpy_helper as nh
import numpy as np
from onnx import helper, TensorProto


def main(src, dst):
    m = onnx.load(src)
    producer = {o: n for n in m.graph.node for o in n.output}
    init_map = {i.name: i for i in m.graph.initializer}

    try:
        mi = onnx.shape_inference.infer_shapes(m, check_type=False, strict_mode=False)
        vi_map = {vi.name: vi for vi in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output)}
    except Exception:
        vi_map = {}

    def get_dtype(name):
        if name in init_map:
            return init_map[name].data_type
        vi = vi_map.get(name)
        if vi:
            return vi.type.tensor_type.elem_type
        p = producer.get(name)
        if p and p.op_type == "Constant":
            for a in p.attribute:
                if a.name == "value" and a.t.data_type:
                    return a.t.data_type
        return None

    gathers = [n for n in m.graph.node if n.op_type == "Gather"]
    print(f"Total Gather nodes: {len(gathers)}")

    int64_indices = [(g, g.input[1]) for g in gathers if len(g.input) >= 2 and get_dtype(g.input[1]) == TensorProto.INT64]
    print(f"Gathers with INT64 index: {len(int64_indices)}")

    by_idx = {}
    for g, idx in int64_indices:
        by_idx.setdefault(idx, []).append(g)

    const_conv = init_conv = cast_added = 0
    new_nodes = []
    for idx_name, gs in by_idx.items():
        if idx_name in init_map:
            arr = nh.to_array(init_map[idx_name]).astype(np.int32)
            init_map[idx_name].CopyFrom(nh.from_array(arr, name=idx_name))
            init_conv += 1; continue
        p = producer.get(idx_name)
        if p and p.op_type == "Constant":
            for a in p.attribute:
                if a.name == "value" and a.t.data_type == TensorProto.INT64:
                    arr = nh.to_array(a.t).astype(np.int32)
                    a.t.CopyFrom(nh.from_array(arr, name=a.t.name if a.t.name else ""))
                    const_conv += 1; break
            continue
        # Dynamic index: insert Cast
        cast_out = f"{idx_name}__cast_int32"
        new_nodes.append(helper.make_node(
            "Cast", [idx_name], [cast_out],
            name=f"cast_{idx_name.replace('/','_').replace(':','_')}_to_int32",
            to=TensorProto.INT32,
        ))
        for g in gs:
            for i, inp in enumerate(g.input):
                if i == 1 and inp == idx_name:
                    g.input[i] = cast_out
        cast_added += 1

    print(f"  const INT64→INT32: {const_conv}")
    print(f"  init  INT64→INT32: {init_conv}")
    print(f"  cast nodes inserted: {cast_added}")

    for n in new_nodes:
        m.graph.node.append(n)
    del m.graph.initializer[:]
    m.graph.initializer.extend(init_map.values())

    onnx.save(m, dst, save_as_external_data=False)
    print(f"Saved: {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src", nargs="?", default="./out/decoder_only.onnx")
    p.add_argument("dst", nargs="?", default="./out/decoder_only_int32gather.onnx")
    args = p.parse_args()
    main(args.src, args.dst)
