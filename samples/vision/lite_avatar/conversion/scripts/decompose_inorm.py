"""
Decompose the 3 affine-free InstanceNormalization nodes in net_decode_deconv.onnx
into 7 plain named ops, so run_on_cpu can force the whole norm subgraph to FP32 CPU.

Why: hb_compile decomposes InstanceNormalization (a composite op) DURING its optimize
pass into mean1/sub1/mul1/mean2/reciprocal/mul2 (all quantized INT16 on BPU), and the
parent node's run_on_cpu flag does NOT survive the decomposition — so `run_on_cpu:
.../norm/InstanceNormalization` was silently a no-op (byte-identical node_info).

By decomposing FIRST in the ONNX, the sub-ops are real named nodes that run_on_cpu
matches (proven for plain Mul/Add/Gemm in the gamma/beta node config). `Pow(x,-0.5)`
replaces Reciprocal(Sqrt(x)) to avoid hb_compile's HzRsqrt fuse (which has no CPU
impl — the earlier HzSwish/bs_proj breakage).

ONNX InstanceNorm (epsilon=1e-5, scale=1, B=0) ==
    norm = (x - mean_hw(x)) / sqrt( mean_hw((x - mean_hw(x))^2) + eps )

Decompose:
    mean     = ReduceMean(x, [2,3], keepdims=1)
    centered = Sub(x, mean)
    sq       = Mul(centered, centered)
    var      = ReduceMean(sq, [2,3], keepdims=1)
    var_eps  = Add(var, eps)          # eps = 1e-5
    inv_std  = Pow(var_eps, -0.5)     # = 1/sqrt, no HzRsqrt fuse
    out      = Mul(centered, inv_std) # reuses the original output name

Usage:
  python scripts/decompose_inorm.py --src conversion/net_decode_deconv.onnx \
      --dst conversion/net_decode_deconv_inorm.onnx [--verify]
"""
import argparse, sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def eps_init(name):
    return numpy_helper.from_array(np.array([1e-5], dtype=np.float32), name=name)


def neg_half_init(name):
    return numpy_helper.from_array(np.array([-0.5], dtype=np.float32), name=name)


def axes_init(name):
    return numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name=name)


def make_decompose(graph, node):
    base = node.name                                # /dec/decoder/{i}/AAD2/norm/InstanceNormalization
    x = node.input[0]
    out = node.output[0]
    prefix = "/dec/" + base.split("/dec/", 1)[1].rsplit("/", 1)[0]  # strip trailing node leaf

    eps = prefix + "/_eps"
    neg_half = prefix + "/_neg_half"
    ax = prefix + "/_axes"
    graph.initializer.append(eps_init(eps))
    graph.initializer.append(neg_half_init(neg_half))
    graph.initializer.append(axes_init(ax))

    mean_n = prefix + "/norm_mean"
    sub_n  = prefix + "/norm_centered"
    sq_n   = prefix + "/norm_sq"
    var_n  = prefix + "/norm_var"
    ve_n   = prefix + "/norm_vareps"
    inv_n  = prefix + "/norm_invstd"
    apply_n = prefix + "/norm_apply"

    nodes = [
        helper.make_node("ReduceMean", [x], [mean_n], axes=[2, 3], keepdims=1,
                         name=prefix + "/norm_mean"),
        helper.make_node("Sub", [x, mean_n], [sub_n], name=prefix + "/norm_centered"),
        helper.make_node("Mul", [sub_n, sub_n], [sq_n], name=prefix + "/norm_sq"),
        helper.make_node("ReduceMean", [sq_n], [var_n], axes=[2, 3], keepdims=1,
                         name=prefix + "/norm_var"),
        helper.make_node("Add", [var_n, eps], [ve_n], name=prefix + "/norm_vareps"),
        helper.make_node("Pow", [ve_n, neg_half], [inv_n], name=prefix + "/norm_invstd"),
        helper.make_node("Mul", [sub_n, inv_n], [out], name=prefix + "/norm_apply"),
    ]

    idx = list(graph.node).index(node)
    for i, nd in enumerate(nodes):
        graph.node.insert(idx + i, nd)
    graph.node.remove(node)

    # drop now-orphaned Constant nodes that produced this InstanceNorm's scale(1)/B(0)
    for inp in (node.input[1], node.input[2]):
        for cn in list(graph.node):
            if cn.op_type == "Constant" and inp in cn.output:
                graph.node.remove(cn)


def find_nodes(model):
    return [n for n in model.graph.node if n.op_type == "InstanceNormalization"]


def verify(src, dst):
    import onnxruntime as ort
    import glob as _g
    a = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    fa = [i.name for i in a.get_inputs()]
    fb = [i.name for i in b.get_inputs()]
    assert fa == fb, f"input mismatch {fa} vs {fb}"
    calib = "calib_data/facegen_decode"
    sample = {}
    for nm in fa:
        files = sorted(_g.glob(f"{calib}/{nm}/*.npy"))
        sample[nm] = np.load(files[0]).astype(np.float32)
    ya = a.run(None, sample)[0]
    yb = b.run(None, sample)[0]
    diff = np.abs(ya - yb)
    cos = float(np.sum(ya * yb) / (np.linalg.norm(ya) * np.linalg.norm(yb) + 1e-12))
    print(f"verify: max|diff|={diff.max():.3e}  mean|diff|={diff.mean():.3e}  cosine={cos:.9f}")
    assert cos > 0.999999, "decompose changed the output — check op ordering"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="conversion/net_decode_deconv.onnx")
    ap.add_argument("--dst", default="conversion/net_decode_deconv_inorm.onnx")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    model = onnx.load(a.src)
    nodes = find_nodes(model)
    print(f"found {len(nodes)} InstanceNormalization nodes")
    assert len(nodes) == 3, f"expected 3, got {len(nodes)}"
    for n in nodes:
        print(f"  {n.name}")
        make_decompose(model.graph, n)

    onnx.checker.check_model(model)
    onnx.save(model, a.dst)
    print(f"saved -> {a.dst}")

    if a.verify:
        verify(a.src, a.dst)


if __name__ == "__main__":
    main()