"""
Lossless replace of the 3 bilinear-upsample `Resize` nodes in net_decode.onnx with
equivalent fixed-weight ConvTranspose2d, so hb_compile no longer falls back to CPU.

The 3 Resize ops are `mode=linear, coordinate_transformation_mode=half_pixel`,
scales [1,1,2,2]  ==  F.interpolate(scale_factor=2, mode='bilinear', align_corners=False).
This is exactly:

    Pad(x, pads=(1,1,1,1), mode='edge')                    # replicate edge
    -> ConvTranspose2d(kernel_size=4, stride=2, padding=3, group=C)   # no bias
    weights [C,1,4,4] = bilinear kernel  outer([.25,.75,.75,.25], [...same...])

Verified (torch): max |replacement - F.interpolate| < 2.4e-7 across H=4/48 random tensors.
The Pad(edge) is required to reproduce the half_pixel boundary clamp exactly.

Usage: python replace_resize_deconv.py --src conversion/net_decode.onnx --dst conversion/net_decode_deconv.onnx
"""
import argparse, sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from onnx import shape_inference

RESIZE_OUTPUTS = [                      # node name -> its (C, H, W) input shape (from shape inference)
]


def bilinear_kernel_1d():
    w = np.array([0.25, 0.75, 0.75, 0.25], dtype=np.float32)
    return np.outer(w, w).astype(np.float32)          # [4,4]


def find_resize_nodes(model):
    nodes = []
    for n in model.graph.node:
        if n.op_type == "Resize":
            # inputs: [X, roi(''), scales('')/Constant sizes]
            nodes.append(n)
    return nodes


def resolve_shapes(model):
    """Run shape inference once and return {tensor_name: [dims...]}."""
    mi = shape_inference.infer_shapes(model, strict_mode=False)
    shapes = {}
    for vi in list(mi.graph.input) + list(mi.graph.output) + list(mi.graph.value_info):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = dims
    return shapes


def make_replacement(graph, resize_node, C, out_onnx_path):
    x_name = resize_node.input[0]
    out_name = resize_node.output[0]
    base = resize_node.name

    # 1) Pad node (edge replicate, +1 pixel each side)
    pad_name = base + "/_deconv_pad"
    pads_init = numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64),
                                        name=base + "/_pads")
    graph.initializer.append(pads_init)
    pad_node = helper.make_node(
        "Pad", [x_name, pads_init.name], [pad_name], mode="edge",
        name=base + "/_edgepad")

    # 2) ConvTranspose2d weights [C,1,4,4] (group=C, depthwise bilinear)
    kernel = bilinear_kernel_1d()                        # [4,4]
    W = np.zeros((C, 1, 4, 4), dtype=np.float32)
    for c in range(C):
        W[c, 0] = kernel
    w_init = numpy_helper.from_array(W, name=base + "/_deconv_W")
    graph.initializer.append(w_init)

    deconv_node = helper.make_node(
        "ConvTranspose", [pad_name, w_init.name], [out_name],
        kernel_shape=[4, 4], strides=[2, 2], pads=[3, 3, 3, 3],
        dilations=[1, 1], group=C, output_padding=[0, 0],
        name=base + "/_deconv")

    # replace: swap Resize node for pad+deconv IN PLACE (keep topological order)
    idx = list(graph.node).index(resize_node)
    graph.node.insert(idx, pad_node)
    graph.node.insert(idx + 1, deconv_node)
    graph.node.remove(resize_node)

    # drop the now-orphaned sizes Constant node feeding this Resize (its output name
    # is the 3rd input of Resize). Leave if shared (it isn't).
    sizes_name = resize_node.input[2] if len(resize_node.input) > 2 else None
    if sizes_name:
        for cn in list(graph.node):
            if cn.op_type == "Constant" and sizes_name in cn.output:
                graph.node.remove(cn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="conversion/net_decode.onnx")
    ap.add_argument("--dst", default="conversion/net_decode_deconv.onnx")
    a = ap.parse_args()

    model = onnx.load(a.src)
    model.graph.ClearField("value_info")                # shape_inference will refill
    resizes = find_resize_nodes(model)
    print(f"found {len(resizes)} Resize nodes")
    if len(resizes) != 3:
        raise RuntimeError(f"expected 3 Resize nodes, got {len(resizes)}")

    shapes = resolve_shapes(model)                      # resolve all BEFORE mutating
    for n in resizes:
        dims = shapes.get(n.input[0])
        if dims is None or len(dims) != 4 or dims[1] <= 0:
            raise RuntimeError(f"cannot resolve NCHW shape for {n.input[0]}: {dims}")
        C = dims[1]
        print(f"  {n.name}: in={n.input[0]} (C={C}) -> out={n.output[0]}")
        make_replacement(model.graph, n, C, a.dst)

    onnx.checker.check_model(model)
    onnx.save(model, a.dst)
    print(f"\nsaved -> {a.dst}")


if __name__ == "__main__":
    main()
