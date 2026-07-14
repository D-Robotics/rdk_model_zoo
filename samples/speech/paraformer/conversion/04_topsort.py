"""
Step 04: Topologically sort ONNX nodes (required by onnx.checker after graph surgery).

Usage:
    python 04_topsort.py <input.onnx> <output.onnx>
"""
import sys, argparse
from collections import deque
import onnx
from onnx import helper


def main(src, dst):
    m = onnx.load(src)
    nodes = list(m.graph.node)
    n = len(nodes)

    producer = {}
    for i, node in enumerate(nodes):
        for o in node.output:
            producer[o] = i

    init_names = {i.name for i in m.graph.initializer}
    graph_input_names = {i.name for i in m.graph.input}

    deps = [set() for _ in range(n)]
    for i, node in enumerate(nodes):
        for inp in node.input:
            if inp in init_names or inp in graph_input_names or inp == "":
                continue
            j = producer.get(inp)
            if j is not None and j != i:
                deps[i].add(j)

    indeg = [len(d) for d in deps]
    reverse = [[] for _ in range(n)]
    for i, d in enumerate(deps):
        for j in d:
            reverse[j].append(i)

    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        i = q.popleft()
        order.append(i)
        for k in reverse[i]:
            indeg[k] -= 1
            if indeg[k] == 0:
                q.append(k)

    if len(order) != n:
        print(f"WARN: cycle detected — sorted {len(order)}/{n}")
    else:
        print(f"topologically sorted {n} nodes")

    sorted_nodes = [nodes[i] for i in order]

    new_graph = helper.make_graph(
        sorted_nodes, m.graph.name, list(m.graph.input), list(m.graph.output),
        initializer=list(m.graph.initializer), value_info=list(m.graph.value_info),
    )
    m_new = helper.make_model(new_graph, opset_imports=list(m.opset_import), ir_version=m.ir_version)
    onnx.save(m_new, dst, save_as_external_data=False)
    onnx.checker.check_model(m_new)
    print(f"Saved: {dst}  onnx.checker OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src", nargs="?", default="./out/decoder_only_int32gather.onnx")
    p.add_argument("dst", nargs="?", default="./out/decoder_only_topo.onnx")
    args = p.parse_args()
    main(args.src, args.dst)
