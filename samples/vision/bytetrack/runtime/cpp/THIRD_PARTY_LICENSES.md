# Third-party components in this C++ runtime

## lap (linear assignment, Jonker-Volgenant)

- `inc/lapjv.h`, `src/lapjv.cpp` are the verbatim C++ sources shipped inside
  the `lap` 0.5.12 source distribution (PyPI sdist `lap-0.5.12.tar.gz`,
  sha256 `570b414ea7ae6c04bd49d0ec8cdac1dc5634737755784d44e37f9f668bab44fd`,
  directory `_lapjv_cpp/`), which is the same solver the Python baseline
  reaches through `lap.lapjv(cost, extend_cost=True, cost_limit=thresh)`.
  No modification; the wrapper semantics around `lapjv_internal` (the
  (rows+cols) cost_limit/2 padding, the zero bottom-right block and the -1
  mapping for assignments into the padding) are re-implemented in
  `src/tracking.cpp` following `lap/_lapjv.pyx` from the same distribution.
- License: BSD 2-Clause, Copyright (c) 2012-2024 Tomas Kazmar — full text in
  [`licenses/lap-LICENSE`](licenses/lap-LICENSE). Upstream:
  https://pypi.org/project/lap/ , https://github.com/gatagat/lap

## IoU convention (cython_bbox)

The IoU formula in `src/tracking.cpp` (`bbox_iou`) reproduces the
`cython_bbox.bbox_overlaps` convention the Python baseline calls: inclusive
+1 pixel extents (`iw = min(x2) - max(x1) + 1`) and
`union = areaA + areaB - iw*ih`, in double precision. The formula
originates in Fast R-CNN (Copyright (c) 2015 Microsoft, MIT License; see
https://github.com/hracny/cython_bbox which the sample pins). No source code
from cython_bbox is included here; the expression is re-implemented.
