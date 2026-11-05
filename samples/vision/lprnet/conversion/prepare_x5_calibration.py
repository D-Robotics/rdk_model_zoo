# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Round-trip the pinned source export's real calibration tensors to X5 raw F32."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration', type=Path, required=True, help='Source export calibration/*.npy directory')
    p.add_argument('--output', type=Path, required=True, help='New calibration-bin directory')
    a = p.parse_args(argv)
    inputs = sorted(a.calibration.glob('*.npy'))
    if len(inputs) != 200:
        raise ValueError('Expected the source export real-image partition of 200 tensors')
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for source in inputs:
        value = np.load(source, allow_pickle=False)
        if value.shape != (1,3,24,94) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Expected normalized BGR float32 NCHW plate tensor')
        destination = a.output / (source.stem + '.bin')
        value.tofile(destination)
        np.testing.assert_array_equal(np.fromfile(destination, np.float32).reshape(value.shape), value)
        rows.append({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                     'filename':destination.name,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()})
    a.output.with_name(a.output.name + '-manifest.json').write_text(json.dumps({'count':200,'dtype':'float32','shape':[1,3,24,94],
        'format':'raw binary; no NPY header','records':rows},indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
