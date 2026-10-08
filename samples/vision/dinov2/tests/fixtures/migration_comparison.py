# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test-only DINOv2 source-parity recipe (no code executes at import).

The constant below holds the exact ``python3 - <<'PY'`` heredoc formerly
published in ``samples/vision/dinov2/evaluator/README.md`` and
``README_cn.md`` at commit f773f3542f01 (the two documents' heredocs were
byte-identical, verified by extraction digest, so a single variant is
preserved and aliased for both fixture runs). The customer-facing evaluator
documents no longer carry the migration recipe; it is preserved here so the
host suite still executes the identical source-vs-unified comparison under
fake-SDK fixtures. Only ``test_readme_contract.py`` compiles and executes
these strings.
"""

README_MD = r'''from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from dinov2 import Dinov2, Dinov2Config
from samples.vision.dinov2.runtime.python.cli import resolve_selection
from samples.vision.dinov2.runtime.python.embedding import DINOv2Embedder, RuntimeModelRunner
from utils.py_utils.platforms import require_execution_target

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

repo = Path.cwd()
target = 's100'  # change to s100p or s600 to select that target's own HBM
selection = resolve_selection(target)
require_execution_target(target)
if not selection.model_path.is_file():
    raise FileNotFoundError(selection.model_path)
image_path = repo / 'samples/vision/dinov2/test_data/dog.jpg'
image = cv2.imread(str(image_path))
if image is None:
    raise ValueError(image_path)
started = datetime.now(timezone.utc)
output_dir = repo / 'evaluator-output' / ('dinov2-' + started.strftime('%Y%m%dT%H%M%S%fZ'))
output_dir.mkdir(parents=True, exist_ok=False)
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
legacy = Dinov2(Dinov2Config(str(selection.model_path)))
legacy.set_scheduling_params(priority=0, bpu_cores=[0])
prepared = DINOv2Embedder(selection, runner=runner).pre_process(image)
old_inputs = legacy.pre_process(image)
np.testing.assert_array_equal(old_inputs[legacy.model_name]['input'], prepared.tensors['input'])
np.save(output_dir / 'input.npy', prepared.tensors['input'], allow_pickle=False)
old_raw = legacy.forward(old_inputs)
new_raw = runner(prepared.tensors)
records = {}
passed = True
for name in ('cls_feat', 'patch_feat'):
    a, b = np.asarray(old_raw[legacy.model_name][name]), np.asarray(new_raw[name])
    np.save(output_dir / ('legacy-raw-' + name + '.npy'), a, allow_pickle=False)
    np.save(output_dir / ('unified-raw-' + name + '.npy'), b, allow_pickle=False)
    same_protocol = a.shape == b.shape and a.dtype == b.dtype
    raw_ok = same_protocol and (np.array_equal(a, b) if np.issubdtype(a.dtype, np.integer)
                               else np.allclose(a, b, rtol=0, atol=1e-5))
    legacy.cfg.output = name
    reference = legacy.post_process(old_raw)
    candidate = DINOv2Embedder(selection, output=name, runner=runner).post_process(new_raw)
    np.save(output_dir / ('legacy-result-' + name + '.npy'), reference, allow_pickle=False)
    np.save(output_dir / ('unified-result-' + name + '.npy'), candidate, allow_pickle=False)
    result_ok = reference.shape == candidate.shape and reference.dtype == candidate.dtype and np.allclose(reference, candidate, rtol=0, atol=1e-5)
    records[name] = {'raw_protocol_equal': same_protocol, 'raw_equal': bool(raw_ok),
                     'result_equal': bool(result_ok), 'shape': list(b.shape), 'raw_dtype': str(b.dtype)}
    passed = passed and raw_ok and result_ok
code_paths = [p for base in ('samples/vision/dinov2/runtime/python', 'utils/py_utils',
                             'platforms/s/samples/vision/dinov2/runtime/python', 'platforms/s/utils/py_utils')
              for p in (repo / base).glob('*.py')]
report = {'started_utc': started.isoformat(), 'ended_utc': datetime.now(timezone.utc).isoformat(),
          'target': target, 'asset_id': selection.asset.reference,
          'model_sha256': sha256(selection.model_path), 'image_sha256': sha256(image_path),
          'code_sha256': {str(p.relative_to(repo)): sha256(p) for p in code_paths},
          'records': records, 'passed': bool(passed),
          'scope': 'same-board legacy/unified migration parity; not float-ONNX accuracy'}
(output_dir / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(output_dir)
print(json.dumps(report, indent=2))
if not passed:
    raise AssertionError('DINOv2 migration parity failed; full arrays are saved.')'''

# README.md and README_cn.md carried the identical heredoc body at that
# commit; one variant serves both fixture runs.
README_CN_MD = README_MD
