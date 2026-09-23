# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Reproducible legacy/unified fixed-image SAM comparison, separate from tasks.

Runs only on the explicitly requested board in normal use. Captures both stage
inputs/raw outputs and results in a new directory, even on comparison failure.
This measures migration consistency, not dataset accuracy or performance.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import cv2
import numpy as np

from samples._shared.platforms import require_execution_target
from samples._shared.sam_binding import resolve_selection
from samples._shared.sam_runner import RuntimeModelRunner
from samples._shared.sam_stages import SAMPipeline
from samples._shared.sam_tensor_io import DEFAULT_BOX, validate_box

_ROOT = Path(__file__).resolve().parents[2]


def _digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _board_identity(resolved):
    """Record only public board identity files, leaving absent facts null."""
    from samples._shared.platforms import (SOC_NAME_PATH, SOCINFO_NAME_PATH,
        BOARD_TYPE_PATH, DEVICE_TREE_MODEL_PATH)
    values = {'resolved_target': resolved}
    for path in (SOC_NAME_PATH, SOCINFO_NAME_PATH, BOARD_TYPE_PATH, DEVICE_TREE_MODEL_PATH):
        try:
            values[str(path)] = path.read_text(encoding='utf-8').strip('\x00 \n\r\t')
        except OSError:
            values[str(path)] = None
    return values


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Unsupported evidence value {type(value).__name__}')


class _RecordingRuntime:
    """Observe native calls without changing values sent to the underlying SDK."""
    def __init__(self, runtime, stage, record):
        self._runtime, self._stage, self._record = runtime, stage, record

    def __getattr__(self, name):
        return getattr(self._runtime, name)

    def run(self, physical):
        name = self.model_names[0]
        flat = physical[name] if name in physical else physical
        self._record['inputs'][self._stage] = {k: np.array(v, copy=True) for k, v in flat.items()}
        result = self._runtime.run(physical)
        self._record['outputs'][self._stage] = {k: np.array(v, copy=True) for k, v in result[name].items()}
        return result


def _arrays_equal(left, right, *, atol):
    if set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if (a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all()
                or not np.isfinite(b).all() or not np.allclose(a, b, rtol=0, atol=atol)):
            return False
    return True


def compare_records(legacy, unified):
    """Strict shapes/dtypes, exact inputs/masks/index, raw 1e-5 and IoU 1e-6."""
    checks = {}
    for kind, tolerance in (('inputs', 0), ('outputs', 1e-5)):
        checks[f'{kind}_equal' if kind == 'inputs' else 'raw_close'] = (
            set(legacy[kind]) == set(unified[kind])
            and all(_arrays_equal(legacy[kind][s], unified[kind][s], atol=tolerance) for s in legacy[kind]))
    a, b = legacy['result'], unified['result']
    checks['mask_equal'] = _arrays_equal({'mask': a['mask']}, {'mask': b['mask']}, atol=0)
    checks['mask_index_equal'] = a['mask_index'] == b['mask_index']
    checks['iou_close'] = bool(np.isfinite(a['iou']) and np.isfinite(b['iou']) and abs(a['iou'] - b['iou']) <= 1e-6)
    checks['low_res_close'] = _arrays_equal({'masks': a['low_res_masks']}, {'masks': b['low_res_masks']}, atol=1e-5)
    changed = int(np.count_nonzero(a['mask'] != b['mask'])) if a['mask'].shape == b['mask'].shape else None
    return dict(checks=checks, passed=all(checks.values()), mask_changed_pixels=changed,
                tolerances=dict(raw_atol=1e-5, result_iou_atol=1e-6, rtol=0),
                decision_policy='No tie, dtype, shape or changed-mask automatic exemption.')


def _load_legacy(sample, group, factory):
    path = _ROOT/'platforms'/group/'samples'/'vision'/sample/'runtime'/'python'/f'{sample}.py'
    name = f'_sam_comparison_legacy_{group}_{sample}'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # The runtime factory already chose the real SDK or an offline fixture.
    # Source import must not force a second SDK import for the fixture case.
    had_sdk_module = 'hbm_runtime' in sys.modules
    if not had_sdk_module:
        bridge = types.ModuleType('hbm_runtime')
        bridge.HB_HBMRuntime = factory
        sys.modules['hbm_runtime'] = bridge
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
        if not had_sdk_module:
            sys.modules.pop('hbm_runtime', None)
    module.HB_HBMRuntime = factory
    return module, path


def _save_arrays(directory, records):
    saved = {}
    for side, record in records.items():
        arrays = {}
        for kind in ('inputs', 'outputs'):
            for stage, values in record[kind].items():
                for name, value in values.items():
                    arrays[f'{side}_{stage}_{kind}_{name}.npy'] = value
        result = record.get('result')
        if result:
            arrays[f'{side}_mask.npy'] = result['mask']
            arrays[f'{side}_low_res_masks.npy'] = result['low_res_masks']
        for filename, array in arrays.items():
            # Names are fixed protocol tensor names checked by native bindings.
            if Path(filename).name != filename:
                raise ValueError(f'Unsafe tensor evidence filename {filename!r}.')
            path = directory/filename
            np.save(path, array, allow_pickle=False)
            saved[filename] = dict(shape=list(array.shape), dtype=str(array.dtype), sha256=_digest(path))
    return saved


def run_comparison(selection, image, image_path, output_dir, *, box=None,
                   priority=0, bpu_cores=None, runtime_factory=None):
    """Capture source baseline then unified inference on the same exact files.

    The optional runtime_factory is an offline test seam. Production invocation
    still always checks hardware identity. A new directory is mandatory; failed
    execution writes error evidence then re-raises, and failed comparisons return
    passed=False with arrays intact. CLI maps either failure to nonzero status.
    """
    directory = Path(output_dir).expanduser().resolve()
    if directory.exists():
        raise FileExistsError(f'Evidence directory must not already exist: {directory}')
    actual_target = require_execution_target(selection.target)
    if priority < 0 or priority > 255:
        raise ValueError('priority must be in 0..255.')
    if bpu_cores is not None and (not bpu_cores or any(type(v) is not int or v < 0 for v in bpu_cores)):
        raise ValueError('bpu-cores must be nonempty nonnegative indexes.')
    if selection.target == 'x5' and bpu_cores is not None:
        raise ValueError('X5 does not expose BPU core selection.')
    if selection.sample == 'efficient_sam' and box is not None:
        raise ValueError('EfficientSAM uses a fixed exported prompt and does not accept a runtime box.')
    box = validate_box(DEFAULT_BOX if box is None else box) if selection.sample == 'mobile_sam' else None
    cores = ([0] if bpu_cores is None else bpu_cores) if selection.target != 'x5' else None
    assets = {stage: dict(asset_id=asset.reference, path=str(path.resolve()),
              publisher_sha256=asset.sha256, observed_sha256=None) for stage, asset, path in (
        ('encoder', selection.encoder_asset, selection.encoder_model_path),
        ('decoder', selection.decoder_asset, selection.decoder_model_path))}
    sample = selection.sample
    group = 'x5' if selection.target == 'x5' else 's'
    code_paths = list((_ROOT/'samples'/'_shared').glob('sam_*.py'))
    code_paths += [_ROOT/'samples'/'_shared'/name for name in ('runtime_meta.py', 'platforms.py', 'assets.py')]
    code_paths += list((_ROOT/'samples'/'vision'/sample/'runtime'/'python').glob('*.py'))
    code_paths += [_ROOT/'platforms'/group/'samples'/'vision'/sample/'runtime'/'python'/f'{sample}.py']
    summary = dict(sample=sample, target=selection.target, started_utc=_now(),
        argv=list(sys.argv), cwd=str(Path.cwd()), output_dir=str(directory), artifacts=assets,
        board_identity=_board_identity(actual_target),
        host_versions=dict(python=sys.version, numpy=np.__version__, opencv=cv2.__version__),
        image=dict(path=str(Path(image_path).resolve()), sha256=None,
                   decoded_shape=list(image.shape), decoded_dtype=str(image.dtype)),
        box=box, priority=priority, bpu_cores=cores, passed=False,
        source_ref='ac115717197920355fc390bb04299b20e6436864' if group == 'x5' else '380e1a2bf42041af54be6f34935e50197cfadff9',
        code_sha256={},
        measurement='fixed-image migration consistency; not accuracy or latency')
    records = {side: dict(inputs={}, outputs={}) for side in ('legacy', 'unified')}
    summary['metadata'] = {side: {} for side in records}
    directory.mkdir(parents=True, exist_ok=False)
    failure = None
    try:
        for entry in assets.values():
            entry['observed_sha256'] = _digest(entry['path'])
        summary['image']['sha256'] = _digest(image_path)
        summary['code_sha256'] = {str(p.relative_to(_ROOT)): _digest(p) for p in code_paths}
        if runtime_factory is None:
            from samples._shared.model_runner import _default_runtime_factory
            runtime_factory = _default_runtime_factory()

        def recording_factory(side):
            def create(path):
                stage = 'encoder' if Path(path) == selection.encoder_model_path else 'decoder'
                runtime = runtime_factory(path)
                from samples._shared.runtime_meta import RuntimeMetadata
                summary['metadata'][side][stage] = asdict(RuntimeMetadata.from_runtime(runtime))
                return _RecordingRuntime(runtime, stage, records[side])
            return create

        source, source_path = _load_legacy(sample, group, recording_factory('legacy'))
        prefix = 'EfficientSAM' if sample == 'efficient_sam' else 'MobileSAM'
        cfg = dict(encoder_model_path=str(selection.encoder_model_path), decoder_model_path=str(selection.decoder_model_path))
        if box is not None:
            cfg['box'] = box
        legacy = getattr(source, prefix+'Segment')(getattr(source, prefix+'Config')(**cfg))
        schedule = dict(priority=priority)
        if cores is not None:
            schedule['bpu_cores'] = cores
        legacy.set_scheduling_params(**schedule)
        records['legacy']['result'] = legacy.predict(image)
        runner = RuntimeModelRunner(selection, runtime_factory=recording_factory('unified'))
        binding = runner.load()
        runner.set_scheduling_params(priority=priority, bpu_cores=cores)
        records['unified']['result'] = SAMPipeline(runner, binding).predict(image, box=box)
        summary.update(compare_records(records['legacy'], records['unified']))
        summary['result_summaries'] = {side: dict(iou=r['result']['iou'], mask_index=r['result']['mask_index']) for side, r in records.items()}
    except Exception as exc:
        summary['error'] = dict(type=type(exc).__name__, message=str(exc))
        failure = exc
    finally:
        summary['arrays'] = _save_arrays(directory, records)
        summary['return_code'] = 2 if failure is not None else (0 if summary['passed'] else 1)
        summary['finished_utc'] = _now()
        (directory/'comparison.json').write_text(json.dumps(summary, indent=2, default=_json_default, allow_nan=False)+'\n', encoding='utf-8')
    if failure is not None:
        raise failure
    return summary


def _parse_box(value):
    try:
        return validate_box(value.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser(sample):
    p = argparse.ArgumentParser(description=f'{sample}: fixed-image legacy/unified board comparison, no download.')
    p.add_argument('--target', choices=('x5', 's100', 's100p', 's600'), required=True)
    p.add_argument('--output-dir', type=Path, required=True, help='New evidence directory; existing paths are rejected.')
    p.add_argument('--test-img', type=Path, default=_ROOT/'samples'/'vision'/sample/'test_data'/'dogs.jpg')
    for stage in ('encoder', 'decoder'):
        p.add_argument(f'--{stage}-model-path', default=None)
        p.add_argument(f'--{stage}-asset-id', default=None)
    p.add_argument('--priority', type=int, default=0)
    p.add_argument('--bpu-cores', type=int, nargs='+', default=None)
    if sample == 'mobile_sam':
        p.add_argument('--box', type=_parse_box, default=DEFAULT_BOX)
    return p


def main(sample, argv=None):
    args = build_parser(sample).parse_args(argv)
    try:
        selection = resolve_selection(sample, args.target,
            encoder_model_path=args.encoder_model_path, decoder_model_path=args.decoder_model_path,
            encoder_asset_id=args.encoder_asset_id, decoder_asset_id=args.decoder_asset_id)
        image = cv2.imread(str(args.test_img), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot read image: {args.test_img}')
        summary = run_comparison(selection, image, args.test_img, args.output_dir,
            box=getattr(args, 'box', None), priority=args.priority, bpu_cores=args.bpu_cores)
        print(json.dumps(dict(passed=summary['passed'], checks=summary['checks'],
                              evidence=str(Path(args.output_dir).resolve()/'comparison.json')), indent=2))
        return 0 if summary['passed'] else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
