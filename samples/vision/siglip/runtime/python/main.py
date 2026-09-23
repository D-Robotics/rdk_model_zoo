# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Native SigLIP CLI: model selection, image/file I/O and feature summaries."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.siglip.runtime.python.model_binding import (
    SAMPLE_DIR, SUBMODELS, VARIANTS, list_available_assets, resolve_selection,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the SDK-free parser; no detection, model loading or downloads."""
    p = argparse.ArgumentParser(description='SigLIP packed vision encoder on S100/S100P.')
    p.add_argument('--target', choices=('auto','x5','s100','s100p','s600'), default='auto', help='Actual execution target; auto detects the board.')
    p.add_argument('--variant', choices=tuple(VARIANTS), default=None, help='Published variant; defaults to base-patch16-224 unless asset-id selects one.')
    p.add_argument('--asset-id', default=None, help='Exact qualified manifest reference; required with model-path.')
    p.add_argument('--model-path', default=None, help='Explicit local HBM path, paired with asset-id.')
    p.add_argument('--test-img', default=str(SAMPLE_DIR/'test_data/dog.jpg'), help='Input BGR image; default bundled dog.jpg.')
    p.add_argument('--image-size', type=int, default=None, help='Optional assertion of the selected artifact size; defaults to its bound size.')
    p.add_argument('--submodel', choices=SUBMODELS, default='pooler_output', help='Packed submodel to execute.')
    p.add_argument('--priority', type=int, default=0, help='Runtime priority 0..255.')
    p.add_argument('--bpu-cores', type=int, nargs='+', default=[0], help='Nonnegative BPU core indexes.')
    p.add_argument('--output-file', default=None, help='Optional NumPy-format result path (used exactly as given).')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--list-models', action='store_true', help='List unique manifest assets without hardware.')
    mode.add_argument('--dry-run', action='store_true', help='Resolve metadata contract; requires an explicit target, no SDK.')
    return p


def main(argv=None) -> int:
    """Print resolved contracts or execute one feature extraction; return 0/2."""
    args=build_parser().parse_args(argv)
    try:
        if args.list_models:
            assets=list_available_assets(args.target)
            for asset in assets:
                print(asset.reference)
            print(f'{len(assets)} unique assets; supported targets: s100, s100p; no model loaded.')
            return 0
        if args.dry_run and args.target == 'auto':
            raise ValueError('Host dry-run requires --target s100 or --target s100p; no board detection performed.')
        selection=resolve_selection(args.target,variant=args.variant,asset_id=args.asset_id,
            model_path=args.model_path,submodel=args.submodel,image_size=args.image_size)
        if args.dry_run:
            size,dim,count=VARIANTS[selection.variant]
            print(json.dumps(dict(target=selection.target,variant=selection.variant,
                asset_id=selection.asset.reference,model_path=str(selection.model_path),
                model_path_exists=selection.model_path.is_file(),submodel=selection.submodel,
                input_shape=[1,3,size,size],output_shapes=[[1,dim],[1,1,dim]] if selection.submodel=='pooler_output' else [[1,count,dim]],
                output_policy='preserve native numeric dtype/shape; no dequantization or activation',
                source_manifest=selection.asset.source_path),indent=2))
            return 0
        from samples._shared.platforms import require_execution_target
        require_execution_target(selection.target)
        if not selection.model_path.is_file():
            raise FileNotFoundError(f'Model not found: {selection.model_path}; prepare it explicitly with model/download.sh.')
        return _run(selection,args)
    except (OSError,ValueError,RuntimeError) as exc:
        print(f'error: {exc}',file=sys.stderr)
        return 2


def _run(selection,args):
    """Read one image, call the task, summarize and optionally save its result."""
    import cv2
    import numpy as np
    from samples.vision.siglip.runtime.python.model_runner import RuntimeModelRunner
    from samples.vision.siglip.runtime.python.embedding import SigLIPTask
    image=cv2.imread(str(Path(args.test_img).expanduser()))
    if image is None:
        raise ValueError(f'Cannot read image: {args.test_img}')
    runner=RuntimeModelRunner(selection)
    binding=runner.load()
    runner.set_scheduling_params(priority=args.priority,bpu_cores=args.bpu_cores)
    result=SigLIPTask(runner,binding).predict(image)
    flat=result.reshape(-1).astype(np.float32)
    summary=dict(submodel=selection.submodel,shape=list(result.shape),dtype=str(result.dtype),
        mean=float(flat.mean()),std=float(flat.std()),min=float(flat.min()),max=float(flat.max()),
        l2_norm=float((flat**2).sum()**0.5))
    print(json.dumps(summary,indent=2,allow_nan=False))
    if args.output_file is not None:
        output=Path(args.output_file).expanduser()
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('wb') as handle:
            np.save(handle,result,allow_pickle=False)
        print(f'Feature tensor saved: {output}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
