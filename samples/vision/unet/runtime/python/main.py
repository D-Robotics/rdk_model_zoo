# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""UNet CLI: image IO, reporting and visualization outside the four stages."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from samples.vision.unet.runtime.python.model_binding import (
    SAMPLE_DIR, SUPPORTED_TARGETS, VARIANTS, resolve_selection, list_available_assets,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='UNet Pascal VOC semantic segmentation')
    p.add_argument('--target',choices=('auto',)+SUPPORTED_TARGETS,default='auto')
    p.add_argument('--variant',choices=VARIANTS,default=None)
    p.add_argument('--asset-id')
    p.add_argument('--model-path',type=Path)
    p.add_argument('--test-img',type=Path,default=SAMPLE_DIR/'test_data/2007_000033.jpg')
    p.add_argument('--mask-save-path',type=Path,default=Path('unet_mask.png'))
    p.add_argument('--img-save-path',type=Path,default=Path('unet_result.png'))
    p.add_argument('--report-path',type=Path,default=Path('unet_runtime_report.json'))
    p.add_argument('--priority',type=int)
    p.add_argument('--bpu-core',type=int)
    p.add_argument('--alpha',type=float,default=0.55)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--dry-run',action='store_true')
    modes.add_argument('--list-models',action='store_true')
    return p


def main(argv=None) -> int:
    args=build_parser().parse_args(argv)
    try:
        if args.list_models:
            print(json.dumps([{'asset_id':a.reference,'filename':a.filename,'sha256':a.sha256,
                               'url':a.url,'target':'x5'} for a in list_available_assets(args.target)],indent=2))
            return 0
        s=resolve_selection(args.target,variant=args.variant,asset_id=args.asset_id,model_path=args.model_path)
        if not 0 <= args.alpha <= 1:
            raise ValueError('alpha must be between 0 and 1')
        if args.priority is not None and not 0 <= args.priority <= 255:
            raise ValueError('priority must be 0..255')
        if args.bpu_core is not None and args.bpu_core < 0:
            raise ValueError('bpu-core must be nonnegative')
        if args.dry_run:
            print(json.dumps({'target':s.target,'variant':s.variant,'asset_id':s.asset.reference,
                              'model_path':str(s.model_path),'sdk_loaded':False,'downloaded':False,
                              'model_path_exists':s.model_path.is_file(),'packed_input_shape':[1,768,512,1],
                              'mask_shape':[512,512]},indent=2))
            return 0
        import cv2
        import numpy as np
        from samples._shared.runtime_meta import metadata_evidence
        from samples.vision.unet.runtime.python.model_runner import RuntimeModelRunner
        from samples.vision.unet.runtime.python.unet import UNetTask
        from samples.vision.unet.runtime.python.visualization import colorize_mask
        runner=RuntimeModelRunner(s)
        binding=runner.load()
        runner.set_scheduling_params(priority=args.priority,
                                     bpu_cores=[args.bpu_core] if args.bpu_core is not None else None)
        image=cv2.imread(str(args.test_img.expanduser()),cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Could not read image {args.test_img}')
        task=UNetTask(runner,binding)
        start=time.perf_counter()
        mask=task.predict(image)
        elapsed=(time.perf_counter()-start)*1000
        resized=cv2.resize(image,(512,512),interpolation=cv2.INTER_LINEAR)
        overlay=cv2.addWeighted(resized,1-args.alpha,colorize_mask(mask),args.alpha,0)
        for path, data in [(args.mask_save_path,mask),(args.img_save_path,overlay)]:
            path=path.expanduser()
            path.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(path),data):
                raise OSError(f'Could not save {path}')
        report={'target':s.target,'variant':s.variant,'asset_id':s.asset.reference,
                'model_path':str(s.model_path),'image_path':str(args.test_img),
                'runtime_version':str(getattr(runner.runtime,'version','unknown')),
                'metadata':metadata_evidence(binding.metadata),'mask_shape':list(mask.shape),
                'classes_present':np.unique(mask).tolist(),'elapsed_ms':elapsed,
                'mask_save_path':str(args.mask_save_path),'img_save_path':str(args.img_save_path)}
        args.report_path.expanduser().parent.mkdir(parents=True,exist_ok=True)
        text=json.dumps(report,indent=2)
        args.report_path.expanduser().write_text(text+'\n')
        print(text)
        return 0
    except (ValueError,OSError,RuntimeError,ImportError) as exc:
        print(f'error: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
