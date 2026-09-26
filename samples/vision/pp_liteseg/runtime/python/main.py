# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""PP-LiteSeg CLI: explicit preparation, image IO, rendering and evidence fields."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from samples.vision.pp_liteseg.runtime.python.model_binding import (
    SAMPLE_DIR,SUPPORTED_TARGETS,resolve_selection,list_available_assets,
)


def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description='PP-LiteSeg Cityscapes class-map inference')
    p.add_argument('--target',choices=('auto',)+SUPPORTED_TARGETS,default='auto')
    p.add_argument('--asset-id')
    p.add_argument('--model-path',type=Path)
    p.add_argument('--test-img',type=Path,default=SAMPLE_DIR/'test_data/street.png')
    p.add_argument('--output',type=Path,default=Path('outputs/pp_liteseg/result.jpg'))
    p.add_argument('--mask-save-path',type=Path,default=Path('outputs/pp_liteseg/labels.npy'))
    p.add_argument('--report-path',type=Path,default=Path('outputs/pp_liteseg/result.json'))
    p.add_argument('--alpha',type=float,default=0.55)
    p.add_argument('--input-width',type=int,default=1024,help='Compiled geometry; must remain 1024')
    p.add_argument('--input-height',type=int,default=512,help='Compiled geometry; must remain 512')
    p.add_argument('--priority',type=int,default=None)
    p.add_argument('--bpu-cores',nargs='+',type=int,default=None)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--list-models',action='store_true')
    modes.add_argument('--dry-run',action='store_true')
    return p


def main(argv=None) -> int:
    args=build_parser().parse_args(argv)
    try:
        if args.list_models:
            print(json.dumps([{'asset_id':a.reference,'url':a.url,'sha256':a.sha256,'target':'x5'}
                              for a in list_available_assets(args.target)],indent=2))
            return 0
        selection=resolve_selection(args.target,asset_id=args.asset_id,model_path=args.model_path)
        if (args.input_width,args.input_height)!=(1024,512):
            raise ValueError('The published model requires input-width=1024 and input-height=512')
        if not 0<=args.alpha<=1:
            raise ValueError('alpha must be in [0,1]')
        if args.priority is not None and not 0<=args.priority<=255:
            raise ValueError('priority must be 0..255')
        if args.bpu_cores is not None and any(i<0 for i in args.bpu_cores):
            raise ValueError('bpu-cores must be nonnegative')
        if args.mask_save_path.suffix!='.npy':
            raise ValueError('mask-save-path must end in .npy')
        if args.dry_run:
            print(json.dumps({'target':selection.target,'asset_id':selection.asset.reference,
                              'model_path':str(selection.model_path),'test_img':str(args.test_img),
                              'input_shape':[768,1024],'mask_shape':[512,1024],
                              'output_semantics':'int32 class IDs 0..18; not logits',
                              'sdk_loaded':False,'downloaded':False},indent=2))
            return 0
        import cv2
        import numpy as np
        from samples._shared.runtime_meta import metadata_evidence
        from samples.vision.pp_liteseg.runtime.python.model_runner import RuntimeModelRunner
        from samples.vision.pp_liteseg.runtime.python.pp_liteseg import PPLiteSegTask
        from samples.vision.pp_liteseg.runtime.python.visualization import render_result,CITYSCAPES_CLASS_NAMES
        runner=RuntimeModelRunner(selection)
        binding=runner.load()
        runner.set_scheduling_params(priority=args.priority,bpu_cores=args.bpu_cores)
        image=cv2.imread(str(args.test_img.expanduser()),cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot read image: {args.test_img}')
        labels=PPLiteSegTask(runner,binding).predict(image)
        result=render_result(image,labels,alpha=args.alpha)
        paths=[args.output.expanduser(),args.mask_save_path.expanduser(),args.report_path.expanduser()]
        for path in paths:
            path.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(paths[0]),result):
            raise OSError(f'Could not save image {paths[0]}')
        np.save(paths[1],labels,allow_pickle=False)
        ids=np.unique(labels).tolist()
        report={'target':selection.target,'asset_id':selection.asset.reference,
                'model_path':str(selection.model_path),'input_path':str(args.test_img),
                'publisher_sha256':selection.asset.sha256,
                'runtime_version':str(getattr(runner.runtime,'version','unknown')),
                'metadata':metadata_evidence(binding.metadata),'class_ids':ids,
                'class_names':[CITYSCAPES_CLASS_NAMES[i] for i in ids],
                'mask_shape':list(labels.shape),'output_shape':list(result.shape),
                'output':str(paths[0]),'mask_save_path':str(paths[1])}
        text=json.dumps(report,indent=2)
        paths[2].write_text(text+'\n')
        print(text)
        return 0
    except (ValueError,OSError,RuntimeError,ImportError) as exc:
        print(f'error: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
