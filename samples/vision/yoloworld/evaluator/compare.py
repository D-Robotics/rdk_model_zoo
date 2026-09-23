# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Capture same-board source/unified YOLOWorld parity evidence.

This command is intentionally board-only: it never downloads or fabricates a
comparison. The output directory must be new and contains raw arrays, results,
code/model/input hashes, and metadata when both implementations execute.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from samples.vision.yoloworld.runtime.python.model_binding import resolve_selection

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def build_parser():
    p=argparse.ArgumentParser(description='Compare fixed X5 legacy and unified YOLOWorld outputs on the same board.')
    p.add_argument('--target',choices=('x5',),required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--model-path',default=None);p.add_argument('--asset-id',default=None)
    p.add_argument('--vocab-file',default=str(ROOT/'samples/vision/yoloworld/test_data/offline_vocabulary_embeddings.json'))
    p.add_argument('--test-img',default=str(ROOT/'samples/vision/yoloworld/test_data/dog.jpeg'));p.add_argument('--prompts',default='dog')
    p.add_argument('--score-thres',type=float,default=.05);p.add_argument('--nms-thres',type=float,default=.45)
    return p

def _save_tree(directory, prefix, value):
    if isinstance(value,dict):
        for k,v in value.items(): _save_tree(directory, f'{prefix}_{k}', v)
    else: np.save(directory/f'{prefix}.npy', np.asarray(value))

def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        from samples._shared.platforms import require_execution_target
        require_execution_target(args.target)
        selection=resolve_selection(args.target,model_path=args.model_path,asset_id=args.asset_id)
        out=args.output_dir.expanduser()
        if out.exists(): raise ValueError('output-dir must not already exist; failed comparisons retain evidence.')
        out.mkdir(parents=True)
        import cv2
        image=cv2.imread(args.test_img,cv2.IMREAD_COLOR)
        if image is None: raise ValueError(f'Cannot read image: {args.test_img}')
        vocab=json.loads(Path(args.vocab_file).read_text(encoding='utf-8')); prompts=[x.strip() for x in args.prompts.split(',')]
        from samples.vision.yoloworld.runtime.python.model_runner import RuntimeModelRunner
        from samples.vision.yoloworld.runtime.python.yoloworld import YOLOWorldTask
        unified_runner=RuntimeModelRunner(selection); binding=unified_runner.load()
        unified=YOLOWorldTask(unified_runner,binding,vocab,score_thres=args.score_thres,nms_thres=args.nms_thres)
        prepared=unified.pre_process(image,prompts); uraw=unified.forward(prepared); uresult=unified.post_process(uraw,prepared.context)
        _save_tree(out,'unified_raw',uraw); np.save(out/'unified_boxes.npy',uresult.boxes);np.save(out/'unified_scores.npy',uresult.scores);np.save(out/'unified_class_ids.npy',uresult.class_ids)
        source_path=ROOT/'platforms/x5/samples/vision/yoloworld/runtime/python/yoloworld_det.py'
        spec=importlib.util.spec_from_file_location('yoloworld_legacy',source_path); legacy_mod=importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = legacy_mod
        spec.loader.exec_module(legacy_mod)
        legacy_cfg=legacy_mod.YOLOWorldConfig(str(selection.model_path),args.vocab_file,args.score_thres,args.nms_thres)
        legacy=legacy_mod.YOLOWorldDetect(legacy_cfg)
        linputs=legacy.pre_process(image,prompts); lraw=legacy.forward(linputs); lresult=legacy.post_process(lraw,image.shape[1],image.shape[0])
        _save_tree(out,'legacy_raw',lraw); np.save(out/'legacy_boxes.npy',lresult[0]);np.save(out/'legacy_scores.npy',lresult[1]);np.save(out/'legacy_class_ids.npy',lresult[2])
        np.testing.assert_array_equal(uraw[binding.score_output_name],lraw[legacy.model_name][legacy.output_names[0]])
        np.testing.assert_array_equal(uraw[binding.box_output_name],lraw[legacy.model_name][legacy.output_names[1]])
        np.testing.assert_allclose(uresult.boxes,lresult[0],atol=0,rtol=0);np.testing.assert_allclose(uresult.scores,lresult[1],atol=0,rtol=0);np.testing.assert_array_equal(uresult.class_ids,lresult[2])
        metadata={'target':args.target,'asset_id':selection.asset.reference,'model_path':str(selection.model_path),'model_sha256':_sha(selection.model_path),'source_code_sha256':_sha(source_path),'input_sha256':_sha(args.test_img),'vocabulary_sha256':_sha(args.vocab_file),'prompts':prompts,'score_thres':args.score_thres,'nms_thres':args.nms_thres,'decision':'pass'}
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8'); return 0
    except AssertionError as exc:
        print(f'comparison failed: {exc}',file=sys.stderr); return 1
    except (ImportError,OSError,RuntimeError,ValueError,KeyError) as exc:
        print(f'execution/setup failed: {exc}',file=sys.stderr); return 2
if __name__=='__main__': raise SystemExit(main())
