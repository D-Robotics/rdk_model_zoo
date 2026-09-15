# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Export OBB predictions; no DOTA AP is claimed by this prediction dumper."""
import argparse,json,os,sys,math
from pathlib import Path
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parent))
from eval_common import add_platform_arguments,add_threshold_arguments,evaluation_types,evaluation_options,resolve_platform_argument,IMAGE_SUFFIXES


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True)
    p.add_argument('--image-dir',required=True)
    p.add_argument('--json-save-path',default='results_obb.json')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--angle-sign',type=float,default=1)
    p.add_argument('--angle-offset',type=float,default=0)
    p.add_argument('--label-path',default=None,help='Legacy option: prediction export only; labels are not scored.')
    add_platform_arguments(p);add_threshold_arguments(p)
    return p


def main(argv=None):
    args=build_parser().parse_args(argv);platform=resolve_platform_argument(args)
    Model,Config=evaluation_types(args,platform,'obb')
    config=dict(model_path=args.model_path,platform=platform,input_shape=args.input_shape,
        nms_thres=args.nms_thres,angle_sign=args.angle_sign,angle_offset=args.angle_offset,**evaluation_options(args,'obb'))
    if args.conf_thres is not None:config['score_thres']=args.conf_thres
    model=Model(Config(**config))
    names=sorted(n for n in os.listdir(args.image_dir) if n.lower().endswith(IMAGE_SUFFIXES))
    if args.limit>0:names=names[:args.limit]
    results=[]
    for index,name in enumerate(names):
        img=cv2.imread(os.path.join(args.image_dir,name))
        if img is None:raise FileNotFoundError(name)
        stem=Path(name).stem
        image_id=int(stem[1:]) if stem.startswith('P') and stem[1:].isdigit() else index
        for item in model.predict(img):
            cx,cy,w,h,angle=item['rrect']
            polygon=cv2.boxPoints(((float(cx),float(cy)),(float(w),float(h)),math.degrees(angle)))
            results.append(dict(image_id=image_id,file_name=name,category_id=int(item['id']),score=float(item['score']),
                rrect=[float(v) for v in item['rrect']],polygon=polygon.reshape(-1).tolist()))
    with open(args.json_save_path,'w',encoding='utf-8') as handle:json.dump(results,handle)
    print(f'Wrote {len(results)} OBB predictions. DOTA AP was not computed.')
    return 0

if __name__=='__main__':sys.exit(main())
