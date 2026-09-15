# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Select task implementations by family protocol, independently of platform I/O."""
import importlib
from yolo_assets import family_registry, is_nms_free, UnsupportedAssetError

DFL_TASKS = {'detect': ('yolo_detect','YoloDetect'), 'cls': ('yolo_cls','YoloCls'),
             'seg': ('yolo_seg','YoloSeg'), 'pose': ('yolo_pose','YoloPose')}
LTRB_TASKS = {'detect': ('yolo26_det','YOLO26Detect'), 'cls': ('yolo26_cls','YOLO26Cls'),
              'seg': ('yolo26_seg','YOLO26Seg'), 'pose': ('yolo26_pose','YOLO26Pose'),
              'obb': ('yolo26_obb','YOLO26OBB')}


def get_task_types(profile, family, task):
    spec=family_registry(profile).get(family)
    if spec is None or task not in spec.tasks:
        raise UnsupportedAssetError(f'{profile.key}/{family} does not support task {task}.')
    module,name=(LTRB_TASKS if family=='yolo26' else DFL_TASKS)[task]
    if task=='detect' and is_nms_free(profile,family):module,name='yolo_v10detect','YoloV10Detect'
    loaded=importlib.import_module(module)
    return getattr(loaded,name),getattr(loaded,name+'Config')


def runtime_resize(profile,family,task):
    from yolo_runtime import default_resize_type
    return 0 if family=='yolo26' and task=='cls' else default_resize_type(profile,task)


def create_runtime_model(profile,args):
    from yolo_runtime import default_nms_thres
    Model,Config=get_task_types(profile,args.family,args.task)
    kw=dict(model_path=args.model_path,platform=profile,input_shape=args.input_shape,
        resize_type=args.resize_type if args.resize_type is not None else runtime_resize(profile,args.family,args.task))
    if args.task=='cls':
        kw['topk']=args.topk
    else:
        kw.update(score_thres=args.score_thres,strides=args.strides)
        if not is_nms_free(profile,args.family):
            kw['nms_thres']=args.nms_thres if args.nms_thres is not None else default_nms_thres(profile,args.task)
        if args.classes_num is not None and args.task in ('detect','seg','obb'):kw['classes_num']=args.classes_num
        if args.family!='yolo26':
            kw['reg']=args.reg
            if args.task=='pose':kw['nkpt']=args.nkpt
            if args.task=='seg':kw['mces_num']=args.mc
        elif args.reg!=16 or args.nkpt!=17 or args.mc!=32:
            raise ValueError('YOLO26 uses direct LTRB, 17 pose points and 32 mask coefficients; DFL overrides do not apply.')
        if args.task=='obb':kw.update(angle_sign=args.angle_sign,angle_offset=args.angle_offset,regularize=bool(args.regularize))
    return Model(Config(**kw))
