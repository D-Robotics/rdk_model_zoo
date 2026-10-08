# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Select task implementations by family protocol, independently of platform I/O.

``get_task_types`` resolves one ``(Model, Config)`` pair with direct imports
of the task modules — DFL families use ``detect``/``yolo_cls``/``yolo_seg``/
``yolo_pose`` (plus ``yolo_v10detect`` for the NMS-free S YOLOv10 head),
the ``yolo26`` family uses the direct-LTRB modules. Task modules import
lazily so the model-free CLI modes stay light. ``prepare_runtime_model``
translates one parsed argument set into the task configuration.
"""
from yolo_assets import family_registry, is_nms_free, UnsupportedAssetError


def get_task_types(profile, family, task):
    """Return the task model class and its config class for one family/task.

    Args:
        profile: Platform profile naming the published families.
        family: Model family, for example ``yolo11``, ``yolov10`` or ``yolo26``.
        task: Task name: ``detect``, ``cls``, ``seg``, ``pose`` or ``obb``.

    Returns:
        Tuple of the task model class and its configuration dataclass.

    Raises:
        UnsupportedAssetError: If the family is unknown or does not publish
            the requested task.
    """
    spec = family_registry(profile).get(family)
    if spec is None or task not in spec.tasks:
        raise UnsupportedAssetError(f'{profile.key}/{family} does not support task {task}.')
    if task == 'detect':
        if is_nms_free(profile, family):
            from samples.vision.ultralytics_yolo.runtime.python.yolo_v10detect import YoloV10Detect, YoloV10DetectConfig
            return YoloV10Detect, YoloV10DetectConfig
        if family == 'yolo26':
            from samples.vision.ultralytics_yolo.runtime.python.yolo26_det import YOLO26Detect, YOLO26DetectConfig
            return YOLO26Detect, YOLO26DetectConfig
        from samples.vision.ultralytics_yolo.runtime.python.detect import YoloDetect, YoloDetectConfig
        return YoloDetect, YoloDetectConfig
    if task == 'cls':
        from samples.vision.ultralytics_yolo.runtime.python.yolo_cls import YoloCls, YoloClsConfig
        return YoloCls, YoloClsConfig
    if task == 'seg':
        if family == 'yolo26':
            from samples.vision.ultralytics_yolo.runtime.python.yolo26_seg import YOLO26Seg, YOLO26SegConfig
            return YOLO26Seg, YOLO26SegConfig
        from samples.vision.ultralytics_yolo.runtime.python.yolo_seg import YoloSeg, YoloSegConfig
        return YoloSeg, YoloSegConfig
    if task == 'pose':
        if family == 'yolo26':
            from samples.vision.ultralytics_yolo.runtime.python.yolo26_pose import YOLO26Pose, YOLO26PoseConfig
            return YOLO26Pose, YOLO26PoseConfig
        from samples.vision.ultralytics_yolo.runtime.python.yolo_pose import YoloPose, YoloPoseConfig
        return YoloPose, YoloPoseConfig
    from samples.vision.ultralytics_yolo.runtime.python.yolo26_obb import YOLO26OBB, YOLO26OBBConfig
    return YOLO26OBB, YOLO26OBBConfig


def runtime_resize(profile, family, task):
    """Return the documented default resize policy for one family/task."""
    from yolo_runtime import default_resize_type
    return 0 if family == 'yolo26' and task == 'cls' else default_resize_type(profile, task)


def prepare_runtime_model(profile, args):
    """Resolve the model class and build its configuration without loading SDKs.

    Args:
        profile: Resolved platform profile.
        args: Parsed command-line arguments from ``yolo_cli.build_parser``.

    Returns:
        Tuple of the model class and its configuration instance. The caller
        constructs the model with ``Model(config)``.

    Raises:
        UnsupportedAssetError: If the family/task combination is not published.
        ValueError: If DFL-only overrides are requested for a YOLO26 model.
    """
    from yolo_runtime import default_nms_thres
    Model, Config = get_task_types(profile, args.family, args.task)
    kw = dict(model_path=args.model_path, platform=profile, input_shape=args.input_shape,
              resize_type=args.resize_type if args.resize_type is not None else runtime_resize(profile, args.family, args.task))
    if args.task == 'cls':
        kw['topk'] = args.topk
    else:
        kw.update(score_thres=args.score_thres, strides=args.strides)
        if not is_nms_free(profile, args.family):
            kw['nms_thres'] = args.nms_thres if args.nms_thres is not None else default_nms_thres(profile, args.task)
        if args.classes_num is not None and args.task in ('detect', 'seg', 'obb'):
            kw['classes_num'] = args.classes_num
        if args.family != 'yolo26':
            kw['reg'] = args.reg
            if args.task == 'pose':
                kw['nkpt'] = args.nkpt
            if args.task == 'seg':
                kw['mces_num'] = args.mc
        elif args.reg != 16 or args.nkpt != 17 or args.mc != 32:
            raise ValueError('YOLO26 uses direct LTRB, 17 pose points and 32 mask coefficients; DFL overrides do not apply.')
        if args.task == 'obb':
            kw.update(angle_sign=args.angle_sign, angle_offset=args.angle_offset, regularize=bool(args.regularize))
    return Model, Config(**kw)
