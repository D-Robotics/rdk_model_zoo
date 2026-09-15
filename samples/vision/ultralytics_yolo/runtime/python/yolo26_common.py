# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""YOLO26 tensor protocol. Platform I/O remains in the shared NV12 adapter."""
import numpy as np
from yolo_platform import PlatformProfile, resolve_platform
from yolo_runtime import open_model
from rdk_yolo_utils import preprocess as pre_utils


def ordered_outputs(names, shapes, height, strides, task, classes):
    """Bind roles by geometry/channels, never by compiler output-list order."""
    names = list(names)
    if len(names) != len(set(names)):
        raise ValueError('Duplicate model output names.')
    if task == 'cls':
        if len(names) != 1:
            raise ValueError('Classification requires one output.')
        return names
    channels = {'detect': (classes,4), 'seg': (classes,4,32),
                'pose': (1,4,51), 'obb': (classes,4,1)}[task]
    expected=[]
    for stride in strides:
        if int(stride) <= 0 or height % int(stride):
            raise ValueError('Strides must divide model geometry.')
        expected.extend((1,height//int(stride),height//int(stride),c) for c in channels)
    if task == 'seg':expected.append((1,height//4,height//4,32))
    if len(names) != len(expected):
        raise ValueError(f'{task} expects {len(expected)} outputs, got {len(names)}.')
    result=[]
    for shape in expected:
        matches=[n for n in names if n not in result and tuple(shapes[n]) == shape]
        if len(matches) != 1:
            raise ValueError(f'Cannot uniquely bind YOLO26 output {shape}; check task and output layout.')
        result.append(matches[0])
    return result


class Yolo26Runtime:
    """Shared loading, pre-processing and inference for LTRB task decoders."""
    task = None

    def __init__(self, config):
        self.cfg=config
        profile=config.platform
        if not isinstance(profile,PlatformProfile):profile=resolve_platform(profile)
        config.platform=profile
        self.model,self.input_adapter=open_model(config,profile,config.input_shape)
        self.model_name=self.input_adapter.model_name
        self.input_names=self.input_adapter.input_names
        self.input_shapes=self.model.input_shapes[self.model_name]
        self.input_h=self.input_adapter.input_height
        self.input_w=self.input_adapter.input_width
        if list(config.strides) != [8,16,32]:
            raise ValueError('YOLO26 currently supports strides 8,16,32.')
        grids=self.input_adapter.expected_anchor_sizes(config.strides)
        if getattr(config,'anchor_sizes',None) is not None and list(config.anchor_sizes)!=grids:
            raise ValueError('anchor_sizes conflict with model input metadata.')
        config.anchor_sizes=grids
        self.output_names=ordered_outputs(self.model.output_names[self.model_name],
            self.model.output_shapes[self.model_name],self.input_h,config.strides,self.task,
            getattr(config,'classes_num',1 if self.task=='pose' else 15))
        self.grids={}
        for stride,grid_size in zip(config.strides,grids):
            grid=np.stack(np.indices((grid_size,grid_size))[::-1],axis=-1)
            self.grids[int(stride)]=grid.reshape(-1,2).astype(np.float32)+.5
        self.map_idx={s:(i*3,i*3+1,i*3+2) for i,s in enumerate(config.strides)}
        self.conf_raw=self.logit_threshold(config.score_thres)
        self.angle_offset_rad=np.deg2rad(getattr(config,'angle_offset',0))

    @staticmethod
    def logit_threshold(score):
        if not 0 < score < 1:raise ValueError('score_thres must lie strictly between 0 and 1.')
        return -np.log(1/score-1)

    def set_scheduling_params(self,priority=None,bpu_cores=None):
        kwargs={}
        if priority is not None:kwargs['priority']={self.model_name:priority}
        if bpu_cores is not None:kwargs['bpu_cores']={self.model_name:bpu_cores}
        if kwargs:self.model.set_scheduling_params(**kwargs)

    def pre_process(self,img,image_format='BGR'):
        if image_format != 'BGR':raise ValueError('Expected BGR input image.')
        resized=pre_utils.resized_image(img,self.input_w,self.input_h,self.cfg.resize_type)
        return self.input_adapter.build(*pre_utils.bgr_to_nv12_planes(resized))

    def forward(self,inputs):
        outputs=self.model.run(inputs)
        shapes=self.model.output_shapes[self.model_name]
        for name in self.output_names:
            value=outputs[self.model_name][name]
            if tuple(value.shape)!=tuple(shapes[name]) or not np.issubdtype(value.dtype,np.floating):
                raise ValueError(f'YOLO26 expects dequantized floating output {name} with shape {shapes[name]}.')
        return outputs

    def predict(self,img,image_format='BGR',score_thres=None,nms_thres=None):
        if score_thres is not None:self.logit_threshold(score_thres)
        height,width=img.shape[:2]
        return self.post_process(self.forward(self.pre_process(img,image_format)),width,height,score_thres,nms_thres)

    def __call__(self,*args,**kwargs):
        return self.predict(*args,**kwargs)
