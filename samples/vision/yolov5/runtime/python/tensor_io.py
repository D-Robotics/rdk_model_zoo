# Copyright (c) 2025 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Per-call YOLOv5 image geometry and explicit packed/split NV12 transport."""
from dataclasses import dataclass
import cv2
import numpy as np
from samples._shared.image import bgr_to_nv12_planes

def resized_image(img: np.ndarray, input_W: int, input_H: int,
                  resize_type: int = 1,
                  interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    """Resize an image using direct resize or letterbox strategy.

    This function resizes the input image to the target resolution required
    by the model. It supports either direct resizing or letterbox resizing
    with padding to preserve the aspect ratio.

    Args:
        img: Input image array with shape `(H, W, 3)`.
        input_W: Target image width.
        input_H: Target image height.
        resize_type: Resize strategy used during preprocessing.
            - 0: Direct resize.
            - 1: Letterbox resize with padding to preserve aspect ratio.
        interpolation: OpenCV interpolation method used for resizing.

    Returns:
        The resized image with shape `(input_H, input_W, 3)`.

    Raises:
        ValueError: If an invalid `resize_type` is provided.
    """
    img_h, img_w = img.shape[:2]

    if resize_type == 0:  # Direct resize
        resized = cv2.resize(img, (input_W, input_H), interpolation=interpolation)
    elif resize_type == 1:  # Letterbox resize (preserve aspect ratio)
        scale = min(input_H / img_h, input_W / img_w)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        pad_w = input_W - new_w
        pad_h = input_H - new_h
        left, right = pad_w // 2, pad_w - pad_w // 2
        top, bottom = pad_h // 2, pad_h - pad_h // 2

        # Pad image with gray (127,127,127)
        resized = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=(127, 127, 127))
    else:
        raise ValueError(f"Invalid resize_type: {resize_type}, must be 0 or 1")

    return resized


@dataclass(frozen=True)
class DetectionContext:
    """Original H/W, bound network size and source inverse-resize mode, per call."""
    original_size: tuple[int, int]
    model_size: int
    resize_type: int

    def __post_init__(self):
        if not isinstance(self.original_size,tuple) or len(self.original_size)!=2 or any(type(v) is not int or v<1 for v in self.original_size):
            raise ValueError('original_size must be a positive (height, width) tuple.')
        if type(self.model_size) is not int or self.model_size<2 or self.model_size%2:
            raise ValueError('model_size must be a positive even integer.')
        if type(self.resize_type) is not int or self.resize_type not in (0,1):
            raise ValueError('resize_type must be 0 or 1.')


@dataclass(frozen=True)
class PreparedInput:
    """Owned uint8 flat packed NV12 or Y/UV tensor map plus immutable geometry."""
    tensors: dict[str, np.ndarray]
    context: DetectionContext


def prepare_image(image, binding, resize_type=None):
    """Prepare positive HWC uint8 BGR input; mode 0 nearest, mode 1 source letterbox.

    X5 returns flat uint8 H*W*3/2; S returns uint8 Y (1,H,W,1) and UV
    (1,H/2,W/2,2). Letterbox uses linear resize, integer floor dimensions,
    symmetric 127 padding; inverse coordinates retain source ideal scale.
    Raises ValueError for invalid pixels/mode or degenerate resized geometry.
    """
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<1:
        raise ValueError('Expected nonempty HWC uint8 BGR image.')
    resize=binding.resize_type if resize_type is None else resize_type
    if type(resize) is not int or resize not in (0,1):raise ValueError('resize_type must be 0 or 1.')
    size=binding.input_size
    if resize==1 and min(int(n*min(size/image.shape[0],size/image.shape[1])) for n in image.shape[:2])<1:
        raise ValueError('Image aspect ratio produces an empty letterbox dimension.')
    pixels=resized_image(image,size,size,resize)
    y,uv=bgr_to_nv12_planes(pixels)
    names=binding.input_names
    tensors=({names[0]:np.concatenate((y.reshape(-1),uv.reshape(-1)))} if binding.selection.target=='x5' else {names[0]:y,names[1]:uv})
    return PreparedInput(tensors,DetectionContext(tuple(image.shape[:2]),size,resize))
