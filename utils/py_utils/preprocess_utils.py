# Copyright (c) 2025 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# flake8: noqa: E501

import cv2
import numpy as np


def bgr_to_nv12_planes(image: np.ndarray) -> tuple:
    """
    @brief Convert a BGR image to NV12 format (Y and UV planes).
    @param image Input BGR image as a NumPy array of shape (H, W, 3).
    @return A tuple of:
        - y: Y plane with shape (1, H, W, 1)
        - uv: UV plane with shape (1, H/2, W/2, 2)
    """
    height, width = image.shape[:2]
    area = height * width

    # Convert to planar YUV I420 format
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
    yuv420p = yuv420p.reshape((area * 3 // 2,))

    # Extract Y, U, V planes
    y = yuv420p[:area].reshape((height, width))
    u = yuv420p[area:area + area // 4].reshape((height // 2, width // 2))
    v = yuv420p[area + area // 4:].reshape((height // 2, width // 2))

    # Interleave U and V to form UV plane
    uv = np.stack((u, v), axis=-1)

    # Add batch and channel dimensions
    y = y[np.newaxis, :, :, np.newaxis]
    uv = uv[np.newaxis, :, :, :]

    return y, uv


def bgr_to_nv12_continuous(image: np.ndarray) -> np.ndarray:
    """
    @brief Convert a BGR image to continuous NV12 format.
    @param image Input BGR image as a NumPy array of shape (H, W, 3).
    @return NV12 data as a 1D array of shape (H * W * 3 // 2,)
    """
    height, width = image.shape[:2]
    area = height * width

    # Convert to planar YUV I420 format
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
    yuv420p = yuv420p.reshape((area * 3 // 2,))

    # Extract Y, U, V planes
    y = yuv420p[:area]  # shape: (H*W,)
    u = yuv420p[area:area + area // 4]  # shape: (H*W//4,)
    v = yuv420p[area + area // 4:]  # shape: (H*W//4,)

    # 重组为 NV12 格式: Y 平面 + 交错的 UV 平面
    # 将 U 和 V 交错排列: U0, V0, U1, V1, ...
    uv_interleaved = np.empty((area // 2,), dtype=yuv420p.dtype)
    uv_interleaved[0::2] = u  # 偶数位置放 U
    uv_interleaved[1::2] = v  # 奇数位置放 V

    # 合并 Y 和交错的 UV
    nv12_data = np.concatenate([y, uv_interleaved])
    
    return nv12_data

# def resized_image(img: np.ndarray, input_W: int, input_H: int,
#                   resize_type: int = 1,
#                   interpolation=cv2.INTER_NEAREST) -> np.ndarray:
#     """
#     @brief Resize image with either direct resize or letterbox strategy.
#     @param img Input image (H, W, 3).
#     @param input_W Target width.
#     @param input_H Target height.
#     @param resize_type Resize method: 0 for direct resize, 1 for letterbox padding.
#     @param interpolation Interpolation method (default: nearest).
#     @return Resized image with shape (input_H, input_W, 3).
#     """
#     img_h, img_w = img.shape[:2]

#     # 确保目标尺寸是偶数
#     input_W = (input_W // 2) * 2
#     input_H = (input_H // 2) * 2

#     if resize_type == 0:  # Direct resize
#         resized = cv2.resize(img, (input_W, input_H), interpolation=interpolation)
#     elif resize_type == 1:  # Letterbox resize (preserve aspect ratio)
#         scale = min(input_H / img_h, input_W / img_w)
#         new_w, new_h = int(img_w * scale), int(img_h * scale)
        
#         # 确保中间尺寸也是偶数
#         new_w = (new_w // 2) * 2
#         new_h = (new_h // 2) * 2
        
#         resized = cv2.resize(img, (new_w, new_h))

#         pad_w = input_W - new_w
#         pad_h = input_H - new_h
#         left, right = pad_w // 2, pad_w - pad_w // 2
#         top, bottom = pad_h // 2, pad_h - pad_h // 2

#         # Pad image with gray (127,127,127)
#         resized = cv2.copyMakeBorder(resized, top, bottom, left, right,
#                                      borderType=cv2.BORDER_CONSTANT,
#                                      value=(127, 127, 127))
#     else:
#         raise ValueError(f"Invalid resize_type: {resize_type}, must be 0 or 1")

#     return resized

def resized_image(img: np.ndarray, input_W: int, input_H: int,
                  resize_type: int = 1,
                  interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    """
    @brief Resize image with either direct resize or letterbox strategy.
    @param img Input image (H, W, 3).
    @param input_W Target width.
    @param input_H Target height.
    @param resize_type Resize method: 0 for direct resize, 1 for letterbox padding.
    @param interpolation Interpolation method (default: nearest).
    @return Resized image with shape (input_H, input_W, 3).
    """
    img_h, img_w = img.shape[:2]

    input_W = (input_W // 2) * 2
    input_H = (input_H // 2) * 2

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


def split_nv12_bytes(nv12_bytes: bytes, width: int, height: int) -> tuple:
    """
    @brief Split raw NV12 bytes into Y and UV planes.
    @param nv12_bytes Input NV12-encoded byte stream.
    @param width Width of the image.
    @param height Height of the image.
    @return Tuple (y, uv), where:
        - y: shape (H, W), dtype uint8
        - uv: shape (H/2, W), dtype uint8 (interleaved UV)
    """
    y_size = width * height
    uv_size = y_size // 2
    nv12_array = np.frombuffer(nv12_bytes, dtype=np.uint8)

    y = nv12_array[:y_size].reshape((height, width))
    uv = nv12_array[y_size:y_size + uv_size].reshape((height // 2, width))

    return y, uv


def letterbox_resize_gray(gray_img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    @brief Resize a grayscale image using letterbox (aspect ratio preserving) strategy.
    @param gray_img Input grayscale image of shape (H, W).
    @param target_w Target width.
    @param target_h Target height.
    @return Resized and padded grayscale image of shape (target_h, target_w).
    """
    h, w = gray_img.shape
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(gray_img, (new_w, new_h))

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    # Pad with value 127 (gray)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT, value=127)
    return padded


def resize_nv12_yuv(y: np.ndarray, uv: np.ndarray,
                    target_h: int = 672, target_w: int = 672,
                    keep_ratio: bool = True) -> tuple:
    """
    @brief Resize Y and UV planes of an NV12 image to target resolution.
    @param y Y plane of shape (H, W).
    @param uv Interleaved UV plane of shape (H/2, W).
    @param target_h Target height.
    @param target_w Target width.
    @param keep_ratio Whether to preserve aspect ratio (uses letterbox if True).
    @return Tuple of resized:
        - y_resized: shape (target_h, target_w)
        - uv_resized: shape (target_h/2, target_w/2, 2)
    """
    # Resize Y
    if keep_ratio:
        y_resized = letterbox_resize_gray(y, target_w, target_h)
    else:
        y_resized = cv2.resize(y, (target_w, target_h))

    # Split UV into U and V components
    u = uv[:, 0::2]
    v = uv[:, 1::2]

    # Resize U and V separately
    if keep_ratio:
        u_resized = letterbox_resize_gray(u, target_w // 2, target_h // 2)
        v_resized = letterbox_resize_gray(v, target_w // 2, target_h // 2)
    else:
        u_resized = cv2.resize(u, (target_w // 2, target_h // 2))
        v_resized = cv2.resize(v, (target_w // 2, target_h // 2))

    # Re-stack into UV plane
    uv_resized = np.stack((u_resized, v_resized), axis=-1)

    return y_resized, uv_resized


def bgr_to_rgb_tensor(image: np.ndarray, 
                      input_W: int, 
                      input_H: int,
                      resize_type: int = 1,
                      normalization: bool = True,
                      mean: list = [0, 0, 0],
                      std: list = [1, 1, 1]) -> np.ndarray:
    """
    @brief Convert a BGR image to RGB tensor format with optional normalization.
    @param image Input BGR image as a NumPy array of shape (H, W, 3).
    @param input_W Target width.
    @param input_H Target height.
    @param resize_type Resize method: 0 for direct resize, 1 for letterbox padding.
    @param normalization Whether to apply normalization.
    @param mean Mean values for each channel [R, G, B].
    @param std Standard deviation values for each channel [R, G, B].
    @return RGB tensor with shape (1, 3, input_H, input_W) and dtype float32.
    """
    # Resize image first
    resized = resized_image(image, input_W, input_H, resize_type)
    
    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Convert to float32
    rgb_float = rgb_img.astype(np.float32)
    
    # Apply normalization if required
    if normalization:
        rgb_float = (rgb_float - mean) / std
    else:
        # Default normalization to [0, 1] if no custom normalization
        rgb_float = rgb_float / 255.0
    
    # Change from HWC to CHW format
    chw_img = rgb_float.transpose(2, 0, 1)
    
    # Add batch dimension
    batch_img = chw_img[np.newaxis, ...]
    
    return batch_img


def bgr_to_rgb_planar(image: np.ndarray,
                      input_W: int,
                      input_H: int,
                      resize_type: int = 1) -> tuple:
    """
    @brief Convert a BGR image to separate R, G, B planes (for some model requirements).
    @param image Input BGR image as a NumPy array of shape (H, W, 3).
    @param input_W Target width.
    @param input_H Target height.
    @param resize_type Resize method: 0 for direct resize, 1 for letterbox padding.
    @return A tuple of:
        - r: R plane with shape (1, input_H, input_W, 1)
        - g: G plane with shape (1, input_H, input_W, 1) 
        - b: B plane with shape (1, input_H, input_W, 1)
    """
    # Resize image
    resized = resized_image(image, input_W, input_H, resize_type)
    
    # Convert BGR to RGB and split channels
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(rgb_img)
    
    # Add batch and channel dimensions
    r = r[np.newaxis, :, :, np.newaxis]
    g = g[np.newaxis, :, :, np.newaxis] 
    b = b[np.newaxis, :, :, np.newaxis]
    
    return r, g, b


def prepare_rgb_input(image: np.ndarray,
                      input_W: int,
                      input_H: int,
                      resize_type: int = 1,
                      input_format: str = "tensor") -> dict:
    """
    @brief Prepare RGB input in various formats for model inference.
    @param image Input BGR image.
    @param input_W Target width.
    @param input_H Target height.
    @param resize_type Resize method.
    @param input_format Output format: "tensor", "planar", or "nhwc".
    @return Dictionary containing prepared input data.
    """
    if input_format == "tensor":
        # NCHW tensor format [1, 3, H, W]
        tensor = bgr_to_rgb_tensor(image, input_W, input_H, resize_type)
        return {"tensor": tensor}
    
    elif input_format == "planar":
        # Separate R, G, B planes
        r, g, b = bgr_to_rgb_planar(image, input_W, input_H, resize_type)
        return {"r": r, "g": g, "b": b}
    
    elif input_format == "nhwc":
        # NHWC format [1, H, W, 3]
        resized = resized_image(image, input_W, input_H, resize_type)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        nhwc_img = rgb_img[np.newaxis, ...]  # Add batch dimension
        return {"nhwc": nhwc_img}
    
    else:
        raise ValueError(f"Unsupported input_format: {input_format}")


def normalize_tensor(tensor: np.ndarray,
                    mean: list = [0.485, 0.456, 0.406],
                    std: list = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    @brief Apply ImageNet-style normalization to RGB tensor.
    @param tensor Input tensor of shape [1, 3, H, W] in range [0, 1].
    @param mean Mean values for each channel.
    @param std Standard deviation values for each channel.
    @return Normalized tensor.
    """
    mean = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
    
    normalized = (tensor - mean) / std
    return normalized