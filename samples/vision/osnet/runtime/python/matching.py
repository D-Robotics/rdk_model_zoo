"""OSNet RGB embeddings using the input bytes verified on RDK X5.

The RGB tensor reports NHWC dimensions, but its payload is signed planar CHW.
This contract is specific to the audited artifacts, not a general RGB adapter.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image



@dataclass(frozen=True)
class OSNetConfig:
    model_path: Path
    target: str = "x5"
    normalize: bool = True


class OSNet:
    """Extract a 512-dimensional feature from a decoded RGB person crop.

    Input: Pillow image or HxWx3 uint8 RGB array.
    Tensors: nested model/input mapping with signed planar bytes reshaped to
        [1, 256, 128, 3]. No geometry Context is needed for whole-crop features.
    RawOutputs: nested runtime mapping with one float32 512-element output.
    Result: an owned float32 array of shape (512,), optionally L2-normalized.

    Args:
        config: Model path, X5 target selection and normalization policy.
        runtime: Optional injected runtime for SDK-free host tests.

    Raises:
        ValueError: The board or tensor metadata violates this RGB contract.

    ``runtime`` is an SDK-free host-test seam. Production construction checks
    the physical target before loading the SDK; the CLI verifies artifact identity.
    """

    def __init__(self, config: OSNetConfig, *, runtime=None):
        self.config = config
        if config.target not in ("x5", "s100", "s100p", "s600", "auto"):
            raise ValueError("Unknown OSNet target")
        self.target = config.target if config.target != "auto" else "x5"
        if runtime is None:
            from utils.py_utils.platforms import require_execution_target

            actual = require_execution_target(config.target)
            self.target = actual
            from hbm_runtime import HB_HBMRuntime

            runtime = HB_HBMRuntime(str(config.model_path))
        self.runtime = runtime
        if len(runtime.model_names) != 1:
            raise ValueError("OSNet requires exactly one model")
        self.name = runtime.model_names[0]
        if len(runtime.input_names[self.name]) != 1 or len(runtime.output_names[self.name]) != 1:
            raise ValueError("OSNet requires one RGB input and one feature output")
        self.input_name = runtime.input_names[self.name][0]
        self.output_name = runtime.output_names[self.name][0]
        self.shape = tuple(runtime.input_shapes[self.name][self.input_name])
        expected_shape = (1, 256, 128, 3) if self.target == "x5" else (1, 3, 256, 128)
        if self.shape != expected_shape:
            raise ValueError(f"Unexpected OSNet input shape: {self.shape}")
        expected_dtype = "RGB" if self.target == "x5" else "F32"
        if runtime.input_dtypes[self.name][self.input_name].name != expected_dtype:
            raise ValueError(f"OSNet {self.target} requires {expected_dtype} input")
        output_shape = tuple(runtime.output_shapes[self.name][self.output_name])
        if np.prod(output_shape) != 512 or output_shape[0] != 1:
            raise ValueError(f"Unexpected OSNet feature shape: {output_shape}")
        if runtime.output_dtypes[self.name][self.output_name].name != "F32":
            raise ValueError("The audited OSNet artifact requires F32 output")

    def preprocess(self, image):
        """Resize RGB with Pillow bilinear, then pack signed planar bytes."""
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Expected a nonempty HxWx3 uint8 RGB array")
            if not image.shape[0] or not image.shape[1]:
                raise ValueError("Expected a nonempty RGB crop")
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a Pillow image or uint8 RGB array")
        rgb = np.asarray(image.convert("RGB").resize((128, 256), getattr(Image, "Resampling", Image).BILINEAR), np.int16)
        if self.target == "x5":
            data = np.ascontiguousarray((rgb - 128).astype(np.int8).transpose(2, 0, 1)).reshape(self.shape)
        else:
            x = rgb.astype(np.float32).transpose(2, 0, 1) / np.float32(255)
            x = (x - np.array([.485, .456, .406], np.float32)[:, None, None]) / np.array([.229, .224, .225], np.float32)[:, None, None]
            data = np.ascontiguousarray(x[None])
        return {self.name: {self.input_name: data}}

    def infer(self, inputs):
        """Run the board SDK with the nested input mapping."""
        return self.runtime.run(inputs)

    def postprocess(self, outputs):
        """Return an owned F32 feature, L2-normalized by default."""
        raw = np.asarray(outputs[self.name][self.output_name])
        if raw.dtype != np.float32 or raw.size != 512:
            raise ValueError("OSNet output must contain exactly 512 float32 elements")
        feature = raw.reshape(512).copy()
        if not np.isfinite(feature).all():
            raise ValueError("Non-finite OSNet output")
        if self.config.normalize:
            feature /= max(float(np.linalg.norm(feature)), 1e-12)
        return feature

    def predict(self, image):
        """Extract a feature from one already-cropped person image."""
        return self.postprocess(self.infer(self.preprocess(image)))

    __call__ = predict
