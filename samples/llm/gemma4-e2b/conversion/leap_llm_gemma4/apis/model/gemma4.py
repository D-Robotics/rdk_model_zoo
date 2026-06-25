import os
from pathlib import Path

import torch
from hbdk4.compiler import leap
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer

from leap_llm.apis.calibration.data_loader import load_text_data
from leap_llm.models.gemma4.model import Gemma4Text, Gemma4Vision, Gemma4VisionConfig


def _patchify_image(image_tensor, patch_size=16):
    """Convert image tensor [1, 3, H, W] -> [1, num_patches, 3*patch_size^2].

    Extracts non-overlapping patches in row-major order and flattens each patch.
    """
    _, C, H, W = image_tensor.shape
    h_patches = H // patch_size
    w_patches = W // patch_size

    # [1, 3, h_patches, patch_size, w_patches, patch_size]
    patches = image_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # [1, h_patches, w_patches, 3, patch_size, patch_size]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    # [1, num_patches, 3*patch_size^2]
    patches = patches.view(1, h_patches * w_patches, C * patch_size * patch_size)
    return patches


def _load_gemma4_image_data(image_path, h_patches=48, w_patches=48, patch_size=16):
    """Load and preprocess images for Gemma4 vision encoder.

    Resizes to (h_patches*patch_size, w_patches*patch_size), normalizes to [0, 1],
    then patchifies.

    Yields: [1, num_patches, 3*patch_size^2] tensors.
    """
    target_h = h_patches * patch_size
    target_w = w_patches * patch_size

    transform = transforms.Compose([
        transforms.Resize((target_h, target_w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),  # [0, 255] -> [0, 1]
    ])

    if image_path is None:
        image_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "calibration", "calibration_data", "images",
        )

    if os.path.isdir(image_path):
        image_files = sorted(
            p for p in Path(image_path).iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    else:
        image_files = [Path(image_path)]

    for img_path in image_files:
        img = Image.open(str(img_path)).convert("RGB")
        tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
        patches = _patchify_image(tensor, patch_size)  # [1, num_patches, patch_dim]
        yield patches


class Gemma4VisionApi:
    """API for Gemma4 vision encoder quantization and compilation."""

    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        calib_image_path: str = None,
        device: str = "cpu",
        model_type: str = "gemma4-e2b",
        core_num: int = 1,
    ):
        self.input_model_path = input_model_path
        self.output_model_path = output_model_path
        self.calib_image_path = calib_image_path
        self.device = device
        self.model_type = model_type
        self.core_num = core_num

        os.makedirs(output_model_path, exist_ok=True)
        self.vit_file_name = os.path.join(
            output_model_path, f"{model_type}_vit_ptq.hbm"
        )

        # Load model
        self.vit_model = Gemma4Vision.load_model(input_model_path)
        self.config = self.vit_model.config

        # Pre-load calibration images
        self.calib_images = list(
            _load_gemma4_image_data(
                calib_image_path,
                h_patches=self.config.h_patches,
                w_patches=self.config.w_patches,
                patch_size=self.config.patch_size,
            )
        )
        print(f"[Gemma4VisionApi] Loaded {len(self.calib_images)} calibration images")

    def compile(self, **kwargs):
        device = self.device if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        vit = self.vit_model

        # Calibration: forward pass to collect statistics
        vit.set_model_device(device, dtype=dtype)
        vit.set_compile_mode(False)

        print("[Gemma4VisionApi] Running calibration...")
        for patches in tqdm(self.calib_images, desc="Calibrating"):
            vit.forward(patches.to(device, dtype=dtype))

        # Compile
        vit.set_model_device("cpu", dtype=torch.float32)
        vit.set_compile_mode(True)
        vit.compile(
            dtype=leap.float16,
            output_model_path=self.vit_file_name,
            vit_core_num=self.core_num,
            **kwargs,
        )

    def get_hbm_path(self):
        return (self.vit_file_name,)


class Gemma4TextCalibrationDataPreparer:
    def __init__(
        self,
        model_dir: str,
        chunk_size: int,
        cache_len: int,
        sliding_window: int,
        device: str = "cpu",
        mask_value: float = -32768.0,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.chunk_size = chunk_size
        self.cache_len = cache_len
        self.sliding_window = sliding_window
        self.device = device
        self.mask_value = mask_value
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id

    def _build_mask(
        self,
        total_seen: int,
        chunk_start: int,
        chunk_valid: int,
        window: int | None,
    ):
        mask = torch.full(
            (self.chunk_size, self.cache_len),
            self.mask_value,
            dtype=torch.float32,
            device=self.device,
        )
        if total_seen <= 0:
            return mask

        valid_total = min(total_seen, self.cache_len)
        cache_start_abs = total_seen - valid_total
        current_pad = self.chunk_size - chunk_valid
        cache_col_start = self.cache_len - current_pad - valid_total

        for row in range(self.chunk_size):
            query_abs = chunk_start + row
            if query_abs >= total_seen:
                query_abs = total_seen - 1
            allowed_end_abs = query_abs
            allowed_start_abs = cache_start_abs
            if window is not None:
                allowed_start_abs = max(allowed_start_abs, allowed_end_abs - window + 1)

            start_col = cache_col_start + (allowed_start_abs - cache_start_abs)
            end_col = cache_col_start + (allowed_end_abs - cache_start_abs)
            mask[row, start_col : end_col + 1] = 0.0

        return mask

    def prepare_inputs(self, prompt: str):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.cache_len - 10,
        )
        input_ids = inputs.input_ids[0].to(self.device)
        valid_len = input_ids.shape[0]

        padded_len = ((valid_len + self.chunk_size - 1) // self.chunk_size) * self.chunk_size
        if padded_len > valid_len:
            pad = torch.full(
                (padded_len - valid_len,),
                self.pad_token_id,
                dtype=input_ids.dtype,
                device=self.device,
            )
            input_ids = torch.cat([input_ids, pad], dim=0)

        position_ids = torch.arange(valid_len, dtype=torch.int32, device=self.device)
        if padded_len > valid_len:
            pad_value = max(valid_len - 1, 0)
            position_pad = torch.full(
                (padded_len - valid_len,),
                pad_value,
                dtype=torch.int32,
                device=self.device,
            )
            position_ids = torch.cat([position_ids, position_pad], dim=0)

        input_chunks = [chunk.unsqueeze(0) for chunk in input_ids.split(self.chunk_size, dim=0)]
        position_chunks = list(position_ids.split(self.chunk_size, dim=0))

        full_masks = []
        sliding_masks = []
        for chunk_idx in range(len(input_chunks)):
            total_seen = min((chunk_idx + 1) * self.chunk_size, valid_len)
            chunk_start = chunk_idx * self.chunk_size
            chunk_valid = min(valid_len - chunk_start, self.chunk_size)
            full_masks.append(
                self._build_mask(total_seen, chunk_start, chunk_valid, None)
            )
            sliding_masks.append(
                self._build_mask(
                    total_seen,
                    chunk_start,
                    chunk_valid,
                    self.sliding_window,
                )
            )

        return input_chunks, position_chunks, full_masks, sliding_masks


class Gemma4TextApi:
    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        calib_text_path: str = None,
        chunk_size: int = 256,
        cache_len: int = 4096,
        device: str = "cpu",
        model_type: str = "gemma4-e2b",
        prefill_core_num: int = 1,
        decode_core_num: int = 1,
    ):
        self.input_model_path = input_model_path
        self.output_model_path = output_model_path
        self.calib_text_data = list(load_text_data(calib_text_path))
        self.chunk_size = chunk_size
        self.cache_len = cache_len
        self.device = device
        self.model_type = model_type
        self.prefill_core_num = prefill_core_num
        self.decode_core_num = decode_core_num

        os.makedirs(output_model_path, exist_ok=True)
        self.lm_file_name = os.path.join(
            output_model_path,
            f"{model_type}_lm_chunk_{chunk_size}_cache_{cache_len}_ptq.hbm",
        )

        self.text_model = Gemma4Text.load_model(
            input_model_path,
            chunk_size=chunk_size,
            cache_len=cache_len,
        )
        self.calib_preparer = Gemma4TextCalibrationDataPreparer(
            input_model_path,
            chunk_size=chunk_size,
            cache_len=cache_len,
            sliding_window=self.text_model.config.sliding_window,
            device=device,
        )

        tok_embs = (
            self.text_model.model.embed_tokens.weight.data
            * self.text_model.model.embed_tokens.embed_scale
        )
        tok_embs.detach().cpu().numpy().tofile(
            os.path.join(output_model_path, "tok_embeddings.bin")
        )

    def compile(self, **kwargs):
        device = self.device if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        text = self.text_model
        text.set_model_device(device, dtype=dtype)
        text.set_compile_mode(False)

        print(f"[Gemma4TextApi] Loaded {len(self.calib_text_data)} calibration prompts")
        print("[Gemma4TextApi] Running calibration...")
        for prompt in tqdm(self.calib_text_data, desc="Calibrating"):
            input_chunks, position_chunks, full_masks, sliding_masks = self.calib_preparer.prepare_inputs(prompt)
            caches = text.build_empty_caches(device=device, transpose_cache=True)

            for input_ids, position_ids, full_mask, sliding_mask in zip(
                input_chunks,
                position_chunks,
                full_masks,
                sliding_masks,
            ):
                input_ids = input_ids.to(device)
                position_ids = position_ids.to(device)
                full_mask = full_mask.to(device)
                sliding_mask = sliding_mask.to(device)
                inputs_embeds = text.get_input_embeddings(input_ids)
                outputs = text.forward(
                    inputs_embeds,
                    input_ids,
                    position_ids,
                    full_mask,
                    sliding_mask,
                    caches,
                )
                caches = text.update_caches(
                    caches,
                    list(outputs[1:]),
                    input_ids.shape[-1],
                )

        text.set_model_device("cpu", dtype=torch.float32)
        text.set_compile_mode(True)
        text.compile(
            stage="all",
            output_model_path=self.lm_file_name,
            prefill_core_num=self.prefill_core_num,
            decode_core_num=self.decode_core_num,
            **kwargs,
        )

    def get_hbm_path(self):
        return (self.lm_file_name,)
