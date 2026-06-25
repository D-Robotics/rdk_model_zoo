import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from hbdk4.compiler import leap, save
from safetensors import safe_open
from torch import nn

from leap_llm.nn.modules import (
    ConstFakeQuant,
    FakeQuantAdd,
    FakeQuantLinear,
    FakeQuantMul,
)
from leap_llm.nn.modules.ops import FakeQuantReduceMean
from leap_llm.nn.modules.rms_norm import FakeQuantRMSNorm
from leap_llm.nn.utils import Model, Module, timeit

from .blocks import (
    Gemma4TextDecoderLayer,
    Gemma4VisionAttention,
    Gemma4VisionEncoderLayer,
    Gemma4VisionMLP,
)


# ============================================================================
# Config
# ============================================================================


@dataclass
class Gemma4VisionConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    rms_norm_eps: float = 1e-6
    patch_size: int = 16
    pooling_kernel_size: int = 3
    position_embedding_size: int = 10240
    rope_theta: float = 100.0
    # Image grid: h_patches * w_patches = num_patches
    # Default: 768x768 image -> 48x48 patches = 2304
    # E4B: 896x896 image -> 56x56 patches = 3136, output = 280
    h_patches: int = 48
    w_patches: int = 48
    default_output_length: int = 256
    # embed_vision projector output dim (text hidden_size)
    text_hidden_size: int = 1536


# ============================================================================
# Patch Embedding
# ============================================================================


class PatchEmbedding(Module):
    """Linear patch projection + 2D learned position embedding.

    Input: pre-patchified pixel_values [num_patches, 3*patch_size^2]
    Output: [num_patches, hidden_size]
    """

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        patch_dim = 3 * config.patch_size ** 2
        self.input_proj = FakeQuantLinear(patch_dim, config.hidden_size, bias=False)
        self.input_norm = ConstFakeQuant(8)

        # Position embedding table: [2, position_embedding_size, hidden_size]
        # Will be loaded from checkpoint
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, config.position_embedding_size, config.hidden_size)
        )

        # Pre-computed position embeddings buffer (set in load_model)
        self.register_buffer("position_embeddings", None)

    def _compute_position_embeddings(self, pixel_position_ids):
        """Pre-compute position embeddings for fixed resolution.

        pixel_position_ids: [num_patches, 2] (x, y coordinates)
        Returns: [num_patches, hidden_size]
        """
        clamped = pixel_position_ids.clamp(min=0)
        one_hot = F.one_hot(
            clamped, num_classes=self.config.position_embedding_size
        )  # [num_patches, 2, pos_emb_size]
        one_hot = one_hot.permute(1, 0, 2).float()  # [2, num_patches, pos_emb_size]
        # [2, num_patches, pos_emb_size] @ [2, pos_emb_size, hidden_size]
        pos_emb = torch.bmm(one_hot, self.position_embedding_table.data.float())
        pos_emb = pos_emb.sum(dim=0)  # [num_patches, hidden_size]
        return pos_emb

    def build(self, pixel_values):
        """pixel_values: [num_patches, patch_dim] already normalized to [0, 1]"""
        # Normalize: 2*(x - 0.5) = 2x - 1, maps [0,1] to [-1,1]
        x = leap.mul(pixel_values, 2.0)
        x = leap.add(x, -1.0)
        x = self.input_norm(x)
        hidden_states = self.input_proj(x)
        # Add pre-computed position embeddings
        pos_emb = self.position_embeddings.data.to(torch.float32)
        hidden_states = leap.add(hidden_states, pos_emb)
        return hidden_states

    def forward(self, pixel_values):
        """pixel_values: [1, num_patches, patch_dim]"""
        x = 2 * (pixel_values - 0.5)
        x = self.input_norm(x)
        hidden_states = self.input_proj(x)
        if self.position_embeddings is not None:
            hidden_states = hidden_states + self.position_embeddings.unsqueeze(0)
        return hidden_states


# ============================================================================
# 2D Rotary Embedding
# ============================================================================


class VisionRotaryEmbedding:
    """Pre-computes 2D rotary position embeddings for the vision encoder.

    Not a Module - just a utility to compute cos/sin constants.
    """

    @staticmethod
    def compute(config: Gemma4VisionConfig):
        """Compute cos/sin for all patch positions.

        Returns: cos [num_patches, head_dim], sin [num_patches, head_dim]
        """
        h, w = config.h_patches, config.w_patches
        head_dim = config.head_dim
        spatial_dim = head_dim // 2  # 32
        base = config.rope_theta

        # inv_freq: [spatial_dim // 2] = [16]
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, spatial_dim, 2, dtype=torch.float32) / spatial_dim
            )
        )

        # Position IDs: [num_patches, 2] as (x, y)
        positions_x = torch.arange(w).repeat(h)  # [num_patches]
        positions_y = torch.arange(h).repeat_interleave(w)  # [num_patches]

        all_cos, all_sin = [], []
        for positions in [positions_x, positions_y]:
            # [num_patches, 1] @ [1, spatial_dim//2] -> [num_patches, spatial_dim//2]
            freqs = positions.float().unsqueeze(1) * inv_freq.unsqueeze(0)
            emb = torch.cat([freqs, freqs], dim=-1)  # [num_patches, spatial_dim]
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())

        cos = torch.cat(all_cos, dim=-1)  # [num_patches, head_dim]
        sin = torch.cat(all_sin, dim=-1)  # [num_patches, head_dim]
        return cos, sin

    @staticmethod
    def compute_pixel_position_ids(config: Gemma4VisionConfig):
        """Compute pixel_position_ids for fixed resolution.

        Returns: [num_patches, 2] as (x, y) coordinates.
        """
        h, w = config.h_patches, config.w_patches
        xs = torch.arange(w).repeat(h)
        ys = torch.arange(h).repeat_interleave(w)
        return torch.stack([xs, ys], dim=-1)


# ============================================================================
# Pooler
# ============================================================================


class VisionPooler(Module):
    """Spatial average pooling + scaling.

    For fixed resolution: uniform k x k average pooling.
    output_length = num_patches / (k * k)
    """

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.root_hidden_size = config.hidden_size ** 0.5
        self.k = config.pooling_kernel_size
        k2 = self.k * self.k

        h, w = config.h_patches, config.w_patches
        self.num_patches = h * w
        self.output_length = self.num_patches // k2
        self.h_out = h // self.k
        self.w_out = w // self.k
        self.pool = FakeQuantReduceMean(quantized=False)

        # Pre-compute pooling weight matrix: [output_length, num_patches]
        # For a regular grid, each output token averages k*k input patches
        self.register_buffer("pool_weights", None)
        self._precompute_pool_weights(config)

    def _precompute_pool_weights(self, config):
        h, w = config.h_patches, config.w_patches
        k = self.k
        k2 = k * k
        h_out = h // k
        w_out = w // k
        output_length = h_out * w_out

        weights = torch.zeros(output_length, h * w)
        for oy in range(h_out):
            for ox in range(w_out):
                out_idx = oy * w_out + ox
                for dy in range(k):
                    for dx in range(k):
                        iy = oy * k + dy
                        ix = ox * k + dx
                        in_idx = iy * w + ix
                        weights[out_idx, in_idx] = 1.0 / k2
        self.pool_weights = weights

    def build(self, hidden_states):
        """hidden_states: [num_patches, hidden_size]"""
        hidden_states = leap.reshape(
            hidden_states,
            [self.h_out, self.k, self.w_out, self.k, self.hidden_size],
        )
        pooled = self.pool(hidden_states, [1, 3])
        pooled = leap.reshape(pooled, [self.output_length, self.hidden_size])
        pooled = leap.mul(pooled, self.root_hidden_size)
        return pooled

    def forward(self, hidden_states):
        """hidden_states: [1, num_patches, hidden_size]"""
        h = hidden_states.squeeze(0).float()
        pooled = h.reshape(
            self.h_out, self.k, self.w_out, self.k, self.hidden_size
        ).mean(dim=(1, 3))
        pooled = pooled.reshape(1, self.output_length, self.hidden_size)
        pooled = pooled * self.root_hidden_size
        return pooled.to(hidden_states.dtype)


# ============================================================================
# Vision Projector (embed_vision)
# ============================================================================


class VisionProjector(Module):
    """Projects vision hidden states to text hidden space.

    RMSNorm (no scale) + Linear projection.
    """

    def __init__(self, vision_hidden_size, text_hidden_size, rms_norm_eps):
        super().__init__()
        self.norm = FakeQuantRMSNorm(
            vision_hidden_size, eps=rms_norm_eps, fuse_norm=True
        )
        self.projection = FakeQuantLinear(
            vision_hidden_size, text_hidden_size, bias=False
        )

    def build(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        return self.projection(hidden_states)

    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        return self.projection(hidden_states)


# ============================================================================
# Full Vision Model
# ============================================================================


class Gemma4VisionModel(Model):
    """Gemma4 Vision Encoder: PatchEmbedder + Encoder + Pooler + Projector.

    Compiled model input: pixel_values [num_patches, patch_dim]
    Compiled model output: [output_length, text_hidden_size]
    """

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.num_patches = config.h_patches * config.w_patches

        self.patch_embedder = PatchEmbedding(config)
        self.layers = nn.ModuleList(
            [
                Gemma4VisionEncoderLayer(
                    layer_id=i,
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    head_dim=config.head_dim,
                    intermediate_size=config.intermediate_size,
                    rms_norm_eps=config.rms_norm_eps,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.pooler = VisionPooler(config)
        self.projector = VisionProjector(
            config.hidden_size, config.text_hidden_size, config.rms_norm_eps
        )

        # Pre-computed RoPE buffers
        self.register_buffer("rope_cos", None)
        self.register_buffer("rope_sin", None)

    def _init_constants(self):
        """Pre-compute position embeddings and RoPE for fixed resolution."""
        config = self.config

        # Position embeddings for patch embedder
        pos_ids = VisionRotaryEmbedding.compute_pixel_position_ids(config)
        pos_emb = self.patch_embedder._compute_position_embeddings(pos_ids)
        self.patch_embedder.position_embeddings = pos_emb

        # RoPE cos/sin
        cos, sin = VisionRotaryEmbedding.compute(config)
        self.rope_cos = cos
        self.rope_sin = sin

    def build(self, pixel_values):
        """
        pixel_values: [num_patches, patch_dim]
        Returns: [output_length, text_hidden_size]
        """
        # Cast to float32 to match FakeQuantLinear weight dtype (avoids si8:f16 vs si8:f32)
        pixel_values = leap.cast_type(pixel_values, output_type=leap.float32)

        # Patch embedding + position embedding
        hidden_states = self.patch_embedder(pixel_values)

        # Encoder layers
        cos = self.rope_cos.data.to(torch.float32)
        sin = self.rope_sin.data.to(torch.float32)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)

        # Pooler
        hidden_states = self.pooler(hidden_states)

        # Vision projector
        hidden_states = self.projector(hidden_states)

        return hidden_states

    def forward(self, pixel_values):
        """
        pixel_values: [1, num_patches, patch_dim]
        Returns: [1, output_length, text_hidden_size]
        """
        hidden_states = self.patch_embedder(pixel_values)

        cos = self.rope_cos.unsqueeze(0).to(pixel_values.device)
        sin = self.rope_sin.unsqueeze(0).to(pixel_values.device)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)

        hidden_states = self.pooler(hidden_states)
        hidden_states = self.projector(hidden_states)

        return hidden_states


# ============================================================================
# Wrapper (load_model / compile / etc.)
# ============================================================================


class Gemma4Vision:
    """Wrapper for Gemma4 Vision model: loading, calibration, compilation."""

    def __init__(self, model: Gemma4VisionModel, config: Gemma4VisionConfig):
        self.model = model
        self.config = config

    @staticmethod
    def load_model(model_path, checkpoint=None, config=None):
        """Load Gemma4 vision model from checkpoint.

        Args:
            model_path: path to the model directory
            checkpoint: pre-loaded state dict (optional)
            config: Gemma4VisionConfig (optional, auto-detected from checkpoint)
        """
        import json

        if config is None:
            config_path = os.path.join(model_path, "config.json")
            with open(config_path) as f:
                raw = json.load(f)
            vc = raw.get("vision_config", {})

            default_output_length = vc.get("default_output_length", 256)
            patch_size = vc.get("patch_size", 16)
            pooling_kernel_size = vc.get("pooling_kernel_size", 3)

            k = pooling_kernel_size
            k2 = k * k
            num_patches_before_pool = default_output_length * k2

            best_h, best_w = None, None
            for h in range(1, int(num_patches_before_pool ** 0.5) + 1):
                if num_patches_before_pool % h == 0:
                    w = num_patches_before_pool // h
                    if h % k == 0 and w % k == 0:
                        if best_h is None or abs(h - w) < abs(best_h - best_w):
                            best_h, best_w = h, w
            h_patches, w_patches = best_h, best_w

            config = Gemma4VisionConfig(
                hidden_size=vc.get("hidden_size", 768),
                intermediate_size=vc.get("intermediate_size", 3072),
                num_hidden_layers=vc.get("num_hidden_layers", 16),
                num_attention_heads=vc.get("num_attention_heads", 12),
                num_key_value_heads=vc.get("num_key_value_heads", 12),
                head_dim=vc.get("head_dim", 64),
                rms_norm_eps=vc.get("rms_norm_eps", 1e-6),
                patch_size=patch_size,
                pooling_kernel_size=pooling_kernel_size,
                position_embedding_size=vc.get("position_embedding_size", 10240),
                rope_theta=vc.get("rope_parameters", {}).get("rope_theta", 100.0),
                h_patches=h_patches,
                w_patches=w_patches,
                default_output_length=default_output_length,
                text_hidden_size=raw.get("text_config", {}).get("hidden_size", 1536),
            )

        model = Gemma4VisionModel(config)

        if checkpoint is None:
            checkpoint = Gemma4Vision._load_checkpoint(model_path)

        # Map weights
        mapped = Gemma4Vision._map_weights(checkpoint, config)
        missing, unexpected = model.load_state_dict(mapped, strict=False)

        # Filter out expected missing keys (pre-computed buffers + no-scale norms)
        expected_missing = {
            "patch_embedder.position_embeddings",
            "rope_cos",
            "rope_sin",
            "pooler.pool_weights",
            "projector.norm.weight",  # HF with_scale=False, our FakeQuantRMSNorm has weight=ones
        }
        # v_norm weights: HF with_scale=False, our FakeQuantRMSNorm keeps weight=ones
        for i in range(config.num_hidden_layers):
            expected_missing.add(f"layers.{i}.self_attn.v_norm.weight")
        real_missing = [k for k in missing if k not in expected_missing]
        if real_missing:
            print(f"[Gemma4Vision] Warning: missing keys: {real_missing}")
        if unexpected:
            print(f"[Gemma4Vision] Warning: unexpected keys: {unexpected[:10]}...")

        # Initialize pre-computed constants
        model._init_constants()

        wrapper = Gemma4Vision(model, config)
        return wrapper

    @staticmethod
    def _load_checkpoint(model_path):
        """Load from safetensors or pytorch_model.bin."""
        from safetensors import safe_open

        st_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(st_path):
            checkpoint = {}
            with safe_open(st_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    checkpoint[key] = f.get_tensor(key)
            return checkpoint

        pt_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(pt_path):
            return torch.load(pt_path, map_location="cpu")

        raise FileNotFoundError(
            f"No model file found at {model_path} "
            "(expected model.safetensors or pytorch_model.bin)"
        )

    @staticmethod
    def _map_weights(checkpoint, config):
        """Map HF checkpoint keys to leap_llm model keys."""
        mapped = {}

        # Prefixes to strip
        vt_prefix = "model.vision_tower."
        ev_prefix = "model.embed_vision."

        for key, tensor in checkpoint.items():
            new_key = None

            if key.startswith(vt_prefix):
                new_key = key[len(vt_prefix) :]
            elif key.startswith(ev_prefix):
                suffix = key[len(ev_prefix) :]
                # embed_vision.embedding_projection.weight -> projector.projection.weight
                if "embedding_projection.weight" in suffix:
                    new_key = "projector.projection.weight"
                # embedding_pre_projection_norm has no weight (with_scale=False)
                else:
                    continue
            else:
                continue

            if new_key is None:
                continue

            # Skip ClippableLinear clipping buffers
            if any(
                s in new_key
                for s in ["input_min", "input_max", "output_min", "output_max"]
            ):
                continue

            # Map .linear.weight -> .weight (ClippableLinear -> FakeQuantLinear)
            new_key = new_key.replace(".linear.weight", ".weight")

            # Strip encoder. prefix: encoder.layers.X -> layers.X
            if new_key.startswith("encoder."):
                new_key = new_key[len("encoder.") :]

            # Convert bfloat16 to float32 for loading
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()

            mapped[new_key] = tensor

        return mapped

    def set_compile_mode(self, compiled: bool):
        self.model.compile_mode(compiled)

    def set_model_device(self, device, dtype=None):
        if dtype is not None:
            self.model = self.model.to(device=device, dtype=dtype)
        else:
            self.model = self.model.to(device=device)

    def forward(self, pixel_values):
        """Run PyTorch forward pass for calibration."""
        with torch.no_grad():
            return self.model(pixel_values)

    def get_leap_inputs(self, dtype):
        """Return leap input types for compilation."""
        patch_dim = 3 * self.config.patch_size ** 2
        num_patches = self.config.h_patches * self.config.w_patches
        return [leap.TensorType([num_patches, patch_dim], dtype)]

    def compile(self, dtype, output_model_path, vit_core_num=1, **kwargs):
        """Export -> convert_mlir -> compile_hbo -> link."""
        from pathlib import Path

        from leap_llm.nn.utils import statistics

        model = self.model
        assert model.is_compiled, "Model must be in compile mode before compiling."

        kwargs["core_num"] = vit_core_num
        if vit_core_num > 1:
            kwargs["max_l2m_size"] = 25165824

        inputs = self.get_leap_inputs(dtype)

        # Step 1: Export
        bc_path = str(Path(output_model_path).with_suffix(".bc"))
        print("[Gemma4Vision] Exporting...")
        bc_module = model.export_module(inputs, "Gemma4VisionModel", bc_path)

        # Step 2: Convert MLIR
        convert_bc_path = str(Path(output_model_path).with_suffix(".convert.bc"))
        print("[Gemma4Vision] Converting MLIR...")
        mlir_module = model.convert_mlir(
            bc_module,
            save_path=convert_bc_path,
            march=kwargs["march"],
        )
        statistics(mlir_module)

        # Step 3: Compile HBO
        hbo_path = str(Path(output_model_path).with_suffix(".hbo"))
        print(f"[Gemma4Vision] Compiling HBO (core_num={vit_core_num})...")
        hbo_model = model.compile_hbo(mlir_module, hbo_path, **kwargs)

        # Step 4: Link
        hbm_path = output_model_path
        if not hbm_path.endswith(".hbm"):
            hbm_path = str(Path(output_model_path).with_suffix(".hbm"))
        print("[Gemma4Vision] Linking HBM...")
        model.link_models([hbo_model], hbm_path)
        print(f"[Gemma4Vision] Done: {hbm_path}")


# ============================================================================
# Text Config
# ============================================================================


@dataclass
class Gemma4TextConfig:
    hidden_size: int = 1536
    intermediate_size: int = 6144
    num_attention_heads: int = 8
    num_hidden_layers: int = 35
    num_key_value_heads: int = 1
    vocab_size: int = 262144
    vocab_size_per_layer_input: int = 262144
    hidden_size_per_layer_input: int = 256
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768
    sliding_window: int = 512
    head_dim: int = 256
    global_head_dim: int = 512
    num_kv_shared_layers: int = 20
    use_double_wide_mlp: bool = True
    full_rope_theta: float = 1000000.0
    sliding_rope_theta: float = 10000.0
    partial_rotary_factor: float = 0.25
    prefill_seq_len: int = 256
    decode_seq_len: int = 1
    chunked_sv_start: int = 24
    layer_types: tuple[str, ...] = field(
        default_factory=lambda: (
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        )
    )


# ============================================================================
# Text Embedding / Head
# ============================================================================


class Gemma4TextScaledEmbedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, embed_scale: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.embed_scale = embed_scale
        self.weight_fake_quant = ConstFakeQuant(8)
        self.scale_mul = FakeQuantMul(quantized=False)

    def build(self, x):
        weight = self.weight_fake_quant(self.weight.data)
        embeds = leap.gather_nd(weight, x, 0)
        return self.scale_mul(embeds, self.embed_scale)

    def forward(self, x):
        weight = self.weight_fake_quant(self.weight.data)
        embeds = weight[x]
        return self.scale_mul(embeds, self.embed_scale)


class Gemma4TextLMHead(FakeQuantLinear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)


# ============================================================================
# Text Model
# ============================================================================


class Gemma4TextModel(Model):
    def __init__(self, config: Gemma4TextConfig, cache_len: int):
        super().__init__()
        self.config = config
        self.cache_len = cache_len
        self.first_kv_shared_layer_idx = (
            config.num_hidden_layers - config.num_kv_shared_layers
        )
        self.non_shared_layer_indices = list(range(self.first_kv_shared_layer_idx))
        self.non_shared_head_dims = [
            self._layer_head_dim(i) for i in self.non_shared_layer_indices
        ]

        prev_layers = list(config.layer_types[: self.first_kv_shared_layer_idx])
        self.shared_source_by_type = {
            layer_type: len(prev_layers) - 1 - prev_layers[::-1].index(layer_type)
            for layer_type in set(prev_layers)
        }

        self.embed_tokens = Gemma4TextScaledEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.hidden_size**0.5,
        )
        self.embed_tokens_per_layer = Gemma4TextScaledEmbedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            config.hidden_size_per_layer_input**0.5,
        )
        self.per_layer_model_projection = FakeQuantLinear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_projection_norm = FakeQuantRMSNorm(
            config.hidden_size_per_layer_input,
            eps=config.rms_norm_eps,
        )
        self.per_layer_projection_mul = FakeQuantMul(quantized=False)
        self.per_layer_add = FakeQuantAdd(quantized=False)
        self.per_layer_input_scale_mul = FakeQuantMul(quantized=False)

        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            is_kv_shared_layer = layer_idx >= self.first_kv_shared_layer_idx > 0
            store_full_length_kv = (
                not is_kv_shared_layer
                and layer_idx == self.shared_source_by_type[config.layer_types[layer_idx]]
            )
            self.layers.append(
                Gemma4TextDecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                    head_dim=self._layer_head_dim(layer_idx),
                    is_kv_shared_layer=is_kv_shared_layer,
                    store_full_length_kv=store_full_length_kv,
                )
            )

        self.norm = FakeQuantRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Gemma4TextLMHead(config.hidden_size, config.vocab_size)

        self.per_layer_projection_scale = config.hidden_size**-0.5
        self.per_layer_input_scale = 2.0**-0.5

        self.register_buffer("full_rope_cos", None)
        self.register_buffer("full_rope_sin", None)
        self.register_buffer("sliding_rope_cos", None)
        self.register_buffer("sliding_rope_sin", None)

        self.full_cos_fq = ConstFakeQuant(16)
        self.full_sin_fq = ConstFakeQuant(16)
        self.sliding_cos_fq = ConstFakeQuant(16)
        self.sliding_sin_fq = ConstFakeQuant(16)
        self.full_mask_fq = ConstFakeQuant(16)
        self.sliding_mask_fq = ConstFakeQuant(16)

    def _layer_head_dim(self, layer_idx: int) -> int:
        if self.config.layer_types[layer_idx] == "full_attention":
            return self.config.global_head_dim
        return self.config.head_dim

    def _build_rope_cache(self, head_dim: int, theta: float, proportion: float):
        rope_angles = int(proportion * head_dim // 2)
        inv_freq_rotated = 1.0 / (
            theta
            ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim)
        )
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat(
                [inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32)],
                dim=0,
            )
        else:
            inv_freq = inv_freq_rotated

        positions = torch.arange(self.config.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[: self.cache_len], emb.sin()[: self.cache_len]

    def _init_constants(self):
        full_cos, full_sin = self._build_rope_cache(
            self.config.global_head_dim,
            self.config.full_rope_theta,
            self.config.partial_rotary_factor,
        )
        sliding_cos, sliding_sin = self._build_rope_cache(
            self.config.head_dim,
            self.config.sliding_rope_theta,
            1.0,
        )
        self.full_rope_cos = full_cos
        self.full_rope_sin = full_sin
        self.sliding_rope_cos = sliding_cos
        self.sliding_rope_sin = sliding_sin

    def _build_per_layer_inputs(self, inputs_embeds, token_ids):
        seq_len = inputs_embeds.type.shape[0]
        per_layer_inputs = self.embed_tokens_per_layer(token_ids)
        per_layer_inputs = leap.reshape(
            per_layer_inputs,
            [seq_len, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input],
        )

        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection = self.per_layer_projection_mul(
            per_layer_projection,
            self.per_layer_projection_scale,
        )
        per_layer_projection = leap.reshape(
            per_layer_projection,
            [seq_len, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input],
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        per_layer_inputs = self.per_layer_add(per_layer_projection, per_layer_inputs)
        return self.per_layer_input_scale_mul(
            per_layer_inputs,
            self.per_layer_input_scale,
        )

    def _forward_per_layer_inputs(self, inputs_embeds, token_ids):
        seq_len = inputs_embeds.shape[0]
        per_layer_inputs = self.embed_tokens_per_layer(token_ids)
        per_layer_inputs = per_layer_inputs.reshape(
            seq_len,
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )

        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection = self.per_layer_projection_mul(
            per_layer_projection,
            self.per_layer_projection_scale,
        )
        per_layer_projection = per_layer_projection.reshape(
            seq_len,
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        per_layer_inputs = self.per_layer_add(per_layer_projection, per_layer_inputs)
        return self.per_layer_input_scale_mul(
            per_layer_inputs,
            self.per_layer_input_scale,
        )

    def build(self, inputs_embeds, token_ids, position_ids, full_mask, sliding_mask, *caches):
        seq_len = inputs_embeds.type.shape[0]
        token_ids = leap.reshape(token_ids, [seq_len, 1])
        position_ids = leap.reshape(position_ids, [seq_len, 1])

        full_cos = leap.gather_nd(self.full_rope_cos, position_ids, 0)
        full_sin = leap.gather_nd(self.full_rope_sin, position_ids, 0)
        sliding_cos = leap.gather_nd(self.sliding_rope_cos, position_ids, 0)
        sliding_sin = leap.gather_nd(self.sliding_rope_sin, position_ids, 0)

        full_cos = self.full_cos_fq(full_cos)
        full_sin = self.full_sin_fq(full_sin)
        sliding_cos = self.sliding_cos_fq(sliding_cos)
        sliding_sin = self.sliding_sin_fq(sliding_sin)
        full_mask = self.full_mask_fq(full_mask)
        sliding_mask = self.sliding_mask_fq(sliding_mask)

        per_layer_inputs = self._build_per_layer_inputs(inputs_embeds, token_ids)

        hidden_states = inputs_embeds
        non_shared_layer_num = len(self.non_shared_layer_indices)
        caches_k = caches[:non_shared_layer_num]
        caches_v = caches[non_shared_layer_num:]
        shared_caches = {}
        new_keys = []
        new_values = []
        cache_idx = 0

        for layer_idx, layer in enumerate(self.layers):
            per_layer_input = leap.slice(
                per_layer_inputs,
                [0, layer_idx, 0],
                [seq_len, layer_idx + 1, self.config.hidden_size_per_layer_input],
                [1, 1, 1],
            )
            per_layer_input = leap.reshape(
                per_layer_input,
                [seq_len, self.config.hidden_size_per_layer_input],
            )

            if layer.is_sliding:
                cos = sliding_cos
                sin = sliding_sin
                mask = sliding_mask
            else:
                cos = full_cos
                sin = full_sin
                mask = full_mask

            if layer.is_kv_shared_layer:
                shared_cache_k, shared_cache_v = shared_caches[layer.layer_type]
                hidden_states, _, _, _, _ = layer(
                    hidden_states,
                    per_layer_input,
                    cos,
                    sin,
                    mask,
                    shared_cache_k=shared_cache_k,
                    shared_cache_v=shared_cache_v,
                )
                continue

            cache_k = leap.transpose(caches_k[cache_idx], [1, 0, 2])
            cache_v = leap.transpose(caches_v[cache_idx], [1, 0, 2])
            hidden_states, new_k, new_v, full_k, full_v = layer(
                hidden_states,
                per_layer_input,
                cos,
                sin,
                mask,
                cache_k=cache_k,
                cache_v=cache_v,
            )

            if layer.store_full_length_kv:
                shared_caches[layer.layer_type] = (full_k, full_v)

            new_keys.append(leap.transpose(new_k, [1, 0, 2]))
            new_values.append(leap.transpose(new_v, [1, 0, 2]))
            cache_idx += 1

        hidden_states = self.norm(hidden_states)
        hidden_states = leap.reshape(hidden_states, [1, seq_len, self.config.hidden_size])
        logits = self.lm_head(hidden_states)
        return logits, *new_keys, *new_values

    def forward(
        self,
        inputs_embeds,
        token_ids,
        position_ids,
        full_mask,
        sliding_mask,
        caches,
    ):
        hidden_states = inputs_embeds.squeeze(0)
        token_ids = token_ids.squeeze(0).long()
        position_ids = position_ids.long()

        full_cos = self.full_cos_fq(self.full_rope_cos.to(hidden_states.device)[position_ids])
        full_sin = self.full_sin_fq(self.full_rope_sin.to(hidden_states.device)[position_ids])
        sliding_cos = self.sliding_cos_fq(
            self.sliding_rope_cos.to(hidden_states.device)[position_ids]
        )
        sliding_sin = self.sliding_sin_fq(
            self.sliding_rope_sin.to(hidden_states.device)[position_ids]
        )
        full_mask = self.full_mask_fq(full_mask)
        sliding_mask = self.sliding_mask_fq(sliding_mask)

        per_layer_inputs = self._forward_per_layer_inputs(hidden_states, token_ids)

        non_shared_layer_num = len(self.non_shared_layer_indices)
        caches_k = caches[:non_shared_layer_num]
        caches_v = caches[non_shared_layer_num:]
        shared_caches = {}
        new_keys = []
        new_values = []
        cache_idx = 0

        for layer_idx, layer in enumerate(self.layers):
            per_layer_input = per_layer_inputs[:, layer_idx, :]

            if layer.is_sliding:
                cos = sliding_cos
                sin = sliding_sin
                mask = sliding_mask
            else:
                cos = full_cos
                sin = full_sin
                mask = full_mask

            if layer.is_kv_shared_layer:
                shared_cache_k, shared_cache_v = shared_caches[layer.layer_type]
                hidden_states, _, _, _, _ = layer(
                    hidden_states,
                    per_layer_input,
                    cos,
                    sin,
                    mask,
                    shared_cache_k=shared_cache_k,
                    shared_cache_v=shared_cache_v,
                )
                continue

            cache_k = caches_k[cache_idx].transpose(1, 0)
            cache_v = caches_v[cache_idx].transpose(1, 0)
            hidden_states, new_k, new_v, full_k, full_v = layer(
                hidden_states,
                per_layer_input,
                cos,
                sin,
                mask,
                cache_k=cache_k,
                cache_v=cache_v,
            )

            if layer.store_full_length_kv:
                shared_caches[layer.layer_type] = (full_k, full_v)

            new_keys.append(new_k.transpose(1, 0))
            new_values.append(new_v.transpose(1, 0))
            cache_idx += 1

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states.unsqueeze(0))
        return logits, *new_keys, *new_values

    def get_input_embeddings(self, token_ids):
        return self.embed_tokens(token_ids.long())


# ============================================================================
# Text Wrapper
# ============================================================================


class Gemma4Text:
    def __init__(self, model: Gemma4TextModel, config: Gemma4TextConfig, cache_len: int):
        self.model = model
        self.config = config
        self.cache_len = cache_len

    @staticmethod
    def _load_checkpoint(model_path):
        st_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(st_path):
            return ("safetensors", st_path)

        pt_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(pt_path):
            return ("pt", pt_path)

        raise FileNotFoundError(
            f"No model file found at {model_path} "
            "(expected model.safetensors or pytorch_model.bin)"
        )

    @staticmethod
    def _parse_config(model_path, chunk_size, cache_len):
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        tc = raw["text_config"]
        rope = tc.get("rope_parameters", {})
        full_rope = rope.get("full_attention", {})
        sliding_rope = rope.get("sliding_attention", {})

        num_hidden_layers = tc.get("num_hidden_layers", 35)
        chunked_sv_start = num_hidden_layers - 11

        return Gemma4TextConfig(
            hidden_size=tc.get("hidden_size", 1536),
            intermediate_size=tc.get("intermediate_size", 6144),
            num_attention_heads=tc.get("num_attention_heads", 8),
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=tc.get("num_key_value_heads", 1),
            vocab_size=tc.get("vocab_size", 262144),
            vocab_size_per_layer_input=tc.get("vocab_size_per_layer_input", 262144),
            hidden_size_per_layer_input=tc.get("hidden_size_per_layer_input", 256),
            rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
            max_position_embeddings=tc.get("max_position_embeddings", cache_len),
            sliding_window=tc.get("sliding_window", 512),
            head_dim=tc.get("head_dim", 256),
            global_head_dim=tc.get("global_head_dim", 512),
            num_kv_shared_layers=tc.get("num_kv_shared_layers", 20),
            use_double_wide_mlp=tc.get("use_double_wide_mlp", True),
            full_rope_theta=full_rope.get("rope_theta", 1000000.0),
            sliding_rope_theta=sliding_rope.get("rope_theta", 10000.0),
            partial_rotary_factor=full_rope.get("partial_rotary_factor", 0.25),
            prefill_seq_len=chunk_size,
            decode_seq_len=1,
            chunked_sv_start=chunked_sv_start,
            layer_types=tuple(tc.get("layer_types", Gemma4TextConfig().layer_types)),
        )

    @staticmethod
    def _load_mapped_state(model: Gemma4TextModel, model_path: str):
        target_state = model.state_dict()
        loaded = set()
        ckpt_type, ckpt_path = Gemma4Text._load_checkpoint(model_path)
        prefix = "model.language_model."

        if ckpt_type == "safetensors":
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if not key.startswith(prefix):
                        continue
                    target_key = key[len(prefix) :]
                    if target_key not in target_state:
                        continue
                    with torch.no_grad():
                        target_state[target_key].copy_(f.get_tensor(key))
                    loaded.add(target_key)
        else:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            for key, tensor in checkpoint.items():
                if not key.startswith(prefix):
                    continue
                target_key = key[len(prefix) :]
                if target_key not in target_state:
                    continue
                with torch.no_grad():
                    target_state[target_key].copy_(tensor)
                loaded.add(target_key)

        if "lm_head.weight" in target_state and "embed_tokens.weight" in target_state:
            with torch.no_grad():
                target_state["lm_head.weight"].copy_(target_state["embed_tokens.weight"])
            loaded.add("lm_head.weight")

        expected_missing = {
            "full_rope_cos",
            "full_rope_sin",
            "sliding_rope_cos",
            "sliding_rope_sin",
        }
        for layer_idx in range(model.config.num_hidden_layers):
            expected_missing.add(f"layers.{layer_idx}.self_attn.v_norm.weight")

        real_missing = sorted(
            key
            for key in target_state.keys()
            if key not in loaded and key not in expected_missing
        )
        if real_missing:
            print(f"[Gemma4Text] Warning: missing keys: {real_missing[:20]}")

    @staticmethod
    @timeit
    def load_model(model_path: str, chunk_size: int = 256, cache_len: int = 4096):
        config = Gemma4Text._parse_config(model_path, chunk_size, cache_len)
        model = Gemma4TextModel(config, cache_len)
        Gemma4Text._load_mapped_state(model, model_path)
        model._init_constants()
        return Gemma4Text(model, config, cache_len)

    def set_compile_mode(self, compiled: bool):
        self.model.compile_mode(compiled)

    def set_model_device(self, device, dtype=None):
        if dtype is None:
            self.model = self.model.to(device=device)
        else:
            self.model = self.model.to(device=device, dtype=dtype)

    def get_input_embeddings(self, token_ids):
        with torch.no_grad():
            return self.model.get_input_embeddings(token_ids)

    def forward(
        self,
        inputs_embeds,
        token_ids,
        position_ids,
        full_mask,
        sliding_mask,
        caches,
    ):
        with torch.no_grad():
            return self.model(
                inputs_embeds,
                token_ids,
                position_ids,
                full_mask,
                sliding_mask,
                caches,
            )

    def build_empty_caches(self, device="cpu", transpose_cache=True):
        caches = []
        for head_dim in self.model.non_shared_head_dims:
            cache = torch.zeros(
                self.cache_len,
                self.config.num_key_value_heads,
                head_dim,
                dtype=torch.float32,
                device=device,
            )
            if not transpose_cache:
                cache = cache.transpose(0, 1)
            caches.append(cache)
        return caches + [cache.clone() for cache in caches]

    def update_caches(self, past_caches, new_caches, seq_len: int):
        cache_num = len(self.model.non_shared_head_dims)
        updated = []
        for past, new_cache in zip(past_caches[:cache_num], new_caches[:cache_num]):
            updated.append(torch.cat([past[seq_len:, :, :], new_cache], dim=0))
        for past, new_cache in zip(past_caches[cache_num:], new_caches[cache_num:]):
            updated.append(torch.cat([past[seq_len:, :, :], new_cache], dim=0))
        return updated

    def get_leap_input_types(self, seq_len, dtype):
        input_types = [
            leap.TensorType([seq_len, self.config.hidden_size], dtype),
            leap.TensorType([1, seq_len], leap.int64),
            leap.TensorType([seq_len], leap.int32),
            leap.TensorType([seq_len, self.cache_len], leap.float32),
            leap.TensorType([seq_len, self.cache_len], leap.float32),
        ]

        for head_dim in self.model.non_shared_head_dims:
            input_types.append(
                leap.TensorType(
                    [self.cache_len, self.config.num_key_value_heads, head_dim],
                    leap.float32,
                )
            )
        for head_dim in self.model.non_shared_head_dims:
            input_types.append(
                leap.TensorType(
                    [self.cache_len, self.config.num_key_value_heads, head_dim],
                    leap.float32,
                )
            )
        return input_types

    def compile(
        self,
        stage: str,
        output_model_path: str,
        prefill_core_num: int = 1,
        decode_core_num: int = 1,
        enable_vpu=True,
        **kwargs,
    ):
        assert self.model.is_compiled, "Model must be compiled before compiling."

        model_list = []
        stages = []
        if stage in {"prefill", "all"}:
            stages.append("prefill")
        if stage in {"decode", "all"}:
            stages.append("decode")

        for stage_name in stages:
            seq_len = (
                self.config.prefill_seq_len
                if stage_name == "prefill"
                else self.config.decode_seq_len
            )
            inputs = self.get_leap_input_types(seq_len, leap.float32)
            bc_path = str(Path(output_model_path).with_suffix(f".{stage_name}.bc"))
            bc_module = self.model.export_module(inputs, stage_name, bc_path)
            model_list.append(bc_module)

        hbos = []
        for bc_module in model_list:
            func_name = bc_module.functions[0].name
            convert_bc_path = str(
                Path(output_model_path).with_suffix(f".{func_name}_convert.bc")
            )
            mlir_module = self.model.convert_mlir(
                bc_module,
                convert_bc_path,
                enable_vpu=enable_vpu,
                march=kwargs["march"],
            )

            func = mlir_module.functions[0]
            func.remove_io_op(["Dequantize", "Quantize"])
            convert_removed_bc_path = str(
                Path(output_model_path).with_suffix(f".{func_name}_convert_removed.bc")
            )
            save(mlir_module, convert_removed_bc_path)

            hbo_path = str(Path(output_model_path).with_suffix(f".{func_name}.hbo"))
            compile_kwargs = kwargs.copy()
            compile_kwargs["core_num"] = (
                prefill_core_num if "prefill" in func_name else decode_core_num
            )
            if compile_kwargs["core_num"] > 1:
                compile_kwargs["max_l2m_size"] = 25165824
            hbo_model = self.model.compile_hbo(mlir_module, hbo_path, **compile_kwargs)
            hbos.append(hbo_model)

        return self.model.link_models(hbos, output_model_path)
