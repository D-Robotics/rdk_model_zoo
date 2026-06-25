import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import FakeQuantAdd, FakeQuantGELU, FakeQuantLinear, FakeQuantMul
from leap_llm.nn.modules.rms_norm import FakeQuantRMSNorm
from leap_llm.nn.utils import Module

from .attention import Gemma4TextAttention, Gemma4VisionAttention
from .mlp import Gemma4TextMLP, Gemma4VisionMLP


class Gemma4VisionEncoderLayer(Module):
    """Single encoder layer with 4 RMSNorms (pre/post for both attn and ffn)."""

    def __init__(
        self,
        layer_id,
        hidden_size,
        num_attention_heads,
        head_dim,
        intermediate_size,
        rms_norm_eps,
    ):
        super().__init__()
        self.layer_id = layer_id

        self.self_attn = Gemma4VisionAttention(
            hidden_size, num_attention_heads, head_dim, rms_norm_eps
        )
        self.mlp = Gemma4VisionMLP(hidden_size, intermediate_size)

        self.input_layernorm = FakeQuantRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = FakeQuantRMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.pre_feedforward_layernorm = FakeQuantRMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.post_feedforward_layernorm = FakeQuantRMSNorm(
            hidden_size, eps=rms_norm_eps
        )

    def build(self, hidden_states, cos, sin):
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = leap.add(residual, hidden_states)

        # FFN block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = leap.add(residual, hidden_states)

        return hidden_states

    def forward(self, hidden_states, cos, sin):
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # FFN block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma4TextDecoderLayer(Module):
    def __init__(
        self,
        config,
        layer_idx,
        head_dim,
        is_kv_shared_layer,
        store_full_length_kv,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.is_kv_shared_layer = is_kv_shared_layer
        self.store_full_length_kv = store_full_length_kv
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        chunked_sv_start = getattr(config, 'chunked_sv_start', 24)
        chunked_sv_layers = set(range(chunked_sv_start, config.num_hidden_layers))

        self.self_attn = Gemma4TextAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=config.rms_norm_eps,
            use_chunked_sv=layer_idx in chunked_sv_layers,
            sv_chunk_size=1024,
        )
        self.mlp = Gemma4TextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            is_kv_shared_layer=is_kv_shared_layer,
            use_double_wide_mlp=config.use_double_wide_mlp,
        )

        self.input_layernorm = FakeQuantRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = FakeQuantRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = FakeQuantRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = FakeQuantRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_per_layer_input_norm = FakeQuantRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.per_layer_input_gate = FakeQuantLinear(
            config.hidden_size,
            config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_projection = FakeQuantLinear(
            config.hidden_size_per_layer_input,
            config.hidden_size,
            bias=False,
        )
        self.per_layer_act = FakeQuantGELU(quantized=True, quant_bits=16)

        self.add_attn = FakeQuantAdd(quantized=False)
        self.add_ffn = FakeQuantAdd(quantized=False)
        self.add_per_layer = FakeQuantAdd(quantized=False)
        self.mul_per_layer = FakeQuantMul(quantized=False)
        self.mul_layer_scalar = FakeQuantMul(quantized=False)
        self.register_buffer("layer_scalar", torch.ones(1))

    def build(
        self,
        hidden_states,
        per_layer_input,
        cos,
        sin,
        mask,
        cache_k=None,
        cache_v=None,
        shared_cache_k=None,
        shared_cache_v=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.is_kv_shared_layer:
            hidden_states = self.self_attn.build_shared(
                hidden_states,
                cos,
                sin,
                shared_cache_k,
                shared_cache_v,
                mask,
            )
            new_k = None
            new_v = None
            full_k = shared_cache_k
            full_v = shared_cache_v
        else:
            hidden_states, new_k, new_v, full_k, full_v = self.self_attn(
                hidden_states,
                cos,
                sin,
                cache_k,
                cache_v,
                mask,
            )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.add_attn(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = self.add_ffn(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.per_layer_input_gate(hidden_states)
        hidden_states = self.per_layer_act(hidden_states)
        hidden_states = self.mul_per_layer(hidden_states, per_layer_input)
        hidden_states = self.per_layer_projection(hidden_states)
        hidden_states = self.post_per_layer_input_norm(hidden_states)
        hidden_states = self.add_per_layer(residual, hidden_states)
        hidden_states = self.mul_layer_scalar(hidden_states, self.layer_scalar)

        return hidden_states, new_k, new_v, full_k, full_v

    def forward(
        self,
        hidden_states,
        per_layer_input,
        cos,
        sin,
        mask,
        cache_k=None,
        cache_v=None,
        shared_cache_k=None,
        shared_cache_v=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.is_kv_shared_layer:
            hidden_states = self.self_attn.forward_shared(
                hidden_states,
                cos,
                sin,
                shared_cache_k,
                shared_cache_v,
                mask,
            )
            new_k = None
            new_v = None
            full_k = shared_cache_k
            full_v = shared_cache_v
        else:
            hidden_states, new_k, new_v, full_k, full_v = self.self_attn(
                hidden_states,
                cos,
                sin,
                cache_k,
                cache_v,
                mask,
            )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.add_attn(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = self.add_ffn(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.per_layer_input_gate(hidden_states)
        hidden_states = self.per_layer_act(hidden_states)
        hidden_states = self.mul_per_layer(hidden_states, per_layer_input)
        hidden_states = self.per_layer_projection(hidden_states)
        hidden_states = self.post_per_layer_input_norm(hidden_states)
        hidden_states = self.add_per_layer(residual, hidden_states)
        hidden_states = self.mul_layer_scalar(hidden_states, self.layer_scalar)

        return hidden_states, new_k, new_v, full_k, full_v
