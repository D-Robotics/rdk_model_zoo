"""Bias-free Llama weight mapping for leap_llm 1.0.0 (S100/S100P)."""
import json
from dataclasses import fields
from pathlib import Path

from leap_llm.models.deepseek.model import DeepSeek, LLM, ModelArgs
from leap_llm.nn.utils import load_safetensors_state_dict


def build_model(model_dir, chunk_size=256, cache_len=4096):
    """Load bias-free MiniCPM weights into the SDK Llama-compatible model.

    Args:
        model_dir (str or pathlib.Path): Original checkpoint directory containing
            config.json and safetensors weights accepted by the SDK loader.
        chunk_size (int): Number of tokens in a prefill chunk. Defaults to 256,
            the shape verified for the distributed S100/S100P artifacts.
        cache_len (int): KV-cache capacity in tokens, shared by input and output.
            Defaults to the verified capacity of 4096.

    Returns:
        DeepSeek: SDK wrapper around the strictly loaded MiniCPM model and its
        ModelArgs, configured for batch size one, W8 and preserve_precision.
        The wrapper name denotes SDK implementation reuse, not a model change.

    Raises:
        OSError: The checkpoint configuration or weight files cannot be read.
        json.JSONDecodeError: config.json is not valid JSON.
        ValueError: The checkpoint is not a bias-free Llama model, uses scaled
            RoPE, or has a head dimension incompatible with legacy attention.
        RuntimeError: Strict weight loading finds missing or unexpected keys,
            incompatible tensor shapes, or an SDK model-loading failure.
    """
    config = json.loads((Path(model_dir) / 'config.json').read_text())
    if config.get('model_type') != 'llama':
        raise ValueError('Expected a Llama checkpoint')
    if config.get('attention_bias', False) or config.get('mlp_bias', False):
        raise ValueError('Only bias-free MiniCPM5 is supported')
    if config.get('rope_scaling'):
        raise ValueError('Scaled RoPE is not supported')
    args = {field.name: config.get(field.name, field.default) for field in fields(ModelArgs)}
    args.update(max_batch_size=1, prefill_seq_len=chunk_size, w_bits=8)
    if args['head_dim'] != args['hidden_size'] // args['num_attention_heads']:
        raise ValueError('Legacy attention requires head_dim = hidden_size / num_attention_heads')
    params = ModelArgs(**args)
    model = LLM(params, cache_len=cache_len, preserve_precision=True)
    # The SDK Qwen implementation creates Q/K/V biases. MiniCPM5 has none;
    # FakeQuantLinear supports bias=None in both PyTorch and LEAP export.
    for layer in model.layers:
        for name in ('q_proj', 'k_proj', 'v_proj'):
            getattr(layer.self_attn, name).bias = None
    model.load_state_dict(load_safetensors_state_dict(model_dir), strict=True)
    return DeepSeek(model, params)
