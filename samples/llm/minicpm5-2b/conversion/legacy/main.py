"""Compile MiniCPM5 with leap_llm 1.0.0, preserving the stock CLI."""
import os
from pathlib import Path

import torch

from leap_llm.apis.calibration.data_loader import load_text_data
from leap_llm.apis.model.deepseek import DeepSeekApi
from leap_llm.apis.model.model_factory import register_model

from legacy_adapter import build_model


@register_model('minicpm5-2b', ['nash-e', 'nash-m'])
def create_minicpm5(args):
    """Register the MiniCPM adapter with the unmodified SDK compilation CLI."""
    if args.march not in ('nash-e', 'nash-m'):
        raise ValueError('Use the S600 environment for nash-p')
    if args.w_bits != 8:
        raise ValueError('Only W8 is currently validated in the legacy adapter')
    if args.cache_len % args.chunk_size or args.cache_len > 131072:
        raise ValueError('Cache length must be a chunk multiple within 131072')
    # Reuse the SDK calibration/export/compile pipeline with our strict loader.
    api = object.__new__(DeepSeekApi)
    api.input_model_path = args.input_model_path
    api.calib_text_data = load_text_data(args.calib_text_path)
    api.chunk_size = args.chunk_size
    api.cache_len = args.cache_len
    api.device = args.device
    api.dtype = 'float32'
    api.w_bits = 8
    api.mask_value = -32767
    api.model_type = 'minicpm5-2b'
    out = Path(args.output_model_path)
    out.mkdir(parents=True, exist_ok=True)
    api.output_model_path = str(out / f'minicpm5-2b_chunk_{args.chunk_size}_cache_{args.cache_len}_q8.hbm')
    api.deepseek_model = build_model(args.input_model_path, args.chunk_size, args.cache_len)
    return api


if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get('OELLM_CPU_THREADS', '8')))
    from leap_llm.apis.oellm_build import main
    main()
