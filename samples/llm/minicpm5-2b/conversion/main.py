"""Run SDK stages with importable multiprocessing worker functions."""
import importlib
import os
import sys


def main():
    """Dispatch a validated SDK stage with importable multiprocessing workers."""
    if len(sys.argv) < 2 or sys.argv[1] not in ('calib', 'compile', 'torch_eval'):
        raise SystemExit('Usage: main.py {calib|compile|torch_eval} --config_path FILE')
    stage = sys.argv.pop(1)
    if stage == 'compile':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        for key in ('DEV_B30_CHECK_LIMIT', 'DEV_B30_ENABLE_VPU', 'DEV_B30_ENABLE_VPU_EXTRA_OP',
                    'DEV_B30_ENABLE_VPU_TRIAL_OP', 'DEV_B30_TRITON_VPU'):
            os.environ.setdefault(key, '1')
    import torch
    torch.set_num_threads(int(os.environ.get('OELLM_CPU_THREADS', '8')))
    import adapter  # noqa: F401
    module = importlib.import_module('llm_compression.tools.' + stage)
    section = {'calib': 'calibration', 'torch_eval': 'evaluation', 'compile': 'compile'}[stage]
    config = module.parse_config(section)
    from pathlib import Path
    if stage == 'torch_eval':
        checkpoint = getattr(config.evaluation, 'calib_ckpt_load_path', None)
        if checkpoint and not (Path(checkpoint) / 'lm_calibration.pth.tar').is_file():
            raise FileNotFoundError('Fake-quant evaluation requires lm_calibration.pth.tar; float fallback is forbidden')
    module.main(config)


if __name__ == '__main__':
    main()
