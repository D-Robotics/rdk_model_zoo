"""Compile an additional legacy target from calibrated, pre-conversion BC files.

Uses the same SDK conversion, IO removal and linking steps as DeepSeek.compile.
Never reuse a nash-e converted BC as a nash-m input.
"""
import argparse
import os
from pathlib import Path


def main():
    """Convert raw calibrated BC independently for the requested target and link HBM."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-hbm', type=Path, required=True,
                        help='Naming prefix of the original build; reads .prefill.bc/.decode.bc')
    parser.add_argument('--output-hbm', type=Path, required=True)
    parser.add_argument('--march', choices=['nash-e', 'nash-m'], required=True)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    for key in ('DEV_B30_TRITON_VPU', 'DEV_B30_ENABLE_VPU_EXTRA_OP',
                'DEV_B30_ENABLE_VPU_TRIAL_OP'):
        os.environ[key] = '1'
    from hbdk4.compiler import load, save
    from leap_llm.nn.utils import Model
    sources = [args.source_hbm.with_suffix(f'.{part}.bc') for part in ('prefill', 'decode')]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    args.output_hbm.parent.mkdir(parents=True, exist_ok=True)
    hbos = []
    for part, source in zip(('prefill', 'decode'), sources):
        print(f'Converting calibrated {part} from {source} for {args.march}', flush=True)
        module = load(str(source))
        # export_module sets these SDK flags before the stock conversion.
        module._llm_extra = True
        module._high_precision_qpp = False
        converted = Model.convert_mlir(module,
            str(args.output_hbm.with_suffix(f'.{part}_convert.bc')),
            enable_vpu=True, march=args.march)
        converted.functions[0].remove_io_op(['Dequantize', 'Quantize'])
        save(converted, str(args.output_hbm.with_suffix(f'.{part}_convert_removed.bc')))
        hbos.append(Model.compile_hbo(converted,
            str(args.output_hbm.with_suffix(f'.{part}.hbo')),
            march=args.march, jobs=args.jobs, progress_bar=True, opt=2,
            max_time_per_fc=0.0, debug=False, advice=0.0, balance=100,
            input_no_padding=False, output_no_padding=False,
            cache_mode='disable', cache_path=''))
    Model.link_models(hbos, str(args.output_hbm))
    print(f'LINK_COMPLETE {args.output_hbm}', flush=True)


if __name__ == '__main__':
    main()
