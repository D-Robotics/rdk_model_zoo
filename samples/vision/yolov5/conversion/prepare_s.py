"""Prepare native YOLOv5 S calibration and specialize a generated full template."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibrate(args):
    import numpy as np
    source = args.source_manifest.resolve()
    original = json.loads(source.read_text())
    assert original['count'] == 200 and original['seed'] == 42
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    rows = []
    for row in original['images']:
        raw = Path(row['tensor'])
        assert sha(raw) == row['sha256']
        tensor = np.fromfile(raw, dtype=np.float32).reshape(1, 3, 640, 640) / 255.0
        assert tensor.dtype == np.float32 and np.isfinite(tensor).all()
        target = output / 'images' / (raw.stem + '.npy')
        np.save(target, tensor)
        np.testing.assert_array_equal(np.load(target), tensor)
        rows.append({'source': row['source'], 'source_sha256': row['source_sha256'],
                     'raw_tensor_sha256': row['sha256'], 'tensor': str(target),
                     'sha256': sha(target), 'shape': list(tensor.shape), 'dtype': 'float32',
                     'min': float(tensor.min()), 'max': float(tensor.max())})
    report = {'count': 200, 'seed': 42, 'source_manifest': str(source),
              'source_manifest_sha256': sha(source), 'producer_sha256': sha(__file__),
              'geometry': original['geometry'], 'normalization': 'RGB float32 NCHW /255; original ONNX boundary',
              'runtime_normalization': 'NV12 -> RGB /255 inserted by compiler; not reapplied to calibration npy',
              'images': rows}
    (output / 'calibration-manifest.json').write_text(json.dumps(report, indent=2) + '\n')


def configure(args):
    import yaml
    template = args.template.resolve()
    config = yaml.safe_load(template.read_text())
    target = args.target
    march = {'s100': 'nash-e', 's600': 'nash-p'}[target]
    size, tag = args.variant.split('-v')
    prefix = f'yolov5{size}_tag_v{tag}_native_640x640_{march.replace("-", "")}_nv12_20261102'
    config['model_parameters'].update(onnx_model=str(args.onnx.resolve()), march=march,
        working_dir=str(args.working.resolve()), output_model_file_prefix=prefix,
        remove_node_name='', remove_node_type='')
    config['input_parameters'].update(input_name='images', input_shape='1x3x640x640',
        input_type_rt='nv12', input_type_train='rgb', input_layout_train='NCHW',
        input_space_and_range='bt601_video', mean_value='', scale_value=str(1/255), std_value='')
    config['calibration_parameters'].update(cal_data_dir=str(args.calibration.resolve() / 'images'), quant_config={})
    config['compiler_parameters'].update(jobs=2, core_num=1, optimize_level='O2',
        compile_mode='latency', input_source={'images': 'pyramid'})
    with args.output.open('x') as f:
        yaml.safe_dump(config, f, sort_keys=False)
    receipt = {'template_sha256': sha(template), 'config_sha256': sha(args.output),
               'onnx_sha256': sha(args.onnx), 'calibration_manifest_sha256': sha(args.calibration / 'calibration-manifest.json'),
               'target': target, 'march': march, 'prefix': prefix, 'producer_sha256': sha(__file__)}
    args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    cal = sub.add_parser('calibrate')
    cal.add_argument('--source-manifest', type=Path, required=True)
    cal.add_argument('--output', type=Path, required=True)
    cfg = sub.add_parser('configure')
    for name in ('template', 'onnx', 'calibration', 'working', 'output'):
        cfg.add_argument('--' + name, type=Path, required=True)
    cfg.add_argument('--target', choices=('s100', 's600'), required=True)
    cfg.add_argument('--variant', choices=('s-v2.0', 'n-v7.0'), required=True)
    args = parser.parse_args()
    calibrate(args) if args.command == 'calibrate' else configure(args)


if __name__ == '__main__':
    main()
