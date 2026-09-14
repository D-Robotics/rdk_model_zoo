"""Generate absolute-path float and fake-quant SDK configurations."""
import argparse
import copy
from pathlib import Path

import yaml


def main():
    """Resolve input directories and write reproducible SDK configuration files."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=Path, default=Path('MiniCPM5-2B'), help='Original HF checkpoint directory')
    parser.add_argument('--data-root', type=Path, default=Path('datasets'), help='Directory containing calibration and test splits')
    parser.add_argument('--output-dir', type=Path, default=Path('output'), help='Directory for stage outputs and generated YAML')
    args = parser.parse_args()
    model = args.model_path.resolve(strict=True)
    data = args.data_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if not (model/'config.json').is_file():
        raise FileNotFoundError(model/'config.json')
    for split in ('wikitext2-calibration-train', 'wikitext2-test'):
        if not list((data/split).glob('test-*.parquet')):
            raise FileNotFoundError(data/split)
    config = yaml.safe_load((Path(__file__).parent/'s600.yaml').read_text())
    config['model']['model_path'] = str(model)
    config['calibration']['data_path'] = str(data/'wikitext2-calibration-train')
    config['evaluation']['data_path'] = str(data/'wikitext2-test')
    for section in ('calibration', 'evaluation', 'compile'):
        for key, value in config[section].items():
            if isinstance(value, str) and value.startswith('./output'):
                config[section][key] = str(output/value.removeprefix('./output/'))
    output.mkdir(parents=True, exist_ok=True)
    (output/'s600.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    fake_quant = copy.deepcopy(config)
    fake_quant['evaluation']['calib_ckpt_load_path'] = str(output/'calibration')
    fake_quant['evaluation']['result_path'] = str(output/'fake_quant_eval')
    (output/'s600.fake-quant.yaml').write_text(yaml.safe_dump(fake_quant, sort_keys=False))
    print(output/'s600.yaml')
    print(output/'s600.fake-quant.yaml')


if __name__ == '__main__':
    main()
