#!/usr/bin/env python3
"""Verify an official S600 package and bind its files to the SDK Runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


MANIFEST = Path(__file__).with_name('artifacts.json')
SAMPLE = Path(__file__).parents[1]
DEFAULT_PROFILE = 'e2b-w4'


def load_manifest(path=None, profile=DEFAULT_PROFILE):
    if path is None:
        index = json.loads(MANIFEST.read_text(encoding='utf-8'))
        if profile not in index['profiles']:
            raise ValueError('Unknown official Gemma profile: ' + profile)
        path = MANIFEST.parent / index['profiles'][profile]
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('target') != 's600':
        raise ValueError('Expected an official Gemma4 S600 manifest')
    for asset in manifest['files']:
        name = asset['name']
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Model filenames must be plain basenames')
    return manifest


def verify_file(path, asset):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != asset['bytes']:
        raise ValueError(f'Missing file or wrong size: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != asset['sha256']:
        raise ValueError(f'SHA256 mismatch: {path}')


def configure(directory, manifest, template=None):
    directory = Path(directory).resolve()
    template = template or SAMPLE / manifest['runtime']['config_template']
    config = json.loads(Path(template).read_text(encoding='utf-8'))
    names = {asset['name'] for asset in manifest['files']}
    for field in ('lm_model_file', 'embed_weight_name', 'ple_embed_weight_name', 'vit_model_file',
                  'audio_model_file', 'chat_template_file'):
        if config[field] not in names:
            raise ValueError(f'{field} is not bound to the recorded model files')
    config['work_dir'] = str(directory)
    config['tokenizer_path'] = str(directory)
    config['soc'] = 's600'
    output = directory / 'oellm_config.json'
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (directory / 'model-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    shutil.copyfile(SAMPLE / 'LICENSE', directory / 'SAMPLE_LICENSE')
    return output


def prepare(source, output, companion=None, *, manifest=None):
    manifest = load_manifest() if manifest is None else manifest
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        for asset in manifest['files']:
            verify_file(output / asset['name'], asset)
        return configure(output, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + '-', dir=output.parent) as work:
        stage = Path(work) / 'model'
        stage.mkdir()
        for asset in manifest['files']:
            origin = Path(companion).resolve() if asset.get('role') == 'runtime-vision-companion' and companion else source
            path = origin / asset['name']
            verify_file(path, asset)
            shutil.copyfile(path, stage / asset['name'])
            verify_file(stage / asset['name'], asset)
        stage.rename(output)
    return configure(output, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('e2b-w4', 'e2b-w8', 'e4b-w4', 'e4b-w8'), default=DEFAULT_PROFILE)
    parser.add_argument('--source-dir', type=Path, help='Directory containing the compiled language model and embeddings')
    parser.add_argument('--companion-dir', type=Path, help='Directory containing the SDK official vision HBM')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).with_name('prepared'))
    parser.add_argument('--check', action='store_true', help='Verify an already prepared directory without loading a runtime')
    args = parser.parse_args()
    manifest = load_manifest(profile=args.profile)
    if args.check:
        for asset in manifest['files']:
            verify_file(args.output_dir / asset['name'], asset)
        print('Model files verified; board execution: not-run')
    else:
        if args.source_dir is None:
            parser.error('--source-dir is required for preparation')
        print(prepare(args.source_dir, args.output_dir, args.companion_dir, manifest=manifest))


if __name__ == '__main__':
    main()
