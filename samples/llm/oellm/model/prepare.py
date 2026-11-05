#!/usr/bin/env python3
"""Verify an official SDK package and prepare its S600 Runtime configuration."""
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile

SAMPLE = Path(__file__).resolve().parents[1]
INDEX = SAMPLE / 'model/artifacts.json'


def load_index():
    return json.loads(INDEX.read_text(encoding='utf-8'))


def load_manifest(profile):
    index = load_index()
    if profile not in index['profiles']:
        raise ValueError('Unknown official S600 profile: ' + profile)
    manifest = json.loads((INDEX.parent / index['profiles'][profile]).read_text(encoding='utf-8'))
    if (manifest['schema_version'], manifest['target'], manifest['source']['kind']) != (1, 's600', 'sdk-official'):
        raise ValueError('Expected an official SDK S600 manifest')
    if manifest['scope'] not in ('text-gen', 'vlm', 'asr'):
        raise ValueError('This sample supports text, vision-language and ASR profiles')
    names = set()
    for asset in manifest['files']:
        name = asset['name']
        rel = PurePosixPath(name)
        if not name or rel.is_absolute() or '..' in rel.parts or '\\' in name or name in names:
            raise ValueError('Unsafe or duplicate model member: ' + name)
        names.add(name)
    return manifest


def verify_file(path, asset):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != asset['bytes']:
        raise ValueError('Missing file or wrong size: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != asset['sha256']:
        raise ValueError('SHA256 mismatch: ' + str(path))


def config_bindings(value):
    """Find the model/data filenames in vendor configs, including draft models."""
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and item and (key.endswith('_file') or key.endswith('_weight_name')):
                yield key, item
            else:
                yield from config_bindings(item)
    elif isinstance(value, list):
        for item in value:
            yield from config_bindings(item)


def configure(directory, manifest, template=None):
    directory = Path(directory).resolve()
    template = template or SAMPLE / manifest['runtime']['config_template']
    config = copy.deepcopy(json.loads(Path(template).read_text(encoding='utf-8')))
    names = {asset['name'] for asset in manifest['files']}
    for field, name in config_bindings(config):
        if name not in names:
            raise ValueError(f'{field} is not bound to the official package: {name}')
    if config['runtime_type'] != manifest['runtime']['type']:
        raise ValueError('Runtime type does not match the official profile')
    config['work_dir'] = str(directory)
    config['soc'] = 's600'
    output = directory / 'oellm_config.json'
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (directory / 'model-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    shutil.copyfile(SAMPLE / 'LICENSE', directory / 'SAMPLE_LICENSE')
    return output


def prepare(source, output, manifest):
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
            origin, target = source / asset['name'], stage / asset['name']
            verify_file(origin, asset)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
        stage.rename(output)
    return configure(output, manifest)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', help='Profile from model/artifacts.json')
    parser.add_argument('--source-dir', type=Path, help='Extracted official model directory')
    parser.add_argument('--output-dir', type=Path, help='Prepared directory; defaults to source directory')
    parser.add_argument('--check', action='store_true', help='Verify files without configuring or loading Runtime')
    parser.add_argument('--list', action='store_true', help='List official model profiles')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        for profile in load_index()['profiles']:
            manifest = load_manifest(profile)
            print(profile + '\t' + manifest['display_name'])
        return 0
    if not args.profile:
        parser.error('--profile is required; use --list to see available models')
    if args.source_dir is None and args.output_dir is None:
        parser.error('--source-dir or --output-dir is required')
    manifest = load_manifest(args.profile)
    output = args.output_dir or args.source_dir
    if args.check:
        for asset in manifest['files']:
            verify_file(output / asset['name'], asset)
        print('Official model files verified; board execution: not-run')
    else:
        if args.source_dir is None:
            parser.error('--source-dir is required for preparation')
        print(prepare(args.source_dir, output, manifest))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
