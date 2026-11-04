#!/usr/bin/env python3
"""Run the supplied SDK C++ program with a prepared official model profile."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

SAMPLE = Path(__file__).resolve().parents[2]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path', type=Path, required=True)
    parser.add_argument('--prompt', default='你好')
    parser.add_argument('--image_path', type=Path)
    parser.add_argument('--audio_path', type=Path)
    parser.add_argument('--sampler_config_path', type=Path)
    parser.add_argument('--build', action='store_true', help='Build the SDK example from its supplied source')
    parser.add_argument('--dry-run', action='store_true', help='Print the native command without building or inference')
    return parser


def native_command(args, runtime_root):
    config_path = args.config_path.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    directory = Path(config['work_dir'])
    if not directory.is_absolute() or config.get('soc') != 's600':
        raise ValueError('Use an absolute-path S600 configuration from model/prepare.py')
    manifest = json.loads((directory / 'model-manifest.json').read_text(encoding='utf-8'))
    index = json.loads((SAMPLE / 'model/artifacts.json').read_text(encoding='utf-8'))
    profile = manifest['profile']
    if profile not in index['profiles'] or manifest.get('target') != 's600':
        raise ValueError('Unknown official S600 profile')
    recorded = json.loads((SAMPLE / 'model' / index['profiles'][profile]).read_text(encoding='utf-8'))
    if recorded != manifest or config['runtime_type'] != manifest['runtime']['type']:
        raise ValueError('Prepared model does not match its official profile')
    if manifest['scope'] not in ('text-gen', 'vlm', 'asr'):
        raise ValueError('Unsupported model task')
    if args.image_path and manifest['scope'] != 'vlm':
        raise ValueError('Image input requires a vision-language profile')
    if args.audio_path and manifest['scope'] != 'asr':
        raise ValueError('Audio input requires an ASR profile')
    asr = manifest['scope'] == 'asr'
    if asr and not args.audio_path:
        raise ValueError('ASR requires --audio_path')
    program = manifest['runtime']['entrypoint']
    build_directory = SAMPLE / 'runtime/cpp/build' / ('asr' if asr else 'basic')
    binary = build_directory / Path(program).name if args.build else runtime_root / program
    command = [str(binary), '--config_path', str(config_path)]
    if asr:
        command += ['--audio_path', str(args.audio_path.resolve())]
    else:
        command += ['--mode', 'sync', '--no_warmup', '--prompt', args.prompt]
        if args.image_path:
            command += ['--image_path', str(args.image_path.resolve())]
        if args.sampler_config_path:
            command += ['--sampler_config_path', str(args.sampler_config_path.resolve())]
    return command, runtime_root / Path(program).parent, build_directory


def main(argv=None):
    args = build_parser().parse_args(argv)
    runtime_root = Path(os.environ.get('OELLM_RUNTIME_ROOT', '/opt/oellm_runtime_s600')).resolve()
    command, source, build = native_command(args, runtime_root)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    identity = Path('/sys/class/boardinfo/soc_name')
    if not identity.is_file() or identity.read_text().strip().upper() != 'S600':
        raise ValueError('Execution requires an identified S600 board')
    if not (runtime_root / 'env.sh').is_file():
        raise ValueError('Set OELLM_RUNTIME_ROOT to the SDK 2.0.0 S600 Runtime directory')
    if args.build:
        subprocess.run(['cmake', '-S', str(source), '-B', str(build)], check=True)
        subprocess.run(['cmake', '--build', str(build), '--parallel', '4'], check=True)
    if not Path(command[0]).is_file():
        raise ValueError('The SDK C++ program is missing; prepare the complete Runtime package or use --build')
    os.execv(command[0], command)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
