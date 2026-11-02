#!/usr/bin/env python3
"""Map accepted OSNet Workbench bundles to one unpublished multi-platform preview."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
from build_catalog import validate_accuracy, validate_end_to_end_record


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(bundles, destination, cover, source_archive, platform_sources=None):
    releases = []
    seen = set()
    for bundle in bundles:
        release = json.loads((bundle / 'release.json').read_text())
        if tuple(release.get(k) for k in ('source', 'family', 'task', 'size')) != ('osnet', 'osnet', 'reid', 'x1.0'):
            raise ValueError('Expected an OSNet x1.0 Workbench release')
        target = release.get('platform')
        if target not in {'x5', 's100', 's100p', 's600'} or target in seen:
            raise ValueError('Expected a unique supported platform per bundle')
        seen.add(target)
        if release.get('status') != 'candidate' or release['validation'].get('technical_acceptance') != 'passed':
            raise ValueError('Local preview requires technically accepted candidates')
        for line in (bundle / 'SHA256SUMS').read_text().splitlines():
            expected, filename = line.split(maxsplit=1)
            if Path(filename).name != filename or digest(bundle / filename) != expected:
                raise ValueError(f'Bundle checksum mismatch: {filename}')
        artifact = release['artifact']
        if digest(bundle / artifact['filename']) != artifact['sha256']:
            raise ValueError('Artifact differs from release.json')
        validate_accuracy(release['accuracy'], 'reid', 'OSNet accuracy')
        performance = release['performance']
        if {m['threads'] for m in performance['measurements']} != {1, 2}:
            raise ValueError('Runtime 1/2 concurrency is required')
        if {validate_end_to_end_record(row, 'OSNet C++') for row in performance['end_to_end']} != {1, 2}:
            raise ValueError('C++ 1/2 streams are required')
        if target != 'x5' and not release['provenance'].get('runtime_variant'):
            raise ValueError('S-series bundles must identify the native runtime variant')
        if releases:
            first = releases[0][1]
            if release['input'] != first['input'] or any(release['model'][k] != first['model'][k] for k in ('parameter_count', 'gflops')):
                raise ValueError('Platform bundles disagree on source model size/input')
        releases.append((bundle, release))
    platform_sources = platform_sources or {}
    if set(platform_sources) - seen:
        raise ValueError('Source override has no matching platform bundle')
    source_commits = {}
    for target in seen:
        with tarfile.open(platform_sources.get(target, source_archive), 'r:gz') as archive:
            commit = archive.pax_headers.get('comment', '')
        if not re.fullmatch(r'[0-9a-f]{40}', commit):
            raise ValueError('Source archive must come from git archive with a full commit identity')
        source_commits[target] = commit
    destination.mkdir(parents=True, exist_ok=False)
    cover_path = destination / ('cover' + cover.suffix)
    shutil.copyfile(cover, cover_path)
    default_source_path = destination / 'osnet-sample-source.tar.gz'
    shutil.copyfile(source_archive, default_source_path)
    platforms, release_inputs = [], {}
    for bundle, release in releases:
        target = release['platform']
        folder = destination / target
        folder.mkdir()
        source_commit = source_commits[target]
        source_path = default_source_path
        if target in platform_sources:
            source_path = folder / 'osnet-sample-source.tar.gz'
            shutil.copyfile(platform_sources[target], source_path)
        for path in bundle.iterdir():
            if path.is_file():
                shutil.copyfile(path, folder / path.name)
        release['artifact']['local_path'] = str((folder / release['artifact']['filename']).resolve())
        platform = {key: release[key] for key in ('platform', 'status', 'artifact', 'accuracy', 'performance', 'provenance')}
        platform['provenance'].update(sample_pending_publication=True, sample_source_commit=source_commit)
        if target == 'x5':
            platform['provenance'].setdefault('runtime_variant', 'spatial-rgb-late8')
        platform['source_archive'] = {'commit': source_commit, 'filename': source_path.name, 'local_path': str(source_path.resolve()),
                                     'sha256': digest(source_path), 'size_bytes': source_path.stat().st_size}
        platforms.append(platform)
        release_inputs[f'osnet/osnet/reid/x1.0/{target}'] = {
            'oe_data': f'{target}/oe_report_data.json', 'oe_html': f'{target}/oe_report.html',
            'oe_html_sha256': release['reports']['oe_report_html_sha256'], 'artifact_sha256': release['artifact']['sha256']}
    first = releases[0][1]
    record = {'schema_version': 1, 'id': 'osnet/osnet/reid', 'name': 'OSNet', 'domain': 'vision', 'source': 'osnet',
              'provider': 'Kaiyang Zhou', 'family': 'osnet', 'task': 'reid',
              'description': {'zh': 'OSNet 通过多尺度卷积分支与统一通道门控，将行人裁剪图编码为 512 维外观特征。',
                              'en': 'OSNet combines omni-scale convolutional branches with unified channel gating to encode person crops as 512-dimensional appearance features.'},
              'license': {key: first['license'][key] for key in ('name', 'url')}, 'sample_path': 'samples/vision/osnet',
              'variants': [{'size': 'x1.0', 'model': first['model'], 'input': first['input'], 'platforms': platforms}]}
    catalog = {'schema_version': 1, 'source': 'local-candidate', 'status': 'candidate', 'models': [record]}
    inputs = {'schema_version': 1, 'models': {'osnet/osnet/reid': {'cover': cover_path.name, 'cover_label': 'OSNet 行人重识别'}}, 'releases': release_inputs}
    for filename, value in (('catalog.json', catalog), ('inputs.json', inputs)):
        (destination / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, action='append', required=True, help='Repeat for each accepted platform')
    for flag in ('output', 'cover', 'source-archive'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--platform-source', action='append', default=[], metavar='TARGET=ARCHIVE',
                        help='Preserve a platform-specific accepted source archive, such as X5')
    args = parser.parse_args()
    sources = {}
    for value in args.platform_source:
        target, separator, filename = value.partition('=')
        if not separator or not filename or target in sources:
            parser.error('--platform-source requires a unique TARGET=ARCHIVE')
        sources[target] = Path(filename).resolve()
    print(prepare([b.resolve() for b in args.bundle], args.output.resolve(), args.cover.resolve(), args.source_archive.resolve(), sources))


if __name__ == '__main__':
    main()
