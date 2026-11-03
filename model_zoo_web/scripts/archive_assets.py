"""Prepare Archive companion assets without changing model or benchmark facts."""
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote, urlparse

ARCHIVE_HTTPS = 'https://archive.d-robotics.cc/downloads/rdk_model_zoo_web'
PLATFORMS = {'x5', 's100', 's100p', 's600'}
PACKAGE_FILES = ('release.json', 'release.source.json', 'oe_report.html', 'oe_report_data.json', 'SHA256SUMS')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def asset_prefix(platform, source, filename, metadata_revision=None):
    if platform not in PLATFORMS or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', source):
        raise ValueError('Unknown platform or unsafe sample name')
    if PurePosixPath(filename).name != filename or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', filename):
        raise ValueError('Deployment filename must be a safe basename')
    prefix = f'assets/rdk_{platform}/{source}/{PurePosixPath(filename).stem}'
    if metadata_revision is not None:
        if isinstance(metadata_revision, bool) or not isinstance(metadata_revision, int) or metadata_revision <= 0:
            raise ValueError('Metadata revision must be a positive integer')
        prefix += f'/v{metadata_revision}'
    return prefix


def checksum_entries(payload):
    result = {}
    for line in payload.decode().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        if not re.fullmatch(r'[0-9a-f]{64}', digest) or name in result:
            raise ValueError('Invalid or duplicated source checksum entry')
        result[name] = digest
    return result


def prepare(release_bytes, html_bytes, oe_bytes, *, platform, source, filename,
            model_url, expected_model_sha256, source_url=None, source_checksums=None):
    release = json.loads(release_bytes)
    artifact = release['artifact']
    if artifact['sha256'] != expected_model_sha256:
        raise ValueError('Source release describes a different model')
    if (release.get('target', {}).get('platform') or release.get('platform')) != platform:
        raise ValueError('Source release describes a different platform')
    if (artifact.get('name') or artifact.get('filename')) != filename:
        raise ValueError('Source release describes a different deployment filename')
    if json.loads(oe_bytes)['provenance']['artifact_sha256'] != expected_model_sha256:
        raise ValueError('OE JSON describes a different model')
    parsed = urlparse(model_url)
    if parsed.scheme != 'https' or parsed.query or parsed.fragment or PurePosixPath(unquote(parsed.path)).name != filename:
        raise ValueError('Model URL must name this deployment artifact directly')
    if not re.search(br'<(?:!doctype\s+html|html)[\s>]', html_bytes[:1024], re.I):
        raise ValueError('OE report is not HTML')
    if source_checksums:
        entries = checksum_entries(source_checksums)
        checks = {filename: expected_model_sha256, 'release.json': sha(release_bytes),
                  'oe_report.html': sha(html_bytes), 'oe_report_data.json': sha(oe_bytes)}
        for name, digest in checks.items():
            if name in entries and entries[name] != digest:
                raise ValueError(f'Source checksum mismatch: {name}')
        if not {filename, 'release.json', 'oe_report.html'}.issubset(entries):
            raise ValueError('Source package lacks model/release/HTML checksums')
    revision = release.get('metadata_revision')
    if release.get('status') == 'converted' and revision is None:
        raise ValueError('Converted models require versioned companion metadata')
    prefix = asset_prefix(platform, source, filename, revision)
    base = ARCHIVE_HTTPS + '/' + prefix
    relocated = copy.deepcopy(release)
    relocated['artifact'].update(url=model_url, release_manifest_url=base + '/release.json',
                                 checksums_url=base + '/SHA256SUMS')
    relocated.setdefault('reports', {}).update(oe_conversion_url=base + '/oe_report.html',
                                              oe_data_url=base + '/oe_report_data.json')
    origin = {'layout': 'archive-assets-v1', 'source_manifest_sha256': sha(release_bytes)}
    if source_url:
        origin['source_manifest_url'] = source_url
    if source_checksums:
        origin['source_checksums_sha256'] = sha(source_checksums)
    relocated.setdefault('provenance', {})['asset_publication'] = origin
    files = {'release.source.json': release_bytes,
             'release.json': (json.dumps(relocated, ensure_ascii=False, indent=2) + '\n').encode(),
             'oe_report.html': html_bytes, 'oe_report_data.json': oe_bytes}
    records = ([(filename, expected_model_sha256)]
               + [(item['filename'], item['sha256']) for item in artifact.get('components', [])]
               + [(name, sha(files[name])) for name in PACKAGE_FILES[:-1]])
    files['SHA256SUMS'] = ''.join(f'{digest}  {name}\n' for name, digest in records).encode()
    return prefix, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'html', 'oe-data', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('platform', 'source', 'filename', 'model-url', 'model-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--source-checksums', type=Path)
    parser.add_argument('--source-url')
    args = parser.parse_args()
    prefix, files = prepare(args.release.read_bytes(), args.html.read_bytes(), args.oe_data.read_bytes(),
        platform=args.platform, source=args.source, filename=args.filename, model_url=args.model_url,
        expected_model_sha256=args.model_sha256, source_url=args.source_url,
        source_checksums=args.source_checksums.read_bytes() if args.source_checksums else None)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (args.output / name).write_bytes(data)
    print(json.dumps({'status': 'prepared-not-uploaded', 'prefix': prefix, 'model_sha256': args.model_sha256,
                      'files': {name: {'sha256': sha(data), 'size_bytes': len(data)} for name, data in files.items()}}))


if __name__ == '__main__':
    main()
