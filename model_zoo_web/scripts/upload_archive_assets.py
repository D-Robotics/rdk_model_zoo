"""Publish a prepared metadata package from the local Mac."""
import argparse
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
from urllib.request import urlopen

from archive_assets import ARCHIVE_HTTPS, PACKAGE_FILES, asset_prefix, checksum_entries, sha

REMOTE_ROOT = '/home/hobot/www/html/downloads/rdk_model_zoo_web'


def transport_options(jump_host=None):
    route = ['-J', jump_host] if jump_host and jump_host != 'none' else [
        '-o', 'ProxyJump=none', '-o', 'ProxyCommand=none']
    return route + ['-o', 'BatchMode=yes', '-o', 'ControlMaster=auto',
        '-o', 'ControlPersist=600', '-o', 'ControlPath=/tmp/rdk-modelzoo-upload-%C',
        '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
        '-o', 'ConnectTimeout=15']


REMOTE_INSTALL = r'''
import hashlib, io, json, os, pathlib, shutil, sys, tarfile, tempfile

def install(root, prefix, files):
    parts = pathlib.PurePosixPath(prefix).parts
    if len(parts) not in (4, 5) or parts[0] != 'assets' or parts[1] not in ('rdk_x5', 'rdk_s100', 'rdk_s100p', 'rdk_s600'):
        raise ValueError('Invalid assets namespace')
    if len(parts) == 5 and not __import__('re').fullmatch(r'v[1-9][0-9]*', parts[4]):
        raise ValueError('Invalid metadata revision')
    if any(p in ('.', '..') or not p for p in parts):
        raise ValueError('Unsafe assets namespace')
    expected = {'release.json', 'release.source.json', 'oe_report.html', 'oe_report_data.json', 'SHA256SUMS'}
    if set(files) != expected:
        raise ValueError('Package must contain only the five public companions')
    root = pathlib.Path(root).resolve(strict=True)
    dest = root.joinpath(*parts)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.parent.resolve() != dest.parent:
        raise ValueError('Destination traverses a symbolic link')
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or not dest.is_dir() or {p.name for p in dest.iterdir()} != expected:
            raise ValueError('Existing destination is not the same five-file package')
        for name, data in files.items():
            if (dest/name).is_symlink() or (dest/name).read_bytes() != data:
                raise ValueError('Refusing to overwrite different published content: ' + str(dest/name))
        status = 'identical-existing'
    else:
        # Keep incomplete files behind mode 0700 on the destination filesystem.
        # Archive's private home may live on another mounted disk.
        stage = pathlib.Path(tempfile.mkdtemp(prefix='.web-assets-upload-', dir=dest.parent))
        try:
            if stage.stat().st_dev != dest.parent.stat().st_dev:
                raise ValueError('Private staging and public destination must share a filesystem')
            for name, data in files.items():
                (stage/name).write_bytes(data)
                (stage/name).chmod(0o644)
            # A populated existing directory cannot be replaced by rename on POSIX.
            os.rename(stage, dest)
            status = 'created'
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    result = {'prefix': prefix, 'status': status, 'files': {}}
    for name, data in files.items():
        actual = (dest/name).read_bytes()
        if actual != data:
            raise ValueError('Published read-back differs: ' + name)
        result['files'][name] = {'sha256': hashlib.sha256(actual).hexdigest(), 'size_bytes': len(actual)}
    dest.chmod(0o755)
    return result

if __name__ == '__main__':
    files = {}
    with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()), mode='r:gz') as archive:
        for member in archive:
            if not member.isfile() or member.name in files:
                raise ValueError('Package contains non-files or duplicate names')
            files[member.name] = archive.extractfile(member).read()
    print(json.dumps(install(sys.argv[1], sys.argv[2], files)))
'''


def validate_package(package):
    if {p.name for p in package.iterdir()} != set(PACKAGE_FILES):
        raise ValueError('Prepared directory must contain exactly the five companion files')
    if any((package/name).is_symlink() or not (package/name).is_file() for name in PACKAGE_FILES):
        raise ValueError('Prepared companions must be regular files')
    files = {name: (package/name).read_bytes() for name in PACKAGE_FILES}
    release = json.loads(files['release.json'])
    source = release.get('source') or release.get('model', {}).get('source')
    platform = release.get('platform') or release.get('target', {}).get('platform')
    artifact = release['artifact']
    filename = artifact.get('name') or artifact.get('filename')
    prefix = asset_prefix(platform, source, filename, release.get('metadata_revision'))
    base = ARCHIVE_HTTPS + '/' + prefix
    if artifact.get('release_manifest_url') != base+'/release.json' or artifact.get('checksums_url') != base+'/SHA256SUMS':
        raise ValueError('Release URLs do not match the assets namespace')
    if release['reports'].get('oe_conversion_url') != base+'/oe_report.html' or release['reports'].get('oe_data_url') != base+'/oe_report_data.json':
        raise ValueError('OE URLs do not match the assets namespace')
    expected = {filename: artifact['sha256'], **{name: sha(files[name]) for name in PACKAGE_FILES[:-1]}}
    expected.update({item['filename']:item['sha256'] for item in artifact.get('components', [])})
    if checksum_entries(files['SHA256SUMS']) != expected:
        raise ValueError('Prepared checksum inventory does not match the package')
    if release['provenance']['asset_publication']['source_manifest_sha256'] != sha(files['release.source.json']):
        raise ValueError('Original release identity was not preserved')
    if json.loads(files['oe_report_data.json'])['provenance']['artifact_sha256'] != artifact['sha256']:
        raise ValueError('OE data belongs to another model')
    from component_package import validate_components
    validate_components(release, platform=platform, source=source, oe=json.loads(files['oe_report_data.json']))
    return prefix, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('--ssh-host', required=True)
    parser.add_argument('--jump-host', help='Optional SSH jump host; omitted for a direct connection')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    prefix, files = validate_package(args.package)
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode='w:gz') as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    command = shlex.join(['python3', '-c', REMOTE_INSTALL, REMOTE_ROOT, prefix])
    result = subprocess.run(['ssh', *transport_options(args.jump_host), args.ssh_host, command],
        input=payload.getvalue(), capture_output=True, check=True)
    receipt = json.loads(result.stdout)
    receipt['anonymous_https'] = {}
    for name, data in files.items():
        with urlopen(ARCHIVE_HTTPS+'/'+prefix+'/'+name, timeout=90) as response:
            actual = response.read()
            if response.status != 200 or actual != data:
                raise ValueError('Anonymous HTTPS read-back differs: ' + name)
        receipt['anonymous_https'][name] = {'sha256': sha(actual), 'size_bytes': len(actual)}
    receipt['status'] = 'verified'
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'status': 'verified', 'prefix': prefix, 'receipt': str(args.receipt)}))


if __name__ == '__main__':
    main()
