#!/usr/bin/env bash
# Download released S100/S100P HBM models and matching metadata/labels.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 - "${1:-auto}" "${2:-n}" "${3:-}" "$SCRIPT_DIR" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.request import urlopen

march, size, destination, script_dir = sys.argv[1:]
if march in ("auto", "--detect"):
    detect_only = march == "--detect"
    root = Path('/sys/class/boardinfo')
    soc = (root / 'soc_name').read_text().strip().lower() if (root / 'soc_name').exists() else ''
    board = (root / 'board_type').read_text().strip().lower() if (root / 'board_type').exists() else ''
    board = re.sub(r'[^a-z0-9]', '', board)
    if soc not in ('s100', 's100p'):
        raise SystemExit('Cannot detect S100/S100P. On a host, specify nash-e or nash-m explicitly.')
    march = 'nash-m' if soc == 's100p' or board in ('p', 'nashm') or board.startswith(('s100p', 'rdks100p')) else 'nash-e'
    if detect_only:
        print(march)
        raise SystemExit(0)
if march not in ('nash-e', 'nash-m') or size not in ('n', 's', 'm', 'l', 'x', 'all'):
    raise SystemExit('Usage: bash download_model.sh [auto|nash-e|nash-m] [n|s|m|l|x|all] [destination]')
dest = Path(destination) if destination else Path(script_dir) / march
dest.mkdir(parents=True, exist_ok=True)
base = f'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/yoloe26_seg/{march}'
with urlopen(base + '/manifest.json', timeout=60) as response:
    manifest = json.load(response)
if manifest.get('march') != march:
    raise SystemExit('Manifest target mismatch')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

for variant in ('nsmlx' if size == 'all' else size):
    stem = f'yoloe_26{variant}_seg_pf'
    names = (f"{stem}_{march.replace('-', '')}_640x640_nv12.hbm", stem + '.json', stem + '.names')
    for name in names:
        record = manifest['files'][name]
        target = dest / name
        if target.is_symlink():
            raise SystemExit(f'Refusing symbolic-link destination: {target}')
        if target.exists():
            if target.stat().st_size != record['bytes'] or digest(target) != record['sha256']:
                raise SystemExit(f'Existing file differs from release, left unchanged: {target}')
            print(f'Verified: {target}', flush=True)
            continue
        with tempfile.NamedTemporaryFile(dir=dest, prefix='.download-', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            print(f'Downloading: {name}', flush=True)
            with urlopen(base + '/' + name, timeout=60) as response, temporary.open('wb') as stream:
                shutil.copyfileobj(response, stream)
            if temporary.stat().st_size != record['bytes'] or digest(temporary) != record['sha256']:
                raise ValueError(f'Checksum mismatch: {name}')
            temporary.chmod(0o644)
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
print(f'Complete: {dest}', flush=True)
PY
