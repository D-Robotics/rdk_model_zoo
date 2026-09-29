"""Read fixed S source files from Git; do not execute legacy code or contact boards."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[5]
PIN = '380e1a2bf42041af54be6f34935e50197cfadff9'
SAMPLES = ['yolo11', 'yolo11_pose', 'yolo11_seg', 'yolov13_imoonlab']
records = []
for sample in SAMPLES:
    prefix = f'samples/vision/{sample}/'
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', PIN, '--', prefix], cwd=ROOT, text=True).splitlines()
    for path in paths:
        data = subprocess.check_output(['git', 'show', f'{PIN}:{path}'], cwd=ROOT)
        archive = ROOT / 'platforms/s' / path
        records.append({'source_path':path, 'source_sha256':hashlib.sha256(data).hexdigest(),
                        'source_bytes':len(data), 'archive_exists':archive.is_file(),
                        'archive_byte_exact':archive.is_file() and archive.read_bytes()==data})
print(json.dumps({'source_commit':PIN,'files':records,'board':'not-run',
                  'note':'File preservation only; not numerical/runtime equivalence.'},indent=2))
