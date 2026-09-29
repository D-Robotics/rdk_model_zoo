"""Exercise the production native reader with real and adversarial NumPy files."""
from pathlib import Path
import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import numpy as np

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
PROBE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT.parent / '.coordination/paraformer-native-io/feature_probe'
SOURCE = ROOT / 'docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-cli/run-5_sjkdzc/features/prepared-manifest.json'
records = []

def run(label, path, good=True, limit=None):
    argv = [str(PROBE), str(path)] + ([] if limit is None else [str(limit)])
    result = subprocess.run(argv, capture_output=True, text=True)
    records.append({'case': label, 'argv': argv, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'expected': 0 if good else 2})
    assert result.returncode == (0 if good else 2), (label, result.stdout, result.stderr)
    return json.loads(result.stdout) if good else None

def digest(data):
    return hashlib.sha256(data).hexdigest()

actual = run('real-two-WAV-features', SOURCE)
for entry, item in zip(json.loads(SOURCE.read_text()), actual, strict=True):
    data = np.load(SOURCE.parent / entry['feature_file'], allow_pickle=False)
    assert item['values_sha256'] == digest(data.tobytes())
    assert item['frames'] == entry['feat_length']
    assert item['source_record'] == entry

with tempfile.TemporaryDirectory(prefix='paraformer-npy-') as tmp:
    tmp = Path(tmp)
    features = np.random.default_rng(417).normal(size=(1, 400, 560)).astype(np.float32)
    base = {'utt_id':'sample', 'feat_length':71, 'original_frames':71, 'truncated':False, 'text':'测试', 'annotation':{'preserved':True}, 'feature_file':'feature.npy'}
    def check(label, data=None, mutate=None, good=True, limit=None):
        if data is None:
            stream = io.BytesIO(); np.save(stream, features, allow_pickle=False); data = stream.getvalue()
        (tmp/'feature.npy').write_bytes(data)
        entry = copy.deepcopy(base); entry['feature_sha256'] = digest(data)
        entries = [entry]
        if mutate:
            mutate(entries)
        manifest = tmp/'prepared.json'; manifest.write_text(json.dumps(entries, ensure_ascii=False))
        output = run(label, manifest, good, limit)
        if good:
            assert output[0]['values_sha256'] == digest(features.tobytes())
            assert output[0]['source_record'] == entries[0]
    for version in ((1,0),(2,0),(3,0)):
        stream=io.BytesIO(); np.lib.format.write_array(stream,features,version=version,allow_pickle=False)
        check('numpy-version-'+str(version[0]),stream.getvalue())
    stream=io.BytesIO(); np.save(stream,features.astype('>f4'),allow_pickle=False)
    check('big-endian-float32',stream.getvalue())
    check('truncated-500-to-400', mutate=lambda e:e[0].update(feat_length=400,original_frames=500,truncated=True))
    check('uppercase-digest',mutate=lambda e:e[0].update(feature_sha256=e[0]['feature_sha256'].upper()))
    for label, mutation in [
        ('duplicate-id', lambda e:e.append(copy.deepcopy(e[0]))),
        ('zero-frames', lambda e:e[0].update(feat_length=0)),
        ('bool-frames', lambda e:e[0].update(feat_length=True)),
        ('wrong-original', lambda e:e[0].update(original_frames=72)),
        ('wrong-truncated', lambda e:e[0].update(truncated=True)),
        ('wrong-text-type', lambda e:e[0].update(text=3)),
        ('missing-length', lambda e:e[0].pop('feat_length')),
        ('bad-id', lambda e:e[0].update(utt_id='../sample')),
        ('wrong-digest', lambda e:e[0].update(feature_sha256='0'*64)),
        ('missing-file', lambda e:e[0].update(feature_file='absent.npy')),
        ('missing-digest', lambda e:e[0].pop('feature_sha256')),
        ('oversize-frames', lambda e:e[0].update(original_frames=2**64-1)),
    ]:
        check(label, mutate=mutation, good=False)
    check('invalid-unselected-record',mutate=lambda e:e.append(dict(e[0],utt_id='other',feat_length=False)),good=False,limit=1)
    check('missing-unselected-file',mutate=lambda e:e.append(dict(e[0],utt_id='other',feature_file='missing.npy')),limit=1)
    for label,array in [('fortran',np.asfortranarray(features)),('float16',features.astype(np.float16)),('wrong-shape',features.reshape(400,560))]:
        stream=io.BytesIO();np.save(stream,array,allow_pickle=False);check(label,stream.getvalue(),good=False)
    bad=features.copy();bad[0,399,559]=np.nan
    stream=io.BytesIO();np.save(stream,bad,allow_pickle=False);check('nonfinite-padding',stream.getvalue(),good=False)
    stream=io.BytesIO();np.save(stream,features,allow_pickle=False);valid=stream.getvalue()
    check('truncated-payload',valid[:-1],good=False)
    check('extra-payload',valid+b'X',good=False)
    check('bad-magic',b'XXXXXX'+valid[6:],good=False)
    check('unsupported-version',valid[:6]+b'\x04\x00'+valid[8:],good=False)
    def raw_header(header):
        header = header.encode(); header += b' ' * ((-10-len(header)-1)%64)+b'\n'
        return b'\x93NUMPY\x01\x00'+len(header).to_bytes(2,'little')+header+features.tobytes()
    check('reordered-double-quoted-header',raw_header('{"shape": (1,400,560,), "fortran_order": False, "descr": "<f4"}'))
    check('duplicate-header-key',raw_header("{'descr':'<f4','descr':'<f4','shape':(1,400,560),'fortran_order':False}"),good=False)
    check('trailing-header-code',raw_header("{'descr':'<f4','shape':(1,400,560),'fortran_order':False};print('x')"),good=False)
(HERE/'summary.json').write_text(json.dumps({'scope':'host actual NumPy/manifest reader, no SDK or model execution', 'numpy':np.__version__, 'cases':records},ensure_ascii=False,indent=2)+'\n')
print(f'{len(records)} native feature-reader cases passed; real frontend arrays byte-identical')
