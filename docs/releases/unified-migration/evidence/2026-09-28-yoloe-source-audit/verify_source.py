"""Offline source, sidecar, vocabulary and active-hash verification. No board access."""
from pathlib import Path
import hashlib,json,re,subprocess
import yaml
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[4]
existing=json.loads((OUT/'existing-source.json').read_text())
for row in existing:
    data=(ROOT/row['snapshot_path']).read_bytes()
    assert row['snapshot_matches'] and hashlib.sha256(data).hexdigest()==row['source_sha256']
    assert data==subprocess.check_output(['git','cat-file','blob',row['git_blob']],cwd=ROOT)
source=json.loads((OUT/'restored-source.json').read_text())
for row in source['files']:
    data=(ROOT/row['destination']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==row['sha256']
    assert data==subprocess.check_output(['git','cat-file','blob',row['git_blob']],cwd=ROOT)
for row in json.loads((OUT/'release-capture.json').read_text()):
    assert row['status']=='captured'
    data=(OUT/row['file']).read_bytes()
    assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
models=yaml.safe_load((ROOT/'docs/release/s/models.yaml').read_text())
assets=next(m['assets'] for m in models['models'] if m['id']=='yoloe26_seg')
facts=json.loads((OUT/'release-facts.json').read_text())['records']
for row in facts:
    assert next(a['sha256'] for a in assets if a['filename']==row['filename'])==row['sha256']
    march=row['march'];stem=f"yoloe_26{row['size']}_seg_pf"
    meta=json.loads((OUT/'release'/march/(stem+'.json')).read_text())
    manifest=json.loads((OUT/'release'/march/'manifest.json').read_text())
    for suffix in ('.json','.names'):
        data=(OUT/'release'/march/(stem+suffix)).read_bytes()
        expected=manifest['files'][stem+suffix]
        assert expected['sha256']==hashlib.sha256(data).hexdigest() and expected['bytes']==len(data)
    assert meta['hbm_sha256']==row['sha256'] and meta['hbm_bytes']==row['bytes']
    assert meta['protocol']=='yoloe26-pf-raw-v1' and meta['reg_max']==1 and meta['end2end'] is True
    assert meta['hbm_output_quantized'] is True and len(meta['names'])==4585
    assert meta['names']==(OUT/'release'/march/(stem+'.names')).read_text().splitlines()
labels=[ROOT/'platforms/x5/datasets/yoloe/yoloe_seg_pf_classes.names',ROOT/'platforms/s/samples/vision/yoloe11_seg/test_data/coco_extended.names',ROOT/'platforms/s/samples/vision/yoloe26_seg/test_data/coco_extended.names']
reference=(OUT/'release/nash-e/yoloe_26n_seg_pf.names').read_text().splitlines()
assert all(p.read_text().splitlines()==reference for p in labels)
logdir=ROOT/'platforms/s/samples/vision/yoloe11_seg/conversion'
dtypes={}
for name in ('hb_model_info_yoloe_11s_seg.txt','hb_combine_yoloe_11s_seg.txt'):
    dtypes[name]=re.findall(r'\boutput\s+\[[^\]]+\]\s+(FLOAT32|INT32|INT16|INT8)',(logdir/name).read_text())
assert dtypes['hb_model_info_yoloe_11s_seg.txt']==['FLOAT32']*10
assert dtypes['hb_combine_yoloe_11s_seg.txt']==['FLOAT32','INT32','INT32']*3+['INT16']
print(json.dumps({'existing_source_files_verified':len(existing),'source_files_verified':len(source['files']),'release_files_verified':22,'active_hbm_hashes':len(facts),'vocabulary_entries':len(reference),'all_source_vocabularies_match':True,'s11_intermediate_and_final_dtypes':dtypes,'hbm_downloaded':False,'board':'not-run'},indent=2))
