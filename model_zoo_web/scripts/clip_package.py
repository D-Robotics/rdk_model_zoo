"""Validate complete CLIP image-BPU/text-CPU bundles with distinct evidence kinds."""
import base64
import hashlib
import io
import json
import tarfile

from component_package import ARCHIVE, SHA, ReportInventory, identity, require

ROLES = ('image', 'text')


def report_fields(role):
    prefix = 'oe' if role == 'image' else 'host'
    return (prefix+'_report_filename', prefix+'_report_sha256',
            prefix+'_data_filename', prefix+'_data_sha256')


def validate_components(row, *, platform, source, oe=None):
    require(source == 'clip', 'CLIP bundle kind requires the CLIP sample')
    artifact = row['artifact']
    bundle = artifact['bundle']
    name = artifact.get('filename') or artifact.get('name') or artifact['url'].rsplit('/',1)[-1]
    require(name.endswith('.tar.gz'), 'CLIP must download as a complete tar.gz')
    components = artifact.get('components')
    require(isinstance(components,list) and [c.get('role') for c in components] == list(ROLES),
            'CLIP requires image and text components in native order')
    members = bundle.get('members')
    require(isinstance(members,list) and bool(members), 'Missing CLIP bundle members')
    names = [identity(item) for item in components+members]
    require(len(names) == len(set(names)), 'Duplicate CLIP bundle member')
    by_name = {m['filename']:m for m in members}
    require(bundle.get('descriptor') == 'deployment.json' and 'deployment.json' in by_name,
            'Missing CLIP deployment descriptor')
    require(any(n.startswith('LICENSE') for n in names), 'Missing CLIP model license')
    for key in ('bpe_filename','tokenizer_filename','text_config_filename'):
        require(bundle.get(key) in by_name, 'CLIP must include BPE, tokenizer and CPU runtime configuration')
    provenance = row.get('provenance',{})
    require('onnx_sha256' not in provenance and 'config_sha256' not in provenance,
            'CLIP cannot identify both encoders with one graph/config')
    prov = provenance.get('components',{})
    conv = row.get('conversion',{}).get('components',{})
    require(set(prov) == set(conv) == set(ROLES), 'Missing CLIP component provenance')
    expected_io = {
        'image': {'inputs':[{'name':'image','shape':[1,3,224,224],'dtype':'float32'}],
                  'outputs':[{'name':'image_features','shape':[1,512],'dtype':'float32'}]},
        'text': {'inputs':[{'name':'tokens','shape':['prompts',77],'dtype':'int32'}],
                 'outputs':[{'name':'text_features','shape':['prompts',512],'dtype':'float32'}]},
    }
    for c in components:
        role = c['role']
        fmt = ('BIN' if platform == 'x5' else 'HBM') if role == 'image' else 'ONNX'
        require(c.get('platform') == platform and c.get('format') == fmt
                and c['filename'].endswith('.'+fmt.lower()), 'Wrong CLIP component format/target')
        require(c.get('execution') == ('BPU' if role == 'image' else 'CPU'), 'Wrong CLIP execution device')
        require(c.get('url') == f'{ARCHIVE}/rdk_{platform}/clip/{c["filename"]}', 'Wrong CLIP component URL')
        require(all(SHA.fullmatch(str(prov[role].get(k,''))) for k in ('onnx_sha256','config_sha256')),
                'Missing CLIP graph/config identity')
        evidence = conv[role]
        require(evidence.get('status') == 'passed' and evidence.get('artifact_sha256') == c['sha256'],
                'CLIP evidence belongs to another component')
        require({key:evidence.get(key) for key in ('inputs','outputs')} == expected_io[role],
                'CLIP ABI differs from the native ViT-B/32 contract')
        require(evidence.get('kind') == ('native-bpu-compile' if role == 'image' else 'cpu-onnx-host-check'),
                'CPU text export must not be represented as BPU compilation')
        rf,rs,df,ds = report_fields(role)
        for file_key,sha_key,suffix in ((rf,rs,'.html'),(df,ds,'.json')):
            item = by_name.get(evidence.get(file_key))
            require(item is not None and item['filename'].endswith(suffix)
                    and item['sha256'] == evidence.get(sha_key), 'Missing exact CLIP component report')
        if role == 'text':
            require(prov[role]['onnx_sha256'] == c['sha256'], 'CPU text artifact must be the checked ONNX')
            require(by_name[bundle['text_config_filename']]['sha256'] == prov[role]['config_sha256'],
                    'CPU text runtime configuration identity differs')
    outputs = expected_io['image']['outputs'] + expected_io['text']['outputs']
    require(row['conversion'].get('outputs') == outputs, 'CLIP must retain image and text feature outputs')
    pipeline = row.get('validation',{}).get('host_pipeline',{})
    require(pipeline.get('status') == 'passed' and pipeline.get('stages') == list(ROLES)
            and pipeline.get('component_sha256') == {c['role']:c['sha256'] for c in components},
            'Missing complete CLIP host matching evidence')
    member = by_name.get(pipeline.get('evidence_file'))
    require(member is not None and member['sha256'] == pipeline.get('evidence_sha256'),
            'CLIP host matching report is absent')
    if oe is not None:
        require(oe.get('provenance',{}).get('artifact_sha256') == artifact['sha256'], 'OE describes another CLIP bundle')
        require(oe.get('aggregate',{}).get('latency_ms') is None
                and oe.get('aggregate',{}).get('fps_derived') is None and oe.get('board_measurement') is None,
                'No CLIP pipeline or board speed was measured')
        records = oe.get('components',{})
        require(set(records) == set(ROLES), 'CLIP reports must cover both execution devices')
        for c in components:
            role=c['role'];expected=dict(prov[role],artifact_sha256=c['sha256'])
            for _,key,_,data_key in (report_fields(role),):
                expected[key]=conv[role][key];expected[data_key]=conv[role][data_key]
            require(all(records[role].get(k) == v for k,v in expected.items()), 'CLIP report component identity differs')
    return components


def validate_bundle_files(row, model_bytes, html_bytes, oe):
    components = validate_components(row,platform=row['target']['platform'],source=row['model']['source'],oe=oe)
    bundle=row['artifact']['bundle'];expected={i['filename']:i for i in components+bundle['members']};files={}
    with tarfile.open(fileobj=io.BytesIO(model_bytes),mode='r:gz') as archive:
        for member in archive:
            require(member.isfile() and member.name in expected and member.name not in files,
                    'Unexpected, duplicated or non-regular CLIP bundle member')
            data=archive.extractfile(member).read();item=expected[member.name]
            require(len(data) == item['size_bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'],
                    'CLIP bundle member checksum differs')
            files[member.name]=data
    require(set(files) == set(expected), 'CLIP bundle is incomplete')
    require(json.loads(files['deployment.json']) == {'schema_version':1,'target':row['target']['platform'],
            'kind':'clip-image-text','components':components}, 'CLIP deployment descriptor differs')
    for c in components:
        role=c['role'];evidence=row['conversion']['components'][role];_,_,df,_=report_fields(role)
        native=json.loads(files[evidence[df]]);prov=native.get('provenance',{})
        inputs=prov.get('derived_metadata',{}).get('source_evidence',{})
        require(prov.get('artifact_sha256') == c['sha256'] and native.get('target',{}).get('platform') == row['target']['platform'],
                'CLIP component report belongs to another artifact/target')
        require(native.get('io') == {k:evidence[k] for k in ('inputs','outputs')}, 'CLIP component report ABI differs')
        for key,field in (('onnx_sha256','onnx'),('config_sha256','config')):
            require(prov.get(key,inputs.get(field,{}).get('sha256')) == row['provenance']['components'][role][key],
                    'CLIP native graph/config identity differs')
        if role == 'text':
            require(native.get('kind') == 'cpu-onnx-host-check' and native.get('board_measurement') is None,
                    'CLIP text host check must not assert board measurements')
    host=json.loads(files[row['validation']['host_pipeline']['evidence_file']])
    require(host.get('status') == 'passed' and host.get('checkpoint',{}).get('sha256') == row['provenance'].get('source_checkpoint_sha256'),
            'CLIP host report belongs to another source checkpoint')
    require({r:host.get('graphs',{}).get(r,{}).get('sha256') for r in ROLES}
            == {r:row['provenance']['components'][r]['onnx_sha256'] for r in ROLES}, 'CLIP host graphs differ')
    cases=host.get('cases',[])
    require(len(cases)>=4 and len({c.get('name') for c in cases}) == len(cases)
            and all(c.get('status') == 'passed' and all(type(c.get(k)) in (int,float) and c[k]>=0
                    for k in ('image_max_abs','text_max_abs','cosine_max_abs')) for c in cases),
            'CLIP host matching checks did not all pass')
    require(host.get('atol') == 1e-4 and host.get('rtol') == 1e-3 and host.get('tokenizer_matches_official') is True,
            'CLIP float or tokenizer gate differs')
    require(host.get('bpe',{}).get('sha256') == expected[bundle['bpe_filename']]['sha256'], 'CLIP BPE differs from host export')
    config=json.loads(files[bundle['text_config_filename']])
    require(config.get('bpe_sha256') == expected[bundle['bpe_filename']]['sha256']
            and config.get('tokenizer_sha256') == expected[bundle['tokenizer_filename']]['sha256'],
            'CLIP CPU runtime configuration uses another tokenizer')
    parser=ReportInventory();parser.feed(html_bytes.decode('utf-8'));require(parser.count == 1,'Missing combined CLIP report inventory')
    reports=json.loads(''.join(parser.parts));require(set(reports) == set(ROLES), 'CLIP report inventory is incomplete')
    for role,data in reports.items():
        rf,_,_,_=report_fields(role)
        require(base64.b64decode(data,validate=True) == files[row['conversion']['components'][role][rf]],
                'CLIP combined HTML embeds another component report')
