"""Validate complete native deployment pipelines and single-file releases."""
import base64
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import PurePosixPath
import re
import tarfile

KINDS = {'encoder-decoder': ('encoder', 'decoder'), 'fasterrcnn-split': ('parta', 'partb'),
         'paraformer-staged': ('encoder', 'predictor', 'decoder')}
SOURCES = {'efficient_sam': 'encoder-decoder', 'mobile_sam': 'encoder-decoder',
           'fasterrcnn': 'fasterrcnn-split', 'clip': 'clip-image-text',
           'paraformer': 'paraformer-staged'}
SHA = re.compile(r'^[0-9a-f]{64}$')
ARCHIVE = 'https://archive.d-robotics.cc/downloads/rdk_model_zoo_web'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(item):
    name = item.get('filename', '')
    require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name),
            'Bundle members must have safe basenames')
    require(bool(SHA.fullmatch(str(item.get('sha256', '')))), 'Missing component SHA256')
    size = item.get('size_bytes')
    require(type(size) is int and size > 0, 'Missing component size')
    return name


def tensors(items):
    require(isinstance(items, list) and bool(items), 'Component I/O must be recorded')
    names = set()
    for item in items:
        name = item.get('name')
        require(isinstance(name, str) and name and name not in names, 'Invalid component tensor name')
        names.add(name)
        require(isinstance(item.get('dtype'), str) and bool(item['dtype']), 'Missing component tensor dtype')
        shape = item.get('shape')
        require(isinstance(shape, list) and bool(shape)
                and all(type(d) is int and d > 0 for d in shape), 'Invalid component tensor shape')


def validate_components(row, *, platform, source, oe=None):
    artifact = row['artifact']
    bundle = artifact.get('bundle')
    if isinstance(bundle, dict) and bundle.get('kind') == 'clip-image-text':
        from clip_package import validate_components as validate_clip
        return validate_clip(row, platform=platform, source=source, oe=oe)
    if bundle is None:
        require(source not in SOURCES and 'components' not in artifact
                and 'components' not in row.get('provenance', {})
                and 'components' not in row.get('conversion', {}),
                'This task requires a complete deployment bundle')
        return None
    require(isinstance(bundle, dict) and bundle.get('kind') in KINDS, 'Unknown deployment bundle kind')
    kind = bundle['kind']
    require(source not in SOURCES or SOURCES[source] == kind, 'Bundle kind differs from its native task')
    require((artifact.get('filename') or artifact.get('name') or artifact['url'].rsplit('/',1)[-1]).endswith('.tar.gz'),
            'Pipeline deployment must download as tar.gz')
    roles = KINDS[kind]
    components = artifact.get('components')
    require(isinstance(components, list) and len(components) == len(roles), 'Incomplete component set')
    require([c.get('role') for c in components] == list(roles), 'Component roles must match pipeline order')
    members = bundle.get('members')
    require(isinstance(members, list) and bool(members), 'Missing bundle member inventory')
    names = [identity(c) for c in components + members]
    require(len(set(names)) == len(names), 'Duplicate bundle member')
    require(bundle.get('descriptor') == 'deployment.json' and 'deployment.json' in names,
            'Missing deployment descriptor')
    require(any(m['filename'].startswith('LICENSE') for m in members), 'Missing model license in bundle')
    provenance = row.get('provenance', {})
    require('onnx_sha256' not in provenance and 'config_sha256' not in provenance,
            'A pipeline cannot use one component as its entire conversion provenance')
    prov = provenance.get('components', {})
    conv = row.get('conversion', {}).get('components', {})
    require(set(prov) == set(conv) == set(roles), 'Missing per-component conversion evidence')
    by_name = {m['filename']:m for m in members}
    for c in components:
        role = c['role']
        require(c.get('platform') == platform, 'Component platform differs from bundle')
        fmt = 'BIN' if platform == 'x5' else 'HBM'
        require(c.get('format') == fmt and c['filename'].endswith('.'+fmt.lower()), 'Wrong component model format')
        require(c.get('url') == f'{ARCHIVE}/rdk_{platform}/{source}/{c["filename"]}',
                'Component URL must use its exact Archive target and sample')
        require(all(SHA.fullmatch(str(prov[role].get(k,''))) for k in ('onnx_sha256','config_sha256')),
                'Missing component ONNX/config identity')
        evidence = conv[role]
        require(evidence.get('status') == 'passed' and evidence.get('artifact_sha256') == c['sha256'],
                'Component conversion describes another artifact')
        tensors(evidence.get('inputs'))
        tensors(evidence.get('outputs'))
        report = by_name.get(evidence.get('oe_report_filename'))
        require(report is not None and report['filename'].endswith('.html')
                and report['sha256'] == evidence.get('oe_report_sha256'), 'Missing exact component OE report')
        data = by_name.get(evidence.get('oe_data_filename'))
        require(data is not None and data['filename'].endswith('.json')
                and data['sha256'] == evidence.get('oe_data_sha256'), 'Missing exact component OE JSON')
    require(row.get('conversion', {}).get('outputs') == conv[roles[-1]]['outputs'],
            'Bundle outputs must identify its final native stage')
    pipeline = row.get('validation', {}).get('host_pipeline', {})
    require(pipeline.get('status') == 'passed' and pipeline.get('stages') == list(roles)
            and SHA.fullmatch(str(pipeline.get('evidence_sha256','')))
            and pipeline.get('component_sha256') == {c['role']:c['sha256'] for c in components},
            'Pipeline release requires actual connected host inference evidence')
    host_member = by_name.get(pipeline.get('evidence_file'))
    require(host_member is not None and host_member['filename'].endswith('.json')
            and host_member['sha256'] == pipeline['evidence_sha256'],
            'Connected host inference report must be included with its exact SHA256')
    if oe is not None:
        require(oe.get('provenance', {}).get('artifact_sha256') == artifact['sha256'], 'OE describes another bundle')
        records = oe.get('components', {})
        require(set(records) == set(roles), 'OE must cover every component')
        for c in components:
            r = records[c['role']]
            expected = dict(prov[c['role']], artifact_sha256=c['sha256'],
                            oe_report_sha256=conv[c['role']]['oe_report_sha256'],
                            oe_data_sha256=conv[c['role']]['oe_data_sha256'])
            require(all(r.get(k) == v for k,v in expected.items()), 'OE component identity differs from conversion')
    return components


class ReportInventory(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.parts = []
        self.count = 0

    def handle_starttag(self, tag, attrs):
        if tag == 'script' and dict(attrs).get('id') == 'component-reports':
            self.count += 1
            self.active = True

    def handle_endtag(self, tag):
        if tag == 'script':
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)


def validate_bundle_files(row, model_bytes, html_bytes, oe):
    if row['artifact'].get('bundle', {}).get('kind') == 'clip-image-text':
        from clip_package import validate_bundle_files as validate_clip
        return validate_clip(row, model_bytes, html_bytes, oe)
    platform, source = row['target']['platform'], row['model']['source']
    components = validate_components(row, platform=platform, source=source, oe=oe)
    if components is None:
        return
    bundle = row['artifact']['bundle']
    expected = {i['filename']:i for i in components + bundle['members']}
    payloads = {}
    with tarfile.open(fileobj=io.BytesIO(model_bytes), mode='r:gz') as archive:
        for member in archive:
            require(member.isfile() and member.name in expected and member.name not in payloads,
                    'Bundle contains an unlisted, duplicated or non-regular member')
            info = expected[member.name]
            require(member.size == info['size_bytes'], 'Bundle component size differs')
            data = archive.extractfile(member).read()
            require(hashlib.sha256(data).hexdigest() == info['sha256'], 'Bundle component checksum differs')
            payloads[member.name] = data
    require(set(payloads) == set(expected), 'Bundle is missing required files')
    require(json.loads(payloads['deployment.json']) == {
        'schema_version': 1, 'target': platform, 'kind': bundle['kind'], 'components': components},
        'Deployment descriptor differs from release components')
    for component in components:
        role = component['role']
        evidence = row['conversion']['components'][role]
        native = json.loads(payloads[evidence['oe_data_filename']])
        provenance = native.get('provenance',{})
        inputs = provenance.get('derived_metadata',{}).get('source_evidence',{})
        require(provenance.get('artifact_sha256') == component['sha256']
                and native.get('target',{}).get('platform') == platform,
                'Native component OE JSON belongs to another model or platform')
        require(native.get('io') == {'inputs':evidence['inputs'],'outputs':evidence['outputs']},
                'Component I/O differs from its native OE JSON')
        for field,key in [('onnx_sha256','onnx'),('config_sha256','config')]:
            actual = provenance.get(field,inputs.get(key,{}).get('sha256'))
            require(actual == row['provenance']['components'][role][field],
                    'Component graph/config differs from its native OE JSON')
    host = json.loads(payloads[row['validation']['host_pipeline']['evidence_file']])
    require(host.get('status') == 'passed', 'Connected host inference did not pass')
    checkpoint = host.get('source_checkpoint_sha256',host.get('checkpoint_sha256'))
    require(checkpoint == row['provenance'].get('source_checkpoint_sha256'),
            'Connected host report belongs to another source checkpoint')
    if bundle['kind'] == 'encoder-decoder':
        graphs = {role:host.get(role+'_onnx_sha256') for role in ('encoder','decoder')}
    else:
        graphs = host.get('onnx_sha256',{})
    require(graphs == {role:p['onnx_sha256'] for role,p in row['provenance']['components'].items()},
            'Connected host report belongs to other component graphs')
    cases = host.get('cases',[])
    require(isinstance(cases,list) and len(cases)>=4
            and len({item.get('input') for item in cases})==len(cases)
            and all(item.get('comparisons') and all(value.get('passed') is True
                    for value in item['comparisons'].values()) for item in cases),
            'Connected host report lacks its passed real and deterministic test cases')
    parser = ReportInventory()
    parser.feed(html_bytes.decode('utf-8'))
    require(parser.count == 1, 'Combined OE HTML must preserve component reports')
    reports = json.loads(''.join(parser.parts))
    require(set(reports) == {c['role'] for c in components}, 'Combined OE HTML is incomplete')
    for role, encoded in reports.items():
        report = row['conversion']['components'][role]['oe_report_filename']
        require(base64.b64decode(encoded, validate=True) == payloads[report],
                'Combined OE HTML embeds another component report')
