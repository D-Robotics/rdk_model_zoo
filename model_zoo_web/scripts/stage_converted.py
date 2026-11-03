#!/usr/bin/env python3
"""Freeze an initial conversion-only package from real model and report files."""
import argparse
import json
from pathlib import Path

from archive_assets import PACKAGE_FILES, prepare, sha
from component_package import validate_bundle_files


def stage(manifest, model, report, oe_data, output):
    if manifest.get('status') != 'converted':
        raise ValueError('Initial conversion staging requires status: converted')
    validation = manifest.get('validation', {})
    if any(validation.get(key) != 'passed' for key in ('conversion','host_smoke')):
        raise ValueError('Conversion and host checks must pass before model publication')
    if any(validation.get(key) != 'not-run' for key in ('board_smoke','accuracy','runtime_1','runtime_2','cpp_1','cpp_2')):
        raise ValueError('Initial staging requires explicit not-run board checks; publish supplements separately')
    if 'accuracy' in manifest or 'performance' in manifest:
        raise ValueError('Initial staging cannot inherit historical measurements')
    artifact = manifest['artifact']
    data = model.read_bytes()
    digest = sha(data)
    if artifact['sha256'] != digest or artifact['size_bytes'] != len(data) or validation.get('artifact_sha256') != digest:
        raise ValueError('Model identity does not match the conversion manifest')
    filename = artifact.get('name') or artifact.get('filename')
    if model.name != filename:
        raise ValueError('Model filename differs from the manifest')
    validate_bundle_files(manifest, data, report.read_bytes(), json.loads(oe_data.read_text()))
    prefix, files = prepare((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode(),
        report.read_bytes(),oe_data.read_bytes(),platform=manifest['target']['platform'],
        source=manifest['model']['source'],filename=filename,model_url=artifact['url'],expected_model_sha256=digest)
    output.mkdir(parents=True,exist_ok=False)
    for name,payload in files.items(): (output/name).write_bytes(payload)
    return {'status':'staged-not-uploaded','prefix':prefix,'model_sha256':digest,
        'model_size_bytes':len(data),'files':{name:sha(files[name]) for name in PACKAGE_FILES}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','model','report','oe-data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=stage(json.loads(args.manifest.read_text()),args.model,args.report,args.oe_data,args.output)
    print(json.dumps(result))


if __name__=='__main__':
    main()
