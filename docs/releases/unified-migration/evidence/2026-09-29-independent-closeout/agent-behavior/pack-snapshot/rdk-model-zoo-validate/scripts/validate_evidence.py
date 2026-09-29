#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate receipt structure and optionally hash evidence files; never certify hardware.

Requires jsonschema. No commands in the receipt are executed. Exit 2 means invalid.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
try:
    import jsonschema
except ImportError:
    jsonschema=None
SCHEMA=Path(__file__).resolve().parent.parent/'schemas/verification.schema.json'


def safe_relative(value:str)->bool:
    p=PurePosixPath(value)
    return bool(value and not p.is_absolute() and '..' not in p.parts and '\\' not in value and not re.match(r'^[A-Za-z]:',value))


def validate(receipt:dict,evidence_root:Path|None=None)->list[str]:
    """Return all detectable structural/provenance errors, not semantic judgments."""
    if jsonschema is None:return ['dependency-missing: jsonschema; do not auto-install']
    schema=json.loads(SCHEMA.read_text(encoding='utf-8'))
    validator=jsonschema.Draft202012Validator(schema,format_checker=jsonschema.FormatChecker())
    errors=[f'{"/".join(map(str,e.absolute_path))}: {e.message}' for e in validator.iter_errors(receipt)]
    if errors:return errors
    ids=[c['id'] for c in receipt['checks']]
    if len(ids)!=len(set(ids)):errors.append('duplicate check ids')
    for c in receipt['checks']:
        prefix=c['id'];executed=c['status'] in ('passed','failed')
        if executed:
            target=receipt['target']
            if target['commit'] is None or target['dirty'] is None:errors.append(f'{prefix}: executed check requires commit and dirty state')
            if target['dirty'] and not target['patch_sha256']:errors.append(f'{prefix}: dirty checkout requires a patch/content digest including relevant untracked inputs')
            if c['execution'] is None or not c['evidence'] or c['result'] is None:errors.append(f'{prefix}: executed check requires execution, result and evidence')
            if not c['environment']['host']:errors.append(f'{prefix}: executed check requires host identity')
            if c['execution']:
                if c['status']=='passed' and c['execution']['exit_code']!=0:errors.append(f'{prefix}: passed check cannot have nonzero exit')
                start=datetime.fromisoformat(c['execution']['started_at'].replace('Z','+00:00'))
                end=datetime.fromisoformat(c['execution']['ended_at'].replace('Z','+00:00'))
                if end<start:errors.append(f'{prefix}: end time precedes start time')
            if c['level']=='board' or c['purpose'] in ('accuracy','consistency','performance'):
                for key in ('platform','model_variant','task','runtime','model_sha256','input_sha256'):
                    if not c['scope'][key]:errors.append(f'{prefix}: model execution requires scope.{key}')
            if c['level']=='board':
                for key in ('board','os','runtime_version'):
                    if not c['environment'][key]:errors.append(f'{prefix}: board execution requires environment.{key}')
            if c['purpose'] in ('accuracy','consistency') and (not c['result'] or not c['result'].get('acceptance')):
                errors.append(f'{prefix}: numeric validation requires declared acceptance criteria')
        else:
            if c['execution'] is not None:errors.append(f'{prefix}: not-run/not-applicable cannot claim an execution')
            if not c['reason']:errors.append(f'{prefix}: not-run/not-applicable requires a reason')
        for item in c['evidence']:
            if not safe_relative(item['path']):errors.append(f'{prefix}: unsafe evidence path');continue
            if evidence_root is not None:
                path=(evidence_root/item['path']).resolve()
                if not path.is_relative_to(evidence_root):errors.append(f'{prefix}: evidence symlink escapes root');continue
                if not path.is_file():errors.append(f'{prefix}: evidence file missing: {item["path"]}');continue
                h=hashlib.sha256()
                with path.open('rb') as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
                if h.hexdigest()!=item['sha256']:errors.append(f'{prefix}: evidence hash mismatch: {item["path"]}')
    return errors


def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipt',type=Path);p.add_argument('--evidence-root',type=Path)
    a=p.parse_args()
    try:
        if a.receipt.stat().st_size>8*1024*1024:raise ValueError('receipt-size-limit')
        receipt=json.loads(a.receipt.read_text(encoding='utf-8'));root=a.evidence_root.resolve() if a.evidence_root else None
        if root is not None and not root.is_dir():raise ValueError('evidence-root-not-found')
        errors=validate(receipt,root)
    except Exception as e:errors=[str(e)]
    valid=not errors
    print(json.dumps({'valid':valid,'errors':errors,'evidence_hashes_checked':bool(valid and a.evidence_root),
                      'board_verified':False,'scope':'Structure and optional file hashes only; results and hardware execution require independent evidence review.'},ensure_ascii=False,indent=2))
    return 0 if valid else 2
if __name__=='__main__':sys.exit(main())
