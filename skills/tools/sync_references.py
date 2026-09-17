#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check generated per-Skill reference copies; --apply updates managed copies only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path, PurePosixPath
import sys
PREFIX='<!-- GENERATED from skills/_shared/'

def resolve(root:Path,relative:str)->Path:
    p=PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or '\\' in relative:raise ValueError('unsafe reference path')
    out=root/relative
    if out.is_symlink() or not out.resolve().is_relative_to(root):raise ValueError('reference path escapes pack or is a symlink')
    return out

def sync(root:Path,apply:bool)->dict:
    manifest=json.loads((root/'pack.json').read_text(encoding='utf-8'))
    errors=[];changes=[];pending=[]
    for skill in manifest['skills']:
        name=skill['name']
        for ref in skill['shared_references']:
            source=resolve(root,'_shared/'+ref);target=resolve(root,name+'/references/'+ref)
            expected=f'<!-- GENERATED from skills/_shared/{ref}; edit source then run skills/tools/sync_references.py --apply. -->\n'+source.read_text(encoding='utf-8')
            current=target.read_text(encoding='utf-8') if target.is_file() else None
            if current!=expected:
                changes.append(target.relative_to(root).as_posix())
                if apply and current is not None and not current.startswith(PREFIX):errors.append('refuse unmanaged overwrite: '+str(target.relative_to(root)))
                pending.append((target,expected))
    # Preflight every destination before any write.
    if apply and not errors:
        for target,content in pending:
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(content,encoding='utf-8')
    if not apply and changes:errors.append('generated references are not current')
    return {'valid':not errors,'applied':bool(apply and not errors),'changed_files':changes,'errors':errors}

def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    try:r=sync(a.pack_root.resolve(),a.apply)
    except (OSError,ValueError,KeyError,TypeError) as e:r={'valid':False,'applied':False,'errors':[str(e)]}
    print(json.dumps(r,ensure_ascii=False,indent=2));return 0 if r['valid'] else 2
if __name__=='__main__':sys.exit(main())
