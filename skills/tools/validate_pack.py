#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check Skill structure, local resource closure, Python/JSON syntax and eval definitions.

This does not run Agent behavior evaluations, samples, network calls, or board tests.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
try:
    import yaml
except ImportError:
    yaml=None
SECTIONS=('Purpose','When to use','Instructions','Safety')
DIMENSIONS={'correctness','discoverability','security','effectiveness','efficiency'}
NAME=re.compile(r'^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$')
VERSION=re.compile(r'^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$')
LINK=re.compile(r'\[[^\]\n]*\]\(([^\s)]+)(?:\s+"[^"\n]*")?\)')

def validate_skill(root:Path)->tuple[list[str],int,str|None]:
    errors=[];count=0;version=None
    def err(s):errors.append(root.name+': '+s)
    skill=root/'SKILL.md'
    if not skill.is_file():return [root.name+': missing SKILL.md'],0,None
    text=skill.read_text(encoding='utf-8')
    match=re.match(r'^---\n(.*?)\n---\n(.*)$',text,re.S)
    if not match:return [root.name+': invalid YAML frontmatter delimiter'],0,None
    if yaml is None:return [root.name+': dependency-missing PyYAML'],0,None
    fm=yaml.safe_load(match[1]);body=match[2]
    if not isinstance(fm,dict):return [root.name+': invalid frontmatter'],0,None
    for field in ('name','description','version','license'):
        if not isinstance(fm.get(field),str) or not fm[field].strip():err('missing/non-string frontmatter '+field)
    name=fm.get('name','')
    if not isinstance(name,str) or len(name)>64 or not NAME.fullmatch(name) or name!=root.name:err('invalid/mismatched name')
    desc=fm.get('description','')
    if isinstance(desc,str) and (len(desc)>1024 or 'Do not use' not in desc):err('description length/negative trigger invalid')
    version=fm.get('version')
    if not isinstance(version,str) or not VERSION.fullmatch(version):err('invalid semantic Skill version')
    if len(body.splitlines())>500:err('body exceeds 500 lines')
    for sec in SECTIONS:
        if not re.search(r'^## '+re.escape(sec)+r'\s*$',body,re.M):err('missing section '+sec)
    card=root/'skill-card.md'
    if not card.is_file() or 'Owner' not in card.read_text(encoding='utf-8'):err('missing governance owner card')
    eval_file=root/'evals/tasks.yaml'
    if not eval_file.is_file():err('missing evals/tasks.yaml')
    else:
        rows=yaml.safe_load(eval_file.read_text(encoding='utf-8'))
        if not isinstance(rows,list) or not rows:err('eval definitions must be a non-empty list')
        else:
            ids=[];dims=set();negative=0
            for row in rows:
                if not isinstance(row,dict):err('invalid eval record');continue
                ids.append(row.get('id'));dims.add(row.get('dimension'))
                ex=row.get('expect',{})
                if not isinstance(ex,dict):err('invalid expect');continue
                if ex.get('skill') not in (root.name,'none'):err('eval primary skill must be this skill or none')
                if ex.get('skill')=='none':negative+=1
                if not isinstance(row.get('prompt'),str) or not row['prompt'].strip():err('empty eval prompt')
                b=ex.get('behavior')
                if not isinstance(b,list) or not b or any(not isinstance(s,str) or not s for s in b):err('invalid behavior assertions')
            count=len(rows)
            if any(not isinstance(i,str) or not i for i in ids) or len(ids)!=len(set(ids)):err('invalid/duplicate eval ids')
            if dims!=DIMENSIONS:err('evals must cover all five declared dimensions')
            if negative<1:err('missing negative routing eval')
    for f in root.rglob('*'):
        if f.is_symlink():err('symlink not allowed in flat distribution: '+f.relative_to(root).as_posix());continue
        if not f.is_file():continue
        if f.suffix=='.md':
            content=f.read_text(encoding='utf-8')
            for raw in LINK.findall(content):
                parsed=urlsplit(raw)
                if parsed.scheme or raw.startswith('#'):continue
                dest=(f.parent/unquote(parsed.path)).resolve()
                if not dest.is_relative_to(root):err('local link escapes independently installed skill: '+raw)
                elif not dest.exists():err('missing local link '+raw+' in '+f.relative_to(root).as_posix())
        elif f.suffix=='.py':
            try:ast.parse(f.read_text(encoding='utf-8'),filename=str(f))
            except SyntaxError as e:err('Python syntax: '+str(e))
        elif f.suffix=='.json':
            try:json.loads(f.read_text(encoding='utf-8'))
            except ValueError as e:err('JSON syntax: '+str(e))
    return errors,count,version

def main()->int:
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    g.add_argument('--pack-root',type=Path);g.add_argument('--skill-root',type=Path)
    a=p.parse_args();errors=[];count=0;dirs=[]
    try:
        if a.skill_root:
            dirs=[a.skill_root.resolve()];expected=None
        else:
            root=(a.pack_root or Path(__file__).resolve().parents[1]).resolve()
            manifest=json.loads((root/'pack.json').read_text(encoding='utf-8'))
            if manifest['version']!=(root/'VERSION').read_text().strip():errors.append('Pack VERSION mismatch')
            expected={r['name']:r['version'] for r in manifest['skills']}
            if len(expected)!=len(manifest['skills']):errors.append('Duplicate Skill registrations')
            dirs=sorted(p for p in root.iterdir() if p.is_dir() and (p/'SKILL.md').is_file())
            if set(expected)!={p.name for p in dirs}:errors.append('Skill set does not match pack.json')
        for directory in dirs:
            err,n,version=validate_skill(directory);errors.extend(err);count+=n
            if expected is not None and expected.get(directory.name)!=version:errors.append(directory.name+': member version mismatch')
        if not dirs:errors.append('No skills found')
    except Exception as e:errors.append(str(e))
    out={'valid':not errors,'skills_checked':len(dirs),'eval_cases':count,'errors':errors,
         'behavior_evaluated':False,'board_verified':False}
    print(json.dumps(out,ensure_ascii=False,indent=2));return 0 if out['valid'] else 2
if __name__=='__main__':sys.exit(main())
