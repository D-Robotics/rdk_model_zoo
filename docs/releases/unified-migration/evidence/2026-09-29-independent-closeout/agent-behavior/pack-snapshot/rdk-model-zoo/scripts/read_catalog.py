#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read local, versioned Model Zoo manifests; never infer missing metrics.

Requires PyYAML. Outputs a working-tree snapshot, not a Release attestation.
Discovery: an explicit --manifest always wins. Otherwise the known flat and
per-platform layouts are probed; a single candidate is selected implicitly,
several candidates require explicit selection, and none means the manifest
stays unknown.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
try:
    import yaml
except ImportError:
    yaml=None
MAX_BYTES=16*1024*1024
MANIFEST_LAYOUTS=('docs/manifests','docs/release','release')
# Unified per-platform layouts and migration-window snapshots, probed in
# preference order: the active unified manifest wins over the frozen snapshot
# copy of the same platform. Declared here instead of imported so one installed
# skill directory stays self-contained; branch names are never consulted.
PLATFORM_MANIFEST_PATTERNS=(
    ('docs/release/{platform}','unified'),
    ('platforms/{platform}/docs/release','snapshot'),
    ('platforms/{platform}/release','snapshot'))
MANIFEST_PLATFORMS=('x5','s','x3')


def safe_path(root: Path, relative: str) -> Path:
    p=PurePosixPath(relative)
    if not relative or p.is_absolute() or '..' in p.parts or '\\' in relative or re.match(r'^[A-Za-z]:',relative):
        raise ValueError('unsafe-manifest-path')
    out=(root/relative).resolve()
    if not out.is_relative_to(root):raise ValueError('manifest-escapes-repository')
    return out


class AmbiguousManifest(ValueError):
    def __init__(self,candidates:list[str]):
        super().__init__('ambiguous-manifest; choose --manifest explicitly; candidates: '+', '.join(candidates))
        self.candidates=tuple(candidates)


def manifest_candidates(root: Path) -> list[str]:
    """Return manifest files actually present; active copies win over snapshots.

    Flat layouts describe single-platform refs. One candidate may be selected
    implicitly; several candidates require an explicit --manifest and are never
    narrowed by branch, filename, or the S release group.
    """
    flat=[f'{directory}/models.yaml' for directory in MANIFEST_LAYOUTS if (root/directory/'models.yaml').is_file()]
    per_platform:dict[str,str]={}
    for platform in MANIFEST_PLATFORMS:
        for pattern,_kind in PLATFORM_MANIFEST_PATTERNS:
            directory=pattern.format(platform=platform)
            if (root/directory/'models.yaml').is_file():
                per_platform.setdefault(platform,f'{directory}/models.yaml');break
    return sorted(set(flat)|set(per_platform.values()))


def guard_tree(value, stack=None, budget=None, depth=0):
    """Bound YAML aliases/depth and reject cycles before JSON expansion."""
    stack=set() if stack is None else stack;budget=[250000] if budget is None else budget
    budget[0]-=1
    if budget[0]<0 or depth>32:raise ValueError('manifest-structure-limit')
    if isinstance(value,(list,dict)):
        if id(value) in stack:raise ValueError('cyclic-yaml-not-supported')
        stack.add(id(value))
        children=value.values() if isinstance(value,dict) else value
        for item in children:guard_tree(item,stack,budget,depth+1)
        stack.remove(id(value))


def load(root: Path, relative: str, collection: str) -> tuple[dict, dict]:
    if yaml is None:raise ValueError('dependency-missing: PyYAML; do not auto-install')
    path=safe_path(root,relative)
    if not path.is_file():raise ValueError('manifest-not-found')
    if path.stat().st_size>MAX_BYTES:raise ValueError('manifest-size-limit')
    data=path.read_bytes()
    # SafeLoader does not execute Python tags. Reject duplicate mapping keys.
    # The repository's current benchmark manifest deliberately reuses short
    # anchor names for successive records.  YAML permits an anchor name to be
    # rebound; PyYAML's default Composer rejects that extension before the
    # SafeLoader can construct the document.  Remove only the previous binding
    # while composing a new anchored node so aliases still resolve to the most
    # recent definition, matching the parser used by the catalog builder.
    from yaml.events import AliasEvent
    class Loader(yaml.SafeLoader):
        def compose_node(self, parent, index):
            if not self.check_event(AliasEvent):
                event=self.peek_event()
                anchor=getattr(event,'anchor',None)
                if anchor is not None:
                    self.anchors.pop(anchor,None)
            return super().compose_node(parent,index)
    def mapping(loader,node,deep=False):
        loader.flatten_mapping(node); out={}
        for kn,vn in node.value:
            key=loader.construct_object(kn,deep=deep)
            if not isinstance(key,(str,int,float,bool,type(None))):raise ValueError('unsupported-yaml-key')
            if key in out:raise ValueError('duplicate-yaml-key')
            out[key]=loader.construct_object(vn,deep=deep)
        return out
    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,mapping)
    parsed=yaml.load(data,Loader=Loader);guard_tree(parsed)
    if not isinstance(parsed,dict) or type(parsed.get('schema_version')) is not int or parsed['schema_version']!=1:
        raise ValueError('unsupported-manifest-schema')
    rows=parsed.get(collection)
    if not isinstance(rows,list) or any(not isinstance(r,dict) or not isinstance(r.get('id'),str) for r in rows):
        raise ValueError('invalid-manifest-collection')
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate-manifest-id')
    if not isinstance(parsed.get('release',{}),dict):raise ValueError('invalid-release-section')
    info={'path':relative,'sha256':hashlib.sha256(data).hexdigest(),'release':parsed.get('release')}
    return parsed,info


def identity(root: Path) -> dict:
    env={**os.environ,'GIT_OPTIONAL_LOCKS':'0','GIT_TERMINAL_PROMPT':'0'}
    def read(*args):
        p=subprocess.run(['git','-c','core.fsmonitor=false','-C',str(root),*args],text=True,capture_output=True,timeout=30,env=env)
        return p.stdout.strip() if p.returncode==0 else None
    return {'root':str(root),'head':read('rev-parse','--verify','HEAD'),
            'dirty':(None if (s:=read('status','--porcelain=v1','--untracked-files=all')) is None else bool(s)),
            'source_mode':'working-tree'}


def catalog(repo:str,model:str|None=None,manifest:str|None=None,benchmarks:str|None=None)->dict:
    root=Path(repo).expanduser().resolve()
    if not root.is_dir():raise ValueError('repository-directory-not-found')
    if manifest is None:
        options=manifest_candidates(root)
        if not options:raise ValueError('manifest-not-found')
        if len(options)!=1:raise AmbiguousManifest(options)
        manifest=options[0]
    models,mi=load(root,manifest,'models')
    warnings=[];bi=None;bench_rows=[]
    if manifest.startswith('platforms/'):
        warnings.append('Selected manifest is a frozen migration-window snapshot under platforms/; it is not the active unified release location.')
    benchmark_path=benchmarks or (PurePosixPath(manifest).parent/'benchmarks.yaml').as_posix()
    if (root/benchmark_path).is_file() or benchmarks:
        raw,bi=load(root,benchmark_path,'benchmarks');bench_rows=raw['benchmarks']
        mr=models.get('release',{});br=raw.get('release',{})
        for field in ('platform','tag'):
            if mr.get(field) and br.get(field) and mr[field]!=br[field]:
                raise ValueError('manifest-release-identity-mismatch')
    else:warnings.append('No companion benchmark manifest; no performance or accuracy values inferred.')
    query=model.casefold() if model else None
    selected=[r for r in models['models'] if query is None or query in str(r['id']).casefold() or query in str(r.get('name','')).casefold()]
    ids={r['id'] for r in selected}
    rows=[r for r in bench_rows if r.get('sample_id') in ids]
    if not selected:warnings.append('No matching inventory record; this does not prove hardware incompatibility.')
    release=models.get('release') or {}
    if str(release.get('platform','')).casefold() == 's':
        warnings.append('S release group does not imply support for S100, S100P, or S600; only explicit per-record hardware scope is evidence.')
    warnings.append('Inventory and published benchmark claims are not new board execution evidence. Preserve null fields and qualifiers.')
    return {'ok':True,'repository':identity(root),'model_manifest':mi,'benchmark_manifest':bi,'models':selected,'benchmarks':rows,'warnings':warnings,'board_verified':False}


def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True)
    p.add_argument('--model',help='Case-insensitive substring in model id/name; no variant inference.')
    p.add_argument('--manifest',help='Safe repository-relative models.yaml path, required if layouts are ambiguous.')
    p.add_argument('--benchmark-manifest',help='Explicit safe companion benchmark manifest path.')
    a=p.parse_args()
    try:r=catalog(a.repo,a.model,a.manifest,a.benchmark_manifest);code=0
    except Exception as e:
        # No commands have been taken from manifests. Parser failures stay explicit.
        r={'ok':False,'reason':str(e),'board_verified':False}
        if isinstance(e,AmbiguousManifest):r['manifest_candidates']=list(e.candidates)
        code=2
    print(json.dumps(r,ensure_ascii=False,indent=2,default=str));return code
if __name__=='__main__':sys.exit(main())
