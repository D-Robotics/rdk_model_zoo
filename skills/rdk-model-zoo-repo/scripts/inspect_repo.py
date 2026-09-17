#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inspect a local Model Zoo checkout without running its code or fetching refs.

This is an inventory, not a hardware probe or a compatibility certification.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit

MANIFEST_LAYOUTS=('docs/manifests','docs/release','release')
PLATFORM_BY_BRANCH={'rdk_x5':'x5','rdk_s':'s','rdk_x3':'x3','rdk_x5_legacy':'x5-legacy'}
# This is the Skills Pack's maintenance source.  It is intentionally exposed
# as branch metadata, never used as a target-platform declaration.
MAINTENANCE_BRANCH='rdk_x5'

def git(root: Path, *args: str, optional: bool = False) -> str | None:
    """Run a fixed read-only Git command with fsmonitor and optional locks disabled."""
    p = subprocess.run(['git', '-c', 'core.fsmonitor=false', '-c', 'core.untrackedCache=false', '-C', str(root), *args],
                       env={**os.environ, 'GIT_OPTIONAL_LOCKS':'0', 'GIT_TERMINAL_PROMPT':'0'},
                       text=True, capture_output=True, timeout=30)
    if p.returncode:
        if optional: return None
        raise ValueError('git-read-failed; verify the path is an accessible Git checkout')
    return p.stdout


def confined(root: Path, relative: str) -> Path:
    """Resolve a repository-relative path, rejecting traversal and symlink escape."""
    p = PurePosixPath(relative)
    if not relative or p.is_absolute() or '..' in p.parts or '\\' in relative or re.match(r'^[A-Za-z]:',relative):
        raise ValueError('unsafe-sample-path')
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root): raise ValueError('sample-path-escapes-repository')
    if not resolved.is_dir(): raise ValueError('sample-directory-not-found')
    return resolved


def redact_remote(value: str) -> str:
    """Remove URL credentials/query/fragment and SCP-style user identifiers."""
    if '://' in value:
        p = urlsplit(value)
        host = p.hostname or ''
        try: port = f':{p.port}' if p.port else ''
        except ValueError: port = ''
        return urlunsplit((p.scheme,host+port,p.path,'',''))
    return re.sub(r'^[^@/:]+@', '', value).split('?',1)[0].split('#',1)[0]


def manifest_candidates(root: Path) -> list[str]:
    """Return supported manifest layouts present in the working tree."""
    return [f'{directory}/models.yaml' for directory in MANIFEST_LAYOUTS if (root/directory/'models.yaml').is_file()]


def inspect(repo: str, sample: str | None = None) -> dict:
    """Return observed checkout facts without parsing or executing repository code.

    Branch names remain hints.  The maintenance branch is reported separately
    so callers do not mistake the Skills Pack source checkout for a user's
    target.  Use read_catalog.py for validated release and platform metadata.
    """
    candidate = Path(repo).expanduser().resolve()
    if not candidate.is_dir(): raise ValueError('repository-directory-not-found')
    root = Path(git(candidate, 'rev-parse', '--show-toplevel').strip()).resolve()
    head = git(root, 'rev-parse', '--verify', 'HEAD').strip()
    branch_raw = git(root,'symbolic-ref','--quiet','--short','HEAD',optional=True)
    branch = branch_raw.strip() if branch_raw else None
    status = git(root,'status','--porcelain=v1','-z','--untracked-files=all')
    files = sorted(set(filter(None,git(root,'ls-files','--cached','--others','--exclude-standard','-z').split('\0'))))
    candidates=manifest_candidates(root)
    is_maintenance=branch==MAINTENANCE_BRANCH
    if is_maintenance:
        branch_role='maintenance'
    elif branch in PLATFORM_BY_BRANCH:
        branch_role='target'
    else:
        branch_role='unknown'
    remotes=[]
    for line in (git(root,'config','--get-regexp',r'^remote\..*\.url$',optional=True) or '').splitlines():
        key, sep, value=line.partition(' ')
        if sep: remotes.append({'name':key.removeprefix('remote.').removesuffix('.url'),'url':redact_remote(value)})
    result={'ok':True,'repo_root':str(root),'head':head,'branch':branch,'dirty':bool(status),
            'platform_hint':PLATFORM_BY_BRANCH.get(branch),
            'maintenance_branch':MAINTENANCE_BRANCH,'is_maintenance_branch':is_maintenance,
            'branch_role':branch_role,
            'remotes':remotes,'layout_roots':[p for p in ['samples','demos','resource','utils','tros'] if (root/p).is_dir()],
            'manifest_candidates':candidates,
            'guideline_candidates':[p for p in ['docs/Model_Zoo_Repository_Guidelines.md','AGENTS.md','docs/RELEASE_cn.md','docs/RELEASE.md'] if (root/p).is_file()],
            'file_count':len(files),'sample':None,'board_verified':False,
            'warnings':['Branch and remote names are context hints, not hardware or artifact evidence.']}
    if is_maintenance:
        result['warnings'].append('The maintenance branch is recorded separately from the target; it does not prove target hardware support.')
    if len(candidates)>1:
        result['warnings'].append('Multiple manifest layouts are present; target identity is withheld until a manifest is selected explicitly.')
    if PLATFORM_BY_BRANCH.get(branch)=='s':
        result['warnings'].append('S is a release group; no S100, S100P, or S600 target is inferred without explicit evidence.')
    if sample:
        target=confined(root,sample);prefix=target.relative_to(root).as_posix().rstrip('/')+'/'
        selected=[f for f in files if f.startswith(prefix)]
        result['sample']={'path':target.relative_to(root).as_posix(),'files':selected,'file_count':len(selected)}
    return result


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True,help='Local target checkout; no clone or network access is performed.')
    parser.add_argument('--sample',help='Optional safe repository-relative sample directory.')
    args=parser.parse_args()
    try: result=inspect(args.repo,args.sample); code=0
    except (OSError,ValueError,subprocess.SubprocessError) as e: result={'ok':False,'reason':str(e),'board_verified':False};code=2
    print(json.dumps(result,ensure_ascii=False,indent=2));return code
if __name__=='__main__': sys.exit(main())
