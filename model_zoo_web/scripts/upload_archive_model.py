#!/usr/bin/env python3
"""Upload one immutable deployment model from the local Tailscale-connected Mac."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from urllib.request import urlopen

from archive_assets import ARCHIVE_HTTPS, asset_prefix
from upload_archive_assets import REMOTE_ROOT

INSTALL_MODEL = r'''
import fcntl, hashlib, json, os, pathlib, shutil, sys, tempfile
root,prefix,filename,expected,size,staged=sys.argv[1:]
root=pathlib.Path(root).resolve(strict=True);source=pathlib.Path(staged)
parts=pathlib.PurePosixPath(prefix).parts
if len(parts)!=2 or parts[0] not in ('rdk_x5','rdk_s100','rdk_s100p','rdk_s600') or any(p in ('.','..') for p in parts):raise ValueError('Invalid model directory')
if pathlib.PurePosixPath(filename).name!=filename:raise ValueError('Model filename must be a basename')
def identity(path):
 h=hashlib.sha256();count=0
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block);count+=len(block)
 return h.hexdigest(),count
if identity(source)!=(expected,int(size)):raise ValueError('Staged upload differs from the reviewed artifact')
directory=root.joinpath(*parts);directory.mkdir(parents=True,exist_ok=True)
if directory.resolve()!=directory:raise ValueError('Model directory traverses a symlink')
locks=pathlib.Path.home()/'.archive-model-locks';locks.mkdir(mode=0o700,exist_ok=True)
with (locks/hashlib.sha256(prefix.encode()).hexdigest()).open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 # The upload home and public data may be separate mounted filesystems.
 # Copy into an owner-only directory on the destination filesystem before
 # publishing complete bytes with a no-replacement hard link.
 private=pathlib.Path(tempfile.mkdtemp(prefix='.archive-model-upload-',dir=directory))
 try:
  dest=directory/filename
  if dest.exists() or dest.is_symlink():
   if dest.is_symlink() or identity(dest)!=(expected,int(size)):raise ValueError('Refusing to overwrite a different model')
  else:
   ready=private/'model'
   shutil.copyfile(source,ready)
   if identity(ready)!=(expected,int(size)):raise ValueError('Destination staging differs from the reviewed artifact')
   ready.chmod(0o644);os.link(ready,dest)
  checksums=directory/'SHA256SUMS';entries={}
  if checksums.exists():
   for line in checksums.read_text().splitlines():
    digest,name=line.split(maxsplit=1);entries[name.lstrip('*')]=digest
  if filename in entries and entries[filename]!=expected:raise ValueError('Existing checksum entry identifies another artifact')
  entries[filename]=expected
  temp=private/'SHA256SUMS'
  temp.write_text(''.join(f'{entries[name]}  {name}\n' for name in sorted(entries)))
  temp.chmod(0o644);os.replace(temp,checksums)
  if identity(dest)!=(expected,int(size)):raise ValueError('Published model read-back differs')
 finally:
  shutil.rmtree(private)
source.unlink()
print(json.dumps({'status':'server-verified','prefix':prefix,'filename':filename,'sha256':expected,'size_bytes':int(size)}))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--ssh-host',required=True)
    parser.add_argument('--jump-host',required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    release=json.loads(args.manifest.read_text());artifact=release['artifact']
    source=release.get('source') or release['model']['source']
    platform=release.get('platform') or release['target']['platform']
    filename=artifact.get('name') or artifact.get('filename')
    asset_prefix(platform,source,filename,release.get('metadata_revision'))
    prefix=f'rdk_{platform}/{source}'
    if artifact['url']!=ARCHIVE_HTTPS+'/'+prefix+'/'+filename:raise ValueError('Manifest model URL does not match Archive layout')
    digest=hashlib.sha256()
    with args.model.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=artifact['sha256'] or args.model.stat().st_size!=artifact['size_bytes']:raise ValueError('Local model identity differs')
    ssh=['ssh','-J',args.jump_host,'-o','BatchMode=yes','-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath=/tmp/rdk-modelzoo-bulk-%r-%h-%p','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','ConnectTimeout=15',args.ssh_host]
    temp_command=shlex.join(['python3','-c','import os,tempfile;fd,p=tempfile.mkstemp(prefix=".archive-model-upload-",dir=os.path.expanduser("~"));os.close(fd);print(p)'])
    stage=subprocess.check_output(ssh+[temp_command],text=True).strip()
    subprocess.run(['scp','-J',args.jump_host,'-o','BatchMode=yes','-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath=/tmp/rdk-modelzoo-bulk-%r-%h-%p','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2',str(args.model),args.ssh_host+':'+stage],check=True)
    command=shlex.join(['python3','-c',INSTALL_MODEL,REMOTE_ROOT,prefix,filename,artifact['sha256'],str(artifact['size_bytes']),stage])
    receipt=json.loads(subprocess.check_output(ssh+[command],text=True))
    digest=hashlib.sha256();size=0
    with urlopen(artifact['url'],timeout=180) as response:
        if response.status!=200:raise ValueError('Anonymous model download failed')
        for block in iter(lambda:response.read(1024*1024),b''):digest.update(block);size+=len(block)
    if digest.hexdigest()!=artifact['sha256'] or size!=artifact['size_bytes']:raise ValueError('Anonymous model download differs')
    receipt.update(status='verified',url=artifact['url'],anonymous_https='verified')
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
