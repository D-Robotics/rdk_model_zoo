#!/usr/bin/env python3
"""Upload an immutable model from a source host or an explicit local path.

Use --source-host for server-held models: relay SSH streams without writing a
local model file, then verify the anonymous download on the source host.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from urllib.request import urlopen

from archive_assets import ARCHIVE_HTTPS, asset_prefix
from upload_archive_assets import REMOTE_ROOT, transport_options

FILE_INFO = r'''
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1])
if not p.exists():
 print(json.dumps({'exists':False}))
else:
 if not p.is_file() or p.is_symlink():raise ValueError('Expected a regular model file')
 h=hashlib.sha256();size=0
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block);size+=len(block)
 result={'exists':True,'sha256':h.hexdigest(),'size_bytes':size}
 checksums=p.parent/'SHA256SUMS'
 if checksums.is_file():
  for line in checksums.read_text().splitlines():
   parts=line.split(maxsplit=1)
   if len(parts)==2 and parts[1].lstrip('*')==p.name:result['directory_checksum']=parts[0]
 print(json.dumps(result))
'''

RECEIVE_MODEL = r'''
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);expected=sys.argv[2];expected_size=int(sys.argv[3])
if not p.is_file() or p.is_symlink():raise ValueError('Missing private upload staging file')
h=hashlib.sha256();size=0
with p.open('wb') as f:
 for block in iter(lambda:sys.stdin.buffer.read(1024*1024),b''):f.write(block);h.update(block);size+=len(block)
if h.hexdigest()!=expected or size!=expected_size:
 p.unlink();raise ValueError('Streamed upload differs from the reviewed model')
print(json.dumps({'status':'stream-verified','sha256':h.hexdigest(),'size_bytes':size}))
'''

VERIFY_HTTP = r'''
import hashlib,json,sys
from urllib.request import urlopen
h=hashlib.sha256();size=0
with urlopen(sys.argv[1],timeout=180) as response:
 if response.status!=200:raise ValueError('Anonymous model download failed')
 for block in iter(lambda:response.read(1024*1024),b''):h.update(block);size+=len(block)
print(json.dumps({'sha256':h.hexdigest(),'size_bytes':size}))
'''


def remote_json(ssh,script,*arguments):
    command=shlex.join(['python3','-c',script,*map(str,arguments)])
    return json.loads(subprocess.check_output(ssh+[command],text=True))


def stream_model(source_ssh,source_path,destination_ssh,staged,artifact):
    """Relay SSH stdout directly to SSH stdin without a local model file."""
    script="import shutil,sys;f=open(sys.argv[1],'rb');shutil.copyfileobj(f,sys.stdout.buffer,1024*1024)"
    source=subprocess.Popen(source_ssh+[shlex.join(['python3','-c',script,str(source_path)])],stdout=subprocess.PIPE)
    try:
        command=shlex.join(['python3','-c',RECEIVE_MODEL,staged,artifact['sha256'],str(artifact['size_bytes'])])
        receipt=json.loads(subprocess.check_output(destination_ssh+[command],stdin=source.stdout,text=True))
        source.stdout.close()
        if source.wait()!=0:raise ValueError('Source SSH model stream failed')
        return receipt
    finally:
        if source.poll() is None:source.terminate();source.wait()

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
    parser.add_argument('--model',type=Path,required=True,help='Model path on --source-host, or a local path when source-host is omitted')
    parser.add_argument('--source-host',help='SSH host holding the model; streams directly without local model staging')
    parser.add_argument('--source-jump-host',help='Optional SSH jump host for the source connection')
    parser.add_argument('--ssh-host',required=True)
    parser.add_argument('--jump-host',help='Optional SSH jump host; omitted for a direct connection')
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    release=json.loads(args.manifest.read_text());artifact=release['artifact']
    source=release.get('source') or release['model']['source']
    platform=release.get('platform') or release['target']['platform']
    filename=artifact.get('name') or artifact.get('filename')
    asset_prefix(platform,source,filename,release.get('metadata_revision'))
    prefix=f'rdk_{platform}/{source}'
    if artifact['url']!=ARCHIVE_HTTPS+'/'+prefix+'/'+filename:raise ValueError('Manifest model URL does not match Archive layout')
    source_ssh=['ssh',*transport_options(args.source_jump_host),args.source_host] if args.source_host else None
    if source_ssh:
        identity=remote_json(source_ssh,FILE_INFO,args.model)
        if not identity.get('exists') or identity.get('sha256')!=artifact['sha256'] or identity.get('size_bytes')!=artifact['size_bytes']:
            raise ValueError('Source-host model identity differs')
    else:
        digest=hashlib.sha256()
        with args.model.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=artifact['sha256'] or args.model.stat().st_size!=artifact['size_bytes']:raise ValueError('Local model identity differs')
    transport=transport_options(args.jump_host)
    ssh=['ssh',*transport,args.ssh_host]
    published=remote_json(ssh,FILE_INFO,str(Path(REMOTE_ROOT)/prefix/filename))
    if published.get('exists') and (published['sha256']!=artifact['sha256'] or published['size_bytes']!=artifact['size_bytes']):
        raise ValueError('Refusing to overwrite a different published model')
    if published.get('directory_checksum') not in (None,artifact['sha256']):
        raise ValueError('Existing model directory checksum identifies different bytes')
    reused=bool(published.get('exists') and published.get('directory_checksum')==artifact['sha256'])
    if reused:
        if published['sha256']!=artifact['sha256'] or published['size_bytes']!=artifact['size_bytes']:
            raise ValueError('Refusing to overwrite a different published model')
        receipt={'status':'identical-existing','prefix':prefix,'filename':filename,'sha256':artifact['sha256'],'size_bytes':artifact['size_bytes']}
    else:
        temp_command=shlex.join(['python3','-c','import os,tempfile;fd,p=tempfile.mkstemp(prefix=".archive-model-upload-",dir=os.path.expanduser("~"));os.close(fd);print(p)'])
        stage=subprocess.check_output(ssh+[temp_command],text=True).strip()
        if source_ssh:stream_model(source_ssh,args.model,ssh,stage,artifact)
        else:subprocess.run(['scp',*transport,str(args.model),args.ssh_host+':'+stage],check=True)
        command=shlex.join(['python3','-c',INSTALL_MODEL,REMOTE_ROOT,prefix,filename,artifact['sha256'],str(artifact['size_bytes']),stage])
        receipt=json.loads(subprocess.check_output(ssh+[command],text=True))
    if source_ssh:
        downloaded=remote_json(source_ssh,VERIFY_HTTP,artifact['url'])
        receipt.update(source_host=args.source_host,source_path=str(args.model),
            transfer='reused-identical; no local model file' if reused else 'ssh-stdio-stream; no local model file',
            anonymous_verification_host=args.source_host)
    else:
        digest=hashlib.sha256();size=0
        with urlopen(artifact['url'],timeout=180) as response:
            if response.status!=200:raise ValueError('Anonymous model download failed')
            for block in iter(lambda:response.read(1024*1024),b''):digest.update(block);size+=len(block)
        downloaded={'sha256':digest.hexdigest(),'size_bytes':size}
    if downloaded['sha256']!=artifact['sha256'] or downloaded['size_bytes']!=artifact['size_bytes']:raise ValueError('Anonymous model download differs')
    receipt.update(status='verified',url=artifact['url'],anonymous_https='verified')
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
