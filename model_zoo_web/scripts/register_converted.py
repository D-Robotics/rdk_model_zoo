#!/usr/bin/env python3
"""Register one uploaded conversion without importing historical measurements."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--web',type=Path,required=True,help='model_zoo_web directory')
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--oe-data',type=Path,required=True)
    parser.add_argument('--supplement',action='store_true',help='Apply a newer metadata revision for the same artifact SHA')
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    if manifest.get('status') not in ({'converted','released'} if args.supplement else {'converted'}):
        raise ValueError('Initial registration requires converted; promotion requires --supplement')
    model=manifest['model'];target=manifest['target'];platform=target['platform'];size=model['size']
    path=args.web/'data'/model.get('domain','vision')/model['source']/model['family']/(model['task']+'.yaml')
    record=yaml.safe_load(path.read_text())
    if record['id']!=f"{model['source']}/{model['family']}/{model['task']}": raise ValueError('Catalog identity mismatch')
    for key in ('accuracy','performance'):
        if key in manifest and not args.supplement: raise ValueError('Initial registration does not import board metrics')
    record.setdefault('provider',model['provider'])
    if record.get('registration')=='pending': record.setdefault('model_license',manifest['license'])
    else: record.setdefault('license',manifest['license'])
    variants=record.setdefault('variants',[])
    variant=next((v for v in variants if v['size']==size),None)
    if variant is None:
        variant={'size':size,'model':{key:model[key] for key in ('parameter_count','gflops','measurement_method') if key in model},
                 'input':manifest['input'],'platforms':[]}
        variants.append(variant)
    elif variant['input']!=manifest['input']:
        raise ValueError('The selected size has a different input contract')
    else:
        facts=variant.setdefault('model',{})
        for key in ('parameter_count','gflops','measurement_method'):
            if key not in model:continue
            if key in facts and facts[key]!=model[key]:raise ValueError('Supplementation cannot change existing model facts')
            if key not in facts and not args.supplement:raise ValueError('New model facts require --supplement')
            facts[key]=model[key]
    fields=('license','status','converted_at','released_at','metadata_revision','artifact','reports','conversion','validation','provenance','runtime_command','accuracy','performance')
    row={'platform':platform,**{key:manifest[key] for key in fields if key in manifest}}
    existing=next((p for p in variant['platforms'] if p['platform']==platform),None)
    if existing is not None:
        if existing!=row:
            if not args.supplement: raise ValueError('Existing platform differs; use --supplement after publishing a new metadata revision')
            if existing['artifact']['sha256']!=row['artifact']['sha256'] or existing['artifact']['url']!=row['artifact']['url']:
                raise ValueError('Supplementation cannot replace the model; register a new artifact variant')
            if row['metadata_revision']<=existing.get('metadata_revision',0):raise ValueError('Metadata revision must increase')
            variant['platforms'][variant['platforms'].index(existing)]=row
    else:variant['platforms'].append(row)
    # The catalog validator resolves pinned samples from this working tree.
    sys.path.insert(0,str(args.web/'scripts'))
    import build_catalog
    normalized=build_catalog.validate_record(record,path)
    oe=json.loads(args.oe_data.read_text())
    if oe['provenance']['artifact_sha256']!=row['artifact']['sha256']:raise ValueError('OE data describes another model')
    report_name=f"{model['source']}-{model['family']}-{model['task']}-{size}-{platform}-v{row['metadata_revision']}-oe-data.json"
    report_path=args.web/'release/reports'/report_name
    if report_path.exists() and report_path.read_bytes()!=args.oe_data.read_bytes():raise ValueError('Existing report differs')
    inputs_path=args.web/'release/inputs.json';inputs=json.loads(inputs_path.read_text())
    inputs['releases'][record['id']+'/'+size+'/'+platform]={'oe_data':'reports/'+report_name}
    hero=record.get('hero_visual',{}).get('file')
    if record['id'] not in inputs['models'] and hero:
        inputs['models'][record['id']]={'cover':str(Path(hero).relative_to('model_zoo_web/release')),
                                      'cover_label':'模型系列参考效果图'}
    path.write_text(yaml.safe_dump(record,allow_unicode=True,sort_keys=False))
    report_path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(args.oe_data,report_path)
    inputs_path.write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'registered-locally','record':record['id'],'variant':size,'platform':platform}))


if __name__=='__main__':
    main()
