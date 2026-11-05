"""Assemble a private OSNet candidate from measured Workbench receipts."""
import argparse,copy,hashlib,importlib.util,json,math,statistics,shutil
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def need(v,msg):
    if not v: raise ValueError(msg)
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--workspace',type=Path,required=True);p.add_argument('--build-id',required=True)
p.add_argument('--float-run-id',required=True);p.add_argument('--oe-run-id',required=True)
a=p.parse_args();wb=a.workspace;bench=wb/'runs/benchmark'/a.build_id;board=bench/'work/board';out=bench/'output'
cpath=wb/'configs'/f'{a.build_id}.json';c=read(cpath)
need(cpath.with_suffix('.sha256').read_text().split()[0]==sha(cpath),'Campaign changed after freeze')
need(sha(board/'input/campaign.json')==sha(cpath),'Board campaign differs')
float_dir=wb/'runs/benchmark'/a.float_run_id/'output/float'
float_acc=read(float_dir/'accuracy.json');board_acc=read(out/'board-score/accuracy.json');capture=read(board/'output/board-capture/capture.json')
ids=read(board/'output/identities.json');artifact=c['artifact']
need(ids['model_sha256']==artifact['sha256']==capture['provenance']['model_sha256'],'Artifact identity mismatch')
need(sha(artifact['path'])==artifact['sha256'],'Staged artifact changed')
need(ids['executable_sha256']==sha(board/'osnet'),'Executable changed')
for result in (float_acc,board_acc):
    need(result['samples']==19281 and result['valid_queries']==3368 and result['normalized'] and not result['reranking'],'Incomplete or different accuracy protocol')
for result in (float_acc,board_acc,capture):
    need(result['provenance']['dataset_manifest_sha256']==c['dataset']['manifest_sha256'],'Evaluation image bytes differ')
need(capture['provenance']['features_sha256']==sha(board/'output/board-capture/features.npy'),'Board feature file changed')
need(board_acc['provenance']['features_sha256']==capture['provenance']['features_sha256'],'Scored features differ')
accuracy={'dataset':'Market1501','task':'person-reidentification','images':19281,'query_images':3368,'gallery_images':15913,
          'evaluation_scope':c['evaluation']['scope'],'float_onnx':{'label':'FP32 spatial ONNX; CPUExecutionProvider'},'runtime':{}}
for src,dst in (('rank1_percent','rank1'),('mAP_percent','map')):
    accuracy['float_onnx'][dst]=float_acc[src]/100;accuracy['runtime'][dst]=board_acc[src]/100
    need(accuracy['float_onnx'][dst]-accuracy['runtime'][dst]<=c['acceptance']['max_'+dst+'_drop'],f'{dst} drop exceeds frozen budget')
performance=read(out/'performance.json')
need({m['threads'] for m in performance['measurements']}=={1,2},'Runtime concurrency missing')
need(len(performance['raw_measurements'])==6 and all(r['frames']==200 for r in performance['raw_measurements']),'Runtime rounds/frames missing')
need(performance['input_sha256']==ids['fixed_input_sha256'],'Performance input changed')
events=[json.loads(s) for s in (board/'events.jsonl').read_text().splitlines()]
need(all(e['exit_code']==0 for e in events),'Board command failed')
trace=[json.loads(s) for s in (board/'logs/environment-trace.jsonl').read_text().splitlines()]
need(len(trace)>10 and all(x['bpu_frequency_hz'] for x in trace),'Environment trace missing')
need(len({x['bpu_frequency_hz'] for x in trace})==1,'BPU frequency changed during measurements')
need({p['scaling_governor'] for x in trace for p in x['cpu'].values()}=={'schedutil'},'Governor changed')
need(len({p['scaling_max_freq'] for x in trace for p in x['cpu'].values()})==1,'CPU frequency ceiling changed')
environment={'board':'D-Robotics RDK X5 V1.0','os':'Ubuntu 22.04.5 LTS','kernel':'6.1.83','runtime':'1.24.5','hbrt':'3.15.55.0',
             'online_cpu_threads':8,'cpu_governor':'schedutil','cpu_frequency_mhz':statistics.median(int(p['scaling_cur_freq'])/1000 for x in trace for p in x['cpu'].values()),
             'cpu_frequency_mhz_observed_range':[min(int(p['scaling_cur_freq'])/1000 for x in trace for p in x['cpu'].values()),max(int(p['scaling_cur_freq'])/1000 for x in trace for p in x['cpu'].values())],
             'cpu_frequency_ceiling_mhz':int(next(iter(trace[0]['cpu'].values()))['scaling_max_freq'])/1000,
             'bpu_frequency_mhz':int(trace[0]['bpu_frequency_hz'])/1e6,'temperature_before_c':max(map(int,trace[0]['thermal'].values()))/1000,
             'temperature_after_c':max(map(int,trace[-1]['thermal'].values()))/1000,'temperature_max_c':max(int(t) for x in trace for t in x['thermal'].values())/1000,
             'thermal_throttling_observed':False,'observation_scope':'1 Hz trace; stable BPU frequency and CPU frequency ceiling; schedutil frequencies vary normally',
             'source_logs':['work/board/logs/environment-before.log','work/board/logs/environment-trace.jsonl','work/board/logs/environment-after.log']}
end=[]
for streams in (1,2):
    raw=read(board/f'output/e2e-{streams}.json');need(raw['pipeline_streams']==streams and raw['timed_frames']==streams*600,'C++ counts differ')
    need(raw['rounds']==3 and raw['frames_per_round']==200 and raw['warmup_frames_per_round']==20,'C++ matrix differs')
    need(len(raw['raw_frame_ms'])==raw['timed_frames'],'C++ per-frame evidence missing')
    need(all(math.isclose(sum(f[:3]),f[3],rel_tol=1e-8,abs_tol=1e-7) for f in raw['raw_frame_ms']),'C++ stage accounting differs')
    need(math.isclose(raw['timed_frames']*1000/raw['aggregate_wall_ms'],raw['throughput_fps'],rel_tol=1e-8),'C++ FPS not wall-based')
    need(raw['functional']['preprocess_byte_exact'] and raw['functional']['max_feature_abs_error']==0,'C++ functional mismatch')
    event=next(e for e in events if e['label']==f'e2e-{streams}')
    samples=[x for x in trace if event['started_at']<=x['utc']<=event['ended_at']]
    need(samples,'No CPU snapshots during C++ run')
    cpus=[int(p['scaling_cur_freq'])/1000 for x in samples for p in x['cpu'].values()]
    record={k:v for k,v in raw.items() if k not in ('raw_frame_ms',)}
    record.update(cpu_governor='schedutil',cpu_frequency_mhz=statistics.median(cpus),cpu_frequency_mhz_observed_range=[min(cpus),max(cpus)],bpu_frequency_mhz=environment['bpu_frequency_mhz'])
    end.append(record)
performance['end_to_end']=end
reference=Path(c['performance_inputs']['spatial-rgb-late8.f32']['path']).read_bytes()[:2048]
need((board/'output/hrt-smoke/model_infer_output_0_embedding.bin').read_bytes()==reference,'Native hrt smoke differs')
need((board/'input/perf-input.bin').read_bytes()==(board/'output/hrt-smoke/model_infer_input_0_images.bin').read_bytes(),'Native hrt input differs')
feature=np.load(board/'output/smoke.npy');expected=np.frombuffer(reference,np.float32).copy();expected/=np.linalg.norm(expected)
need(feature.shape==(512,) and np.allclose(feature,expected,atol=1e-6,rtol=1e-6),'Python smoke differs')
conv=wb/'runs/conversion'/a.oe_run_id;export=read(conv/'work/export/runs/export.json')
need(export['onnx_sha256']==float_acc['provenance']['model_sha256'],'Float ONNX identity mismatch')
need(export['weights_sha256']==c['model']['weights_sha256'],'Checkpoint differs')
report_dir=conv/'work/hb_perf_result/spatial-rgb-late8';report=report_dir/'torch_jit_subgraph_0.html'
need(report.exists(),'Original hb_perf HTML missing')
spec=importlib.util.spec_from_file_location('oe_extractor',wb/'scripts/extract_oe_report.py');oe=importlib.util.module_from_spec(spec);spec.loader.exec_module(oe)
subgraph=oe.parse_perf_html(report);need(subgraph and subgraph['perf']['latency_ms']>0,'OE report cannot be parsed')
oe_data={'schema_version':1,'generated_from_run':a.oe_run_id,'model':{'sample':'osnet','family':'osnet','task':'reid','size':'x1.0'},
         'target':{'platform':'x5','march':'bayes-e'},'toolchain':{'hb_mapper':'1.24.3','hbdk':'3.49.15','horizon_nn':'1.1.0','runtime':'3.15.55.0'},
         'io':{'inputs':[{'name':'images','shape':[1,256,128,3],'dtype':'int8','layout':'NHWC','payload_layout':'CHW','format':'RGB'}],
               'outputs':[{'name':'embedding','shape':[1,512,1,1],'dtype':'float32'}]},
         'quantization':{'summary':{},'nodes':[],'outputs':[]},'subgraphs':[subgraph],
         'aggregate':{'latency_ms':subgraph['perf']['latency_ms'],'ddr_mb_per_frame':subgraph['perf']['ddr_mb_per_frame'],'bpu_ops_per_frame':subgraph['perf']['bpu_ops_per_frame'],
                      'derivation':'Single BPU subgraph static estimate from hb_perf; CPU nodes excluded'},
         'similarity_provenance':{'stage':'unavailable','method':'Original per-layer PTQ similarity report was not recovered; this run does not invent it'},
         'performance_provenance':{'stage':'compiler-static-estimate','source':'hb_perf-exact-bin','method':'Regenerated analysis of the exact preserved BIN, not a new compile'},
         'board_measurement':None,'provenance':{'artifact_sha256':artifact['sha256'],'oe_report_html_sha256':sha(report),'source_log_sha256':sha(conv/'logs/oe-report.log')}}
validation={k:'passed' for k in ('board_smoke','cpp_functional','runtime_perf','cpp_e2e','input_contract_review','environment_review','full_dataset_review')}
validation.update(license_review='local-evaluation-only',release_approval='not-run',approved_by=None,technical_acceptance='passed',publication_authorized=False,
                  evidence={'campaign_sha256':sha(cpath),'board_identities':ids,'float_accuracy_sha256':sha(float_dir/'accuracy.json'),'board_accuracy_sha256':sha(out/'board-score/accuracy.json'),
                            'environment_trace_sha256':sha(board/'logs/environment-trace.jsonl'),'official_dataset_manifest_sha256':c['dataset']['manifest_sha256']})
release={'schema_version':1,'status':'candidate','build_id':a.build_id,'source':'osnet','family':'osnet','task':'reid','size':'x1.0','platform':'x5',
         'model':c['model'],'input':{'width':128,'height':256,'source':{'format':'rgb','dtype':'float32','layout':'NCHW','shape':[1,3,256,128],
                                 'scale':1/255,'mean':[.485,.456,.406],'std':[.229,.224,.225]}},
         'artifact':{'filename':'osnet_x1_0_bayes_e_256x128_rgb.bin','format':'bin','march':'bayes-e','runtime_input':'rgb','sha256':artifact['sha256'],'size_bytes':artifact['size_bytes'],
                     'runtime_tensor':{'dtype':'int8','payload_layout':'CHW','metadata_layout':'NHWC','shape':[1,256,128,3],'packing':'Signed planar RGB minus 128, reshaped to metadata NHWC; use the OSNet sample adapter'}},
         'accuracy':accuracy,'performance':performance,'environment':environment,
         'reports':{'oe_report_html':'oe_report.html','oe_report_html_sha256':sha(report),'oe_report_data':'oe_report_data.json','scope':'Original hb_perf BPU subgraph HTML; full generated report tree retained in the private conversion run'},
         'license':{'name':'Research use only','url':'https://github.com/KaiyangZhou/deep-person-reid/issues/520','scope':'model','dataset':'Market1501; research only, no distribution or commercial use'},
         'provenance':{'repository_commit':c['source_commit'],'build_id':a.build_id,'precision_policy':artifact['precision_policy'],'sample_pending_publication':True,
                       'float_run':a.float_run_id,'oe_run':a.oe_run_id,'float_source_commit':read(wb/'configs'/f'{a.float_run_id}.json')['source_commit'],
                       'reuse_reason':'Only benchmark fixture locking and legacy C++ return changed; export/evaluator/model source bytes unchanged',
                       'source_weight_url':'https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view','onnx_sha256':export['onnx_sha256'],
                       'weights_sha256':export['weights_sha256'],'campaign_sha256':sha(cpath),'oe_image_id':c['conversion']['image_id'],
                       'dataset_manifest_sha256':c['dataset']['manifest_sha256'],'original_conversion_archive_available':False},
         'validation':validation}
dest=wb/'releases/models/osnet/osnet/reid/x1.0/x5';dest.mkdir(parents=True,exist_ok=False)
shutil.copyfile(artifact['path'],dest/release['artifact']['filename']);shutil.copyfile(report,dest/'oe_report.html')
write(dest/'oe_report_data.json',oe_data);release['reports']['oe_report_data_sha256']=sha(dest/'oe_report_data.json');write(dest/'release.json',release)
files=[release['artifact']['filename'],'release.json','oe_report.html','oe_report_data.json']
(dest/'SHA256SUMS').write_text(''.join(f'{sha(dest/f)}  {f}\n' for f in files))
write(out/'environment.json',environment);write(out/'validation.json',validation);write(out/'release-gate.json',{'technical_acceptance':'passed','publication':'not-authorized','staging':str(dest),'checksums_sha256':sha(dest/'SHA256SUMS')})
print(json.dumps({'staging':str(dest),'accuracy':accuracy,'runtime':performance['measurements'],'technical_acceptance':'passed'},indent=2))
