"""Validate a frozen S-series OSNet Workbench campaign and stage a private candidate."""
import argparse,copy,hashlib,importlib.util,json,math,re,shutil,statistics,tarfile
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def need(x,msg):
    if not x:raise ValueError(msg)
def profile(p):
    parts=p.read_text().split('***');return [json.loads(s[s.index('{'):]) for s in parts if '{' in s]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--build-id',required=True);a=p.parse_args()
wb=a.workspace;bench=wb/'runs/benchmark'/a.build_id;board=bench/'work/board';out=bench/'output';cpath=bench/'input/campaign.json';c=read(cpath);target=c['platform'];artifact=c['artifact'];conv=Path(artifact['path']).parents[2];compiled=Path(artifact['path']).parent
need(cpath.with_suffix('.sha256').read_text().split()[0]==sha(cpath),'Frozen campaign changed')
need(sha(cpath)==sha(board/'input/campaign.json'),'Board campaign differs')
cc=conv/'input/campaign.json';need(sha(cc)==c['conversion_campaign_sha256'],'Conversion campaign differs');conversion=read(cc)
need(sha(conv/'input/compile.yaml')==conversion['config_sha256'],'Compiler configuration differs')
need(sha(artifact['path'])==artifact['sha256'],'Artifact changed')
ids=read(board/'output/identities.json');capture=read(board/'output/board-capture/capture.json');acc=read(out/'board-score/accuracy.json');float_run=wb/'runs/benchmark/osnet-s-float-20261030-r02';fp=read(float_run/'output/float/accuracy.json')
need(ids['model_sha256']==capture['provenance']['model_sha256']==artifact['sha256'],'Board artifact differs')
need(ids['source_archive_sha256']==c['source_archive_sha256']==sha(board/'input/source.tar'),'Runtime source differs')
with tarfile.open(board/'input/source.tar') as t:need(t.pax_headers.get('comment')==c['source_commit'],'Runtime source commit differs')
need(ids['runner_sha256']==c['board_runner_sha256']==sha(board/'input/board_run.py'),'Board controller differs')
need(ids['fixture_sha256']==c['fixture_sha256'],'Fixture differs')
need(ids['executable_sha256']==sha(board/'osnet'),'C++ executable differs')
need(capture['samples']==19281 and capture['query']==3368 and capture['gallery']==15913 and capture['provenance']['board_test']=='executed','Board capture incomplete')
need(acc['provenance']['features_sha256']==capture['provenance']['features_sha256']==sha(board/'output/board-capture/features.npy'),'Scored feature identity differs')
for item in (fp,acc,capture):need(item['provenance']['dataset_manifest_sha256']==c['dataset_manifest_sha256'],'Dataset bytes differ')
for item in (fp,acc):need(item['samples']==19281 and item['valid_queries']==3368 and item['normalized'] and not item['reranking'] and item['distance']=='squared Euclidean','Accuracy protocol differs')
accuracy={'dataset':'Market1501','task':'person-reidentification','images':19281,'query_images':3368,'gallery_images':15913,'evaluation_scope':'market1501_single_query_l2_no_rerank','float_onnx':{'label':'FP32 spatial opset19 ONNX; CPUExecutionProvider'},'runtime':{}}
for src,dst in [('rank1_percent','rank1'),('mAP_percent','map')]:
    accuracy['float_onnx'][dst]=fp[src]/100;accuracy['runtime'][dst]=acc[src]/100
    need((fp[src]-acc[src])/100<=c['acceptance']['max_'+dst+'_drop'],dst+' accuracy drop exceeds frozen budget')
need(fp['provenance']['model_sha256']==conversion['onnx_sha256'],'FP32 model differs')
float_cfg=wb/'configs/osnet-s-float-20261030-r02.json';need(float_cfg.with_suffix('.sha256').read_text().split()[0]==sha(float_cfg),'Frozen float campaign differs')
export_run=wb/'runs/conversion/osnet-s-20261030-export-r02';export=read(export_run/'work/export/runs/export.json');need(export['onnx_sha256']==conversion['onnx_sha256'] and export['weights_sha256']==conversion['checkpoint_sha256'],'Frozen source export differs')
need(read(export_run/'run.yaml')['status']=='verified-byte-identical','Exporter reproduction was not verified')
fixed=read(out/'fixed-input-review.json');need(fixed['status']=='passed' and fixed['board_model_sha256']==artifact['sha256'] and fixed['input_sha256']==ids['input_sha256'],'Fixed-input stage review missing or mismatched')
launcher=read(board/'output/native-launcher.json');need(launcher['exit_code']==0,'Native run.sh smoke failed')
launch_result=read(board/'output/native-launcher-smoke.json');need(launch_result['functional']['max_feature_abs_error']==0 and launch_result['pipeline_streams']==2,'Native run.sh output differs')
events=[json.loads(s) for s in (board/'events.jsonl').read_text().splitlines()];need(all(e['exit_code']==0 for e in events),'Board command failed')
need({'cpp-build','python-smoke','hrt-fixed-infer','e2e-1','e2e-2','board-capture','environment-before','environment-after'}.issubset({e['label'] for e in events}),'Board stages missing')
trace=[json.loads(s) for s in (board/'logs/environment-trace.jsonl').read_text().splitlines()];need(len(trace)>10,'Environment trace missing')
cpus=[int(p['scaling_cur_freq'])/1000 for x in trace for p in x['cpu'].values()];governors={p['scaling_governor'] for x in trace for p in x['cpu'].values()};bpus=[int(p['cur_freq'])/1e6 for x in trace for name,p in x['devfreq'].items() if 'bpu' in name and p['cur_freq']]
need(cpus and bpus,'CPU/BPU frequencies missing');need(len(set(bpus))==1,'BPU frequency change requires review')
for policy in trace[0]['cpu']:
    need(len({x['cpu'][policy]['scaling_max_freq'] for x in trace})==1,'CPU ceiling changed')
    need(len({x['cpu'][policy]['scaling_governor'] for x in trace})==1,'CPU governor changed')
before=(board/'logs/environment-before.log').read_text();need('\n'+target.upper()+'\n' in before,'Physical target identity differs')
version=re.search(r'ucp_version: ([\d.]+)_\(([\d.]+) HBRT\)',(board/'output/profile/perf-t1-r1/profiler.log').read_text());need(version,'Runtime versions missing')
temps=[int(t)/1000 for x in trace for t in x['thermal'].values() if t is not None]
environment={'board':'D-Robotics RDK '+target.upper(),'os':re.search(r'PRETTY_NAME="([^"]+)"',before)[1],'kernel':before.splitlines()[0].split()[2],'runtime':version[1],'hbrt':version[2],'cpu_governor':' / '.join(sorted(governors)),'cpu_governor_by_policy':{k:v['scaling_governor'] for k,v in trace[0]['cpu'].items()},'cpu_frequency_mhz':statistics.median(cpus),'cpu_frequency_mhz_observed_range':[min(cpus),max(cpus)],'cpu_frequency_ceiling_mhz_by_policy':{k:int(v['scaling_max_freq'])/1000 for k,v in trace[0]['cpu'].items()},'bpu_frequency_mhz':bpus[0],'temperature_before_c':max(int(t)/1000 for t in trace[0]['thermal'].values()),'temperature_after_c':max(int(t)/1000 for t in trace[-1]['thermal'].values()),'temperature_max_c':max(temps),'observation_scope':'0.2 second trace; stable BPU frequency, CPU governor and CPU frequency ceilings; no policy changes made','source_logs':['work/board/logs/environment-before.log','work/board/logs/environment-trace.jsonl','work/board/logs/environment-after.log']}
performance={'tool':'hrt_model_exec perf','implementation':'native_cpp_cli','timing_scope':'model_runtime','thread_semantics':'runtime_submission_concurrency','core_id':1,'warmup_frames_per_condition':20,'warmup_scope':'separate_process_device_warmup; enable_warmup=false for timed runs','runs_per_condition':3,'frames_per_run':200,'input_kind':'fixed normalized RGB float32 NCHW featuremap','input_sha256':ids['input_sha256'],'stages':{'preprocess':'not_measured','runtime':'measured','postprocess':'not_measured','end_to_end':'not_measured'},'measurements':[],'raw_measurements':[],'end_to_end':[]}
for threads in (1,2):
    rows=[]
    for round_id in (1,2,3):
        label=f'perf-t{threads}-r{round_id}';path=board/'output/profile'/label/'profiler.log';prof,detail=profile(path);cond=prof['running_condition'];res=prof['perf_result'];task=detail['task_latency']['TaskRunningTime'];need(cond['frame_count']==200 and cond['thread_num']==threads and str(cond['core_id'])=='1','Runtime counts/core differ')
        event=next(e for e in events if e['label']==label);need('--enable_warmup=false' in event['argv'],'Warmup semantics differ')
        rows.append({'threads':threads,'round':round_id,'frames':cond['frame_count'],'average_latency_ms':res['average_latency'],'minimum_latency_ms':min(task['min_time'],res['average_latency']),'maximum_latency_ms':max(task['max_time'],res['average_latency']),'fps':res['FPS'],'wall_time_ms':cond['run_time'],'log':str(path.relative_to(bench)),'sha256':sha(path)})
    warm=next(e for e in events if e['label']==f'perf-t{threads}-warmup');need(warm['argv'][warm['argv'].index('--frame_count')+1]=='20','Warmup count differs')
    performance['raw_measurements'].extend(rows);performance['measurements'].append({'threads':threads,'average_latency_ms':statistics.mean(r['average_latency_ms'] for r in rows),'observed_min_latency_ms':min(r['minimum_latency_ms'] for r in rows),'observed_max_latency_ms':max(r['maximum_latency_ms'] for r in rows),'aggregate_fps':600000/sum(r['wall_time_ms'] for r in rows)})
    raw=read(board/f'output/e2e-{threads}.json');frames=np.array(raw['raw_frame_ms']);need(raw['timed_frames']==threads*600 and raw['rounds']==3 and raw['frames_per_round']==200 and raw['warmup_frames_per_round']==20 and raw['pipeline_streams']==threads,'C++ counts differ')
    need(frames.shape==(threads*600,4) and np.isfinite(frames).all() and (frames>0).all(),'C++ raw frames missing')
    need(np.allclose(frames[:,:3].sum(1),frames[:,3],atol=1e-7,rtol=1e-8),'Same-frame stages differ')
    for i,stage in enumerate(('preprocess','runtime','postprocess','end_to_end')):
        vals=frames[:,i];expected={'mean':vals.mean(),'p50':np.quantile(vals,.5),'p95':np.quantile(vals,.95),'min':vals.min(),'max':vals.max()}
        need(all(math.isclose(raw['metrics_ms'][stage][k],v,abs_tol=1e-7,rel_tol=1e-8) for k,v in expected.items()),'C++ statistics differ')
    need(math.isclose(raw['timed_frames']*1000/raw['aggregate_wall_ms'],raw['throughput_fps'],rel_tol=1e-8),'C++ FPS differs')
    need(raw['functional']['fixtures_per_worker']==32 and raw['functional']['preprocess_byte_exact'] and raw['functional']['max_feature_abs_error']==0,'C++ function differs from Python SDK')
    entry={k:v for k,v in raw.items() if k!='raw_frame_ms'};entry.update(cpu_governor=environment['cpu_governor'],cpu_frequency_mhz=environment['cpu_frequency_mhz'],bpu_frequency_mhz=environment['bpu_frequency_mhz']);performance['end_to_end'].append(entry)
environment['online_cpu_threads']=performance['end_to_end'][0]['online_cpu_threads']
reference=(board/'input/reference.f32').read_bytes()[:2048];need((board/'output/hrt-smoke/model_infer_output_0_embedding.bin').read_bytes()==reference,'Native CLI feature differs from Python SDK')
need(sha(board/'input/perf-input.bin')==ids['input_sha256'] and (board/'output/hrt-smoke/model_infer_input_0_images.bin').read_bytes()==(board/'input/perf-input.bin').read_bytes(),'CLI fixed input differs')
# The Python smoke uses a named query crop, while C++/CLI use the fixture; each has a separate evidence source.
smoke=np.load(board/'output/smoke.npy');need(smoke.shape==(512,) and np.isfinite(smoke).all() and np.isclose(np.linalg.norm(smoke),1,atol=1e-6),'Python smoke invalid')
log=(conv/'logs/hb-compile.log').read_text();need('Calibration using batch 8' in log and 'reset batch' not in log.lower(),'Actual PTQ batch 8 missing or reset')
# Preserve the original OE files; expose copies where the guide extractor expects them.
for src,dest in [(compiled/'osnet_x1_0_node_info.csv',conv/'output/osnet_x1_0_node_info.csv'),(compiled/'osnet_x1_0.json',conv/'output/oe_report_data_source.json'),(conv/'logs/hb-compile.log',conv/'output/hb_compile.log')]:
    if dest.exists():need(sha(dest)==sha(src),'Existing report differs')
    else:shutil.copyfile(src,dest)
spec=importlib.util.spec_from_file_location('oe',wb/'scripts/extract_oe_report.py');oe=importlib.util.module_from_spec(spec);spec.loader.exec_module(oe)
data=oe.extract_s_run(conv,{'run_id':conv.name,'created_at':json.loads((conv/'events.jsonl').read_text().splitlines()[0])['started_at']},{'model':{'sample':'osnet','family':'osnet','task':'reid','size':'x1.0'},'target':{'platform':target,'march':artifact['march']},'artifacts':{'model':artifact}})
need(data and data['quantization']['summary']['measured_nodes']>0 and data['aggregate']['latency_ms']>0,'OE report incomplete')
report=compiled/'osnet_x1_0.html';need(report.exists(),'Original OE HTML absent');data['provenance']['oe_report_html_sha256']=sha(report)
need('There are 128 samples in the dataset.' in log,'Calibration sample count missing');data['similarity_provenance']['calibration_samples']=128
# Source model dimensions/statistics are shared with the independently checked X5 export, not its measurements.
x5=read(wb/'releases/models/osnet/osnet/reid/x1.0/x5/release.json');need(x5['model']['weights_sha256']==conversion['checkpoint_sha256'],'Source model differs')
precision='int16';dtypes=data['quantization']['summary']['dtype_counts'];
if any('si8' in k for k in dtypes):precision='mixed-int8-int16'
validation={k:'passed' for k in ('board_smoke','native_launcher','cpp_functional','runtime_perf','cpp_e2e','input_contract_review','environment_review','full_dataset_review')};validation.update(license_review='local-evaluation-only',release_approval='not-run',approved_by=None,technical_acceptance='passed',publication_authorized=False,evidence={'campaign_sha256':sha(cpath),'board_identities':ids,'float_accuracy_sha256':sha(float_run/'output/float/accuracy.json'),'board_accuracy_sha256':sha(out/'board-score/accuracy.json'),'environment_trace_sha256':sha(board/'logs/environment-trace.jsonl'),'official_dataset_manifest_sha256':c['dataset_manifest_sha256'],'fixed_input_review_sha256':sha(out/'fixed-input-review.json')})
release={'schema_version':1,'status':'candidate','build_id':a.build_id,'source':'osnet','family':'osnet','task':'reid','size':'x1.0','platform':target,'model':x5['model'],'input':x5['input'],'artifact':{'filename':'osnet_x1_0_'+artifact['march'].replace('-','')+'_256x128_f32_int16.hbm','format':'hbm','march':artifact['march'],'runtime_input':'featuremap','sha256':artifact['sha256'],'size_bytes':artifact['size_bytes'],'runtime_tensor':{'dtype':'float32','payload_layout':'NCHW','metadata_layout':'NCHW','shape':[1,3,256,128],'packing':'RGB /255, ImageNet mean/std; contiguous NCHW float32'}},'accuracy':accuracy,'performance':performance,'environment':environment,'reports':{'oe_report_html':'oe_report.html','oe_report_html_sha256':sha(report),'oe_report_data':'oe_report_data.json','scope':'Original hb_compile HTML, full node similarity CSV and native JSON preserved'},'license':{'name':'Research use only','url':'https://github.com/KaiyangZhou/deep-person-reid/issues/520','scope':'model','dataset':'Market1501; research only, no distribution or commercial use'},'provenance':{'repository_commit':c['source_commit'],'build_id':a.build_id,'precision_policy':precision,'requested_quant_config':{'model_config':{'all_node_type':'int16'}},'runtime_variant':c['runtime_variant'],'sample_pending_publication':True,'float_run':float_run.name,'export_run':export_run.name,'oe_run':conv.name,'conversion_source_commit':conversion['source_commit'],'float_source_commit':read(wb/'configs/osnet-s-float-20261030-r02.json')['source_commit'],'source_weight_url':x5['provenance']['source_weight_url'],'onnx_sha256':conversion['onnx_sha256'],'weights_sha256':conversion['checkpoint_sha256'],'campaign_sha256':sha(cpath),'oe_image_id':conversion['image_id'],'dataset_manifest_sha256':c['dataset_manifest_sha256'],'original_conversion_archive_available':True,'actual_calibration_batch':8,'calibration_samples':128},'validation':validation}
dest=wb/'releases/models/osnet/osnet/reid/x1.0'/target;dest.mkdir(parents=True,exist_ok=False)
shutil.copyfile(artifact['path'],dest/release['artifact']['filename']);shutil.copyfile(report,dest/'oe_report.html');write(dest/'oe_report_data.json',data);release['reports']['oe_report_data_sha256']=sha(dest/'oe_report_data.json');write(dest/'release.json',release)
files=[release['artifact']['filename'],'release.json','oe_report.html','oe_report_data.json'];(dest/'SHA256SUMS').write_text(''.join(f'{sha(dest/f)}  {f}\n' for f in files))
for name,value in [('environment',environment),('performance',performance),('validation',validation),('release-gate',{'technical_acceptance':'passed','publication':'not-authorized','staging':str(dest),'checksums_sha256':sha(dest/'SHA256SUMS')})]:write(out/(name+'.json'),value)
write(bench/'run.yaml',{'status':'technically-accepted','source_commit':c['source_commit'],'campaign_sha256':sha(cpath),'publication_authorized':False})
print(json.dumps({'platform':target,'accuracy':accuracy,'runtime':performance['measurements'],'cpp':[{k:r[k] for k in ('pipeline_streams','throughput_fps','metrics_ms')} for r in performance['end_to_end']],'staging':str(dest)},indent=2))
