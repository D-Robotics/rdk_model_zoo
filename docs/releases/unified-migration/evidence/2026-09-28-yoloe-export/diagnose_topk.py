from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()))
import torch,cv2
from ultralytics import YOLOE
from samples.vision.yoloe.conversion.export_heads import RawPFModel
from samples._shared.yoloe26_geometry import prepare_rgb
torch.set_num_threads(2)
results={}
for variant in ('26m','26l','26x'):
 network=YOLOE('../.coordination/yoloe-checkpoints/yoloe-'+variant+'-seg-pf.pt').model.cpu().float().eval()
 wrapper=RawPFModel(network,variant);head=wrapper.raw_head.head
 sample=torch.from_numpy(prepare_rgb(cv2.imread('samples/vision/yoloe/test_data/office_desk.jpg')))
 head.export,head.dynamic,head.format,head.agnostic_nms=True,False,'onnx',True
 captured={};original=head._inference
 def capture(raw):
  captured['raw']=raw;captured['decoded']=original(raw);return captured['decoded']
 head._inference=capture
 with torch.inference_mode():
  features=wrapper.features(sample);expected=wrapper.raw_head(features);upstream=head(list(features))
  source=captured['raw'];source_decoded=captured['decoded']
  raw={'boxes':torch.cat([expected[i].permute(0,3,1,2).reshape(1,4,-1) for i in (1,4,7)],2),'scores':torch.cat([expected[i].permute(0,3,1,2).reshape(1,4585,-1) for i in (0,3,6)],2),'mask_coefficient':torch.cat([expected[i].permute(0,3,1,2).reshape(1,32,-1) for i in (2,5,8)],2),'feats':features,'index':None}
  raw_errors={k:float((raw[k]-source[k]).abs().max()) for k in ('boxes','scores','mask_coefficient')}
  decoded=original(raw)
  vals_a,labels_a,idx_a=head.get_topk_index(decoded[:,4:4589,:].permute(0,2,1),head.max_det)
  vals_b,labels_b,idx_b=head.get_topk_index(source_decoded[:,4:4589,:].permute(0,2,1),head.max_det)
  keys_a=(idx_a.long()*4585+labels_a.long()).flatten();keys_b=(idx_b.long()*4585+labels_b.long()).flatten()
  changed=torch.nonzero(keys_a!=keys_b).flatten()
  rows=[]
  for i in changed.tolist():
   key=int(keys_a[i]);j=torch.nonzero(keys_b==key).flatten()
   rows.append({'rank':i,'key_a':key,'key_b':int(keys_b[i]),'source_rank_of_a':j.tolist(),'a_score':float(vals_a.flatten()[i]),'b_score':float(vals_b.flatten()[i]),'same_rank_gap':float(abs(vals_a.flatten()[i]-vals_b.flatten()[i]))})
  result={'raw_max_abs':raw_errors,'same_key_set':bool(torch.equal(keys_a.sort().values,keys_b.sort().values)),'changed_rows':rows}
  for k in ('boxes','scores','mask_coefficient'):torch.testing.assert_close(raw[k],source[k],rtol=1e-4,atol=1e-4)
  torch.testing.assert_close(decoded,source_decoded,rtol=1e-4,atol=1e-4)
  result['raw_and_decoded_before_topk']='passed'
  results[variant]=result
print(json.dumps(results,indent=2))
