import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch,onnxruntime as ort,numpy as np,cv2
from ultralytics import YOLOE
from samples.vision.yoloe.conversion.export_heads import RawPFModel,OUTPUT_NAMES
from samples.vision.yoloe.conversion.calibration import calibration_tensor
torch.set_num_threads(2)
network=YOLOE('../.coordination/yoloe-checkpoints/yoloe-11m-seg-pf.pt').model.cpu().float().eval()
sample=torch.from_numpy(calibration_tensor(cv2.imread('samples/vision/yoloe/test_data/office_desk.jpg'),'x5','11m')/255)
with torch.inference_mode(): before=[v.numpy() for v in RawPFModel(network,'11m')(sample)]
network.fuse(verbose=False)
with torch.inference_mode(): after=[v.numpy() for v in RawPFModel(network,'11m')(sample)]
def compare(a,b):
 return {name:{'max_abs':float(np.max(np.abs(x-y))),'violations':int(np.count_nonzero(np.abs(x-y)>.002+.002*np.abs(y)))} for name,x,y in zip(OUTPUT_NAMES,a,b)}
results={'fused_vs_original_torch':compare(after,before)}
for label,level in [('default',ort.GraphOptimizationLevel.ORT_ENABLE_ALL),('disabled',ort.GraphOptimizationLevel.ORT_DISABLE_ALL)]:
 options=ort.SessionOptions();options.intra_op_num_threads=2;options.graph_optimization_level=level
 session=ort.InferenceSession('../.coordination/yoloe-export-final-20260928/11m/yoloe_11m_seg_pf.onnx',options,providers=['CPUExecutionProvider'])
 actual=session.run(list(OUTPUT_NAMES),{'images':sample.numpy()})
 results[label+'_vs_original']=compare(actual,before)
 results[label+'_vs_fused']=compare(actual,after)
 del session,actual
print(json.dumps(results,indent=2))
