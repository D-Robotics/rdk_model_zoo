"""State/phase contracts and actual CPU tracker regression, no board runtime."""
import importlib,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from samples.vision.bytetrack.runtime.python.tracking import ByteTrackTask,TrackingConfig
from samples.vision.yolov5.runtime.python.detection import DetectionResult,YOLOv5Task
from samples.vision.yolov5.runtime.python.cli import resolve_selection
from samples.vision.yolov5.runtime.python.detection import bind_model
from samples.vision.yolov5.tests.test_yolov5 import FakeRuntime
from utils.py_utils.tests.legacy_platforms import legacy_path, legacy_tree  # noqa: E402
ROOT=Path(__file__).resolve().parents[4]


class FakeDetector:
    def __init__(self):
        from samples.vision.yolov5.runtime.python.detection import DetectionContext,PreparedInput
        self.prepared=lambda x:PreparedInput({'image':x},DetectionContext(x.shape[:2],672,1))
        self.result=DetectionResult(np.array([[1,2,11,22],[20,30,40,50]],np.float32),np.array([.9,.8],np.float32),np.array([0,1],np.int32))
    def preprocess(self,image):return self.prepared(image)
    def infer(self,tensors):return tensors
    def postprocess(self,raw,context):return self.result


class FakeTracker:
    def __init__(self,config):self.frame_id=0;self.calls=[]
    def update(self,detections,shape1,shape2):
        self.frame_id+=1;self.calls.append((detections.copy(),shape1,shape2))
        return [types.SimpleNamespace(track_id=7,tlbr=detections[0,:4],score=detections[0,4])] if len(detections) else []


class TrackingTests(unittest.TestCase):
    def test_zero_area_person_does_not_corrupt_real_kalman_state(self):
        detector=FakeDetector();task=ByteTrackTask(detector)
        detector.result=DetectionResult(np.array([[1,2,11,2],[4,2,4,22]],np.float32),np.array([.9,.8],np.float32),np.array([0,0],np.int32))
        self.assertEqual(task.predict(np.zeros((17,31,3),np.uint8)),())
        self.assertEqual(task.frame_index,1)
        detector.result=FakeDetector().result
        task.predict(np.zeros((17,31,3),np.uint8))
        result=task.predict(np.zeros((17,31,3),np.uint8))
        self.assertEqual(len(result),1)
        self.assertTrue(np.isfinite(result[0].tlbr).all())

    def test_stages_and_predict_have_one_state_transition_and_filter_person(self):
        task=ByteTrackTask(FakeDetector(),tracker_factory=FakeTracker);image=np.zeros((17,31,3),np.uint8)
        prepared=task.preprocess(image);raw=task.infer(prepared.tensors);result=task.postprocess(raw,prepared.context)
        self.assertEqual(task.frame_index,1);self.assertEqual(len(result),1);self.assertEqual(result[0].track_id,7)
        self.assertEqual(task.tracker.calls[0][0].shape,(1,5));self.assertEqual(task.tracker.calls[0][1:],((17,31),(17,31)))
        result2=task.predict(image);self.assertEqual(task.frame_index,2);self.assertEqual(result[0].tlbr,result2[0].tlbr)
        task.reset();self.assertEqual(task.frame_index,0);self.assertEqual(task.predict(image)[0].tlbr,result[0].tlbr)

    def test_outputs_owned_and_no_person_still_advances_tracker(self):
        detector=FakeDetector();task=ByteTrackTask(detector,tracker_factory=FakeTracker)
        result=task.predict(np.zeros((17,31,3),np.uint8));original=result[0].tlbr
        detector.result.boxes.fill(99);self.assertEqual(result[0].tlbr,original)
        detector.result.class_ids.fill(3);self.assertEqual(task.predict(np.zeros((31,17,3),np.uint8)),())
        self.assertEqual(task.tracker.calls[-1][0].shape,(0,5));self.assertEqual(task.frame_index,2)

    def test_real_detector_binding_preserves_bytetrack_s100p_identity(self):
        runtime=FakeRuntime('s100p','int8');selection=resolve_selection('s100p',consumer='bytetrack')
        detector=YOLOv5Task(runner=lambda _:runtime.outputs,binding=bind_model(selection,runtime.facts))
        task=ByteTrackTask(detector,tracker_factory=FakeTracker);task.predict(np.zeros((97,151,3),np.uint8))
        self.assertEqual(task.frame_index,1);self.assertEqual(selection.asset.sample_id,'bytetrack')

    def test_x5_default_rebases_model_dir_explicit_model_path_wins(self):
        from samples.vision.bytetrack.runtime.python.cli import resolve_selection as bytetrack_selection
        model_dir=ROOT/'samples/vision/bytetrack/model'
        default=bytetrack_selection('x5')
        # Catalog default lands under this sample's model/ dir, identity kept on yolov5.
        self.assertEqual(default.model_path,model_dir/default.asset.filename)
        self.assertEqual(default.consumer,'yolov5')
        explicit=bytetrack_selection('x5',model_path='/tmp/external_yolov5n.bin',
                                     asset_id=default.asset.reference)
        self.assertEqual(explicit.model_path,Path('/tmp/external_yolov5n.bin'))
        self.assertEqual(explicit.asset.reference,default.asset.reference)

    def test_x5_fake_runtime_three_stages_feed_real_tracker_ids_empty_reset(self):
        from samples.vision.bytetrack.runtime.python.cli import resolve_selection as bytetrack_selection
        runtime=FakeRuntime('x5','float32')
        task=ByteTrackTask.from_model(bytetrack_selection('x5'),runtime_factory=lambda _:runtime)
        task.detector.set_scheduling_params(priority=1,bpu_cores=[0])
        self.assertEqual(runtime.schedule,{'priority':{'detector':1},'bpu_cores':{'detector':[0]}})
        image=np.zeros((97,151,3),np.uint8)
        # X5 binding: one flat packed-NV12 tensor at 640x640, raw F32 heads.
        prepared=task.preprocess(image)
        (tensor,)=prepared.tensors.values()
        self.assertEqual(tensor.shape,(640*640*3//2,));self.assertEqual(tensor.dtype,np.uint8)
        staged=task.postprocess(task.infer(prepared.tensors),prepared.context)
        composed=task.predict(image)
        self.assertEqual(len(staged),1);self.assertEqual(len(composed),1)
        self.assertEqual(staged[0].track_id,composed[0].track_id)
        person_id=composed[0].track_id
        # Empty frame: every score below the detector gate still advances the tracker.
        saved={name:value.copy() for name,value in runtime.outputs.items()}
        for value in runtime.outputs.values():value.fill(-30)
        self.assertEqual(task.predict(image),())
        # Restored detections re-activate the buffered track with the same ID.
        for name,value in runtime.outputs.items():value[...]=saved[name]
        self.assertEqual(task.predict(image)[0].track_id,person_id)
        # Reset clears history; the process-global counter never reissues an ID.
        task.reset();self.assertEqual(task.frame_index,0)
        self.assertNotEqual(task.predict(image)[0].track_id,person_id)
        self.assertGreater(task.frame_index,0)

    def test_frame_rate_controls_real_tracker_buffer(self):
        cfg=TrackingConfig(frame_rate=15,track_buffer=60)
        task=ByteTrackTask(FakeDetector(),config=cfg)
        self.assertEqual(task.tracker.max_time_lost,30)
        for kw in [dict(track_thresh=float('nan')),dict(track_buffer=-1),dict(frame_rate=0)]:
            with self.assertRaises(ValueError):TrackingConfig(**kw)

    def test_real_tracker_matches_fixed_source_sequence(self):
        from samples.vision.bytetrack.runtime.python.tracker_backend.byte_tracker import BYTETracker
        source=legacy_tree('s/samples/vision/bytetrack/3rdparty')
        old_path=list(sys.path);sys.path.insert(0,str(source))
        try:legacy=importlib.import_module('tracker.byte_tracker').BYTETracker
        finally:sys.path[:]=old_path
        cfg=TrackingConfig();left=legacy(cfg,frame_rate=cfg.frame_rate);right=BYTETracker(cfg,frame_rate=cfg.frame_rate)
        sequence=[[[10,10,30,40,.95]],[[11,10,31,40,.8]],[[12,11,32,41,.2]],[],[[14,11,34,41,.9]],[[15,11,35,41,.9],[70,10,90,40,.85]],[[16,11,36,41,.9],[71,10,91,40,.85]]]
        # The source counter is process-global. Normalize only initial numeric ID offsets,
        # keeping the per-frame identity correspondence and all geometry/scores exact.
        maps=[{},{}]
        for frame in sequence:
            values=np.asarray(frame,np.float64).reshape(-1,5)
            outputs=[tracker.update(values.copy(),(100,100),(100,100)) for tracker in (left,right)]
            rows=[]
            for i,tracks in enumerate(outputs):
                rows.append([(maps[i].setdefault(t.track_id,len(maps[i])),tuple(t.tlbr),float(t.score)) for t in tracks])
            self.assertEqual(rows[0],rows[1])

if __name__=='__main__':unittest.main()
