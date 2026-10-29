from dataclasses import replace
import numpy as np
import pytest
from movement_tracker.pose_data import Landmark
from movement_tracker.pose_corrections import PoseCorrections
from movement_tracker.tracking import guided_estimate
from test_analysis import synthetic


def test_crop_coordinate_mapping_and_estimate_undo(monkeypatch):
    frame = synthetic()[0]
    anchors = {'head':(.2,.4), 'near_shoulder':(.3,.4), 'near_hip':(.5,.4),
               'near_knee':(.6,.6), 'near_ankle':(.8,.6)}
    captured = {}
    real_warp = guided_estimate.cv2.warpAffine
    def warp(rgb, matrix, size, **kwargs):
        captured.update(matrix=matrix, size=size)
        return real_warp(rgb, matrix, size, **kwargs)
    monkeypatch.setattr(guided_estimate.cv2, 'warpAffine', warp)
    class Tracker:
        def process_frame(self, crop, timestamp, index):
            xy = captured['matrix'] @ np.array([.3*499,.4*599,1])
            w,h = captured['size']
            return replace(frame, landmarks={'left_shoulder':Landmark(xy[0]/(w-1),xy[1]/(h-1),.1,.8,.9)})
    points = guided_estimate.estimate_pose(np.zeros((600,500,3),dtype=np.uint8),frame,anchors,Tracker())
    assert points['left_shoulder'].x == pytest.approx(.3)
    assert points['left_shoulder'].y == pytest.approx(.4)
    assert points['left_shoulder'].z is None
    store = PoseCorrections()
    store.rough(frame, anchors)
    rough = store.apply(frame)
    store.estimate(frame,points,anchors)
    effective = store.apply(frame)
    assert effective.manual_landmarks == ()
    assert effective.landmarks['left_shoulder'].confidence == .8
    assert store.metadata([frame])['manual_pose_replacements'][0]['guide_anchors'] == anchors
    store.undo(frame.frame_index)
    assert store.apply(frame) == rough


def test_no_detection_and_insufficient_guides_do_not_create_pose():
    frame = synthetic()[0]
    class Tracker:
        def process_frame(self, *args):
            return replace(frame, landmarks={})
    image = np.zeros((600,500,3),dtype=np.uint8)
    with pytest.raises(ValueError, match='shoulder'):
        guided_estimate.estimate_pose(image,frame,{},Tracker())
    with pytest.raises(ValueError, match='No pose'):
        guided_estimate.estimate_pose(image,frame,{'near_shoulder':(.5,.3),'near_hip':(.5,.5)},Tracker())
