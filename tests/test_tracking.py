from types import SimpleNamespace
import numpy as np
from movement_tracker.tracking.mediapipe_tracker import MediaPipeTracker

def test_adapter_selects_prominent_pose_and_preserves_source_time():
    def body(size):
        return [SimpleNamespace(x=.1 + (i % 2) * size, y=.1 + (i % 3) * size,
                                z=-.1, visibility=.8, presence=.9) for i in range(33)]
    timestamps = []
    class Model:
        def detect_for_video(self, image, timestamp):
            timestamps.append(timestamp)
            return SimpleNamespace(pose_landmarks=[body(.1), body(.3)])
    tracker = MediaPipeTracker.__new__(MediaPipeTracker)
    tracker._last_ms = -1
    tracker._model = Model()
    tracker._mp = SimpleNamespace(Image=lambda **kwargs: kwargs, ImageFormat=SimpleNamespace(SRGB=1))
    rgb = np.zeros((64, 96, 3), np.uint8)
    first = tracker.process_frame(rgb, .0001, 0)
    second = tracker.process_frame(rgb, .0002, 1)
    assert timestamps == [0, 1]
    assert first.timestamp == .0001 and second.timestamp == .0002
    assert len(first.landmarks) == 33
    assert first.landmarks['left_eye_inner'].x == .4
    assert first.landmarks['nose'].confidence == .8

def test_no_detection_never_reuses_previous_pose():
    class Model:
        def detect_for_video(self, *args):
            return SimpleNamespace(pose_landmarks=[])
    tracker = MediaPipeTracker.__new__(MediaPipeTracker)
    tracker._last_ms = -1
    tracker._model = Model()
    tracker._mp = SimpleNamespace(Image=lambda **kwargs: kwargs, ImageFormat=SimpleNamespace(SRGB=1))
    frame = tracker.process_frame(np.zeros((64, 96, 3), np.uint8), 1, 30)
    assert frame.landmarks == {}
    assert frame.confidence is None

def test_job_coalesces_results_and_surfaces_errors():
    from movement_tracker.jobs import Job
    def run(job):
        job.publish("old")
        job.publish("new")
        raise ValueError("test failure")
    job = Job(run)
    assert job.done.wait(timeout=2)
    assert job.take_latest() == "new"
    assert job.take_latest() is None
    assert job.error == "test failure"
