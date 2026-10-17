import json
from dataclasses import asdict
from pathlib import Path
from threading import Event
import numpy as np
import pytest
from movement_tracker.pose_data import Landmark, PoseFrame
from movement_tracker.experiments.rotations import (
    ANGLES, rotate_input, to_source, run_rotations,
)
from movement_tracker.experiments.report import export_report, review_summary, comparison_image
from movement_tracker.video.processing import Cancelled
from test_pipeline import make_video, FakeTracker

@pytest.mark.parametrize("angle,input_xy", [(0, (.2, .7)), (90, (.7, .8)),
                                           (180, (.8, .3)), (270, (.3, .2))])
def test_inverse_rotation_preserves_anatomical_identity_depth_and_confidence(angle, input_xy):
    width, height = 1080, 1920
    iw, ih = (height, width) if angle in (90, 270) else (width, height)
    point = Landmark(*input_xy, z=-.25 * width / iw, visibility=.99, presence=.77)
    raw = PoseFrame(.4, 12, iw, ih, {"left_knee": point}, source_timestamp=2.4)
    before = asdict(raw)
    mapped = to_source(raw, angle, width, height)
    assert asdict(raw) == before
    assert set(mapped.landmarks) == {"left_knee"}
    knee = mapped.landmarks["left_knee"]
    assert (knee.x, knee.y, knee.z) == pytest.approx((.2, .7, -.25))
    assert (knee.visibility, knee.presence, knee.confidence) == (.99, .77, .77)
    assert (mapped.width, mapped.height, mapped.frame_index, mapped.source_timestamp) == (1080, 1920, 12, 2.4)

@pytest.mark.parametrize("angle,expected", [(0, (0, 0)), (90, (2, 0)), (180, (2, 1)), (270, (0, 1))])
def test_counterclockwise_pixel_rotation(angle, expected):
    rgb = np.zeros((2, 3, 3), np.uint8)
    rgb[0, 0] = (255, 0, 0)
    rotated = rotate_input(rgb, angle)
    if angle in (0, 180):
        # Expected source top-left is top-left / bottom-right.
        row, column = (0, 0) if angle == 0 else (1, 2)
    else:
        row, column = expected
    assert tuple(rotated[row, column]) == (255, 0, 0)
    assert rotated.flags.c_contiguous

def test_out_of_bounds_and_missing_depth_are_not_fabricated_or_clipped():
    raw = PoseFrame(0, 0, 1920, 1080, {"nose": Landmark(-.2, 1.4)})
    nose = to_source(raw, 90, 1080, 1920).landmarks["nose"]
    assert (nose.x, nose.y) == pytest.approx((-.4, -.2))
    assert nose.z is None and nose.confidence is None
    empty = to_source(PoseFrame(0, 0, 1920, 1080), 90, 1080, 1920)
    assert not empty.landmarks

def test_mismatched_dimensions_rejected():
    with pytest.raises(ValueError, match="dimensions"):
        to_source(PoseFrame(0, 0, 1080, 1920), 90, 1080, 1920)

class RecordingTracker(FakeTracker):
    def __init__(self):
        self.calls = []
        self.closed = False
    def process_frame(self, rgb, timestamp, frame_index):
        self.calls.append((frame_index, timestamp, rgb.shape))
        return super().process_frame(rgb, timestamp, frame_index)
    def close(self):
        self.closed = True

@pytest.mark.parametrize("mode,expected_indices", [("video", [0, 1, 2, 3]), ("image", [2, 3])])
def test_separate_histories_and_precise_selected_frames(tmp_path, mode, expected_indices):
    source = tmp_path / "clip.mp4"
    make_video(source)
    trackers = []
    def factory():
        tracker = RecordingTracker()
        trackers.append(tracker)
        return tracker
    progress = []
    result = run_rotations(source, tmp_path / "unused.task", 2, 3, Event(),
                           lambda n, total: progress.append((n, total)), mode, factory)
    try:
        assert len(trackers) == 4 and all(t.closed for t in trackers)
        for tracker, angle in zip(trackers, ANGLES):
            assert [call[0] for call in tracker.calls] == expected_indices
            expected_shape = (96, 64, 3) if angle in (90, 270) else (64, 96, 3)
            assert all(call[2] == expected_shape for call in tracker.calls)
        assert [f.frame_index for f in result.frames] == [2, 3]
        assert result.frames[1].timestamp == pytest.approx(.15)
        assert [c.angle_ccw for c in result.frames[0].candidates] == list(ANGLES)
        assert progress[-1] == (4 * len(expected_indices), 4 * len(expected_indices))
        assert result.preview(0).shape == (64, 96, 3)
        target = tmp_path / "report.json"
        reviews = {"2": {"ratings": {"0": "wrong_anatomy", "180": "correct_anatomy"}}}
        export_report(result, reviews, target)
        report = json.loads(target.read_text())
        candidate = report["frames"][0]["candidates"][1]
        assert candidate["raw_rotated_input_pose"]["width"] == 64
        assert candidate["source_coordinates_pose"]["width"] == 96
        assert candidate["raw_rotated_input_pose"]["landmarks"]["left_knee"]["confidence"] == .8
        assert report["review_summary"]["human_reviewed_recovery_frames"] == [2]
        assert not candidate["interpolated"]
        assert candidate["anatomy_correctness"] == "unreviewed"  # Raw evidence remains separate.
        assert not result.metadata["correction_applied"]
        assert comparison_image(result.preview(0), result.frames[0]).shape == (304, 192, 3)
        with pytest.raises(ValueError, match="overwritten"):
            export_report(result, {}, source)
    finally:
        cache = Path(result.cache.name)
        result.close()
        assert not cache.exists()

def test_partial_initialization_closes_already_created_trackers(tmp_path):
    source = tmp_path / "clip.mp4"
    make_video(source)
    first = RecordingTracker()
    calls = []
    def factory():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("failed initialization")
        return first
    with pytest.raises(RuntimeError, match="initialization"):
        run_rotations(source, tmp_path / "unused", 0, 1, Event(), tracker_factory=factory)
    assert first.closed

def test_cancelled_experiment_closes_trackers(tmp_path):
    source = tmp_path / "clip.mp4"
    make_video(source)
    trackers = []
    cancel = Event()
    def factory():
        tracker = RecordingTracker()
        trackers.append(tracker)
        return tracker
    with pytest.raises(Cancelled):
        run_rotations(source, tmp_path / "unused", 0, 3, cancel,
                      lambda *args: cancel.set(), tracker_factory=factory)
    assert all(t.closed for t in trackers)

def test_out_of_range_is_error_instead_of_silent_truncation(tmp_path):
    source = tmp_path / "clip.mp4"
    make_video(source)
    with pytest.raises(ValueError, match="beyond"):
        run_rotations(source, tmp_path / "unused", 0, 9, Event(), tracker_factory=RecordingTracker)

def test_high_confidence_does_not_become_anatomy_judgment(tmp_path):
    source = tmp_path / "clip.mp4"
    make_video(source)
    result = run_rotations(source, tmp_path / "unused", 0, 1, Event(), tracker_factory=RecordingTracker)
    try:
        summary = review_summary(result.frames, {})
        assert summary["frames_with_any_human_rating"] == 0
        assert summary["human_reviewed_recovery_frames"] == []
        assert all(not c.input_pose.landmarks for c in result.frames[1].candidates)
    finally:
        result.close()
