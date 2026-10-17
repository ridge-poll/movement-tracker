import csv
from fractions import Fraction
from pathlib import Path
from threading import Event
import av
import numpy as np
import pytest
from movement_tracker.pose_data import Landmark, PoseFrame, LANDMARK_NAMES
from movement_tracker.video.video_reader import VideoReader
from movement_tracker.video.processing import process_video, Cancelled
from movement_tracker.export.csv_export import export_csv
from movement_tracker.export.video_export import export_video
from movement_tracker.visualization.skeleton_renderer import render, OverlayOptions

def make_video(path, pts=(0, 33333, 66667, 150000), rotation=0, rate=30):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height = 96, 64
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1_000_000)
        stream.codec_context.time_base = stream.time_base
        stream.codec_context.max_b_frames = 0
        if rotation:
            stream.set_display_rotation(rotation)
        for i, timestamp in enumerate(pts):
            rgb = np.zeros((64, 96, 3), np.uint8)
            rgb[:, :48] = (180, 30, 30)
            rgb[:, 48:] = (30, 30, 180)
            rgb[i: i + 5] = 240
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = timestamp
            frame.time_base = stream.time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

class FakeTracker:
    def process_frame(self, rgb, timestamp, frame_index):
        # Exercise explicit no-detection rows as well as all 33 named landmarks.
        points = {} if frame_index == 1 else {
            name: Landmark(.3 + i / 100, .3 + i / 100, -.1, .8, .9)
            for i, name in enumerate(LANDMARK_NAMES)
        }
        return PoseFrame(timestamp, frame_index, rgb.shape[1], rgb.shape[0], points)

@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "source.mp4"
    make_video(path)
    result = process_video(path, FakeTracker(), Event())
    yield result
    result.close()

def test_full_processing_preserves_variable_timestamps_and_all_frames(clip):
    assert len(clip.frames) == 4
    assert [p.timestamp for p in clip.frames] == pytest.approx([0, .033333, .066667, .15], abs=2e-6)
    assert len(clip.frames[0].landmarks) == 33
    assert not clip.frames[1].landmarks
    assert clip.preview(3).shape == (64, 96, 3)
    assert not any(p.timestamp_estimated for p in clip.frames)

def test_csv_keeps_missing_pose_rows_and_source_metadata(clip, tmp_path):
    target = tmp_path / "data.csv"
    export_csv(clip.frames, target)
    with target.open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert rows[1]["pose_detected"] == "0"
    assert rows[1]["left_knee_x"] == ""
    assert rows[0]["left_knee_confidence"] == "0.8"
    assert "right_foot_index_presence" in rows[0]
    assert rows[3]["source_timestamp"] != ""

def test_video_export_preserves_frame_count_timing_and_overlays(clip, tmp_path):
    target = tmp_path / "export.mp4"
    export_video(clip, target, OverlayOptions(), Event())
    with VideoReader(target) as reader:
        frames = list(reader)
    assert len(frames) == len(clip.frames)
    assert [f.timestamp for f in frames] == pytest.approx([p.timestamp for p in clip.frames], abs=2e-6)
    with VideoReader(clip.source) as reader:
        source = list(reader)
    assert np.mean(np.abs(frames[0].rgb.astype(float) - source[0].rgb)) > 2
    assert frames[-1].duration == pytest.approx(clip.last_duration, abs=2e-6)

@pytest.mark.parametrize("rotation", [90, -90, 180])
def test_rotation_is_applied_to_pixels(tmp_path, rotation):
    source = tmp_path / "rotated.mp4"
    make_video(source, rotation=rotation)
    with VideoReader(source) as reader:
        frame = next(iter(reader))
    expected_shape = (96, 64, 3) if abs(rotation) == 90 else (64, 96, 3)
    assert frame.rgb.shape == expected_shape
    if rotation == 90:
        assert frame.rgb[10, 30, 2] > frame.rgb[10, 30, 0]
    elif rotation == -90:
        assert frame.rgb[10, 30, 0] > frame.rgb[10, 30, 2]

def test_240fps_does_not_drop_frames(tmp_path):
    source = tmp_path / "fast.mp4"
    make_video(source, pts=tuple(round(i * 1_000_000 / 240) for i in range(24)), rate=240)
    clip = process_video(source, FakeTracker(), Event())
    try:
        assert len(clip.frames) == 24
        assert clip.frames[-1].timestamp == pytest.approx(23 / 240, abs=2e-6)
    finally:
        clip.close()

def test_cancelled_export_preserves_existing_output(clip, tmp_path):
    target = tmp_path / "export.mp4"
    target.write_bytes(b"existing")
    cancel = Event()
    cancel.set()
    with pytest.raises(Cancelled):
        export_video(clip, target, OverlayOptions(), cancel)
    assert target.read_bytes() == b"existing"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["export.mp4", "source.mp4"]

def test_source_cannot_be_overwritten(clip):
    with pytest.raises(ValueError, match="different"):
        export_video(clip, clip.source, OverlayOptions(), Event())

def test_changed_source_rejected(clip, tmp_path):
    with clip.source.open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        export_video(clip, tmp_path / "out.mp4", OverlayOptions(), Event())

def test_overlay_does_not_change_raw_frame_or_pose():
    rgb = np.zeros((100, 100, 3), np.uint8)
    pose = PoseFrame(0, 0, 100, 100, {"nose": Landmark(.5, .5, visibility=.2)})
    assert render(rgb, pose).sum() > 0
    assert rgb.sum() == 0
    assert not render(rgb, pose, OverlayOptions(False, False, False)).any()

def test_cancelled_processing(tmp_path):
    source = tmp_path / "source.mp4"
    make_video(source)
    cancel = Event()
    cancel.set()
    with pytest.raises(Cancelled):
        process_video(source, FakeTracker(), cancel)
