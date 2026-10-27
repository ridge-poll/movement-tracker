import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from dataclasses import replace
from types import SimpleNamespace
import json
import csv
import numpy as np
import pytest
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from movement_tracker.pose_corrections import PoseCorrections, corrections_for
from movement_tracker.analysis import analyze
from movement_tracker.export.analysis_export import export_analysis
from movement_tracker.ui.pose_editor import VideoView
from movement_tracker.ui.analysis_panel import AnalysisPanel
from movement_tracker.visualization.skeleton_renderer import OverlayOptions
from test_analysis import synthetic, parameters


def test_non_destructive_frame_local_history():
    frames = synthetic()
    frame = frames[12]
    store = PoseCorrections()
    original = frame.landmarks.copy()
    p = original['left_shoulder']
    assert store.move(frame, 'left_shoulder', .2, .3)
    edited = store.apply(frame)
    assert frame.landmarks == original
    assert edited.landmarks['right_shoulder'] is original['right_shoulder']
    assert edited.landmarks['left_shoulder'] == replace(p, x=.2, y=.3)
    assert store.apply(frames[11]) is frames[11]
    assert store.reset(12)
    assert store.apply(frame) is frame
    assert store.undo(12)
    assert store.apply(frame) == edited
    assert store.undo(12)
    assert store.apply(frame) is frame
    assert not store.undo(12)
    with pytest.raises(ValueError):
        store.move(frame, 'left_shoulder', float('nan'), .3)


def test_corrected_rotation_recalculation_and_export_provenance(tmp_path):
    good = synthetic()
    broken = list(good)
    points = good[16].landmarks.copy()
    for side in ('left', 'right'):
        name = side + '_shoulder'
        points[name] = replace(points[name], x=.1, y=.1, visibility=.01)
    broken[16] = replace(good[16], landmarks=points)
    store = PoseCorrections()
    assert analyze(broken, parameters()).summary['total_rotation'].value is None
    for side in ('left', 'right'):
        name = side + '_shoulder'
        p = good[16].landmarks[name]
        store.move(broken[16], name, p.x, p.y)
    effective = [store.apply(f) for f in broken]
    result = analyze(effective, parameters(), store.metadata(broken))
    assert result.summary['total_rotation'].value == pytest.approx(720)
    assert effective[16].landmarks['left_shoulder'].visibility == .01
    assert broken[16].landmarks['left_shoulder'].x == .1
    assert result.metadata['pose_correction_applied']
    assert len(result.metadata['manual_pose_corrections']) == 2
    csv_path, json_path = export_analysis(result, tmp_path/'corrected.csv')
    rows = list(csv.DictReader(csv_path.open()))
    assert 'manual_landmark:left_shoulder' in rows[16]['quality_flags']
    payload = json.loads(json_path.read_text())
    assert payload['metadata']['manual_pose_corrections'][0]['raw_xy'] == [.1, .1]
    assert payload['metadata']['manual_landmarks_by_frame']['16'] == ['left_shoulder', 'right_shoulder']


@pytest.mark.parametrize('size', [(800, 400), (400, 800)])
def test_drag_letterbox_single_commit_and_selection(size):
    app = QApplication.instance() or QApplication([])
    view = VideoView()
    view.resize(*size)
    frame = synthetic()[0]
    view.set_frame(np.zeros((600, 500, 3), dtype=np.uint8), frame, OverlayOptions(), True)
    view.selected = 'left_shoulder'
    view.show()
    app.processEvents()
    ox, oy, w, h = view.image_geometry()
    p = frame.landmarks['left_shoulder']
    start = QPoint(round(ox+p.x*w), round(oy+p.y*h))
    end = QPoint(round(ox+.4*w), round(oy+.3*h))
    emitted = []
    view.landmark_moved.connect(lambda *args: emitted.append(args))
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view, end)
    assert not emitted
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=end)
    assert len(emitted) == 1
    assert emitted[0][0] == 'left_shoulder'
    assert emitted[0][1:] == pytest.approx((.4, .3), abs=.004)
    assert frame.manual_landmarks == ()
    view.close()


def test_panel_preserves_events_and_invalidates_corrected_analysis(tmp_path):
    app = QApplication.instance() or QApplication([])
    clip = SimpleNamespace(frames=synthetic(), source=tmp_path/'test.mov', source_size=0, source_mtime_ns=0)
    panel = AnalysisPanel()
    panel.load_clip(clip)
    panel.facing.setCurrentIndex(panel.facing.findData('right'))
    for name, time in [('takeoff', 100), ('landing', 101)]:
        panel.set_position(time, 0)
        panel.set_event_from_video(name)
    panel.recalculate()
    store = corrections_for(clip)
    store.move(clip.frames[20], 'left_knee', .4, .4)
    panel.settings_changed()
    assert panel.stale and not panel.export_button.isEnabled()
    panel.load_clip(clip)
    assert panel.boundaries_set == {'takeoff', 'landing'}
    panel.recalculate()
    assert panel.result.metadata['pose_correction_applied']
    assert not panel.stale
    assert panel.result.metadata['manual_pose_corrections'][0]['landmark'] == 'left_knee'
    store.undo(20)
    panel.settings_changed()
    panel.recalculate()
    assert not panel.result.metadata['pose_correction_applied']
    new_clip = SimpleNamespace(frames=synthetic())
    assert not corrections_for(new_clip).overrides
    panel.close()
