import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from movement_tracker.pose_corrections import PoseCorrections
from movement_tracker.analysis import analyze
from movement_tracker.ui.main_window import MainWindow
from test_analysis import synthetic, parameters


def test_clear_rough_drag_reset_and_undo():
    raw = synthetic()[0]
    store = PoseCorrections()
    store.clear(raw.frame_index)
    assert not store.apply(raw).landmarks
    assert raw.landmarks
    store.rough(raw, {'near_shoulder':(.3,.3), 'near_hip':(.3,.5)})
    rough = store.apply(raw)
    assert set(rough.landmarks) == {'near_shoulder','near_hip'}
    assert rough.landmarks['near_hip'].confidence is None
    assert rough.landmarks['near_hip'].z is None
    store.move( raw, 'near_hip', .4,.5)
    assert store.apply(raw).landmarks['near_hip'].x == .4
    store.undo(raw.frame_index)
    assert store.apply(raw) == rough
    store.undo(raw.frame_index)
    assert store.apply(raw).manual_pose_mode == 'cleared'
    store.undo(raw.frame_index)
    assert store.apply(raw) is raw
    store.rough(raw, {'head':(.3,.2)})
    store.reset(raw.frame_index)
    assert store.apply(raw) is raw


def test_missing_detection_rebuilt_analysis_and_provenance():
    good = synthetic()
    frames = list(good)
    frames[16] = replace(good[16], landmarks={})
    store = PoseCorrections()
    positions = {'near_'+joint:(good[16].landmarks['left_'+joint].x,
                              good[16].landmarks['left_'+joint].y)
                 for joint in ('shoulder','hip','knee')}
    store.rough(frames[16], positions)
    result = analyze([store.apply(f) for f in frames], parameters(), store.metadata(frames))
    assert result.summary['total_rotation'].value == pytest.approx(720)
    assert result.series['tuck_ratio'][16] is not None
    assert 'manual_pose:rough_single_side' in result.quality_flags[16]
    assert all(r['raw_xy'] is None for r in result.metadata['manual_pose_corrections'])
    store.clear(16)
    result = analyze([store.apply(f) for f in frames], parameters(), store.metadata(frames))
    assert result.metadata['pose_correction_applied']
    assert result.series['tuck_ratio'][16] is None


def test_guided_placement_back_skip_cancel_and_single_commit(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path/'model')
    frames = synthetic()
    window.clip = SimpleNamespace(frames=frames, fps=40, source=Path('test.mov'),
        preview=lambda i: np.zeros((600,500,3),dtype=np.uint8), close=lambda:None)
    window.times = [f.timestamp for f in frames]
    window.slider.setRange(0,len(frames)-1)
    window.show_frame(0)
    window.start_rough()
    window.advance_rough((.4,.2))
    window.advance_rough(None)
    window.back_rough()
    assert window.rough_step == 1
    window.advance_rough((.4,.3))
    window.step(1)
    window.play()
    assert window.index == 0 and not window.playing
    window.advance_rough((.4,.5))
    window.advance_rough((.4,.7))
    window.advance_rough((.4,.9))
    store = window.clip.corrections
    assert len(store.history[0]) == 1
    assert len(store.apply(frames[0]).landmarks) == 5
    window.start_rough()
    window.advance_rough((.1,.1))
    window.cancel_rough()
    assert len(store.history[0]) == 1
    assert store.apply(frames[0]).landmarks['head'].x == .4
    window.change_correction('undo')
    assert store.apply(frames[0]) is frames[0]
    window.close()
