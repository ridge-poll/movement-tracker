import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from movement_tracker.pose_data import PoseFrame, Landmark
from movement_tracker.pose_corrections import PoseCorrections
from movement_tracker.ui.pose_editor import VideoView
from movement_tracker.visualization.skeleton_renderer import OverlayOptions

@pytest.fixture
def editor():
    app = QApplication.instance() or QApplication([])
    frame = PoseFrame(0, 0, 500, 500, {
        'left_ankle': Landmark(.3, .5), 'left_heel': Landmark(.3, .6),
        'left_foot_index': Landmark(.4, .6), 'right_knee': Landmark(.7, .3),
        'left_wrist': Landmark(.6, .7), 'left_index': Landmark(.7, .7),
        'left_thumb': Landmark(.65, .8), 'left_pinky': Landmark(.7, .8)})
    view = VideoView()
    view.resize(700, 500)
    view.set_frame(np.zeros((500, 500, 3), dtype=np.uint8), frame, OverlayOptions(), True)
    view.show()
    app.processEvents()
    yield view, frame
    view.close()


def point(view, x, y):
    ox, oy, w, h = view.image_geometry()
    return QPoint(round(ox+x*w), round(oy+y*h))


def click(view, x, y, shift=False):
    QTest.mouseClick(view, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier,
                     point(view,x,y))


def drag(view, start, end):
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=point(view,*start))
    QTest.mouseMove(view, point(view,*end))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=point(view,*end))


def test_anatomical_selection_shift_and_individual(editor):
    view, frame = editor
    click(view,.3,.5)
    assert view.selection == {'left_ankle','left_heel','left_foot_index'}
    click(view,.7,.3,True)
    assert 'right_knee' in view.selection
    click(view,.3,.6,True)
    assert 'left_heel' not in view.selection
    click(view,.1,.1)
    assert not view.selection
    click(view,.6,.7)
    assert view.selection == {'left_wrist','left_index','left_thumb','left_pinky'}
    click(view,.1,.1)
    view.group_extremities = False
    click(view,.3,.5)
    assert view.selection == {'left_ankle'}


def test_marquee_is_exact_and_does_not_expand_groups(editor):
    view, frame = editor
    drag(view,(.25,.45),(.35,.65))
    assert view.selection == {'left_ankle','left_heel'}
    assert view.marquee is None
    assert not view.grab().isNull()


def test_group_move_single_undo_preserves_offsets_and_raw(editor):
    view, frame = editor
    store = PoseCorrections()
    events = []
    def commit(positions):
        events.append(positions)
        store.move_many(frame, positions)
        view.set_frame(view.rgb, store.apply(frame), OverlayOptions(), True)
    view.landmarks_moved.connect(commit)
    click(view,.3,.5)
    drag(view,(.3,.5),(.4,.4))
    assert len(events) == 1
    assert len(store.history[0]) == 1
    effective = store.apply(frame)
    deltas = [(effective.landmarks[n].x-frame.landmarks[n].x,
               effective.landmarks[n].y-frame.landmarks[n].y) for n in view.selection]
    assert all(delta == pytest.approx(deltas[0]) for delta in deltas)
    assert effective.landmarks['right_knee'] is frame.landmarks['right_knee']
    assert frame.landmarks['left_ankle'].x == .3
    assert store.undo(0)
    assert store.apply(frame) is frame


def test_boundary_clamps_group_without_deformation(editor):
    view, frame = editor
    events = []
    view.landmarks_moved.connect(events.append)
    drag(view,(.3,.5),(1.,1.))
    positions = events[0]
    assert positions['left_foot_index'] == pytest.approx((1.,1.))
    assert positions['left_ankle'] == pytest.approx((.9,.9))


def test_click_is_selection_only_and_batch_validation_atomic(editor):
    view, frame = editor
    events = []
    view.landmarks_moved.connect(events.append)
    click(view,.3,.5)
    assert events == []
    store = PoseCorrections()
    with pytest.raises(ValueError):
        store.move_many(frame, {'left_ankle':(.4,.4),'left_heel':(2.,.4)})
    assert store.revision == 0 and not store.history and not store.overrides
    view.set_frame(view.rgb, PoseFrame(1,1,500,500,frame.landmarks), OverlayOptions(), True)
    assert not view.selection
