import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from threading import Event
from PySide6.QtWidgets import QApplication
from movement_tracker.ui.rotation_window import RotationWindow
from movement_tracker.experiments.rotations import run_rotations
from test_pipeline import make_video
from test_rotations import RecordingTracker

def test_comparison_navigation_review_and_raw_preservation(tmp_path):
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "clip.mp4"
    make_video(source)
    result = run_rotations(source, tmp_path / "unused", 0, 3, Event(), tracker_factory=RecordingTracker)
    window = RotationWindow(source, tmp_path / "unused", 0, 3)
    assert not window.json_button.isEnabled()
    window.load_result(result)
    assert window.json_button.isEnabled()
    original = result.frames[0].candidates[0].input_pose
    window.ratings[0].setCurrentIndex(window.ratings[0].findData("wrong_anatomy"))
    window.ratings[2].setCurrentIndex(window.ratings[2].findData("correct_anatomy"))
    assert "1 manually identified" in window.status.text()
    window.step(1)
    assert window.offset == 1
    assert all("NO POSE" in heading.text() for heading in window.headings)
    assert window.ratings[0].currentData() == "unreviewed"
    window.step(-1)
    assert window.ratings[0].currentData() == "wrong_anatomy"
    assert window.ratings[2].currentData() == "correct_anatomy"
    assert result.frames[0].candidates[0].input_pose is original
    window.slider.setValue(3)
    assert window.offset == 3
    window.step(1)
    assert window.offset == 3
    window.angle_select.setCurrentIndex(1)
    assert window.table.item(0, 1).text() != "—"
    window.dirty = False
    window.close()
    app.processEvents()
