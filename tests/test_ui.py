import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from movement_tracker.ui.main_window import MainWindow
from movement_tracker.models import DEFAULT_MODEL
from movement_tracker.video.processing import process_video
from threading import Event
from test_pipeline import make_video, FakeTracker

def test_frame_stepping_scrubbing_and_controls(tmp_path):
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "clip.mp4"
    make_video(source)
    window = MainWindow(DEFAULT_MODEL)
    assert not window.play_button.isEnabled()
    window.clip = process_video(source, FakeTracker(), Event())
    window.times = [p.timestamp for p in window.clip.frames]
    window.slider.setRange(0, len(window.times) - 1)
    window.update_controls()
    window.show_frame(0)
    assert window.play_button.isEnabled()
    window.step(1)
    assert window.index == 1
    assert "N/A" in window.metrics.text()
    window.slider.setValue(3)
    assert window.index == 3
    window.step(1)
    assert window.index == 3
    window.step(-1)
    assert window.index == 2
    window.play()
    assert window.playing
    window.pause()
    assert not window.playing
    window.close()
    app.processEvents()
