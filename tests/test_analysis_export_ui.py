import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PySide6.QtWidgets import QApplication
from movement_tracker.analysis import analyze
from movement_tracker.export.analysis_export import export_analysis
from movement_tracker.ui.analysis_panel import AnalysisPanel
from test_analysis import synthetic, parameters

def test_export_units_exact_events_nulls_and_metadata(tmp_path):
    result = analyze(synthetic(), parameters(event_overrides={"opening": 100.8123}))
    csv_path, json_path = export_analysis(result, tmp_path / "measurements.csv")
    with csv_path.open() as file:
        rows = list(csv.DictReader(file))
    payload = json.loads(json_path.read_text())
    assert len(rows) == 41
    assert rows[0]["source_timestamp"] == "100.0"
    assert sum(int(row["event_opening"]) for row in rows) == 1
    assert payload["events"]["opening"]["source_timestamp"] == 100.8123
    assert payload["units"]["angular_velocity_deg_s"] == "deg/s"
    assert payload["smoothing"]["method"] == "local_polynomial_on_actual_timestamps"
    assert payload["export"]["csv_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert payload["analysis_version"] == "2.1.0"

def test_export_pair_rolls_back_if_second_replace_fails(tmp_path, monkeypatch):
    from movement_tracker.export import analysis_export
    csv_path, json_path = tmp_path / "data.csv", tmp_path / "data.json"
    csv_path.write_text("old csv")
    json_path.write_text("old json")
    original = analysis_export.os.replace
    def fail_json(source, target):
        if Path(source).name == "new.json":
            raise OSError("simulated write failure")
        return original(source, target)
    monkeypatch.setattr(analysis_export.os, "replace", fail_json)
    with pytest.raises(OSError):
        export_analysis(analyze(synthetic(), parameters()), csv_path)
    assert csv_path.read_text() == "old csv"
    assert json_path.read_text() == "old json"

def test_analysis_panel_source_times_manual_events_stale_result_and_render(tmp_path):
    app = QApplication.instance() or QApplication([])
    clip = SimpleNamespace(frames=synthetic(), source=tmp_path / "synthetic.mp4", source_size=0, source_mtime_ns=0)
    panel = AnalysisPanel()
    panel.load_clip(clip)
    with pytest.raises(ValueError, match="both"):
        panel.parameters()
    panel.facing.setCurrentIndex(panel.facing.findData("right"))
    panel.set_position(100, 0)
    panel.set_event_from_video("takeoff")
    panel.set_position(101, 40)
    panel.set_event_from_video("landing")
    panel.recalculate()
    assert panel.result.summary["total_rotation"].value == pytest.approx(720)
    assert panel.export_button.isEnabled()
    panel.set_position(100.5, 20)
    assert all(plot.position == 100.5 for plot in panel.plots)
    assert "100.5000" in panel.position.text()
    panel.resize(620, 960)
    panel.show()
    app.processEvents()
    assert not panel.grab().isNull()
    panel.window.setValue(.2)
    assert panel.stale
    assert not panel.export_button.isEnabled()
    panel.recalculate()
    assert not panel.stale
    panel.close()

def test_plot_seek_and_video_scrub_use_source_time_bidirectionally(tmp_path):
    import numpy as np
    from movement_tracker.ui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path / "unused.task")
    window.clip = SimpleNamespace(frames=synthetic(), source=tmp_path / "synthetic.mp4", source_size=0,
                                  source_mtime_ns=0, fps=40, preview=lambda i: np.zeros((120,100,3),np.uint8),
                                  close=lambda: None)
    window.times = [frame.timestamp for frame in window.clip.frames]
    window.slider.setRange(0, 40)
    window.open_analysis()
    window.analysis_panel.plots[0].time_clicked.emit(100.5)
    assert window.index == 20
    assert window.analysis_panel.current_time == 100.5
    window.slider.setValue(10)
    assert window.analysis_panel.current_time == 100.25
    assert all(plot.position == 100.25 for plot in window.analysis_panel.plots)
    window.close()
    app.processEvents()
