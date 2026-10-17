from pathlib import Path
import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QTabWidget, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QCheckBox,
    QPlainTextEdit, QFileDialog, QMessageBox, QScrollArea,
)
from movement_tracker.analysis import AnalysisParameters, analyze
from movement_tracker.analysis.models import EVENT_NAMES
from movement_tracker.analysis.kinematics import source_times
from movement_tracker.export.analysis_export import export_analysis
from .analysis_plot import AnalysisPlot

class AnalysisPanel(QWidget):
    seek_time = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.clip = None
        self.result = None
        self.current_time = None
        self.current_frame = 0
        self.boundaries_set = set()
        self.loading = False
        self.stale = False
        layout = QVBoxLayout(self)
        self.position = QLabel("Choose takeoff, landing, and facing direction.")
        self.position.setWordWrap(True)
        layout.addWidget(self.position)
        self.tabs = QTabWidget()
        self.plots = []
        plot_page = QWidget()
        plot_layout = QVBoxLayout(plot_page)
        self.flight_only = QCheckBox("Focus plots on flight")
        self.flight_only.setChecked(True)
        self.flight_only.toggled.connect(self.refresh_plots)
        plot_layout.addWidget(self.flight_only)
        definitions = [
            ("Hip midpoint proxy (upward positive; not COM)", "px", [("hip_vertical_px", "Measured · teal", "#55dab4"), ("fitted_hip_vertical_px", "Ballistic fit · yellow", "#f1ce6c")]),
            ("Cumulative rotation from takeoff (+ backward)", "deg", [("cumulative_rotation_deg", "Measured continuous segments", "#67d5ff")]),
            ("Angular velocity", "deg/s", [("angular_velocity_raw_deg_s", "Raw · gray", "#788494"), ("angular_velocity_deg_s", "Local polynomial · teal", "#55dab4")]),
            ("Tuck: shoulder–knee distance / torso length", "1", [("tuck_ratio", "Visibility-weighted sides", "#d9a5ff")]),
        ]
        for title, unit, curves in definitions:
            plot = AnalysisPlot(title, unit, curves)
            plot.time_clicked.connect(self.seek_time.emit)
            plot_layout.addWidget(plot, 1)
            self.plots.append(plot)
        legend = QLabel("Click a plot to seek video. White line: video position. Red bands: quality warnings.\nTO: takeoff · A: apex · T: tuck start · MT: max tuck · O: opening · L: landing")
        legend.setWordWrap(True)
        plot_layout.addWidget(legend)
        self.tabs.addTab(plot_page, "Plots")
        settings = QWidget()
        form = QVBoxLayout(settings)
        instructions = QLabel("Use the video controls to find takeoff and landing, then set each from the current frame. Times below are source-video seconds. Automatic events are suggestions, not ground truth.")
        instructions.setWordWrap(True)
        form.addWidget(instructions)
        event_grid = QGridLayout()
        self.event_times, self.event_modes = {}, {}
        for row, name in enumerate(EVENT_NAMES):
            event_grid.addWidget(QLabel(name.replace("_", " ").title()), row, 0)
            time = QDoubleSpinBox()
            time.setDecimals(6)
            time.setRange(-1e9, 1e9)
            time.setSingleStep(.01)
            mode = QComboBox()
            if name in ("takeoff", "landing"):
                mode.addItem("Manual", "manual")
            else:
                mode.addItem("Auto", "auto")
                mode.addItem("Manual", "manual")
                mode.addItem("Unset", "unset")
            button = QPushButton("Set from video")
            button.clicked.connect(lambda checked=False, name=name: self.set_event_from_video(name))
            time.valueChanged.connect(lambda value, name=name: self.event_edited(name))
            mode.currentIndexChanged.connect(lambda index, name=name: self.mode_changed(name))
            self.event_times[name], self.event_modes[name] = time, mode
            event_grid.addWidget(time, row, 1)
            event_grid.addWidget(mode, row, 2)
            event_grid.addWidget(button, row, 3)
        form.addLayout(event_grid)
        self.facing = QComboBox()
        for label, value in (("Set facing direction…", None), ("Facing right", "right"), ("Facing left", "left")):
            self.facing.addItem(label, value)
        self.facing.currentIndexChanged.connect(self.settings_changed)
        form.addWidget(self.facing)
        form.addWidget(QLabel("Viewpoint: assumed side view · fixed camera · no perspective correction"))
        numeric = QGridLayout()
        self.quality = self.number(.5, .05, 1, .05)
        self.jump = self.number(90, 20, 179, 5)
        self.window = self.number(.15, .03, 1, .01, 3)
        self.max_points = QSpinBox(); self.max_points.setRange(3, 15); self.max_points.setValue(5)
        self.order = QSpinBox(); self.order.setRange(1, 4); self.order.setValue(2)
        for i, (label, widget) in enumerate((("Minimum landmark confidence", self.quality),
                                            ("Orientation jump warning (°/frame)", self.jump),
                                            ("Local polynomial window (s)", self.window),
                                            ("Maximum fit samples", self.max_points),
                                            ("Polynomial order", self.order))):
            numeric.addWidget(QLabel(label), i, 0)
            numeric.addWidget(widget, i, 1)
            widget.valueChanged.connect(self.settings_changed)
        form.addLayout(numeric)
        form.addWidget(QLabel("Optional athlete-height calibration (leave both zero to omit):"))
        calibration = QGridLayout()
        self.height_m = self.number(0, 0, 3, .01)
        self.height_px = self.number(0, 0, 16000, 1, 1)
        self.reference = QLineEdit()
        self.reference.setPlaceholderText("Reference: standing frame/time and how full pixel height was measured")
        for i, (label, widget) in enumerate((("Athlete height (m)", self.height_m), ("Standing full height (source pixels)", self.height_px))):
            calibration.addWidget(QLabel(label), i, 0)
            calibration.addWidget(widget, i, 1)
            widget.valueChanged.connect(self.settings_changed)
        form.addLayout(calibration)
        form.addWidget(self.reference)
        self.reference.textEdited.connect(self.settings_changed)
        form.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(settings)
        self.tabs.addTab(scroll, "Events & settings")
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.tabs.addTab(self.summary, "Summary & quality")
        layout.addWidget(self.tabs, 1)
        actions = QHBoxLayout()
        self.run_button = QPushButton("Analyze / Recalculate")
        self.run_button.clicked.connect(self.recalculate)
        self.export_button = QPushButton("Export analysis CSV + JSON")
        self.export_button.clicked.connect(self.export)
        self.export_button.setEnabled(False)
        actions.addWidget(self.run_button)
        actions.addWidget(self.export_button)
        layout.addLayout(actions)
        self.state = QLabel("No analysis yet")
        self.state.setWordWrap(True)
        layout.addWidget(self.state)
        self.tabs.setCurrentIndex(1)

    @staticmethod
    def number(value, minimum, maximum, step, decimals=2):
        widget = QDoubleSpinBox()
        widget.setDecimals(decimals)
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setValue(value)
        return widget

    def load_clip(self, clip):
        if self.clip is clip:
            return
        self.loading = True
        self.clip, self.result = clip, None
        self.boundaries_set.clear()
        self.stale = False
        times = source_times(clip.frames)
        for name in EVENT_NAMES:
            self.event_times[name].setRange(float(times[0]), float(times[-1]))
            self.event_times[name].setValue(float(times[-1] if name == "landing" else times[0]))
            self.event_modes[name].setCurrentIndex(0)
        self.facing.setCurrentIndex(0)
        self.height_m.setValue(0)
        self.height_px.setValue(0)
        self.reference.clear()
        self.loading = False
        for name in EVENT_NAMES:
            self.mode_changed(name)
        self.export_button.setEnabled(False)
        self.summary.clear()
        self.tabs.setCurrentIndex(1)
        self.state.setText("Set takeoff and landing explicitly, and choose facing direction.")
        self.refresh_plots()

    def event_edited(self, name):
        if not self.loading:
            if name in ("takeoff", "landing"):
                self.boundaries_set.add(name)
            self.settings_changed()

    def mode_changed(self, name):
        self.event_times[name].setEnabled(self.event_modes[name].currentData() == "manual")
        if self.event_modes[name].currentData() == "manual":
            self.event_times[name].setSpecialValueText("")
        self.settings_changed()

    def settings_changed(self, *args):
        if not self.loading and self.result is not None:
            self.stale = True
            self.export_button.setEnabled(False)
            self.state.setText("Settings changed. Displayed plots are the previous result; recalculate before exporting.")

    def set_event_from_video(self, name):
        if self.current_time is None:
            return
        self.event_modes[name].setCurrentIndex(self.event_modes[name].findData("manual"))
        self.event_times[name].setValue(self.current_time)
        self.event_edited(name)

    def set_position(self, time, frame_index):
        self.current_time, self.current_frame = float(time), frame_index
        detail = ""
        if self.result is not None:
            i = int(np.argmin(np.abs(np.asarray(self.result.source_timestamps) - time)))
            flags = self.result.quality_flags[i]
            quality = self.result.series["orientation_quality"][i]
            detail = f" · torso quality {quality:.2f}\n" + (", ".join(flags) if flags else "No quality flags on this frame")
        self.position.setText(f"Video frame {frame_index} · source time {time:.4f} s" + detail)
        for plot in self.plots:
            plot.position = time
            plot.update()

    def parameters(self):
        if self.boundaries_set != {"takeoff", "landing"}:
            raise ValueError("Set both takeoff and landing from the video, or edit both source-time fields.")
        if self.facing.currentData() is None:
            raise ValueError("Choose the athlete's facing direction before takeoff.")
        overrides = {}
        times = source_times(self.clip.frames)
        def event_value(name):
            value = self.event_times[name].value()
            nearest = float(times[np.argmin(np.abs(times - value))])
            # Spinboxes display microseconds; retain exact PTS when selecting a frame.
            return nearest if abs(nearest - value) <= .000001 else value
        for name in EVENT_NAMES:
            if name in ("takeoff", "landing"):
                continue
            mode = self.event_modes[name].currentData()
            if mode != "auto":
                overrides[name] = event_value(name) if mode == "manual" else None
        return AnalysisParameters(
            facing=self.facing.currentData(), takeoff=event_value("takeoff"),
            landing=event_value("landing"), event_overrides=overrides,
            min_confidence=self.quality.value(), max_orientation_step_deg=self.jump.value(),
            smoothing_window_seconds=self.window.value(), smoothing_max_points=self.max_points.value(),
            smoothing_order=self.order.value(), athlete_height_m=self.height_m.value() or None,
            athlete_height_pixels=self.height_px.value() or None, calibration_source=self.reference.text() or None)

    def recalculate(self):
        if self.clip is None:
            return
        try:
            parameters = self.parameters()
            result = analyze(self.clip.frames, parameters,
                             {"source_path": str(self.clip.source.resolve()), "source_size": self.clip.source_size,
                              "source_mtime_ns": self.clip.source_mtime_ns})
        except Exception as error:
            QMessageBox.warning(self, "Analysis needs attention", str(error))
            return
        self.result, self.stale = result, False
        self.loading = True
        for name, event in result.events.items():
            if self.event_modes[name].currentData() == "auto":
                self.event_times[name].setSpecialValueText("Unavailable" if event.source_timestamp is None else "")
                self.event_times[name].setValue(self.event_times[name].minimum() if event.source_timestamp is None else event.source_timestamp)
            self.event_times[name].setToolTip(event.provenance + ": " + event.note)
        self.loading = False
        self.refresh_plots()
        self.update_summary()
        self.export_button.setEnabled(True)
        self.state.setText(f"Analysis {result.analysis_version} · raw poses unchanged · automatic events require review")
        self.tabs.setCurrentIndex(0)
        self.set_position(self.current_time if self.current_time is not None else result.source_timestamps[0], self.current_frame)

    def refresh_plots(self, *args):
        for plot in self.plots:
            plot.result = self.result
            plot.flight_only = self.flight_only.isChecked()
            plot.update()

    def update_summary(self):
        result = self.result
        lines = ["SIDE-VIEW KINEMATICS — HIP PROXY, NOT COM", ""]
        for name, metric in result.summary.items():
            decimals = 0 if metric.unit in ("deg", "deg/s") else 2
            value = "Unavailable" if metric.value is None else f"{metric.value:.{decimals}f} {metric.unit}"
            lines.append(f"{name.replace('_', ' ')}: {value}")
            if metric.note:
                lines.append("  " + metric.note)
        lines += ["", "SMOOTHING (also applies to peak velocity)",
                  f"Actual-time local polynomial: {result.smoothing['window_seconds']:.3f} s window, "
                  f"up to {result.smoothing['max_points']} samples, order {result.smoothing['polynomial_order']}.",
                  "No smoothing across quality gaps or orientation jumps.", "", "EVENTS (source-video seconds)"]
        for name, event in result.events.items():
            value = "Unavailable" if event.source_timestamp is None else f"{event.source_timestamp:.3f} s"
            lines.append(f"{name}: {value} ({event.provenance})")
        lines += ["", "WARNINGS", *["• " + warning for warning in result.warnings], "", "FLAG COUNTS"]
        all_flags = sorted({flag for flags in result.quality_flags for flag in flags})
        lines += [f"{flag}: {sum(flag in flags for flags in result.quality_flags)} frames" for flag in all_flags]
        self.summary.setPlainText("\n".join(lines))

    def export(self):
        if self.result is None or self.stale:
            return
        default = self.clip.source.with_name(self.clip.source.stem + "_analysis.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Export analysis CSV and JSON sidecar", str(default), "CSV (*.csv)")
        if not path:
            return
        csv_path = Path(path).with_suffix(".csv")
        sidecar = csv_path.with_suffix(".json")
        if Path(path) != csv_path and csv_path.exists():
            if QMessageBox.question(self, "Replace CSV?", f"Replace {csv_path}?") != QMessageBox.StandardButton.Yes:
                return
        if sidecar.exists() and QMessageBox.question(self, "Replace sidecar?", f"Also replace {sidecar}?") != QMessageBox.StandardButton.Yes:
            return
        try:
            export_analysis(self.result, csv_path)
        except Exception as error:
            QMessageBox.warning(self, "Export failed", str(error))
            return
        self.state.setText(f"Saved {csv_path.name} and {sidecar.name}")
