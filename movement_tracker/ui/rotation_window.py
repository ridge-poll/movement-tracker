"""Experimental comparison view. Phase 1 results are never edited here."""
from pathlib import Path
import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QGridLayout, QSpinBox, QComboBox, QSlider, QCheckBox, QLineEdit,
    QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox,
)
from movement_tracker.jobs import Job
from movement_tracker.pose_data import LANDMARK_NAMES
from movement_tracker.ui.main_window import VideoView
from movement_tracker.experiments.rotations import ANGLES, run_rotations
from movement_tracker.experiments.report import (
    RATINGS, TAGS, export_report, comparison_image, review_summary,
)
from movement_tracker.visualization.skeleton_renderer import render, OverlayOptions

class RotationWindow(QMainWindow):
    def __init__(self, source, model, start_frame=0, end_frame=60, mode="video", max_frame=None):
        super().__init__()
        self.source = Path(source)
        self.model = Path(model)
        self.result = None
        self.job = None
        self.closing = False
        self.reviews = {}
        self.dirty = False
        self.offset = 0
        self.rgb = None
        self.setWindowTitle("Orientation experiment · " + self.source.name)
        self.resize(1120, 920)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        description = QLabel("Compare raw pose hypotheses · All views use the original image orientation · No correction is applied")
        description.setWordWrap(True)
        layout.addWidget(description)
        setup = QHBoxLayout()
        self.start = QSpinBox()
        self.end = QSpinBox()
        for box, value in ((self.start, start_frame), (self.end, end_frame)):
            box.setRange(0, max_frame if max_frame is not None else 1_000_000)
            box.setValue(value)
        self.mode = QComboBox()
        self.mode.addItem("VIDEO · independent histories from frame 0", "video")
        self.mode.addItem("IMAGE · independent per-frame detections", "image")
        self.mode.setCurrentIndex(0 if mode == "video" else 1)
        self.run_button = QPushButton("Run four orientations")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        for widget in (QLabel("Frames (inclusive):"), self.start, QLabel("to"), self.end,
                       self.mode, self.run_button, self.cancel_button):
            setup.addWidget(widget)
        layout.addLayout(setup)
        self.tabs = QTabWidget()
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        self.views, self.headings, self.ratings = [], [], []
        for i, angle in enumerate(ANGLES):
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(2, 2, 2, 2)
            heading = QLabel(f"Input rotation {angle}° CCW")
            heading.setWordWrap(True)
            panel_layout.addWidget(heading)
            view = VideoView()
            view.setMinimumSize(160, 160)
            view.setText("Awaiting experiment")
            panel_layout.addWidget(view, 1)
            rating = QComboBox()
            for value in RATINGS:
                rating.addItem("Human review: " + value.replace("_", " "), value)
            rating.currentIndexChanged.connect(self.save_review)
            panel_layout.addWidget(rating)
            self.views.append(view)
            self.headings.append(heading)
            self.ratings.append(rating)
            grid.addWidget(panel, i // 2, i % 2)
        self.tabs.addTab(grid_widget, "Four skeletons")
        details = QWidget()
        detail_layout = QVBoxLayout(details)
        self.angle_select = QComboBox()
        for angle in ANGLES:
            self.angle_select.addItem(f"Input rotation {angle}° CCW", angle)
        self.angle_select.currentIndexChanged.connect(self.populate_table)
        detail_layout.addWidget(self.angle_select)
        self.table = QTableWidget(len(LANDMARK_NAMES), 10)
        self.table.setHorizontalHeaderLabels(["Landmark", "Source x", "Source y", "Source z",
                                             "Raw x", "Raw y", "Raw z", "Visibility", "Presence", "Confidence"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        for row, name in enumerate(LANDMARK_NAMES):
            for column in range(10):
                self.table.setItem(row, column, QTableWidgetItem(name if column == 0 else "—"))
        detail_layout.addWidget(self.table)
        detail_layout.addWidget(QLabel("Raw = rotated model-input coordinates. Source = inverse rotation. z is relative, not metric."))
        self.tabs.addTab(details, "Landmark measurements")
        layout.addWidget(self.tabs, 1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.show_frame)
        layout.addWidget(self.slider)
        navigation = QHBoxLayout()
        self.previous = QPushButton("< Frame")
        self.next = QPushButton("Frame >")
        self.labels = QCheckBox("Landmark labels")
        self.labels.toggled.connect(self.redraw)
        self.points = QCheckBox("Points")
        self.points.setChecked(True)
        self.points.toggled.connect(self.redraw)
        self.skeleton = QCheckBox("Skeleton")
        self.skeleton.setChecked(True)
        self.skeleton.toggled.connect(self.redraw)
        self.json_button = QPushButton("Export measurements + review JSON")
        self.png_button = QPushButton("Save frame comparison PNG")
        for widget in (self.previous, self.next, self.skeleton, self.points, self.labels,
                       self.json_button, self.png_button):
            navigation.addWidget(widget)
        layout.addLayout(navigation)
        review_row = QHBoxLayout()
        self.tag = QComboBox()
        for tag in TAGS:
            self.tag.addItem(tag.replace("_", " "), tag)
        self.note = QLineEdit()
        self.note.setPlaceholderText("Optional review notes: identify limb swaps, occlusion, or uncertain anatomy")
        self.tag.currentIndexChanged.connect(self.save_review)
        self.note.textEdited.connect(self.save_review)
        review_row.addWidget(QLabel("Frame category:"))
        review_row.addWidget(self.tag)
        review_row.addWidget(self.note, 1)
        layout.addLayout(review_row)
        self.progress = QProgressBar()
        self.progress.hide()
        layout.addWidget(self.progress)
        self.status = QLabel("Choose a short range spanning correct tracking, failure, and recovery. Confidence is not correctness.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self.cancel)
        self.previous.clicked.connect(lambda: self.step(-1))
        self.next.clicked.connect(lambda: self.step(1))
        self.json_button.clicked.connect(self.save_json)
        self.png_button.clicked.connect(self.save_png)
        self.shortcuts = []
        for key, delta in (("Left", -1), ("Right", 1)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda delta=delta: self.step(delta))
            self.shortcuts.append(shortcut)
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        self.set_ready(False)

    def set_ready(self, ready):
        for widget in (self.slider, self.previous, self.next, self.json_button, self.png_button,
                       self.tag, self.note, *self.ratings):
            widget.setEnabled(ready)

    def options(self):
        return OverlayOptions(self.skeleton.isChecked(), self.points.isChecked(), self.labels.isChecked())

    def run(self):
        if self.job or self.result:
            return
        start, end, mode = self.start.value(), self.end.value(), self.mode.currentData()
        if start > end:
            QMessageBox.warning(self, "Invalid range", "Start frame must be at or before the end frame.")
            return
        self.job = Job(lambda job: run_rotations(self.source, self.model, start, end,
                                                job.cancel, job.update, mode))
        for widget in (self.start, self.end, self.mode, self.run_button):
            widget.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.status.setText("Processing four independent orientation streams; VIDEO mode includes all earlier frames for history…")

    def cancel(self):
        if self.job:
            self.job.cancel.set()
            self.cancel_button.setEnabled(False)
            self.status.setText("Stopping after the current model call…")

    def tick(self):
        if self.job:
            with self.job.lock:
                count, total = self.job.progress
            self.progress.setRange(0, total)
            self.progress.setValue(count)
            self.progress.setFormat(f"{count} / {total} model calls (includes history)")
            if self.job.done.is_set():
                job, self.job = self.job, None
                self.progress.hide()
                self.cancel_button.setEnabled(False)
                if job.error or job.cancelled:
                    self.status.setText("Cancelled" if job.cancelled else "Experiment failed: " + job.error)
                    for widget in (self.start, self.end, self.mode, self.run_button):
                        widget.setEnabled(True)
                else:
                    self.load_result(job.result)
        if self.closing and self.job is None:
            self.close()

    def load_result(self, result):
        self.result = result
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(result.frames) - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.set_ready(True)
        self.show_frame(0)

    def show_frame(self, offset):
        if self.result is None:
            return
        self.offset = offset
        try:
            self.rgb = self.result.preview(offset)
        except Exception as error:
            self.status.setText(str(error))
            return
        self.slider.blockSignals(True)
        self.slider.setValue(offset)
        self.slider.blockSignals(False)
        frame = self.result.frames[offset]
        review = self.reviews.get(str(frame.frame_index), {})
        widgets = [*self.ratings, self.tag, self.note]
        for widget in widgets:
            widget.blockSignals(True)
        for angle, box in zip(ANGLES, self.ratings):
            box.setCurrentIndex(box.findData(review.get("ratings", {}).get(str(angle), "unreviewed")))
        self.tag.setCurrentIndex(self.tag.findData(review.get("category", "unlabeled")))
        self.note.setText(review.get("note", ""))
        for widget in widgets:
            widget.blockSignals(False)
        self.redraw()

    def step(self, delta):
        if self.result:
            self.show_frame(max(0, min(len(self.result.frames) - 1, self.offset + delta)))

    def redraw(self, *args):
        if self.result is None or self.rgb is None:
            return
        frame = self.result.frames[self.offset]
        for view, heading, candidate in zip(self.views, self.headings, frame.candidates):
            pose = candidate.source_pose
            score = f"mean confidence {pose.confidence:.3f}" if pose.confidence is not None else "NO POSE"
            heading.setText(f"Input {candidate.angle_ccw}° CCW · {score}")
            view.set_image(render(self.rgb, pose, self.options()))
        self.populate_table()
        self.update_status()

    def populate_table(self, *args):
        if self.result is None:
            return
        candidate = self.result.frames[self.offset].candidates[self.angle_select.currentIndex()]
        for row, name in enumerate(LANDMARK_NAMES):
            source = candidate.source_pose.landmarks.get(name)
            raw = candidate.input_pose.landmarks.get(name)
            values = ([source.x, source.y, source.z, raw.x, raw.y, raw.z,
                       raw.visibility, raw.presence, raw.confidence] if raw else [None] * 9)
            for column, value in enumerate(values, 1):
                self.table.item(row, column).setText(f"{value:.4f}" if value is not None else "—")

    def save_review(self, *args):
        if self.result is None:
            return
        frame = self.result.frames[self.offset]
        self.reviews[str(frame.frame_index)] = {
            "category": self.tag.currentData(), "note": self.note.text(),
            "ratings": {str(angle): box.currentData() for angle, box in zip(ANGLES, self.ratings)},
        }
        self.dirty = True
        self.update_status()

    def update_status(self):
        if self.result is None:
            return
        frame = self.result.frames[self.offset]
        summary = review_summary(self.result.frames, self.reviews)
        recovered = len(summary["human_reviewed_recovery_frames"])
        self.status.setText(f"Frame {frame.frame_index} · {frame.timestamp:.3f} s · "
                            f"{summary['frames_with_any_human_rating']}/{len(self.result.frames)} frames reviewed · "
                            f"{recovered} manually identified rotation recoveries · "
                            + ("Unsaved review" if self.dirty else "Raw measurements preserved; export JSON to keep results"))

    def destination(self, title, suffix):
        default = self.source.with_name(self.source.stem + f"_rotations_{self.start.value()}_{self.end.value()}" + suffix)
        if suffix == ".png":
            frame_index = self.result.frames[self.offset].frame_index
            default = default.with_name(default.stem + f"_frame{frame_index}" + suffix)
        path, _ = QFileDialog.getSaveFileName(self, title, str(default), f"{suffix[1:].upper()} (*{suffix})")
        if not path:
            return None
        target = Path(path)
        if target.suffix.lower() != suffix:
            target = target.with_suffix(suffix)
            if target.exists() and QMessageBox.question(self, "Replace file?", f"Replace {target}?") != QMessageBox.StandardButton.Yes:
                return None
        if target.resolve() == self.source.resolve():
            QMessageBox.warning(self, "Choose another file", "The source video cannot be overwritten.")
            return None
        return target

    def save_json(self):
        target = self.destination("Save measurements and human review", ".json")
        if target is None:
            return False
        try:
            export_report(self.result, self.reviews, target)
        except Exception as error:
            QMessageBox.warning(self, "Export failed", str(error))
            return False
        self.dirty = False
        self.update_status()
        return True

    def save_png(self):
        target = self.destination("Save this frame's four skeletons", ".png")
        if target is None:
            return
        try:
            rgb = comparison_image(self.rgb, self.result.frames[self.offset], self.options(), self.reviews)
            if not cv2.imwrite(str(target), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError("Unable to write image")
        except Exception as error:
            QMessageBox.warning(self, "Export failed", str(error))

    def closeEvent(self, event):
        if self.job:
            self.closing = True
            self.cancel()
            event.ignore()
            return
        if self.dirty:
            answer = QMessageBox.question(self, "Save review?", "Export your review labels before closing?",
                                          QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                          QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Cancel or (answer == QMessageBox.StandardButton.Save and not self.save_json()):
                self.closing = False
                event.ignore()
                return
        if self.result:
            self.result.close()
            self.result = None
        self.timer.stop()
        event.accept()
