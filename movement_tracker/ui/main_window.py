from bisect import bisect_right
from pathlib import Path
from time import perf_counter
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QCheckBox, QSlider, QFileDialog, QMessageBox, QProgressBar, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QComboBox, QDockWidget,
)
from movement_tracker.jobs import Job
from movement_tracker.pose_data import LANDMARK_NAMES
from movement_tracker.tracking.mediapipe_tracker import MediaPipeTracker
from movement_tracker.video.processing import process_video
from movement_tracker.video.camera import run_camera
from movement_tracker.visualization.skeleton_renderer import OverlayOptions
from movement_tracker.export.csv_export import export_csv
from movement_tracker.export.video_export import export_video

from .pose_editor import VideoView
from movement_tracker.pose_corrections import corrections_for, ROUGH_NAMES
from movement_tracker.pose_data import Landmark
from dataclasses import replace

class MainWindow(QMainWindow):
    def __init__(self, model_path, camera_index=0):
        super().__init__()
        self.model_path = Path(model_path)
        self.rough_draft = None
        self.clip = None
        self.job = None
        self.job_kind = None
        self.closing = False
        self.playing = False
        self.index = 0
        self.times = []
        self.current = None
        self.analysis_panel = None
        self.analysis_dock = None
        self.setWindowTitle("Movement Tracker · Side-view analysis")
        self.resize(1100, 850)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        sources = QHBoxLayout()
        self.open_button = QPushButton("Open Video")
        self.live_button = QPushButton("Live Camera")
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 15)
        self.camera_index.setValue(camera_index)
        self.camera_index.setToolTip("OpenCV camera index. Built-in camera is often 0; try another if an external camera appears.")
        self.stop_button = QPushButton("Stop / Cancel")
        self.analysis_button = QPushButton("Analyze flip…")
        for widget in (self.open_button, self.live_button, QLabel("Camera index:"),
                       self.camera_index, self.stop_button, self.analysis_button):
            sources.addWidget(widget)
        sources.addStretch()
        layout.addLayout(sources)
        splitter = QSplitter()
        self.view = VideoView()
        splitter.addWidget(self.view)
        inspector = QWidget()
        inspection_layout = QVBoxLayout(inspector)
        inspection_layout.addWidget(QLabel("Landmark confidence"))
        self.table = QTableWidget(len(LANDMARK_NAMES), 3)
        self.table.setHorizontalHeaderLabels(["Landmark", "Visibility", "Presence"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        for row, name in enumerate(LANDMARK_NAMES):
            for column, text in enumerate((name, "—", "—")):
                self.table.setItem(row, column, QTableWidgetItem(text))
        inspection_layout.addWidget(self.table)
        note = QLabel("Green: confidence ≥ 0.5\nAmber: low confidence\nPurple: manually placed XY\nConfidence = min(visibility, presence).\nHigh confidence does not guarantee accuracy.")
        note.setWordWrap(True)
        inspection_layout.addWidget(note)
        splitter.addWidget(inspector)
        splitter.setSizes([740, 360])
        layout.addWidget(splitter, 1)
        toggles = QHBoxLayout()
        self.skeleton = QCheckBox("Skeleton")
        self.points = QCheckBox("Landmark points")
        self.labels = QCheckBox("Landmark labels")
        for box in (self.skeleton, self.points, self.labels):
            box.setChecked(box is not self.labels)
            box.toggled.connect(self.redraw)
            toggles.addWidget(box)
        toggles.addStretch()
        layout.addLayout(toggles)
        toggles = QHBoxLayout()
        self.edit_button = QPushButton("Edit Pose")
        self.edit_button.setCheckable(True)
        self.undo_button = QPushButton("Undo")
        self.reset_button = QPushButton("Reset Frame")
        self.recalculate_button = QPushButton("Estimate Pose")
        self.landmark_choice = QComboBox()
        self.landmark_choice.addItem("Grab nearest landmark", None)
        for name in LANDMARK_NAMES:
            self.landmark_choice.addItem(name, name)
        self.landmark_choice.setToolTip("Choose a landmark to disambiguate overlapping points, then drag it")
        self.landmark_choice.currentIndexChanged.connect(self.select_exact_landmark)
        self.edit_button.toggled.connect(self.edit_toggled)
        self.undo_button.clicked.connect(lambda: self.change_correction("undo"))
        self.reset_button.clicked.connect(lambda: self.change_correction("reset"))
        self.recalculate_button.clicked.connect(self.estimate_pose)
        self.view.landmarks_moved.connect(self.move_landmarks)
        for widget in (self.edit_button, self.undo_button, self.reset_button, self.recalculate_button, self.landmark_choice):
            toggles.addWidget(widget)
        toggles.addStretch()
        layout.addLayout(toggles)
        selection_tools = QHBoxLayout()
        self.group_extremities = QCheckBox("Group hands / feet")
        self.group_extremities.setChecked(True)
        self.group_extremities.toggled.connect(self.set_grouping)
        selection_tools.addWidget(self.group_extremities)
        selection_tools.addWidget(QLabel("White rings: selected · Shift-click: toggle point · Drag empty image: select area"))
        selection_tools.addStretch()
        layout.addLayout(selection_tools)
        rough_tools = QHBoxLayout()
        self.clear_pose_button = QPushButton("Clear Pose")
        self.rough_pose_button = QPushButton("Place Rough Pose")
        self.rough_back = QPushButton("Back")
        self.rough_skip = QPushButton("Skip")
        self.rough_cancel = QPushButton("Cancel Placement")
        self.rough_prompt = QLabel("")
        self.clear_pose_button.clicked.connect(lambda: self.change_correction("clear"))
        self.rough_pose_button.clicked.connect(self.start_rough)
        self.rough_back.clicked.connect(self.back_rough)
        self.rough_skip.clicked.connect(lambda: self.advance_rough(None))
        self.rough_cancel.clicked.connect(self.cancel_rough)
        self.view.image_clicked.connect(lambda x, y: self.advance_rough((x, y)))
        for widget in (self.clear_pose_button, self.rough_pose_button, self.rough_back,
                       self.rough_skip, self.rough_cancel, self.rough_prompt):
            rough_tools.addWidget(widget)
        layout.addLayout(rough_tools)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.seek)
        layout.addWidget(self.slider)
        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.back_button = QPushButton("< Frame")
        self.next_button = QPushButton("Frame >")
        self.speed = QComboBox()
        for label, value in (("0.25×", .25), ("0.5×", .5), ("1×", 1), ("2×", 2)):
            self.speed.addItem(label, value)
        self.speed.setCurrentIndex(2)
        self.csv_button = QPushButton("Export Raw CSV")
        self.video_button = QPushButton("Export Overlay MP4")
        for widget in (self.play_button, self.pause_button, self.back_button,
                       self.next_button, self.speed, self.csv_button, self.video_button):
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.metrics = QLabel("No video loaded")
        self.metrics.setWordWrap(True)
        layout.addWidget(self.metrics)
        self.progress = QProgressBar()
        self.progress.hide()
        layout.addWidget(self.progress)
        self.status = QLabel(f"Ready · {self.model_path.stem} · Video exports are silent")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.open_button.clicked.connect(self.open_video)
        self.live_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.cancel)
        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.back_button.clicked.connect(lambda: self.step(-1))
        self.next_button.clicked.connect(lambda: self.step(1))
        self.speed.currentIndexChanged.connect(self.pause)
        self.csv_button.clicked.connect(self.save_csv)
        self.video_button.clicked.connect(self.save_video)
        self.analysis_button.clicked.connect(self.open_analysis)
        self.shortcuts = []
        for key, callback in (("Space", self.toggle_play), ("Left", lambda: self.step(-1)), ("Right", lambda: self.step(1))):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)
        self.timer = QTimer(self)
        self.timer.setInterval(15)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        self.update_controls()

    def options(self):
        return OverlayOptions(self.skeleton.isChecked(), self.points.isChecked(), self.labels.isChecked())

    def update_controls(self):
        idle = self.job is None and not self.closing
        placing = self.rough_draft is not None
        ready = idle and self.clip is not None and not placing
        self.open_button.setEnabled(idle and not placing)
        self.live_button.setEnabled(idle and not placing)
        self.camera_index.setEnabled(idle)
        self.stop_button.setEnabled(self.job is not None and not self.job.cancel.is_set())
        for widget in (self.play_button, self.pause_button, self.back_button, self.next_button,
                       self.slider, self.speed, self.csv_button, self.video_button, self.analysis_button):
            widget.setEnabled(ready)
        for widget in (self.edit_button, self.recalculate_button, self.landmark_choice):
            widget.setEnabled(ready)
        self.clear_pose_button.setEnabled(ready)
        self.rough_pose_button.setEnabled(ready)
        for widget in (self.rough_back, self.rough_skip, self.rough_cancel, self.rough_prompt):
            widget.setVisible(placing)
        self.rough_back.setEnabled(placing and self.rough_step > 0)
        store = corrections_for(self.clip) if self.clip else None
        frame = self.clip.frames[self.index].frame_index if self.clip else 0
        self.undo_button.setEnabled(bool(ready and store.history.get(frame)))
        self.reset_button.setEnabled(bool(ready and store.overrides.get(frame)))
        if self.analysis_panel is not None:
            self.analysis_panel.setEnabled(ready)

    def start_rough(self):
        if self.clip is None or self.job:
            return
        self.pause()
        self.edit_button.setChecked(True)
        self.rough_draft, self.rough_step = {}, 0
        self.view.selection.clear()
        self.refresh_rough()

    def refresh_rough(self):
        self.rough_prompt.setText(f"{self.rough_step+1}/5: click " +
                                  ("head", "near shoulder", "near hip", "near knee", "near ankle")[self.rough_step])
        rgb, raw = self.current
        landmarks = {name: Landmark(*xy) for name, xy in self.rough_draft.items()}
        self.view.set_frame(rgb, replace(raw, landmarks=landmarks, manual_landmarks=tuple(landmarks),
                                        manual_pose_mode="rough_single_side"), self.options(), True)
        self.view.placing = True
        self.update_controls()

    def advance_rough(self, xy):
        if self.rough_draft is None:
            return
        if xy is not None:
            self.rough_draft[ROUGH_NAMES[self.rough_step]] = xy
        self.rough_step += 1
        if self.rough_step == len(ROUGH_NAMES):
            draft = self.rough_draft
            self.rough_draft = None
            self.view.placing = False
            if draft:
                corrections_for(self.clip).rough(self.clip.frames[self.index], draft)
                self.correction_changed()
            else:
                self.redraw()
        else:
            self.refresh_rough()

    def back_rough(self):
        if self.rough_draft is not None and self.rough_step:
            self.rough_step -= 1
            self.rough_draft.pop(ROUGH_NAMES[self.rough_step], None)
            self.refresh_rough()

    def cancel_rough(self):
        self.rough_draft = None
        self.view.placing = False
        self.redraw()

    def edit_toggled(self, enabled):
        self.pause()
        self.redraw()
        if enabled:
            self.status.setText("Select / drag landmarks; white rings show selection, purple marks manual XY. Confidence stays raw; manual points use analysis weight 1. Recalculate when done.")

    def select_exact_landmark(self):
        self.view.selected = self.landmark_choice.currentData()
        self.view.selection.clear()
        self.view.update()

    def set_grouping(self, enabled):
        self.view.group_extremities = enabled
        self.view.selection.clear()
        self.view.update()

    def move_landmarks(self, positions):
        if self.clip is not None and self.job is None:
            if corrections_for(self.clip).move_many(self.clip.frames[self.index], positions):
                self.correction_changed()

    def move_landmark(self, name, x, y):
        if self.clip is not None and self.job is None:
            if corrections_for(self.clip).move(self.clip.frames[self.index], name, x, y):
                self.correction_changed()

    def change_correction(self, operation):
        if self.clip is not None and self.job is None:
            if getattr(corrections_for(self.clip), operation)(self.clip.frames[self.index].frame_index):
                self.correction_changed()

    def correction_changed(self):
        if self.analysis_panel is not None:
            self.analysis_panel.settings_changed()
        self.redraw()
        self.update_controls()
        self.status.setText("Pose correction changed · Recalculate to refresh measurements, plots and events")

    def estimate_pose(self):
        if self.clip is None or self.job or self.rough_draft is not None:
            return
        raw = self.clip.frames[self.index]
        store = corrections_for(self.clip)
        effective = store.apply(raw)
        anchors = {name: (p.x, p.y) for name, p in effective.landmarks.items() if name in ROUGH_NAMES}
        if not anchors:
            anchors = store.overrides.get(raw.frame_index, {}).get("__anchors__", {})
        if not {"near_shoulder", "near_hip"} <= anchors.keys():
            self.status.setText("Place Rough Pose first, including shoulder and hip, then click Estimate Pose.")
            self.start_rough()
            return
        from movement_tracker.tracking.guided_estimate import estimate_pose
        rgb = self.current[0].copy()
        def work(job):
            tracker = self.tracker()
            try:
                landmarks = estimate_pose(rgb, raw, anchors, tracker)
                if job.cancel.is_set():
                    from movement_tracker.video.processing import Cancelled
                    raise Cancelled()
                return raw, anchors, landmarks
            finally:
                tracker.close()
        self.begin_job("estimate", work)

    def recalculate(self):
        self.open_analysis()
        if self.analysis_panel is not None and self.job is None:
            self.analysis_panel.recalculate()

    def tracker(self):
        return MediaPipeTracker(self.model_path)

    def open_analysis(self):
        if self.clip is None or self.job:
            return
        from movement_tracker.ui.analysis_panel import AnalysisPanel
        self.pause()
        if self.analysis_panel is None:
            self.analysis_panel = AnalysisPanel(self)
            self.analysis_panel.seek_time.connect(self.seek_source_time)
            self.analysis_dock = QDockWidget("Side-view measurements", self)
            self.analysis_dock.setWidget(self.analysis_panel)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analysis_dock)
        self.analysis_panel.load_clip(self.clip)
        self.analysis_dock.show()
        self.analysis_dock.raise_()
        self.show_frame(self.index)

    def seek_source_time(self, timestamp):
        if self.clip is not None and self.job is None:
            self.pause()
            index = min(range(len(self.clip.frames)), key=lambda i: abs(
                (self.clip.frames[i].source_timestamp if self.clip.frames[i].source_timestamp is not None
                 else self.clip.frames[i].timestamp) - timestamp))
            self.show_frame(index)

    def check_model(self):
        if self.model_path.is_file():
            return True
        QMessageBox.warning(self, "Model missing", f"Model not found:\n{self.model_path}\n\nRun python -m movement_tracker.models first, or launch with --model PATH.")
        return False

    def begin_job(self, kind, function):
        self.pause()
        self.edit_button.setChecked(False)
        self.job_kind = kind
        self.job = Job(function)
        self.progress.setRange(0, 0)
        self.progress.setVisible(kind != "camera")
        self.status.setText({"process": "Analyzing every frame…", "camera": "Opening camera…",
                             "video": "Exporting full-resolution silent MP4…", "csv": "Exporting landmarks…", "estimate": "Estimating pose from guided image crop…"}[kind])
        self.update_controls()

    def open_video(self):
        if not self.check_model():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Videos (*.mp4 *.mov *.avi *.m4v *.mkv);;All files (*)")
        if path:
            def process(job):
                tracker = self.tracker()
                try:
                    return process_video(Path(path), tracker, job.cancel, job.update)
                finally:
                    tracker.close()
            self.begin_job("process", process)

    def start_camera(self):
        if self.check_model():
            index = self.camera_index.value()
            self.begin_job("camera", lambda job: run_camera(job, index, self.tracker))

    def cancel(self):
        if self.job:
            self.job.cancel.set()
            self.status.setText("Stopping; finishing the current operation…")
            self.update_controls()

    def pause(self, *args):
        self.playing = False

    def toggle_play(self):
        self.pause() if self.playing else self.play()

    def play(self):
        if self.clip is None or self.job or self.rough_draft is not None:
            return
        if self.index >= len(self.clip.frames) - 1:
            self.show_frame(0)
        self.edit_button.setChecked(False)
        self.play_start = perf_counter()
        self.play_origin = self.times[self.index]
        self.playing = True

    def step(self, amount):
        if self.clip and self.job is None and self.rough_draft is None:
            self.pause()
            self.show_frame(max(0, min(len(self.clip.frames) - 1, self.index + amount)))

    def seek(self, index):
        if self.clip and self.job is None and self.rough_draft is None:
            self.pause()
            self.show_frame(index)

    def show_frame(self, index):
        self.index = index
        try:
            self.current = (self.clip.preview(index), self.clip.frames[index])
        except Exception as error:
            self.pause()
            self.status.setText(str(error))
            return
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self.redraw()
        if self.analysis_panel is not None:
            pose = self.clip.frames[index]
            self.analysis_panel.set_position(pose.source_timestamp if pose.source_timestamp is not None else pose.timestamp,
                                             pose.frame_index)

    def redraw(self, *args):
        if self.current is None:
            return
        if self.rough_draft is not None:
            self.refresh_rough()
            return
        rgb, pose = self.current
        if self.clip is not None and self.job_kind != "camera":
            pose = corrections_for(self.clip).apply(pose)
        self.view.set_frame(rgb, pose, self.options(), self.edit_button.isChecked() and self.job is None)
        self.update_controls()
        for row, name in enumerate(LANDMARK_NAMES):
            self.table.item(row, 0).setText(name + (" [manual]" if name in pose.manual_landmarks else ""))
            point = pose.landmarks.get(name)
            for column, field in ((1, "visibility"), (2, "presence")):
                value = getattr(point, field) if point else None
                self.table.item(row, column).setText(f"{value:.3f}" if value is not None else "—")
        fps = 1 / pose.inference_seconds if pose.inference_seconds else 0
        confidence = f"{pose.confidence:.3f}" if pose.confidence is not None else "N/A"
        source = "Live" if self.job_kind == "camera" and self.job else f"Source FPS: {self.clip.fps:.3f}" if self.clip else ""
        self.metrics.setText(f"Frame: {pose.frame_index}  |  Time: {pose.timestamp:.3f} s  |  {pose.width} × {pose.height}  |  {source}  |  Inference FPS: {fps:.1f}  |  Mean confidence: {confidence}")

    def destination(self, title, suffix, filter_text):
        path, _ = QFileDialog.getSaveFileName(self, title, str(self.clip.source.with_name(self.clip.source.stem + "_tracked" + suffix)), filter_text)
        if not path:
            return None
        target = Path(path)
        if target.suffix.lower() != suffix:
            target = target.with_suffix(suffix)
            if target.exists() and QMessageBox.question(self, "Replace file?", f"Replace {target}?") != QMessageBox.StandardButton.Yes:
                return None
        if target.resolve() == self.clip.source.resolve():
            QMessageBox.warning(self, "Choose another file", "The source video cannot be overwritten.")
            return None
        return target

    def save_csv(self):
        self.pause()
        target = self.destination("Export landmarks", ".csv", "CSV (*.csv)")
        if target:
            self.begin_job("csv", lambda job: export_csv(self.clip.frames, target))

    def save_video(self):
        self.pause()
        target = self.destination("Export overlay video (silent)", ".mp4", "MP4 (*.mp4)")
        if target:
            options = self.options()
            corrected_clip = replace(self.clip, frames=[corrections_for(self.clip).apply(f) for f in self.clip.frames])
            self.begin_job("video", lambda job: export_video(corrected_clip, target, options, job.cancel, job.update))

    def tick(self):
        if self.job:
            latest = self.job.take_latest()
            if latest is not None:
                self.current = latest
                self.redraw()
                self.status.setText("Live camera · Latest frames only · Stop to return to video")
            with self.job.lock:
                count, total = self.job.progress
            if self.job_kind != "camera":
                self.progress.setRange(0, max(total, count) if total else 0)
                self.progress.setValue(count)
                self.progress.setFormat(f"{count} / {total} frames" if total else f"{count} frames")
            if self.job.done.is_set():
                job, kind = self.job, self.job_kind
                self.job = None
                self.job_kind = None
                self.progress.hide()
                if job.error:
                    self.status.setText("Operation failed")
                    if not self.closing:
                        QMessageBox.warning(self, "Operation failed", job.error)
                elif job.cancelled:
                    self.status.setText("Cancelled")
                elif kind == "estimate":
                    raw, anchors, landmarks = job.result
                    corrections_for(self.clip).estimate(raw, landmarks, anchors)
                    self.correction_changed()
                    self.status.setText("Second image estimate ready · Review the pose; Undo restores your guide · Recalculate analysis in the plot panel")
                elif kind == "process":
                    if self.clip:
                        self.clip.close()
                    self.clip = job.result
                    if self.analysis_panel is not None:
                        self.analysis_panel.load_clip(self.clip)
                    self.times = [pose.timestamp for pose in self.clip.frames]
                    self.index = 0
                    self.slider.blockSignals(True)
                    self.slider.setRange(0, len(self.times) - 1)
                    self.slider.setValue(0)
                    self.slider.blockSignals(False)
                    detected = sum(bool(p.landmarks) for p in self.clip.frames)
                    estimates = sum(p.timestamp_estimated for p in self.clip.frames)
                    self.status.setText(f"Ready · {len(self.times)} frames · Pose found in {detected} · Estimated timestamps: {estimates} · {self.clip.source.name}")
                else:
                    self.status.setText("Camera stopped" if kind == "camera" else "Export complete")
                if self.clip:
                    self.show_frame(self.index)
                self.update_controls()
        if self.closing and self.job is None:
            self.close()
            return
        if self.playing and self.clip:
            timestamp = self.play_origin + (perf_counter() - self.play_start) * self.speed.currentData()
            index = min(len(self.times) - 1, bisect_right(self.times, timestamp) - 1)
            if index != self.index:
                self.show_frame(index)
            if timestamp >= self.times[-1] + self.clip.last_duration:
                self.pause()

    def closeEvent(self, event):
        self.pause()
        if self.job:
            self.closing = True
            self.cancel()
            event.ignore()
            return
        if self.clip:
            self.clip.close()
            self.clip = None
        event.accept()
