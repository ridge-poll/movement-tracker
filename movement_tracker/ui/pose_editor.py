"""Letterbox-aware direct manipulation of existing 2D landmarks."""
from dataclasses import replace
import math
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import QLabel
from movement_tracker.visualization.skeleton_renderer import render

class VideoView(QLabel):
    landmark_moved = Signal(str, float, float)
    landmarks_moved = Signal(object)
    image_clicked = Signal(float, float)

    def __init__(self):
        super().__init__("Open a video or start the live camera")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 320)
        self.setStyleSheet("background: #151a20; color: #c5cdd6;")
        self.setMouseTracking(True)
        self.image = None
        self.pose = None
        self.editing = False
        self.drag = None
        self.selected = None  # Optional exact-point hit-test filter.
        self.selection = set()
        self.group_extremities = True
        self.marquee = None
        self.preview_pose = None
        self.press_position = None
        self.placing = False
        self.pending = {}

    def set_frame(self, rgb, pose, options, editing=False):
        if self.pose is None or self.pose.frame_index != pose.frame_index or not editing or not self.editing:
            self.selection.clear()
        self.selection.intersection_update(pose.landmarks)
        self.marquee = None
        self.pending = {}
        self.rgb, self.pose, self.options, self.editing = rgb, pose, options, editing
        self.drag = None
        self.draw_pose(pose)

    def draw_pose(self, pose):
        self.preview_pose = pose
        options = replace(self.options, points=True, skeleton=True) if self.editing else self.options
        self.set_image(render(self.rgb, pose, options))

    def set_image(self, rgb):
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        self.image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.refresh()

    def refresh(self):
        if self.image is not None:
            self.setPixmap(QPixmap.fromImage(self.image).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                               Qt.TransformationMode.SmoothTransformation))

    def image_geometry(self):
        pix = self.pixmap()
        return ((self.width()-pix.width())/2, (self.height()-pix.height())/2,
                max(1, pix.width()-1), max(1, pix.height()-1))

    def nearest(self, position):
        if not self.editing or self.pose is None or not self.pose.landmarks:
            return None
        ox, oy, w, h = self.image_geometry()
        candidates = []
        for name, p in self.pose.landmarks.items():
            if math.isfinite(p.x) and math.isfinite(p.y) and 0 <= p.x <= 1 and 0 <= p.y <= 1:
                distance = (position.x()-ox-p.x*w)**2 + (position.y()-oy-p.y*h)**2
                # Inspector selection disambiguates coincident left/right landmarks.
                if self.selected is None or name == self.selected:
                    candidates.append((distance, name))
        if candidates:
            distance, name = min(candidates)
            return name if distance <= 14**2 else None

    def cluster(self, name):
        if self.group_extremities and self.selected is None:
            for side in ("left", "right"):
                for joints in (("wrist", "pinky", "index", "thumb"),
                               ("ankle", "heel", "foot_index")):
                    group = {side + "_" + joint for joint in joints}
                    if name in group:
                        return group & self.visible_names()
        return {name}

    def visible_names(self):
        return {name for name, p in self.pose.landmarks.items()
                if math.isfinite(p.x) and math.isfinite(p.y) and 0 <= p.x <= 1 and 0 <= p.y <= 1}

    def inside_image(self, position):
        ox, oy, w, h = self.image_geometry()
        return ox <= position.x() <= ox+w and oy <= position.y() <= oy+h

    def mousePressEvent(self, event):
        if not self.editing or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        if not self.inside_image(event.position()):
            return
        if self.placing:
            ox, oy, w, h = self.image_geometry()
            self.image_clicked.emit((event.position().x()-ox)/w, (event.position().y()-oy)/h)
            event.accept()
            return
        self.press_position = QPointF(event.position())
        self.pending = {}
        name = self.nearest(event.position())
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if name:
            if shift:
                # Shift always toggles one point, even with anatomical grouping enabled.
                self.selection.symmetric_difference_update({name})
            elif name not in self.selection:
                self.selection = self.cluster(name)
            self.drag = name if name in self.selection else None
        else:
            self.drag = None
            self.marquee_base = self.selection.copy() if shift else set()
            self.selection = self.marquee_base.copy()
            self.marquee = QRectF(self.press_position, self.press_position)
        self.update()
        event.accept()

    def update_gesture(self, position):
        if self.drag:
            ox, oy, w, h = self.image_geometry()
            dx = (position.x()-self.press_position.x())/w
            dy = (position.y()-self.press_position.y())/h
            points = [self.pose.landmarks[name] for name in self.selection]
            # Clamp the translation once for the whole group, preserving its shape.
            dx = min(1-max(p.x for p in points), max(-min(p.x for p in points), dx))
            dy = min(1-max(p.y for p in points), max(-min(p.y for p in points), dy))
            self.pending = {name: (self.pose.landmarks[name].x+dx, self.pose.landmarks[name].y+dy)
                            for name in self.selection} if dx or dy else {}
            landmarks = self.pose.landmarks.copy()
            for name, (x, y) in self.pending.items():
                landmarks[name] = replace(landmarks[name], x=x, y=y)
            self.draw_pose(replace(self.pose, landmarks=landmarks,
                                  manual_landmarks=tuple(set(self.pose.manual_landmarks) | self.pending.keys())))
        elif self.marquee is not None:
            self.marquee = QRectF(self.press_position, position).normalized()
            ox, oy, w, h = self.image_geometry()
            enclosed = {name for name in self.visible_names()
                        if self.marquee.contains(QPointF(ox+self.pose.landmarks[name].x*w,
                                                         oy+self.pose.landmarks[name].y*h))}
            self.selection = self.marquee_base | enclosed
            self.update()

    def mouseMoveEvent(self, event):
        if self.drag or self.marquee is not None:
            self.update_gesture(event.position())
        else:
            name = self.nearest(event.position())
            self.setToolTip(name or "")
            self.setCursor(Qt.CursorShape.OpenHandCursor if name else Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and (self.drag or self.marquee is not None):
            self.update_gesture(event.position())
            positions = self.pending.copy()
            self.drag = None
            self.marquee = None
            self.pending = {}
            if positions:
                self.landmarks_moved.emit(positions)
                if len(positions) == 1:
                    name, (x, y) = next(iter(positions.items()))
                    self.landmark_moved.emit(name, x, y)
            self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.editing or self.preview_pose is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#ffffff"), 2))
        ox, oy, w, h = self.image_geometry()
        for name in self.selection:
            p = self.preview_pose.landmarks[name]
            painter.drawEllipse(QPointF(ox+p.x*w, oy+p.y*h), 7, 7)
        if self.marquee is not None:
            painter.setPen(QPen(QColor("#80ceff"), 1, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(80, 170, 255, 35))
            painter.drawRect(self.marquee)
        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh()
