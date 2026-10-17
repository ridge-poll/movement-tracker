"""Small native Qt plots: no web runtime or additional plotting dependency."""
import math
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QPainterPath
from PySide6.QtWidgets import QWidget

class AnalysisPlot(QWidget):
    time_clicked = Signal(float)

    def __init__(self, title, unit, curves, parent=None):
        super().__init__(parent)
        self.title, self.unit, self.curves = title, unit, curves
        self.result = None
        self.position = None
        self.flight_only = True
        self.setMinimumHeight(125)
        self.setMinimumWidth(360)
        self.setMouseTracking(True)

    def bounds(self):
        if self.result is None:
            return (0, 1, 0, 1)
        times = self.result.source_timestamps
        if self.flight_only:
            xmin = max(times[0], self.result.parameters["takeoff"] - .1)
            xmax = min(times[-1], self.result.parameters["landing"] + .1)
        else:
            xmin, xmax = times[0], times[-1]
        values = [v for key, _, _ in self.curves for t, v in zip(times, self.result.series[key])
                  if xmin <= t <= xmax and v is not None and math.isfinite(v)]
        if not values:
            return xmin, xmax, 0, 1
        low, high = min(values), max(values)
        pad = max((high - low) * .1, .05)
        return xmin, xmax, low - pad, high + pad

    def plot_rect(self):
        return QRectF(58, 39, max(1, self.width() - 72), max(1, self.height() - 64))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#17212c"))
        painter.setPen(QColor("#eff4f8"))
        painter.drawText(8, 16, f"{self.title} [{self.unit}]")
        legend = " · ".join(label for _, label, _ in self.curves)
        painter.setPen(QColor("#b7c5d4"))
        painter.drawText(8, 32, legend)
        if self.result is None:
            painter.drawText(65, 78, "Select flight events, then Analyze / Recalculate")
            return
        xmin, xmax, ymin, ymax = self.bounds()
        rect = self.plot_rect()
        def x(t): return rect.left() + (t - xmin) / max(1e-12, xmax - xmin) * rect.width()
        def y(v): return rect.bottom() - (v - ymin) / (ymax - ymin) * rect.height()
        painter.setPen(QColor("#405264"))
        for fraction in (0, .5, 1):
            value = ymin + fraction * (ymax - ymin)
            painter.drawLine(rect.left(), y(value), rect.right(), y(value))
            painter.setPen(QColor("#b7c5d4"))
            painter.drawText(3, int(y(value) + 4), f"{value:.1f}")
            painter.setPen(QColor("#405264"))
        painter.setPen(QColor("#b7c5d4"))
        painter.drawText(int(rect.left()), self.height() - 6, f"{xmin:.2f} s")
        painter.drawText(int(rect.right()) - 65, self.height() - 6, f"{xmax:.2f} s")
        times = self.result.source_timestamps
        painter.save()
        painter.setClipRect(rect)
        severe = {"no_pose", "low_torso_quality", "orientation_jump", "rotation_continuity_lost",
                  "large_ballistic_residual", "timestamp_gap", "low_hip_quality"}
        for i, (time, flags) in enumerate(zip(times, self.result.quality_flags)):
            if severe.intersection(flags):
                left = (times[i - 1] + time) / 2 if i else time
                right = (times[i + 1] + time) / 2 if i + 1 < len(times) else time
                painter.fillRect(QRectF(x(left), rect.top(), max(2, x(right) - x(left)), rect.height()), QColor(220, 90, 60, 45))
        for key, _, color in self.curves:
            path = QPainterPath()
            drawing = False
            previous_segment = None
            for i, (time, value) in enumerate(zip(times, self.result.series[key])):
                segment = self.result.series["orientation_segment"][i] if "velocity" in key else 0
                if value is None or not math.isfinite(value):
                    drawing = False
                    continue
                if not drawing or segment != previous_segment:
                    path.moveTo(x(time), y(value))
                else:
                    path.lineTo(x(time), y(value))
                previous_segment, drawing = segment, True
            painter.setPen(QPen(QColor(color), 1.6))
            painter.drawPath(path)
        for i, (name, marker) in enumerate(self.result.events.items()):
            if marker.source_timestamp is None:
                continue
            xx = x(marker.source_timestamp)
            painter.setPen(QPen(QColor("#dfc16c"), 1, Qt.PenStyle.DashLine))
            painter.drawLine(xx, rect.top(), xx, rect.bottom())
            painter.drawText(int(min(xx + 3, rect.right() - 23)), int(rect.top() + 12 + (i % 3) * 11),
                             {"takeoff":"TO", "landing":"L", "apex":"A", "tuck_start":"T", "maximum_tuck":"MT", "opening":"O"}[name])
        if self.position is not None:
            xx = x(self.position)
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawLine(xx, rect.top(), xx, rect.bottom())
        painter.restore()

    def mousePressEvent(self, event):
        if self.result is not None:
            rect = self.plot_rect()
            if rect.contains(event.position()):
                xmin, xmax, _, _ = self.bounds()
                time = xmin + (event.position().x() - rect.left()) / rect.width() * (xmax - xmin)
                self.time_clicked.emit(time)

    def mouseMoveEvent(self, event):
        if self.result is None:
            return
        rect = self.plot_rect()
        if rect.contains(event.position()):
            xmin, xmax, _, _ = self.bounds()
            time = xmin + (event.position().x() - rect.left()) / rect.width() * (xmax - xmin)
            index = min(range(len(self.result.source_timestamps)), key=lambda i: abs(self.result.source_timestamps[i] - time))
            flags = self.result.quality_flags[index]
            self.setToolTip(f"Frame {self.result.frame_indices[index]} · {self.result.source_timestamps[index]:.4f} s\n"
                            + (", ".join(flags) if flags else "No quality flags"))
