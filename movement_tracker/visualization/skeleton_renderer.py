from dataclasses import dataclass
import math
import cv2
from movement_tracker.pose_data import CONNECTIONS

@dataclass(frozen=True)
class OverlayOptions:
    skeleton: bool = True
    points: bool = True
    labels: bool = False

def render(rgb, pose, options=OverlayOptions()):
    result = rgb.copy()
    h, w = result.shape[:2]
    thickness = max(1, round(max(w, h) / 600))
    points = {}
    for name, p in pose.landmarks.items():
        if math.isfinite(p.x) and math.isfinite(p.y) and 0 <= p.x <= 1 and 0 <= p.y <= 1:
            points[name] = (round(p.x * (w - 1)), round(p.y * (h - 1)))
    def color(name):
        if name in pose.manual_landmarks:
            return (235, 100, 255)
        score = pose.landmarks[name].confidence
        return (255, 172, 65) if score is not None and score < 0.5 else (62, 230, 180)
    if options.skeleton:
        for a, b in CONNECTIONS:
            if a in points and b in points:
                c = (255, 172, 65) if color(a) == (255, 172, 65) or color(b) == (255, 172, 65) else color(a)
                cv2.line(result, points[a], points[b], c, thickness, cv2.LINE_AA)
    for name, point in points.items():
        if options.points:
            cv2.circle(result, point, thickness + 2, color(name), -1, cv2.LINE_AA)
        if options.labels:
            score = pose.landmarks[name].confidence
            label = f"{name} {score:.2f}" if score is not None else name
            cv2.putText(result, label, point, cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.3, max(w, h) / 2400), color(name), 1, cv2.LINE_AA)
    return result
