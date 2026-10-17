"""Image-plane rotations; no inference, candidate selection, or correction."""
from dataclasses import replace
import numpy as np
from movement_tracker.pose_data import PoseFrame

def rotate_image(rgb, angle_ccw):
    if angle_ccw not in (0, 90, 180, 270):
        raise ValueError("Expected 0, 90, 180, or 270 degrees counterclockwise")
    return np.ascontiguousarray(np.rot90(rgb, angle_ccw // 90))

def pose_to_source(pose: PoseFrame, angle_ccw: int, width: int, height: int) -> PoseFrame:
    if angle_ccw not in (0, 90, 180, 270) or width <= 0 or height <= 0:
        raise ValueError("Invalid rotation or image dimensions")
    expected = (height, width) if angle_ccw in (90, 270) else (width, height)
    if (pose.width, pose.height) != expected:
        raise ValueError("Pose dimensions do not match rotated image dimensions")
    points = {}
    for name, p in pose.landmarks.items():
        x, y = p.x, p.y
        if angle_ccw == 90:
            x, y = 1 - y, x
        elif angle_ccw == 180:
            x, y = 1 - x, 1 - y
        elif angle_ccw == 270:
            x, y = y, 1 - x
        points[name] = replace(p, x=x, y=y, z=None if p.z is None else p.z * pose.width / width)
    return replace(pose, width=width, height=height, landmarks=points)
