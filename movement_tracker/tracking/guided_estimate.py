"""Image-based second pass: manual anchors guide an upright, padded body crop."""
from dataclasses import replace
import cv2
import numpy as np

def estimate_pose(rgb, raw, anchors, tracker):
    if not {'near_shoulder', 'near_hip'} <= anchors.keys():
        raise ValueError('Place at least the near shoulder and near hip first.')
    h, w = rgb.shape[:2]
    scale = np.array([w-1, h-1])
    shoulder = np.array(anchors['near_shoulder']) * scale
    hip = np.array(anchors['near_hip']) * scale
    torso = np.linalg.norm(shoulder-hip)
    if torso < 5:
        raise ValueError('Shoulder and hip must be distinct points.')
    up = (shoulder-hip)/torso
    right = np.array([-up[1], up[0]])
    rotation = np.array([right, -up])
    pts = np.array([np.array(xy)*scale for xy in anchors.values()]) @ rotation.T
    # Full-body margin around the guide chain, including arms and omitted extremities.
    lo = pts.min(axis=0) - torso*np.array([1.5, 1.0])
    hi = pts.max(axis=0) + torso*np.array([1.5, 2.0])
    factor = 768 / max(hi-lo)
    matrix = np.column_stack((rotation*factor, -lo*factor))
    size = tuple(np.maximum(2, np.ceil((hi-lo)*factor).astype(int)))
    crop = cv2.warpAffine(rgb, matrix, size, borderMode=cv2.BORDER_CONSTANT)
    predicted = tracker.process_frame(crop, 0, raw.frame_index)
    if not predicted.landmarks:
        raise ValueError('No pose found in the guided crop. Your placed points are unchanged.')
    inverse = cv2.invertAffineTransform(matrix)
    landmarks = {}
    for name, p in predicted.landmarks.items():
        xy = inverse @ np.array([p.x*(size[0]-1), p.y*(size[1]-1), 1.])
        if np.isfinite(xy).all():
            # Crop-space relative depth is not a source-image measurement.
            landmarks[name] = replace(p, x=float(xy[0]/(w-1)), y=float(xy[1]/(h-1)), z=None)
    if not landmarks:
        raise ValueError('No usable landmarks returned. Your placed points are unchanged.')
    return landmarks
