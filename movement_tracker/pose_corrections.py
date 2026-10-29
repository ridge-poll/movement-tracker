"""Session-local XY overrides. Raw landmarks and detector confidence never change."""
from dataclasses import replace, asdict
import math
from movement_tracker.pose_data import Landmark

ROUGH_NAMES = ("head", "near_shoulder", "near_hip", "near_knee", "near_ankle")

class PoseCorrections:
    def __init__(self):
        self.overrides = {}
        self.history = {}
        self.revision = 0

    def _save(self, index, values):
        old = self.overrides.get(index, {})
        if old == values:
            return False
        self.history.setdefault(index, []).append(old.copy())
        self.overrides[index] = values
        self.revision += 1
        return True

    def move(self, frame, name, x, y):
        return self.move_many(frame, {name: (x, y)})

    def move_many(self, frame, positions):
        """Validate and commit an entire gesture atomically as one undo entry."""
        values = self.overrides.get(frame.frame_index, {}).copy()
        effective = self.apply(frame)
        for name, (x, y) in positions.items():
            if name not in effective.landmarks:
                raise ValueError("Only existing landmarks can be moved")
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y)):
                raise ValueError("Coordinates must lie inside the image")
            raw = frame.landmarks.get(name)
            if "__mode__" not in values and raw is not None and (x, y) == (raw.x, raw.y):
                values.pop(name, None)
            else:
                values[name] = (float(x), float(y))
        return self._save(frame.frame_index, values)

    def estimate(self, frame, landmarks, anchors):
        return self._save(frame.frame_index, {"__mode__": "guided_image_estimate",
                          "__estimate__": landmarks.copy(), "__anchors__": dict(anchors)})

    def clear(self, index):
        return self._save(index, {"__mode__": "cleared"})

    def rough(self, frame, positions):
        if not positions or not set(positions) <= set(ROUGH_NAMES):
            raise ValueError("Place at least one rough-pose point")
        if any(not all(math.isfinite(v) and 0 <= v <= 1 for v in xy) or len(xy) != 2
               for xy in positions.values()):
            raise ValueError("Coordinates must lie inside the image")
        return self._save(frame.frame_index, {"__mode__": "rough_single_side", **positions})

    def reset(self, index):
        return self._save(index, {})

    def undo(self, index):
        history = self.history.get(index, [])
        if not history:
            return False
        self.overrides[index] = history.pop()
        self.revision += 1
        return True

    def apply(self, frame):
        values = self.overrides.get(frame.frame_index, {})
        if not values:
            return frame
        mode = values.get("__mode__", "detected")
        landmarks = frame.landmarks.copy() if mode == "detected" else values.get("__estimate__", {}).copy()
        for name, xy in values.items():
            if name.startswith("__"):
                continue
            x, y = xy
            landmarks[name] = replace(landmarks[name], x=x, y=y) if name in landmarks else Landmark(x, y)
        return replace(frame, landmarks=landmarks, manual_landmarks=tuple(sorted(name for name in values if not name.startswith("__"))),
                       manual_pose_mode=mode)

    def metadata(self, frames):
        records, modes = [], []
        for frame in frames:
            values = self.overrides.get(frame.frame_index, {})
            if "__mode__" in values:
                modes.append(dict(frame_index=frame.frame_index, mode=values["__mode__"],
                                  source_timestamp=frame.source_timestamp,
                                  guide_anchors=values.get("__anchors__"),
                                  estimated_landmarks={n: asdict(p) for n, p in values.get("__estimate__", {}).items()}))
            for name, xy in sorted(values.items()):
                if name.startswith("__"):
                    continue
                raw = frame.landmarks.get(name)
                records.append(dict(frame_index=frame.frame_index, source_timestamp=frame.source_timestamp,
                                    landmark=name, raw_xy=[raw.x, raw.y] if raw else None, corrected_xy=list(xy)))
        return {"manual_pose_corrections": records, "manual_pose_replacements": modes,
                "correction_revision": self.revision,
                "correction_coordinates": "normalized oriented-image XY; rough points have no inferred Z or detector confidence"}

def corrections_for(clip):
    if not hasattr(clip, "corrections"):
        clip.corrections = PoseCorrections()
    return clip.corrections
