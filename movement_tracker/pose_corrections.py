"""Session-local XY overrides. Raw landmarks and detector confidence never change."""
from dataclasses import replace
import math

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
        for name, (x, y) in positions.items():
            if name not in frame.landmarks:
                raise ValueError("Only existing landmarks can be moved")
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y)):
                raise ValueError("Coordinates must lie inside the image")
            raw = frame.landmarks[name]
            if (x, y) == (raw.x, raw.y):
                values.pop(name, None)
            else:
                values[name] = (float(x), float(y))
        return self._save(frame.frame_index, values)

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
        landmarks = frame.landmarks.copy()
        for name, (x, y) in values.items():
            landmarks[name] = replace(landmarks[name], x=x, y=y)
        return replace(frame, landmarks=landmarks, manual_landmarks=tuple(sorted(values)))

    def metadata(self, frames):
        records = []
        for frame in frames:
            for name, (x, y) in sorted(self.overrides.get(frame.frame_index, {}).items()):
                raw = frame.landmarks[name]
                records.append(dict(frame_index=frame.frame_index, source_timestamp=frame.source_timestamp,
                                    landmark=name, raw_xy=[raw.x, raw.y], corrected_xy=[x, y]))
        return {"manual_pose_corrections": records, "correction_revision": self.revision,
                "correction_coordinates": "normalized oriented-image XY; Z and detector confidence unchanged"}

def corrections_for(clip):
    if not hasattr(clip, "corrections"):
        clip.corrections = PoseCorrections()
    return clip.corrections
