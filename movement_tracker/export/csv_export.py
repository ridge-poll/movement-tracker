import csv
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from movement_tracker.pose_data import LANDMARK_NAMES

FIELDS = ("x", "y", "z", "confidence", "visibility", "presence")

def export_csv(frames, path):
    path = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=path.parent,
                                suffix=".csv", delete=False) as file:
            temporary = Path(file.name)
            writer = csv.writer(file)
            writer.writerow(["timestamp", "frame", "width", "height", "source_timestamp",
                             "timestamp_estimated", "pose_detected"] +
                            [f"{name}_{field}" for name in LANDMARK_NAMES for field in FIELDS])
            for pose in frames:
                row = [pose.timestamp, pose.frame_index, pose.width, pose.height,
                       pose.source_timestamp, int(pose.timestamp_estimated), int(bool(pose.landmarks))]
                for name in LANDMARK_NAMES:
                    landmark = pose.landmarks.get(name)
                    row.extend(getattr(landmark, field) if landmark else None for field in FIELDS)
                writer.writerow(row)
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
