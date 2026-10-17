from pathlib import Path
from time import perf_counter
import numpy as np
from movement_tracker.pose_data import LANDMARK_NAMES, Landmark, PoseFrame

def prominence(landmarks) -> float:
    """Largest visible bounding box; no identity tracking across people."""
    visible = [p for p in landmarks if (p.visibility or 0) >= 0.5]
    if not visible:
        return 0.0
    xs = [min(1, max(0, p.x)) for p in visible]
    ys = [min(1, max(0, p.y)) for p in visible]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))

class MediaPipeTracker:
    def __init__(self, model_path: Path):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self._mp = mp
        self._last_ms = -1
        self._model = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_path),
                                                delegate=python.BaseOptions.Delegate.CPU),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=3,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

    def process_frame(self, rgb, timestamp, frame_index):
        # Only the backend clock is rounded/adjusted; source times stay intact.
        milliseconds = max(self._last_ms + 1, round(timestamp * 1000))
        self._last_ms = milliseconds
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        start = perf_counter()
        result = self._model.detect_for_video(image, milliseconds)
        elapsed = perf_counter() - start
        landmarks = {}
        if result.pose_landmarks:
            selected = max(result.pose_landmarks, key=prominence)
            landmarks = {
                name: Landmark(p.x, p.y, p.z, p.visibility, p.presence)
                for name, p in zip(LANDMARK_NAMES, selected)
            }
        h, w = rgb.shape[:2]
        return PoseFrame(timestamp, frame_index, w, h, landmarks, inference_seconds=elapsed)

    def close(self):
        self._model.close()
