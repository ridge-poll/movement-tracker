"""Four independent orientation hypotheses, with original measurements retained."""
from dataclasses import dataclass, replace
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Callable
import hashlib
from importlib.metadata import version
import cv2
import numpy as np
from movement_tracker.pose_data import PoseFrame
from movement_tracker.tracking.pose_tracker import PoseTracker
from movement_tracker.tracking.mediapipe_tracker import MediaPipeTracker
from movement_tracker.video.processing import Cancelled
from movement_tracker.video.video_reader import VideoReader

ANGLES = (0, 90, 180, 270)  # Counterclockwise input rotation, after display orientation.

def rotate_input(rgb: np.ndarray, angle: int) -> np.ndarray:
    if angle not in ANGLES:
        raise ValueError("Rotation must be 0, 90, 180, or 270 degrees counterclockwise.")
    return np.ascontiguousarray(np.rot90(rgb, angle // 90))

def to_source(pose: PoseFrame, angle: int, width: int, height: int) -> PoseFrame:
    """Inverse continuous normalized image rotation; anatomical names never swap.

    z is not metric 3D. Its width-normalized scale is converted to source-width
    units; its original value is always preserved in RotationCandidate.input_pose.
    """
    if angle not in ANGLES:
        raise ValueError("Unsupported rotation")
    expected = (height, width) if angle in (90, 270) else (width, height)
    if (pose.width, pose.height) != expected:
        raise ValueError("Pose dimensions do not match the rotated source dimensions.")
    landmarks = {}
    for name, point in pose.landmarks.items():
        x, y = point.x, point.y
        if angle == 90:
            x, y = 1 - y, x
        elif angle == 180:
            x, y = 1 - x, 1 - y
        elif angle == 270:
            x, y = y, 1 - x
        z = point.z * pose.width / width if point.z is not None else None
        landmarks[name] = replace(point, x=x, y=y, z=z)
    return replace(pose, width=width, height=height, landmarks=landmarks)

@dataclass(frozen=True)
class RotationCandidate:
    angle_ccw: int
    input_pose: PoseFrame
    source_pose: PoseFrame
    provenance: str = "mediapipe_measurement_inverse_image_rotation"
    anatomy_correctness: str = "unreviewed"
    interpolated: bool = False

@dataclass(frozen=True)
class RotationFrame:
    frame_index: int
    timestamp: float
    candidates: tuple[RotationCandidate, ...]

@dataclass
class RotationExperiment:
    source: Path
    frames: list[RotationFrame]
    metadata: dict
    cache: TemporaryDirectory

    def preview(self, offset):
        index = self.frames[offset].frame_index
        bgr = cv2.imread(str(Path(self.cache.name) / f"{index:08d}.jpg"))
        if bgr is None:
            raise OSError("Comparison preview is unavailable; rerun the experiment.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cache.cleanup()

def run_rotations(source: Path, model: Path, start_frame: int, end_frame: int,
                  cancel: Event, progress=lambda *args: None, mode="video",
                  tracker_factory: Callable[[], PoseTracker] | None = None):
    """Inclusive frame range. VIDEO warms each independent stream from frame 0.

    IMAGE skips inference before the selected range and has no temporal state.
    The injectable factory keeps media/geometry tests independent of model access.
    """
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("Choose an inclusive frame range with 0 ≤ start ≤ end.")
    if mode not in ("video", "image"):
        raise ValueError("Unknown inference mode")
    source, model = Path(source).resolve(), Path(model).resolve()
    stat = source.stat()
    metadata = {
        "schema_version": 1, "source_path": str(source),
        "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
        "start_frame": start_frame, "end_frame": end_frame,
        "input_angles_ccw": list(ANGLES), "inference_mode": mode,
        "history_start_frame": 0 if mode == "video" else None,
        "independent_orientation_trackers": True,
        "model_path": str(model), "model_sha256": None,
        "backend": "mediapipe_tasks" if tracker_factory is None else "injected_test_backend",
        "mediapipe_version": version("mediapipe") if tracker_factory is None else None,
        "max_detected_people": 3,
        "person_selection": "largest_visible_landmark_bounding_box_per_orientation",
        "confidence_definition": "min(visibility, presence); not anatomy correctness",
        "mapped_z_definition": "input_z * rotated_input_width / source_width; not metric depth",
        "correction_applied": False,
    }
    if tracker_factory is None:
        metadata["model_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
        tracker_factory = lambda: MediaPipeTracker(model, mode=mode)
    cache = TemporaryDirectory(prefix="movement-rotations-")
    frames = []
    try:
        with ExitStack() as resources:
            trackers = []
            for angle in ANGLES:
                if cancel.is_set():
                    raise Cancelled()
                tracker = tracker_factory()
                resources.callback(tracker.close)
                trackers.append(tracker)
            reader = resources.enter_context(VideoReader(source))
            metadata["source_average_fps"] = reader.fps
            total = (end_frame + 1 if mode == "video" else end_frame - start_frame + 1) * 4
            count = 0
            for decoded in reader:
                if cancel.is_set():
                    raise Cancelled()
                if decoded.index > end_frame:
                    break
                if mode == "image" and decoded.index < start_frame:
                    continue
                candidates = []
                height, width = decoded.rgb.shape[:2]
                for angle, tracker in zip(ANGLES, trackers):
                    if cancel.is_set():
                        raise Cancelled()
                    raw = tracker.process_frame(rotate_input(decoded.rgb, angle),
                                                decoded.timestamp, decoded.index)
                    raw = replace(raw, source_timestamp=decoded.source_timestamp,
                                  timestamp_estimated=decoded.estimated)
                    candidates.append(RotationCandidate(angle, raw, to_source(raw, angle, width, height)))
                    count += 1
                    progress(count, total)
                if decoded.index < start_frame:
                    continue
                frames.append(RotationFrame(decoded.index, decoded.timestamp, tuple(candidates)))
                scale = min(1.0, 960 / max(width, height))
                preview = cv2.resize(decoded.rgb, (round(width * scale), round(height * scale)),
                                     interpolation=cv2.INTER_AREA) if scale < 1 else decoded.rgb
                if not cv2.imwrite(str(Path(cache.name) / f"{decoded.index:08d}.jpg"),
                                   cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise OSError("Cannot write comparison preview cache.")
            if cancel.is_set():
                raise Cancelled()
            if not frames or frames[-1].frame_index != end_frame:
                raise ValueError(f"Requested end frame {end_frame} is beyond the decodable clip; choose a smaller range.")
            after = source.stat()
            if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("The source changed during analysis; rerun with an unchanged file.")
        return RotationExperiment(source, frames, metadata, cache)
    except BaseException:
        cache.cleanup()
        raise
