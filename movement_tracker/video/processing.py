"""Sequential full-resolution inference with a bounded-memory preview cache."""
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import cv2
from movement_tracker.pose_data import PoseFrame
from movement_tracker.video.video_reader import VideoReader

class Cancelled(Exception):
    pass

@dataclass
class Clip:
    source: Path
    frames: list[PoseFrame]
    fps: float
    last_duration: float
    cache: TemporaryDirectory
    source_size: int
    source_mtime_ns: int

    def preview(self, index):
        bgr = cv2.imread(str(Path(self.cache.name) / f"{index:08d}.jpg"))
        if bgr is None:
            raise OSError("Could not read preview cache. Reopen the video.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cache.cleanup()

def process_video(path, tracker, cancel: Event, progress=lambda *args: None):
    path = Path(path)
    stat = path.stat()
    cache = TemporaryDirectory(prefix="movement-tracker-")
    frames = []
    try:
        with VideoReader(path) as reader:
            duration = 0.0
            for decoded in reader:
                if cancel.is_set():
                    raise Cancelled()
                pose = tracker.process_frame(decoded.rgb, decoded.timestamp, decoded.index)
                frames.append(replace(pose, source_timestamp=decoded.source_timestamp,
                                      timestamp_estimated=decoded.estimated))
                # Preview only: inference and export retain source resolution.
                h, w = decoded.rgb.shape[:2]
                scale = min(1.0, 960 / max(h, w))
                preview = cv2.resize(decoded.rgb, (round(w * scale), round(h * scale)),
                                     interpolation=cv2.INTER_AREA) if scale < 1 else decoded.rgb
                if not cv2.imwrite(str(Path(cache.name) / f"{decoded.index:08d}.jpg"),
                                   cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise OSError("Unable to write temporary preview cache (disk full?).")
                duration = decoded.duration
                progress(len(frames), reader.total_hint)
            if not frames:
                raise ValueError("No decodable video frames found.")
            if cancel.is_set():
                raise Cancelled()
            return Clip(path, frames, reader.fps, duration, cache, stat.st_size, stat.st_mtime_ns)
    except BaseException:
        cache.cleanup()
        raise
