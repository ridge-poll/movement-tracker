"""Decode in presentation order with real PTS and display rotation."""
from dataclasses import dataclass
from pathlib import Path
import av
import numpy as np
from movement_tracker.geometry import rotate_image

@dataclass
class DecodedFrame:
    rgb: np.ndarray
    index: int
    timestamp: float
    source_timestamp: float | None
    estimated: bool
    duration: float

class VideoReader:
    def __init__(self, path: Path):
        self.container = av.open(str(path))
        if not self.container.streams.video:
            self.container.close()
            raise ValueError("This file has no video stream.")
        self.stream = self.container.streams.video[0]
        self.stream.codec_context.thread_count = 2
        self.rate = self.stream.average_rate or self.stream.guessed_rate
        self.total_hint = self.stream.frames or 0
        self.fps = float(self.rate) if self.rate else 0.0

    def __iter__(self):
        origin = None
        previous = None
        step = 1 / self.fps if self.fps > 0 else None
        for index, frame in enumerate(self.container.decode(self.stream)):
            raw = float(frame.pts * frame.time_base) if frame.pts is not None else None
            estimated = raw is None
            if raw is None:
                if step is None:
                    raise ValueError("Video has neither usable timestamps nor frame rate.")
                timestamp = 0.0 if previous is None else previous + step
            else:
                if origin is None:
                    origin = raw - (0 if previous is None else previous + (step or 0))
                timestamp = raw - origin
                if previous is not None and timestamp <= previous:
                    if step is None:
                        raise ValueError("Non-increasing video timestamps and no frame rate.")
                    timestamp = previous + step
                    estimated = True
            previous = timestamp
            duration = float(frame.duration * frame.time_base) if frame.duration else (step or 0.033333)
            rgb = frame.to_ndarray(format="rgb24")
            rotation = float(frame.rotation)
            turns = round(rotation / 90)
            if abs(rotation - turns * 90) > 0.1:
                raise ValueError(f"Unsupported display rotation {rotation}°; use a standard 90° orientation.")
            if turns % 4:
                rgb = rotate_image(rgb, (turns * 90) % 360)
            yield DecodedFrame(rgb, index, timestamp, raw, estimated, duration)

    def close(self):
        self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
